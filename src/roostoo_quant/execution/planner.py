from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import pandas as pd


@dataclass(frozen=True)
class PlannedOrder:
    pair: str
    action: str  # BUY, SELL, SHORT_OPEN, SHORT_CLOSE
    quantity: float | None = None
    collateral: float | None = None
    close_pct: float | None = None
    reason: str = "rebalance"

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


def floor_to_precision(value: float, precision: int) -> float:
    quantum = Decimal("1").scaleb(-precision)
    return float(Decimal(str(max(value, 0.0))).quantize(quantum, rounding=ROUND_DOWN))


def extract_trade_rules(exchange_info: dict) -> dict[str, dict]:
    return {
        pair: info
        for pair, info in exchange_info.get("TradePairs", {}).items()
        if info.get("CanTrade", False)
    }


def portfolio_nav(balance: dict, tickers: dict, shorts: dict) -> float:
    """Estimate NAV without double-counting short collateral.

    Assumption: short collateral is reflected in USD Lock. Therefore we add only
    unrealized short P&L, not PositionValue. This assumption is documented and
    should be reconciled against the competition account before live deployment.
    """
    wallet = balance.get("Wallet", {})
    usd = wallet.get("USD", {})
    nav = float(usd.get("Free", 0.0)) + float(usd.get("Lock", 0.0))
    data = tickers.get("Data", tickers)
    for coin, b in wallet.items():
        if coin == "USD":
            continue
        qty = float(b.get("Free", 0.0)) + float(b.get("Lock", 0.0))
        pair = f"{coin}/USD"
        px = float(data.get(pair, {}).get("LastPrice", 0.0) or 0.0)
        nav += qty * px
    for p in shorts.get("Positions", []) or []:
        nav += float(p.get("UnrealizedPNL", 0.0))
    return nav


def current_weights(balance: dict, tickers: dict, shorts: dict, nav: float) -> pd.Series:
    data = tickers.get("Data", tickers)
    out: dict[str, float] = {}
    if nav <= 0:
        return pd.Series(dtype=float)
    wallet = balance.get("Wallet", {})
    for coin, b in wallet.items():
        if coin == "USD":
            continue
        pair = f"{coin}/USD"
        px = float(data.get(pair, {}).get("LastPrice", 0.0) or 0.0)
        qty = float(b.get("Free", 0.0)) + float(b.get("Lock", 0.0))
        out[pair] = out.get(pair, 0.0) + qty * px / nav
    for p in shorts.get("Positions", []) or []:
        pair = p["Pair"]
        px = float(data.get(pair, {}).get("LastPrice", p.get("CurrentPrice", 0.0)) or 0.0)
        notional = float(p.get("ShortQty", 0.0)) * px
        out[pair] = out.get(pair, 0.0) - notional / nav
    return pd.Series(out, dtype=float)


def plan_rebalance(
    target_weights: pd.Series,
    current: pd.Series,
    nav: float,
    tickers: dict,
    trade_rules: dict[str, dict],
    min_weight_change: float = 0.03,
) -> list[PlannedOrder]:
    """Create a conservative close-first rebalance plan.

    It closes sign-opposite exposure before opening new exposure. Exact short
    resizing is expressed as close+reopen when reducing, because Roostoo shorts
    are collateral-sized on open and quantity-sized/percentage-sized on close.
    """
    data = tickers.get("Data", tickers)
    pairs = sorted(set(target_weights.index) | set(current.index))
    closes: list[PlannedOrder] = []
    opens: list[PlannedOrder] = []

    for pair in pairs:
        if pair not in trade_rules or pair not in data:
            continue
        tw = float(target_weights.get(pair, 0.0))
        cw = float(current.get(pair, 0.0))
        if abs(tw - cw) < min_weight_change:
            continue
        px = float(data[pair].get("LastPrice", 0.0) or 0.0)
        if px <= 0 or nav <= 0:
            continue
        rule = trade_rules[pair]
        prec = int(rule.get("AmountPrecision", 6))
        min_order = float(rule.get("MiniOrder", 1.0))

        # Close sign conflicts first.
        if cw > 0 and tw <= 0:
            qty = floor_to_precision(cw * nav / px, prec)
            if qty * px >= min_order and qty > 0:
                closes.append(PlannedOrder(pair, "SELL", quantity=qty, reason="close_long"))
            cw = 0.0
        elif cw < 0 and tw >= 0:
            closes.append(PlannedOrder(pair, "SHORT_CLOSE", close_pct=100.0, reason="close_short"))
            cw = 0.0

        # Same-side resize or fresh open.
        delta = tw - cw
        if tw > 0:
            qty = floor_to_precision(abs(delta) * nav / px, prec)
            if qty * px >= min_order and qty > 0:
                action = "BUY" if delta > 0 else "SELL"
                opens.append(PlannedOrder(pair, action, quantity=qty, reason="resize_long"))
        elif tw < 0:
            if cw < 0 and abs(tw) < abs(cw):
                # Reduce by percentage of current notional.
                reduction = min(100.0, 100.0 * (abs(cw) - abs(tw)) / max(abs(cw), 1e-12))
                if reduction > 0.5:
                    opens.append(PlannedOrder(pair, "SHORT_CLOSE", close_pct=reduction, reason="reduce_short"))
            elif delta < 0:
                collateral = abs(delta) * nav
                if collateral >= max(1.0, min_order):
                    opens.append(PlannedOrder(pair, "SHORT_OPEN", collateral=collateral, reason="increase_short"))
    return closes + opens
