from __future__ import annotations

import logging
import os
from pathlib import Path
import time

import pandas as pd
from dotenv import load_dotenv

from roostoo_quant.config import load_yaml, env_bool
from roostoo_quant.data.market import BinanceLiveKlines
from roostoo_quant.execution.executor import ExecutionEngine
from roostoo_quant.execution.planner import (
    current_weights,
    extract_trade_rules,
    plan_rebalance,
    portfolio_nav,
)
from roostoo_quant.portfolio.construct import (
    gross_from_stress,
    long_budget_from_regime,
    target_weights_from_score,
)
from roostoo_quant.roostoo.client import RoostooClient
from roostoo_quant.signals.correlation_regime import market_stress
from roostoo_quant.signals.ensemble import core_ensemble
from roostoo_quant.signals.volatility import realized_vol
from .state import load_state, save_state


LOG = logging.getLogger("roostoo_quant.live")


def roostoo_to_binance(pair: str) -> str:
    coin = pair.split("/")[0]
    return f"{coin}USDT"


def binance_to_roostoo(symbol: str) -> str:
    return f"{symbol.removesuffix('USDT')}/USD"


def _build_live_panels(pairs: list[str], interval: str, limit: int = 240) -> dict[str, pd.DataFrame]:
    frames = {}
    source = BinanceLiveKlines()
    for pair in pairs:
        sym = roostoo_to_binance(pair)
        try:
            df = source.get(sym, interval=interval, limit=limit + 1)
            # Binance may return the currently forming candle. Drop the newest
            # row so signals only use completed bars.
            if len(df) > 1:
                df = df.iloc[:-1].copy()
            frames[sym] = df.set_index("timestamp")
        except Exception as exc:  # live system skips unavailable external symbols safely
            LOG.warning("Binance data unavailable for %s: %s", pair, exc)
    if len(frames) < 2:
        raise RuntimeError("Fewer than two tradable pairs have usable market data")
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    panels = {}
    for field in ["open", "high", "low", "close", "volume"]:
        panels[field] = pd.DataFrame({sym: df.loc[common, field] for sym, df in frames.items()})
    return panels


def compute_latest_target(panels: dict[str, pd.DataFrame], cfg: dict, previous: pd.Series | None = None) -> pd.Series:
    close, high, low = panels["close"], panels["high"], panels["low"]
    sig = core_ensemble(close, high, low, cfg)
    vol = realized_vol(close, cfg["strategy"]["vol_bars"])
    stress = market_stress(close, cfg["strategy"]["corr_short_bars"], cfg["strategy"]["corr_long_bars"])
    ts = close.index[-1]
    s = cfg["strategy"]
    lb = long_budget_from_regime(float(sig["btc_mom"].loc[ts]), float(sig["breadth"].loc[ts]), s["min_long_budget"], s["max_long_budget"])
    gross = gross_from_stress(float(stress.loc[ts]), s["normal_gross"], s["stressed_gross"], s["severe_gross"])
    return target_weights_from_score(
        sig["score"].loc[ts], vol.loc[ts], gross, lb, s["top_k"], s["bottom_k"], s["max_asset_weight"],
        previous=previous, rank_buffer=s.get("hysteresis_rank_buffer", 0)
    )


def _interval_seconds(interval: str) -> int:
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    raise ValueError(f"Unsupported interval for cadence: {interval}")


def _rebalance_due(ts: pd.Timestamp, interval: str, every_bars: int) -> bool:
    sec = _interval_seconds(interval)
    bar_no = int(ts.timestamp()) // sec
    return bar_no % max(1, every_bars) == max(1, every_bars) - 1

def ensure_competition_trading_confirmed() -> None:
    """Require an explicit opt-in before sending competition orders."""
    load_dotenv()
    env = os.getenv("ROOSTOO_ENV", "test").strip().lower()
    confirmation = os.getenv("ROOSTOO_LIVE_TRADING_CONFIRM", "").strip()
    live_trading = env_bool("LIVE_TRADING", default=False)
    if env == "competition" and live_trading and confirmation != "YES":
        raise RuntimeError(
            "Competition trading blocked. Set ROOSTOO_LIVE_TRADING_CONFIRM=YES explicitly."
        )


def run_once(cfg_path: str = "config/default.yaml") -> dict:
    load_dotenv()
    ensure_competition_trading_confirmed()
    cfg = load_yaml(cfg_path)
    api_key = os.environ.get("ROOSTOO_API_KEY", "")
    secret = os.environ.get("ROOSTOO_SECRET_KEY", "")
    if not api_key or not secret:
        raise RuntimeError("ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY must be set")
    dry_run = env_bool("LIVE_TRADING", default=False) is False
    client = RoostooClient(api_key, secret, os.environ.get("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com"))
    info = client.exchange_info()
    rules = extract_trade_rules(info)
    tickers = client.ticker()
    bal = client.balance()
    shorts = client.short_positions()
    nav = portfolio_nav(bal, tickers, shorts)
    current = current_weights(bal, tickers, shorts, nav)

    # Research uses 4h Binance spot candles; only pairs dynamically reported as
    # tradable by Roostoo are considered live.
    panels = _build_live_panels(list(rules), cfg["research"]["interval"], limit=240)
    last_bar = panels["close"].index[-1]
    state_path = Path("logs/state.json")
    state = load_state(state_path)
    force = env_bool("FORCE_REBALANCE", default=False)
    due = _rebalance_due(last_bar, cfg["research"]["interval"], int(cfg["strategy"]["rebalance_every_bars"]))
    if not force and (not due or state.get("last_rebalanced_bar") == str(last_bar)):
        return {"nav": nav, "dry_run": dry_run, "status": "no_rebalance_due", "last_completed_bar": str(last_bar)}

    prev_binance = pd.Series({roostoo_to_binance(k): v for k, v in current.items()})
    target = compute_latest_target(panels, cfg, previous=prev_binance)
    target.index = [binance_to_roostoo(x) if x.endswith("USDT") else x for x in target.index]
    target = target.reindex(list(rules)).fillna(0.0)

    orders = plan_rebalance(target, current, nav, tickers, rules, cfg["strategy"]["min_trade_weight"])
    # Commit the rebalance bar before sending orders. A crash can therefore
    # under-trade but cannot automatically duplicate the entire rebalance.
    state.update({
        "last_rebalanced_bar": str(last_bar),
        "last_target": target.to_dict(),
        "last_planned_orders": [o.__dict__ for o in orders],
        "last_plan_epoch": int(time.time()),
    })
    save_state(state_path, state)
    engine = ExecutionEngine(client, "logs/trade_ledger.jsonl", dry_run=dry_run)
    responses = engine.execute(orders)
    state["last_responses"] = responses
    state["last_success_epoch"] = int(time.time())
    save_state(state_path, state)
    return {"nav": nav, "target": target.to_dict(), "current": current.to_dict(), "orders": [o.__dict__ for o in orders], "responses": responses, "dry_run": dry_run, "last_completed_bar": str(last_bar)}


def run_forever(cfg_path: str = "config/default.yaml") -> None:
    load_dotenv()
    ensure_competition_trading_confirmed()
    cfg = load_yaml(cfg_path)
    poll = int(cfg["execution"]["poll_seconds"])
    while True:
        try:
            result = run_once(cfg_path)
            LOG.info("cycle complete: %s", result)
        except Exception:
            LOG.exception("cycle failed safely")
        time.sleep(poll)
