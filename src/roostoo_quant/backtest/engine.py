from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from roostoo_quant.backtest.metrics import performance_metrics
from roostoo_quant.portfolio.construct import (
    apply_deadband,
    gross_from_stress,
    long_budget_from_regime,
    target_weights_from_score,
)
from roostoo_quant.signals.correlation_regime import market_stress
from roostoo_quant.signals.ensemble import core_ensemble
from roostoo_quant.signals.volatility import realized_vol


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    targets: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]
    signals: dict[str, pd.DataFrame | pd.Series]


def _next_open_returns(open_px: pd.DataFrame) -> pd.DataFrame:
    return open_px.shift(-1) / open_px - 1.0


def build_targets(ohlcv: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, dict]:
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    s = cfg["strategy"]
    sig = core_ensemble(close, high, low, cfg)
    score = sig["score"]
    vol = realized_vol(close, s["vol_bars"])
    stress = market_stress(close, s["corr_short_bars"], s["corr_long_bars"])
    sig["volatility"] = vol
    sig["stress"] = stress

    targets = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    prev = pd.Series(0.0, index=close.columns)
    rebalance = int(s["rebalance_every_bars"])

    for i, ts in enumerate(close.index):
        if i % rebalance != 0:
            targets.loc[ts] = prev
            continue
        row = score.loc[ts]
        vrow = vol.loc[ts]
        if row.notna().sum() < 2 or vrow.notna().sum() < 2:
            targets.loc[ts] = prev
            continue
        lb = long_budget_from_regime(
            float(sig["btc_mom"].loc[ts]) if ts in sig["btc_mom"].index else np.nan,
            float(sig["breadth"].loc[ts]) if ts in sig["breadth"].index else np.nan,
            s["min_long_budget"],
            s["max_long_budget"],
        )
        gross = gross_from_stress(float(stress.loc[ts]), s["normal_gross"], s["stressed_gross"], s["severe_gross"])
        desired = target_weights_from_score(
            row,
            vrow,
            gross=gross,
            long_budget_fraction=lb,
            top_k=s["top_k"],
            bottom_k=s["bottom_k"],
            max_asset_weight=s["max_asset_weight"],
            previous=prev,
            rank_buffer=s.get("hysteresis_rank_buffer", 0),
        )
        desired = apply_deadband(desired, prev, s["min_trade_weight"])
        targets.loc[ts] = desired
        prev = desired
    return targets, sig


def run_backtest(ohlcv: dict[str, pd.DataFrame], cfg: dict, targets: pd.DataFrame | None = None) -> BacktestResult:
    open_px = ohlcv["open"].astype(float)
    close = ohlcv["close"].astype(float)
    if targets is None:
        targets, sig = build_targets(ohlcv, cfg)
    else:
        sig = {}

    # Signals use candle t close; position can only start at candle t+1 open.
    positions = targets.shift(int(cfg["research"].get("signal_delay_bars", 1))).fillna(0.0)
    positions = positions.reindex(index=open_px.index, columns=open_px.columns).fillna(0.0)

    fwd = _next_open_returns(open_px).fillna(0.0)
    gross_ret = (positions * fwd).sum(axis=1)
    delta = positions.diff().fillna(positions)
    turnover_by_asset = delta.abs()
    turnover = turnover_by_asset.sum(axis=1)
    gross_exposure = positions.abs().sum(axis=1)

    # Conservative default: all simulated fills are taker fills. Short fee docs
    # are also 10 bps, so a common turnover fee is defensible until organizer
    # clarification. Slippage is charged on each unit of turnover.
    fee = float(cfg["competition"]["taker_fee"])
    slip = float(cfg["research"].get("slippage_bps", 0.0)) / 10000.0
    trading_cost = turnover * (fee + slip)
    net_ret = gross_ret - trading_cost
    equity = (1.0 + net_ret).cumprod()
    dd = equity / equity.cummax() - 1.0

    # Order ledger: one row per changed asset target.
    records: list[dict] = []
    for ts in delta.index:
        changed = delta.loc[ts]
        for sym, d in changed[changed.abs() > 1e-12].items():
            records.append({
                "timestamp": ts,
                "symbol": sym,
                "delta_weight": float(d),
                "target_weight": float(positions.loc[ts, sym]),
                "reference_open": float(open_px.loc[ts, sym]) if pd.notna(open_px.loc[ts, sym]) else np.nan,
                "estimated_cost_fraction": abs(float(d)) * (fee + slip),
            })
    trades = pd.DataFrame(records)

    detail = pd.DataFrame({
        "gross_return": gross_ret,
        "turnover": turnover,
        "trading_cost": trading_cost,
        "net_return": net_ret,
        "equity": equity,
        "drawdown": dd,
        "gross_exposure": gross_exposure,
        "net_exposure": positions.sum(axis=1),
    })
    metrics = performance_metrics(
        net_ret.iloc[:-1],
        turnover=turnover.iloc[:-1],
        exposure=gross_exposure.iloc[:-1],
        transaction_costs=trading_cost.iloc[:-1],
        annual_periods=int(cfg["research"]["annual_periods"]),
        trades=len(trades),
    )
    return BacktestResult(detail, targets, positions, trades, metrics, sig)
