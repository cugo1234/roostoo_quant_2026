from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd

from roostoo_quant.backtest.engine import run_backtest
from roostoo_quant.portfolio.construct import (
    apply_deadband,
    gross_from_stress,
    long_budget_from_regime,
    target_weights_from_score,
)
from roostoo_quant.signals.correlation_regime import market_stress
from roostoo_quant.signals.momentum import returns_over
from roostoo_quant.signals.volatility import realized_vol


def targets_from_score(ohlcv: dict[str, pd.DataFrame], score: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    close = ohlcv["close"]
    s = cfg["strategy"]
    vol = realized_vol(close, s["vol_bars"])
    stress = market_stress(close, s["corr_short_bars"], s["corr_long_bars"])
    btc_mom = returns_over(close[["BTCUSDT"]], s["momentum_72_bars"])["BTCUSDT"]
    breadth = (returns_over(close, s["momentum_72_bars"]) > 0).mean(axis=1) * 2 - 1
    targets = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    prev = pd.Series(0.0, index=close.columns)
    rebalance = int(s["rebalance_every_bars"])
    for i, ts in enumerate(close.index):
        if i % rebalance:
            targets.loc[ts] = prev
            continue
        row, vrow = score.loc[ts], vol.loc[ts]
        if row.notna().sum() < 2 or vrow.notna().sum() < 2:
            targets.loc[ts] = prev
            continue
        lb = long_budget_from_regime(float(btc_mom.loc[ts]), float(breadth.loc[ts]), s["min_long_budget"], s["max_long_budget"])
        gross = gross_from_stress(float(stress.loc[ts]), s["normal_gross"], s["stressed_gross"], s["severe_gross"])
        desired = target_weights_from_score(
            row, vrow, gross, lb, s["top_k"], s["bottom_k"], s["max_asset_weight"],
            previous=prev, rank_buffer=s.get("hysteresis_rank_buffer", 0),
        )
        desired = apply_deadband(desired, prev, s["min_trade_weight"])
        targets.loc[ts] = desired
        prev = desired
    return targets


def evaluate_score(ohlcv: dict[str, pd.DataFrame], score: pd.DataFrame, cfg: dict):
    targets = targets_from_score(ohlcv, score, cfg)
    return run_backtest(ohlcv, cfg, targets=targets)


def candidate_config(base: dict, candidate: str) -> dict:
    c = deepcopy(base)
    # Candidate-specific cadence reflects its intended economic horizon rather
    # than tuning to the supplied sample.
    if candidate == "short_term_reversal_4h":
        c["strategy"]["rebalance_every_bars"] = 1
        c["strategy"]["min_trade_weight"] = 0.08
    elif candidate in {"time_series_momentum_72h", "cross_sectional_momentum_72h"}:
        c["strategy"]["rebalance_every_bars"] = 6
    elif candidate == "realized_volatility_rank_24h":
        c["strategy"]["rebalance_every_bars"] = 6
    return c
