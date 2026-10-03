from __future__ import annotations

import pandas as pd

from roostoo_quant.signals.breakout import cross_sectional_breakout
from roostoo_quant.signals.momentum import cross_sectional_momentum, time_series_momentum
from roostoo_quant.signals.reversal import short_term_reversal
from roostoo_quant.signals.ensemble import core_ensemble
from roostoo_quant.signals.correlation_regime import market_stress


def candidate_scores(ohlcv: dict[str, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    close, high, low = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    s = cfg["strategy"]
    ensemble = core_ensemble(close, high, low, cfg)["score"]
    return {
        "time_series_momentum_72h": time_series_momentum(close, s["momentum_72_bars"]),
        "cross_sectional_momentum_72h": cross_sectional_momentum(close, s["momentum_72_bars"]),
        "breakout_80h": cross_sectional_breakout(high, low, close, s["breakout_bars"]),
        "short_term_reversal_4h": short_term_reversal(close, 1),
        "multi_factor_ensemble": ensemble,
    }


def risk_regime(ohlcv: dict[str, pd.DataFrame], cfg: dict) -> pd.Series:
    return market_stress(ohlcv["close"], cfg["strategy"]["corr_short_bars"], cfg["strategy"]["corr_long_bars"])
