from __future__ import annotations
import pandas as pd
from .momentum import blended_momentum, time_series_momentum, returns_over
from .breakout import cross_sectional_breakout


def core_ensemble(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame | pd.Series]:
    s = cfg["strategy"]
    cs = blended_momentum(
        close,
        bars=(s["momentum_24_bars"], s["momentum_72_bars"], s["momentum_168_bars"]),
        weights=(0.2, 0.5, 0.3),
    )
    ts = time_series_momentum(close, s["momentum_72_bars"])
    bo = cross_sectional_breakout(high, low, close, s["breakout_bars"])
    sw = s["score_weights"]
    score = sw["cs_mom"] * cs + sw["ts_mom"] * ts + sw["breakout"] * bo
    btc_mom = returns_over(close[["BTCUSDT"]], s["momentum_72_bars"])["BTCUSDT"]
    breadth = (returns_over(close, s["momentum_72_bars"]) > 0).mean(axis=1) * 2.0 - 1.0
    return {"score": score, "cs_mom": cs, "ts_mom": ts, "breakout": bo, "btc_mom": btc_mom, "breadth": breadth}
