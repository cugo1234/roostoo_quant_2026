from __future__ import annotations
import numpy as np
import pandas as pd


def realized_vol(prices: pd.DataFrame, bars: int = 42) -> pd.DataFrame:
    return prices.pct_change().rolling(bars, min_periods=bars).std(ddof=0)


def inverse_vol_weights(vol: pd.Series, cap: float = 0.25) -> pd.Series:
    inv = 1.0 / vol.replace(0, np.nan)
    inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
    if inv.empty or inv.sum() <= 0:
        return pd.Series(dtype=float)
    w = inv / inv.sum()
    # Iterative cap-and-redistribute.
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w.loc[over] = cap
        under = ~over
        if under.any() and w[under].sum() > 0:
            w.loc[under] += excess * w[under] / w[under].sum()
    return w / w.sum()
