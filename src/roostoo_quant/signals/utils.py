from __future__ import annotations
import numpy as np
import pandas as pd


def cross_sectional_rank_score(df: pd.DataFrame) -> pd.DataFrame:
    """Map cross-sectional percentile ranks into [-1, 1], preserving NaNs."""
    return df.rank(axis=1, pct=True, method="average") * 2.0 - 1.0


def rolling_zscore(df: pd.DataFrame, window: int, clip: float | None = 3.0) -> pd.DataFrame:
    mean = df.rolling(window, min_periods=window).mean()
    std = df.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    z = (df - mean) / std
    return z.clip(-clip, clip) if clip is not None else z
