from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import cross_sectional_rank_score


def breakout_location(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, bars: int = 20) -> pd.DataFrame:
    hi = high.rolling(bars, min_periods=bars).max()
    lo = low.rolling(bars, min_periods=bars).min()
    width = (hi - lo).replace(0, np.nan)
    loc = ((close - lo) / width) * 2.0 - 1.0
    return loc.clip(-1, 1)


def cross_sectional_breakout(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, bars: int = 20) -> pd.DataFrame:
    return cross_sectional_rank_score(breakout_location(high, low, close, bars))
