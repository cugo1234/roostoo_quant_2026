from __future__ import annotations
import pandas as pd
from .utils import cross_sectional_rank_score


def returns_over(prices: pd.DataFrame, bars: int) -> pd.DataFrame:
    return prices / prices.shift(bars) - 1.0


def time_series_momentum(prices: pd.DataFrame, bars: int = 18) -> pd.DataFrame:
    r = returns_over(prices, bars)
    # Smooth continuous direction rather than binary sign; cross-asset scale handled later.
    vol = prices.pct_change().rolling(bars, min_periods=bars).std(ddof=0).replace(0, pd.NA)
    return (r / (vol * (bars ** 0.5))).clip(-3, 3) / 3.0


def cross_sectional_momentum(prices: pd.DataFrame, bars: int = 18) -> pd.DataFrame:
    return cross_sectional_rank_score(returns_over(prices, bars))


def blended_momentum(prices: pd.DataFrame, bars=(6, 18, 42), weights=(0.2, 0.5, 0.3)) -> pd.DataFrame:
    out = 0.0
    for b, w in zip(bars, weights):
        out = out + w * cross_sectional_momentum(prices, b)
    return out
