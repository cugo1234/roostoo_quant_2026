from __future__ import annotations
import pandas as pd
from .utils import cross_sectional_rank_score


def short_term_reversal(prices: pd.DataFrame, bars: int = 1) -> pd.DataFrame:
    """Contrarian score. Research candidate only; fees often dominate at short horizons."""
    ret = prices / prices.shift(bars) - 1.0
    return -cross_sectional_rank_score(ret)
