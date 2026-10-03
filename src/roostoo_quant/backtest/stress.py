from __future__ import annotations

from copy import deepcopy
import pandas as pd


def inject_market_shock(ohlcv: dict[str, pd.DataFrame], at: pd.Timestamp, shock: float = -0.20) -> dict[str, pd.DataFrame]:
    """Apply a one-bar common market gap from `at` onward for stress testing."""
    out = {k: v.copy() for k, v in ohlcv.items()}
    factor = 1.0 + shock
    mask = out["close"].index >= at
    for field in ["open", "high", "low", "close"]:
        out[field].loc[mask] = out[field].loc[mask] * factor
    return out


def inject_stale_asset(ohlcv: dict[str, pd.DataFrame], symbol: str, start: pd.Timestamp, bars: int = 6) -> dict[str, pd.DataFrame]:
    out = {k: v.copy() for k, v in ohlcv.items()}
    idx = out["close"].index
    pos = idx.get_indexer([start], method="nearest")[0]
    stale_idx = idx[pos:pos+bars]
    for field in ["open", "high", "low", "close", "volume"]:
        if symbol in out[field].columns:
            out[field].loc[stale_idx, symbol] = pd.NA
    return out
