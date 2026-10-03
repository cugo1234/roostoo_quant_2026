from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd

from .engine import run_backtest
from .metrics import performance_metrics


def sensitivity_grid(ohlcv: dict[str, pd.DataFrame], cfg: dict, grid: list[dict]) -> pd.DataFrame:
    rows = []
    for params in grid:
        c = deepcopy(cfg)
        for key, value in params.items():
            section, field = key.split(".", 1)
            c[section][field] = value
        result = run_backtest(ohlcv, c)
        row = dict(params)
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def walk_forward_splits(index: pd.Index, train_fraction: float = 0.60, validation_fraction: float = 0.20):
    n = len(index)
    a = int(n * train_fraction)
    b = int(n * (train_fraction + validation_fraction))
    return index[:a], index[a:b], index[b:]


def anchored_walk_forward_windows(index: pd.Index, initial_train_bars: int, test_bars: int, step_bars: int | None = None):
    """Yield anchored train/test indices without future leakage."""
    step = step_bars or test_bars
    train_end = initial_train_bars
    while train_end + test_bars <= len(index):
        train = index[:train_end]
        test = index[train_end:train_end + test_bars]
        yield train, test
        train_end += step


def slice_ohlcv(ohlcv: dict[str, pd.DataFrame], index: pd.Index) -> dict[str, pd.DataFrame]:
    return {k: v.loc[index] for k, v in ohlcv.items()}


def block_bootstrap_return_distribution(
    returns: pd.Series,
    horizon_bars: int,
    block_bars: int = 6,
    simulations: int = 1000,
    seed: int = 7,
) -> pd.DataFrame:
    """Approximate uncertainty using a moving-block bootstrap of realized strategy returns.

    This is not a substitute for real rolling tournament windows; it preserves
    short serial dependence better than IID resampling and is used only as a
    supplementary uncertainty diagnostic.
    """
    x = returns.dropna().to_numpy(float)
    if len(x) < max(horizon_bars, block_bars):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(x) - block_bars + 1)
    rows = []
    for _ in range(simulations):
        sample = []
        while len(sample) < horizon_bars:
            s = int(rng.choice(starts))
            sample.extend(x[s:s+block_bars])
        arr = np.asarray(sample[:horizon_bars])
        equity = np.cumprod(1 + arr)
        rows.append({
            "total_return": float(equity[-1] - 1),
            "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
        })
    return pd.DataFrame(rows)
