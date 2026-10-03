import numpy as np
import pandas as pd
import yaml

from roostoo_quant.backtest.engine import run_backtest


def synthetic_ohlcv(n=700, k=6):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"][:k]
    common = rng.normal(0.0002, 0.012, n)
    close = {}
    for j, s in enumerate(syms):
        idio = rng.normal(0, .006 + j*.0007, n)
        trend = np.sin(np.arange(n) / (40 + j*3)) * 0.001
        rets = .75 * common + idio + trend
        close[s] = 100 * np.exp(np.cumsum(rets))
    close = pd.DataFrame(close, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    wiggle = pd.DataFrame(rng.uniform(.001, .01, size=close.shape), index=idx, columns=close.columns)
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    volume = pd.DataFrame(rng.lognormal(10, .5, size=close.shape), index=idx, columns=close.columns)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def test_backtest_runs_and_no_leverage():
    cfg = yaml.safe_load(open("config/default.yaml"))
    r = run_backtest(synthetic_ohlcv(), cfg)
    assert len(r.returns) > 0
    assert r.positions.abs().sum(axis=1).max() <= 1.000001
    assert (r.returns["trading_cost"] >= 0).all()
