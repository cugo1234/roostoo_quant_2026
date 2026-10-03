import numpy as np
import pandas as pd

from roostoo_quant.signals.correlation_regime import teammate_cov_corr_feature
from roostoo_quant.backtest.metrics import max_drawdown


def test_cov_corr_identity_equal_window():
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(size=500))
    y = 0.7 * x + pd.Series(rng.normal(scale=0.5, size=500))
    cov = x.rolling(50).cov(y)
    corr = x.rolling(50).corr(y)
    lhs = cov / corr
    rhs = x.rolling(50).std() * y.rolling(50).std()
    mask = lhs.notna() & rhs.notna()
    assert np.allclose(lhs[mask], rhs[mask], rtol=1e-10, atol=1e-12)


def test_teammate_feature_has_expected_columns():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"BTCUSDT": rng.normal(size=200), "ETHUSDT": rng.normal(size=200)})
    f = teammate_cov_corr_feature(df, cov_window=20, corr_window=85)
    assert list(f.columns) == ["ETHUSDT"]
    assert f["ETHUSDT"].notna().sum() > 0


def test_max_drawdown():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert abs(max_drawdown(equity) + 0.25) < 1e-12
