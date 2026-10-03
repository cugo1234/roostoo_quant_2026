from __future__ import annotations
import numpy as np
import pandas as pd


def btc_alt_corr_ratio(returns: pd.DataFrame, btc_symbol: str = "BTCUSDT", short: int = 18, long: int = 42) -> pd.DataFrame:
    """Per-alt ratio of recent BTC correlation to longer correlation.

    This is a decomposition-friendly version of the teammate's cov/corr idea.
    The raw cov/corr quantity is not a regression slope; with equal windows it
    reduces to sigma_BTC*sigma_alt. Here correlation change is kept explicit.
    """
    btc = returns[btc_symbol]
    out = {}
    for c in returns.columns:
        if c == btc_symbol:
            continue
        cshort = returns[c].rolling(short, min_periods=short).corr(btc)
        clong = returns[c].rolling(long, min_periods=long).corr(btc)
        out[c] = cshort / clong.replace(0, np.nan)
    return pd.DataFrame(out)


def teammate_cov_corr_feature(returns: pd.DataFrame, anchor: str = "BTCUSDT", cov_window: int = 20, corr_window: int = 85) -> pd.DataFrame:
    anchor_r = returns[anchor]
    out = {}
    for c in returns.columns:
        if c == anchor:
            continue
        cov = anchor_r.rolling(cov_window, min_periods=cov_window).cov(returns[c])
        corr = anchor_r.rolling(corr_window, min_periods=corr_window).corr(returns[c])
        out[c] = cov / corr.where(corr.abs() >= 0.10)
    return pd.DataFrame(out)


def market_stress(close: pd.DataFrame, short: int = 18, long: int = 42) -> pd.Series:
    """0..1 stress from median volatility expansion and mean pair correlation."""
    r = close.pct_change()
    short_vol = r.rolling(short, min_periods=short).std(ddof=0).median(axis=1)
    long_vol = r.rolling(long, min_periods=long).std(ddof=0).median(axis=1).replace(0, np.nan)
    vol_ratio = (short_vol / long_vol).clip(0.5, 2.5)

    pair_corrs = []
    cols = list(r.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair_corrs.append(r[cols[i]].rolling(short, min_periods=short).corr(r[cols[j]]))
    avgc = pd.concat(pair_corrs, axis=1).mean(axis=1) if pair_corrs else pd.Series(np.nan, index=r.index)
    stress = 0.5 * ((vol_ratio - 1.0) / 1.5).clip(0, 1) + 0.5 * ((avgc - 0.5) / 0.5).clip(0, 1)
    return stress.clip(0, 1)
