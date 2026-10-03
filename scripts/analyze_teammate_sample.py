#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read_any(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="results/teammate_sample_diagnostics.json")
    args = ap.parse_args()
    df = read_any(args.path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")
    if not {"BTCUSDT", "ETHUSDT"}.issubset(df.columns):
        raise SystemExit("Expected BTCUSDT and ETHUSDT columns")
    br = df["BTCUSDT"].pct_change()
    er = df["ETHUSDT"].pct_change()
    w = 20
    cov = br.rolling(w).cov(er)
    corr = br.rolling(w).corr(er)
    lhs = cov / corr.replace(0, np.nan)
    rhs = br.rolling(w).std() * er.rolling(w).std()
    mask = lhs.notna() & rhs.notna() & (rhs.abs() > 0)
    rel_err = ((lhs[mask] - rhs[mask]).abs() / rhs[mask].abs()).max() if mask.any() else np.nan

    # Reproduce the submitted warmup chain.
    cov20 = br.rolling(20, min_periods=20).cov(er)
    corr85 = br.rolling(85, min_periods=85).corr(er)
    raw = (cov20 / corr85.where(corr85.abs() >= 0.10)).replace([np.inf, -np.inf], np.nan)
    neutral = raw - raw.rolling(126, min_periods=126).mean()
    std = neutral.rolling(126, min_periods=126).std(ddof=0)
    z = (neutral / std.replace(0, np.nan)).clip(-3, 3)
    smooth = z.ewm(span=18, adjust=False, min_periods=18).mean()
    alpha = smooth.shift(1)

    result = {
        "rows": int(len(df)),
        "start": str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None,
        "end": str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
        "btc_eth_return_correlation_full_sample": float(br.corr(er)),
        "equal_window_cov_over_corr_identity_max_relative_error": float(rel_err) if pd.notna(rel_err) else None,
        "valid_cov20": int(cov20.notna().sum()),
        "valid_corr85": int(corr85.notna().sum()),
        "valid_raw_signal": int(raw.notna().sum()),
        "valid_neutralized": int(neutral.notna().sum()),
        "valid_zscore": int(z.notna().sum()),
        "valid_smooth": int(smooth.notna().sum()),
        "valid_alpha": int(alpha.notna().sum()),
        "conclusion": "Sample is too short for the submitted multi-stage warmup if valid_alpha is zero; a flat backtest then tests no positions rather than no alpha.",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
