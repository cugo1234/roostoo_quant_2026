#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from roostoo_quant.config import load_yaml
from roostoo_quant.data.binance import load_cached_symbol, panel_from_frames
from roostoo_quant.backtest.engine import run_backtest
from roostoo_quant.backtest.tournament import rolling_tournament_windows, tournament_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--universe", default="config/universe.yaml")
    ap.add_argument("--cache", default="data/binance")
    ap.add_argument("--out", default="results/full_backtest")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    u = yaml.safe_load(Path(args.universe).read_text())
    frames = {}
    for sym in u["symbols"]:
        try:
            frames[sym] = load_cached_symbol(args.cache, sym, cfg["research"]["interval"], cfg["research"]["start"], cfg["research"]["end"])
        except FileNotFoundError:
            pass
    if len(frames) < 2:
        raise SystemExit("Need >=2 cached symbols. Run scripts/download_data.py first.")
    fields = {f: panel_from_frames(frames, f) for f in ["open", "high", "low", "close", "volume"]}
    # Keep rows with at least two assets; listing dates naturally create NaNs.
    good = fields["close"].notna().sum(axis=1) >= 2
    fields = {k: v.loc[good] for k, v in fields.items()}
    result = run_backtest(fields, cfg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result.returns.to_csv(out / "backtest_detail.csv")
    result.positions.to_csv(out / "positions.csv")
    result.trades.to_csv(out / "trades.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2, default=str))
    bars_per_day = 6 if cfg["research"]["interval"] == "4h" else 24
    tw = rolling_tournament_windows(result.returns.iloc[:-1], bars_per_day, cfg["research"]["tournament_days"], cfg["research"]["annual_periods"])
    tw.to_csv(out / "tournament_windows.csv")
    (out / "tournament_summary.json").write_text(json.dumps(tournament_summary(tw), indent=2, default=str))
    print(json.dumps({"metrics": result.metrics, "tournament": tournament_summary(tw)}, indent=2, default=str))


if __name__ == "__main__":
    main()
