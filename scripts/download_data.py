#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from roostoo_quant.data.binance import BinanceArchiveDownloader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="config/universe.yaml")
    ap.add_argument("--cache", default="data/binance")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-10-01")
    args = ap.parse_args()
    u = yaml.safe_load(Path(args.universe).read_text())
    dl = BinanceArchiveDownloader(Path(args.cache))
    for sym in u["symbols"]:
        listing = u.get("listing_dates", {}).get(sym, args.start)
        start = max(args.start, listing)
        print(f"Downloading {sym} {args.interval} {start} -> {args.end}")
        paths = dl.download_range(sym, args.interval, start, args.end)
        print(f"  cached {len(paths)} monthly files")


if __name__ == "__main__":
    main()
