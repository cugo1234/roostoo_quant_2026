#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
import os
from roostoo_quant.live.bot import run_forever, run_once


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.once:
        print(run_once(args.config))
    else:
        run_forever(args.config)


if __name__ == "__main__":
    main()
