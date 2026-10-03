#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import requests
import yaml

URL = "https://mock-api.roostoo.com/v3/exchangeInfo"


def main():
    r = requests.get(URL, timeout=15)
    r.raise_for_status()
    info = r.json()
    pairs = {p:v for p,v in info.get("TradePairs",{}).items() if v.get("CanTrade")}
    symbols = [f"{v['Coin']}USDT" for p,v in pairs.items() if v.get("Unit") == "USD"]
    out = {"source": URL, "initial_wallet": info.get("InitialWallet"), "roostoo_pairs": pairs, "binance_spot_proxy_symbols": symbols}
    Path("results").mkdir(exist_ok=True)
    Path("results/discovered_roostoo_universe.json").write_text(json.dumps(out,indent=2))
    Path("config/live_universe.yaml").write_text(yaml.safe_dump({"symbols": symbols, "roostoo_pairs": list(pairs)}, sort_keys=False))
    print(json.dumps(out,indent=2))

if __name__ == "__main__":
    main()
