#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

from roostoo_quant.roostoo.client import RoostooClient
from roostoo_quant.execution.planner import extract_trade_rules


def main() -> int:
    load_dotenv()
    api_key = os.getenv("ROOSTOO_API_KEY", "").strip()
    secret = os.getenv("ROOSTOO_SECRET_KEY", "").strip()
    base = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com").strip()
    env = os.getenv("ROOSTOO_ENV", "test").strip().lower()
    live = os.getenv("LIVE_TRADING", "0").strip()
    confirm = os.getenv("ROOSTOO_LIVE_TRADING_CONFIRM", "").strip()

    missing = [k for k, v in [("ROOSTOO_API_KEY", api_key), ("ROOSTOO_SECRET_KEY", secret)] if not v]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2

    if env == "competition" and live == "1" and confirm != "YES":
        print("LIVE_TRADING=1 in competition requires ROOSTOO_LIVE_TRADING_CONFIRM=YES", file=sys.stderr)
        return 2

    client = RoostooClient(api_key, secret, base)
    st = client.server_time()
    info = client.exchange_info()
    ticker = client.ticker()
    bal = client.balance()
    shorts = client.short_positions()
    pending = client.pending_count()

    rules = extract_trade_rules(info)
    data = ticker.get("Data", ticker)
    wallet = bal.get("Wallet", {})
    usd = wallet.get("USD", {})
    print("Roostoo preflight OK")
    print(f"server_time={st.get('ServerTime')}")
    print(f"exchange_running={info.get('IsRunning')}")
    print(f"tradable_pairs={len(rules)}")
    print(f"ticker_pairs={len(data)}")
    print(f"usd_free={usd.get('Free', 0)}")
    print(f"usd_lock={usd.get('Lock', 0)}")
    print(f"short_positions={len(shorts.get('Positions', []) or [])}")
    print(f"pending_orders={pending.get('TotalPending', 0)}")
    print(f"env={env} live_trading={live}")
    if env == "competition" and live == "1":
        print("LIVE ORDER MODE IS ENABLED")
    else:
        print("NO ORDERS WILL BE SENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
