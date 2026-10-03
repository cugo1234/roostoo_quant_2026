from __future__ import annotations

from dataclasses import dataclass
import requests
import pandas as pd


@dataclass
class BinanceLiveKlines:
    base_url: str = "https://data-api.binance.vision"
    timeout: int = 10

    def get(self, symbol: str, interval: str = "4h", limit: int = 200) -> pd.DataFrame:
        url = f"{self.base_url}/api/v3/klines"
        r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=self.timeout)
        r.raise_for_status()
        rows = r.json()
        cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","count","taker_buy_volume","taker_buy_quote_volume","ignore"]
        df = pd.DataFrame(rows, columns=cols)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume","quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["timestamp","open","high","low","close","volume","quote_volume"]]
