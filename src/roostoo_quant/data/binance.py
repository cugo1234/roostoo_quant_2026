from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
import calendar
import zipfile
import time
import requests
import pandas as pd

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def _parse_epoch(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.dropna().abs().median()
    if pd.isna(med):
        return pd.to_datetime(series, utc=True, errors="coerce")
    unit = "us" if med > 1e14 else "ms" if med > 1e11 else "s"
    return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")


@dataclass
class BinanceArchiveDownloader:
    cache_dir: Path
    session: requests.Session | None = None
    timeout: int = 30
    pause_seconds: float = 0.05

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = self.session or requests.Session()

    @staticmethod
    def monthly_url(symbol: str, interval: str, year: int, month: int) -> str:
        ym = f"{year:04d}-{month:02d}"
        return (
            "https://data.binance.vision/data/spot/monthly/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        )

    @staticmethod
    def daily_url(symbol: str, interval: str, date: pd.Timestamp) -> str:
        ds = pd.Timestamp(date).strftime("%Y-%m-%d")
        return (
            "https://data.binance.vision/data/spot/daily/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{ds}.zip"
        )

    def _get(self, url: str) -> bytes:
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.content

    def download_month(self, symbol: str, interval: str, year: int, month: int) -> Path | None:
        out = self.cache_dir / interval / symbol / f"{symbol}-{interval}-{year:04d}-{month:02d}.csv"
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        url = self.monthly_url(symbol, interval, year, month)
        try:
            blob = self._get(url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        with zipfile.ZipFile(BytesIO(blob)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            raw = zf.read(name)
        out.write_bytes(raw)
        time.sleep(self.pause_seconds)
        return out

    def download_range(self, symbol: str, interval: str, start: str, end: str) -> list[Path]:
        start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
        months = pd.period_range(start_ts.tz_localize(None).to_period("M"), end_ts.tz_localize(None).to_period("M"), freq="M")
        paths: list[Path] = []
        for p in months:
            got = self.download_month(symbol, interval, p.year, p.month)
            if got is not None:
                paths.append(got)
        return paths


def read_binance_kline_csv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    path = Path(path)
    probe = pd.read_csv(path, nrows=2, header=None)
    first = str(probe.iloc[0, 0]).lower()
    has_header = any(c in first for c in ["open", "time", "date", "timestamp"])
    if has_header:
        df = pd.read_csv(path)
        lower = {str(c).lower(): c for c in df.columns}
        # Handles both Binance archive names and normalized files.
        rename = {}
        for want, opts in {
            "open_time": ["open_time", "timestamp", "date"],
            "open": ["open"], "high": ["high"], "low": ["low"], "close": ["close"], "volume": ["volume"]
        }.items():
            for opt in opts:
                if opt in lower:
                    rename[lower[opt]] = want
                    break
        df = df.rename(columns=rename)
    else:
        df = pd.read_csv(path, header=None)
        df.columns = KLINE_COLS[: df.shape[1]]
    if "open_time" not in df.columns:
        raise ValueError(f"No timestamp column in {path}")
    if pd.api.types.is_numeric_dtype(df["open_time"]):
        df["timestamp"] = _parse_epoch(df["open_time"])
    else:
        df["timestamp"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    if symbol:
        df["symbol"] = symbol
    return df[["timestamp", "open", "high", "low", "close", "volume"] + (["symbol"] if symbol else [])]


def load_cached_symbol(cache_dir: str | Path, symbol: str, interval: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    root = Path(cache_dir) / interval / symbol
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No cached files for {symbol} {interval} under {root}")
    parts = [read_binance_kline_csv(p, symbol=symbol) for p in files]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] < pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)


def panel_from_frames(frames: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    cols = []
    for sym, df in frames.items():
        s = df.set_index("timestamp")[field].rename(sym)
        cols.append(s)
    return pd.concat(cols, axis=1).sort_index()
