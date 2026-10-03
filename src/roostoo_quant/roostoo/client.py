from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlencode
from typing import Any

import requests


class RoostooAPIError(RuntimeError):
    pass


class RoostooClient:
    """Thin, auditable wrapper around the documented Roostoo public API."""

    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://mock-api.roostoo.com", timeout: int = 15):
        self.api_key = api_key
        self.secret_key = secret_key.encode("utf-8")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    @staticmethod
    def _local_ms() -> int:
        return int(time.time() * 1000)

    def _signed(self, params: dict[str, Any] | None = None) -> tuple[dict[str, str], dict[str, str], str]:
        payload = {k: str(v) for k, v in (params or {}).items() if v is not None}
        payload["timestamp"] = str(self._local_ms())
        ordered = sorted(payload.items(), key=lambda kv: kv[0])
        query = urlencode(ordered)
        signature = hmac.new(self.secret_key, query.encode("utf-8"), hashlib.sha256).hexdigest()
        headers = {"RST-API-KEY": self.api_key, "MSG-SIGNATURE": signature}
        return headers, payload, query

    @staticmethod
    def _validate_json(data: dict[str, Any], allow_no_pending: bool = False) -> dict[str, Any]:
        if "Success" in data and not data.get("Success"):
            if allow_no_pending and data.get("TotalPending") == 0:
                return data
            raise RoostooAPIError(data.get("ErrMsg") or f"Roostoo request failed: {data}")
        return data

    def _get_public(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        r = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _get_ts(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        p = dict(params or {})
        p["timestamp"] = str(self._local_ms())
        r = self.session.get(self.base_url + path, params=p, timeout=self.timeout)
        r.raise_for_status()
        return self._validate_json(r.json())

    def _get_signed(self, path: str, params: dict[str, Any] | None = None, allow_no_pending: bool = False) -> dict[str, Any]:
        headers, payload, _ = self._signed(params)
        r = self.session.get(self.base_url + path, headers=headers, params=payload, timeout=self.timeout)
        r.raise_for_status()
        return self._validate_json(r.json(), allow_no_pending=allow_no_pending)

    def _post_signed(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        headers, _, body = self._signed(params)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        r = self.session.post(self.base_url + path, headers=headers, data=body, timeout=self.timeout)
        r.raise_for_status()
        return self._validate_json(r.json())

    def server_time(self) -> dict[str, Any]:
        return self._get_public("/v3/serverTime")

    def exchange_info(self) -> dict[str, Any]:
        return self._get_public("/v3/exchangeInfo")

    def ticker(self, pair: str | None = None) -> dict[str, Any]:
        params = {"pair": pair} if pair else {}
        return self._get_ts("/v3/ticker", params)

    def balance(self) -> dict[str, Any]:
        return self._get_signed("/v3/balance")

    def pending_count(self) -> dict[str, Any]:
        return self._get_signed("/v3/pending_count", allow_no_pending=True)

    def place_order(self, pair: str, side: str, quantity: float, order_type: str = "MARKET", price: float | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires price")
            params["price"] = price
        return self._post_signed("/v3/place_order", params)

    def query_order(self, order_id: str | int | None = None, pair: str | None = None, pending_only: bool | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if order_id is not None:
            params["order_id"] = order_id
        elif pair is not None:
            params["pair"] = pair
            if pending_only is not None:
                params["pending_only"] = "TRUE" if pending_only else "FALSE"
        return self._post_signed("/v3/query_order", params)

    def cancel_order(self, order_id: str | int | None = None, pair: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if order_id is not None:
            params["order_id"] = order_id
        elif pair is not None:
            params["pair"] = pair
        return self._post_signed("/v3/cancel_order", params)

    def short_open(self, pair: str, collateral: float, price: float | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"pair": pair, "collateral": collateral}
        if price is not None:
            params.update({"order_type": "LIMIT", "price": price})
        return self._post_signed("/v6/short_open", params)

    def short_close(self, pair: str, close_qty: float | None = None, close_pct: float | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"pair": pair}
        if close_qty is not None:
            params["close_qty"] = close_qty
        elif close_pct is not None:
            params["close_pct"] = close_pct
        return self._post_signed("/v6/short_close", params)

    def short_positions(self) -> dict[str, Any]:
        return self._get_signed("/v6/short_positions")
