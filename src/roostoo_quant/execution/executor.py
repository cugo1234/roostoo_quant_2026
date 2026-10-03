from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from roostoo_quant.execution.planner import PlannedOrder
from roostoo_quant.roostoo.client import RoostooClient


class ExecutionEngine:
    def __init__(self, client: RoostooClient, ledger_path: str | Path, dry_run: bool = True):
        self.client = client
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run

    def _log(self, event: dict) -> None:
        event = {"logged_at_ms": int(time.time() * 1000), **event}
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str, sort_keys=True) + "\n")

    def execute(self, orders: Iterable[PlannedOrder]) -> list[dict]:
        responses = []
        for order in orders:
            fp = order.fingerprint()
            self._log({"event": "planned_order", "fingerprint": fp, "order": order.__dict__, "dry_run": self.dry_run})
            if self.dry_run:
                resp = {"Success": True, "DryRun": True, "Fingerprint": fp, "Order": order.__dict__}
            elif order.action in {"BUY", "SELL"}:
                resp = self.client.place_order(order.pair, order.action, order.quantity or 0.0, order_type="MARKET")
            elif order.action == "SHORT_OPEN":
                resp = self.client.short_open(order.pair, order.collateral or 0.0)
            elif order.action == "SHORT_CLOSE":
                resp = self.client.short_close(order.pair, close_pct=order.close_pct)
            else:
                raise ValueError(f"Unknown action: {order.action}")
            self._log({"event": "order_response", "fingerprint": fp, "response": resp})
            responses.append(resp)
        return responses
