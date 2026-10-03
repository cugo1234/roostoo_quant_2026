from __future__ import annotations

import json
from pathlib import Path


def load_state(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(path: str | Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(p)
