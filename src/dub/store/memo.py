"""Memo: stores the (small, JSON) result of every call under its call key.

`memo/<first 2 chars>/<call_key>.json` holds {op, provider, result}. Results only contain
small data and CAS keys; the actual audio lives in the CAS.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class Memo:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt file counts as missing and will be recomputed

    def put(self, key: str, value: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".json", dir=p.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)

    def count(self) -> int:
        return sum(1 for _ in self.root.glob("*/*.json"))
