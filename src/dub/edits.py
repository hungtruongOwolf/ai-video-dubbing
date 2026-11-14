"""Human edits, stored on the laptop.

Each (source audio, target language) has one append-only file:
    dub-workspace/edits/<audio key prefix>-<lang>.jsonl

The laptop is the source of truth, so you can edit while the pod is off; every run sends the
full list along with the job. Lines are fsync'ed before an edit is acknowledged. Undo appends
an `undo` record instead of rewriting history. Edits apply to every profile (draft or final),
so promoting a draft keeps your fixes.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from dub.contracts.v1 import Edit
from dub.ops.edits import effective


class EditStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def for_job(cls, ws_root: Path, audio_key: str, tgt_lang: str) -> EditStore:
        return cls(Path(ws_root) / "edits" / f"{audio_key[:16]}-{tgt_lang}.jsonl")

    def all(self) -> list[Edit]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(Edit.model_validate_json(line))
            except ValueError:
                continue  # a torn last line after a crash: ignore it
        return out

    def effective(self) -> list[Edit]:
        return effective(self.all())

    def add(self, unit_id: str, field: str, value: Any = None) -> Edit:
        e = Edit(id=f"{time.time_ns():020d}-{secrets.token_hex(2)}", unit_id=unit_id,
                 field=field, value=value, ts=time.time())  # fmt: skip
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(e.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        return e

    def undo(self, edit_id: str | None = None) -> Edit | None:
        """Cancel one edit (the latest one if no id is given)."""
        live = self.effective()
        target = next((e for e in reversed(live) if edit_id in (None, e.id)), None)
        if target is None:
            return None
        return self.add(target.unit_id, "undo", target.id)

    def summary(self) -> list[dict[str, Any]]:
        return [json.loads(e.model_dump_json()) for e in self.effective()]
