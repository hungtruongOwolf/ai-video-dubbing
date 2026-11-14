"""Garbage collection of the CAS: delete files that no cached result points to.

A CAS file stays when any memo entry references it (a field named *_key in a stored result,
at any depth) or when `sources.json` names it (the laptop's extracted audio). Everything else
is unreachable: experiment outputs, intermediates of runs whose results were replaced, and
leftovers of crashes. Deleting a file can never give a wrong result: a memo entry whose files
are missing is treated as a cache miss and recomputed.

Files younger than `min_age_s` are kept, so a job that is writing files right now (results are
stored after their files) is not disturbed. On the pod, run it when no job is running:
`dub gc --remote`. The network volume has a hard 50 GB quota.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from dub.store import Workspace


def cas_keys(obj: Any) -> list[str]:
    """All CAS keys referenced by a stored result (fields named *_key)."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("_key") and isinstance(v, str) and v:
                out.append(v)
            else:
                out.extend(cas_keys(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(cas_keys(v))
    return out


@dataclass
class GcReport:
    kept: int = 0
    deleted: int = 0
    freed_bytes: int = 0
    tmp_deleted: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kept": self.kept,
            "deleted": self.deleted,
            "freed_mb": round(self.freed_bytes / 1e6, 1),
            "tmp_deleted": self.tmp_deleted,
            "errors": self.errors[:10],
        }


def referenced_keys(ws: Workspace) -> set[str]:
    keys: set[str] = set()
    for f in ws.memo.root.glob("*/*.json"):
        try:
            keys.update(cas_keys(json.loads(f.read_text(encoding="utf-8")).get("result")))
        except (OSError, json.JSONDecodeError):
            continue  # a broken memo entry is a cache miss anyway
    sources = ws.root / "sources.json"
    if sources.exists():
        try:
            keys.update(v["audio_key"] for v in json.loads(sources.read_text("utf-8")).values())
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    return keys


def collect(ws: Workspace, *, min_age_s: float = 3600, dry_run: bool = False) -> GcReport:
    rep = GcReport()
    keep = referenced_keys(ws)
    now = time.time()
    for path in ws.cas.root.glob("??/*"):
        if not path.is_file():
            continue
        try:
            st = path.stat()
            if path.name in keep or now - st.st_mtime < min_age_s:
                rep.kept += 1
                continue
            if not dry_run:
                path.unlink()
            rep.deleted += 1
            rep.freed_bytes += st.st_size
        except OSError as e:
            rep.errors.append(f"{path.name}: {e}")
    for path in (ws.cas.root / "tmp").glob("*"):
        try:
            st = path.stat()
            if path.is_file() and now - st.st_mtime >= min_age_s:
                if not dry_run:
                    path.unlink()
                rep.tmp_deleted += 1
                rep.freed_bytes += st.st_size
        except OSError as e:
            rep.errors.append(f"tmp/{path.name}: {e}")
    return rep
