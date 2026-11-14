"""Storage: CAS (files by hash), memo (results by call key), SQLite (job index)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dub.store.cas import CAS
from dub.store.db import JobDB
from dub.store.memo import Memo


@dataclass
class Workspace:
    """Working directory. On the laptop it defaults to `./dub-workspace`; on the pod it will
    be `/workspace/dub` on the network volume."""

    root: Path
    cas: CAS
    memo: Memo
    db: JobDB

    @classmethod
    def open(cls, root: Path, db_path: Path | None = None) -> Workspace:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            cas=CAS(root / "cas"),
            memo=Memo(root / "memo"),
            db=JobDB(db_path or root / "dub.sqlite"),
        )

    def job_dir(self, job_id: str) -> Path:
        d = self.root / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d


__all__ = ["CAS", "Memo", "JobDB", "Workspace"]
