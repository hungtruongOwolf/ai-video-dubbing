"""SQLite: an index of jobs and calls (spans). It is NOT the source of truth.

The source of truth is CAS + memo (+ edits.jsonl from M2). If this file is lost, a rerun
still produces the same result because every step is a cache hit.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS job (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    src_lang    TEXT,
    tgt_lang    TEXT NOT NULL,
    profile     TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS span (
    job_id      TEXT NOT NULL,
    op          TEXT NOT NULL,
    provider    TEXT NOT NULL,
    subject     TEXT,
    call_key    TEXT NOT NULL,
    cache_hit   INTEGER NOT NULL,
    started_at  REAL NOT NULL,
    dur_ms      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS span_job ON span(job_id);
"""


class JobDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def upsert_job(
        self,
        job_id: str,
        *,
        source: str,
        src_lang: str | None,
        tgt_lang: str,
        profile: str,
        status: str,
    ) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                """INSERT INTO job(id, source, src_lang, tgt_lang, profile, status, created_at,
                                   updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                        src_lang=excluded.src_lang, updated_at=excluded.updated_at""",
                (job_id, source, src_lang, tgt_lang, profile, status, now, now),
            )

    def add_spans(self, rows: list[tuple]) -> None:
        if not rows:
            return
        with self._conn() as c:
            c.executemany("INSERT INTO span VALUES(?,?,?,?,?,?,?,?)", rows)

    def jobs(self) -> list[tuple]:
        with self._conn() as c:
            return c.execute(
                "SELECT id, tgt_lang, profile, status, updated_at FROM job ORDER BY updated_at DESC"
            ).fetchall()
