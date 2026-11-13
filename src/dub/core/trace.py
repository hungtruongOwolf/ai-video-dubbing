"""Timeline trace in Chrome Trace Event format. Open `trace.json` at https://ui.perfetto.dev.

Each lane ("gpu · tts", "llm", "cpu", ...) is shown as a thread. Calls that overlap on the
same lane are put on numbered sub-lanes ("llm #2") so the picture shows real parallelism.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self._lanes: dict[str, int] = {}
        self._busy: dict[int, bool] = {}
        self._lock = threading.Lock()

    def now_us(self) -> float:
        return (time.perf_counter() - self.t0) * 1e6

    def _acquire(self, lane: str) -> int:
        with self._lock:
            base = self._lanes.setdefault(lane, (len(self._lanes) + 1) * 100)
            slot = 0
            while self._busy.get(base + slot):
                slot += 1
            tid = base + slot
            self._busy[tid] = True
            if not any(e.get("tid") == tid and e.get("ph") == "M" for e in self.events):
                name = lane if slot == 0 else f"{lane} #{slot + 1}"
                self.events.append(
                    {"ph": "M", "pid": 1, "tid": tid, "name": "thread_name", "args": {"name": name}}
                )
            return tid

    def _release(self, tid: int) -> None:
        with self._lock:
            self._busy[tid] = False

    @contextmanager
    def span(self, name: str, lane: str, args: dict[str, Any] | None = None):
        tid = self._acquire(lane)
        start = self.now_us()
        try:
            yield
        finally:
            dur = self.now_us() - start
            self._release(tid)
            with self._lock:
                self.events.append(
                    {
                        "ph": "X",
                        "pid": 1,
                        "tid": tid,
                        "name": name,
                        "ts": start,
                        "dur": dur,
                        "args": args or {},
                    }
                )

    def instant(self, name: str, lane: str, args: dict[str, Any] | None = None) -> None:
        tid = self._acquire(lane)
        self._release(tid)
        with self._lock:
            self.events.append(
                {
                    "ph": "i",
                    "s": "t",
                    "pid": 1,
                    "tid": tid,
                    "name": name,
                    "ts": self.now_us(),
                    "args": args or {},
                }
            )

    def save(self, path: Path) -> None:
        meta = {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "dub"}}
        Path(path).write_text(
            json.dumps({"traceEvents": [meta, *self.events], "displayTimeUnit": "ms"}),
            encoding="utf-8",
        )
