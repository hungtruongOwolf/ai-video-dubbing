"""Single flight: two jobs asking for the same call at the same time compute it once."""

from __future__ import annotations

import asyncio
import threading
import time

from dub.contracts.v1 import SynthReq
from dub.core.ctx import Ctx
from dub.core.executor import Executor
from dub.core.router import Router
from dub.core.trace import Tracer
from dub.providers.fake import FakeTTS


def test_identical_calls_in_flight_are_computed_once(cfg, ws, monkeypatch):
    runs = []
    original = FakeTTS.op_tts_synthesize

    def slow(self, req, ctx):
        runs.append(req.unit_id)
        time.sleep(0.3)
        return original(self, req, ctx)

    monkeypatch.setattr(FakeTTS, "op_tts_synthesize", slow)
    req = SynthReq(unit_id="u1", text="xin chào", lang="vi", speaker="S1", target_ms=800)
    out = []

    def job(i: int) -> None:
        cx = Ctx(
            cfg=cfg, ws=ws, router=Router(cfg, overrides=[("tts.synthesize", ["fake_tts"])]),
            executor=Executor(cfg, ws.cas), tracer=Tracer(), job_id=f"j{i}",
            facts={"tgt_lang": "vi"}, log=lambda m: None,
        )  # fmt: skip
        out.append(asyncio.run(cx.call("tts.synthesize", req)))

    threads = [threading.Thread(target=job, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert runs == ["u1"]
    assert len({r.audio_key for r in out}) == 1 and len(out) == 3
