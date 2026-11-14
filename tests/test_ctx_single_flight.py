"""Single flight: two jobs asking for the same call at the same time compute it once."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

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


def test_a_cancelled_waiter_does_not_cancel_the_shared_call(cfg, ws):
    """A job stopped while waiting for another job's call must not break that call: the
    owner still stores its result, and other waiters still get it."""
    from dub.core import ctx as ctx_mod

    cx = Ctx(
        cfg=cfg, ws=ws, router=Router(cfg), executor=Executor(cfg, ws.cas), tracer=Tracer(),
        job_id="j", facts={}, log=lambda m: None,
    )  # fmt: skip

    async def scenario():
        key = "test-cancelled-waiter"
        assert await cx._lookup_or_claim(key) is None  # this caller owns the call
        waiter = asyncio.create_task(cx._lookup_or_claim(key))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        shared = ctx_mod._INFLIGHT[key]
        assert not shared.cancelled()
        ctx_mod._release(key)  # no InvalidStateError
        assert shared.done() and key not in ctx_mod._INFLIGHT

    asyncio.run(scenario())
