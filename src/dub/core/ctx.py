"""Ctx: the single doorway every pipeline step goes through.

`await cx.call(op, req)`:
  1. routes the op to a chain of providers (RouteTrace);
  2. computes the call key from (op, provider, version, params, seed, request);
  3. returns the memoized result if the key already exists (cache hit);
  4. otherwise runs the provider, stores the result, and returns it;
  5. on a classified provider error, waits and retries (usage limits) or falls back to the
     next provider in the chain.

`await cx.local(op, fn, req, ResType)` does the same for pure CPU functions in `dub.ops`.

Because everything is memoized, re-running a job only executes what is missing. That gives
resume-after-crash, cheap edits and "what changed?" for free.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import inspect
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from dub.contracts import v1
from dub.core.config import Config, ProviderManifest
from dub.core.executor import Executor
from dub.core.keys import call_key, hash_bytes
from dub.core.provider_api import ProviderError
from dub.core.router import Router
from dub.core.trace import Tracer
from dub.store import Workspace
from dub.store.gc import cas_keys

T = TypeVar("T", bound=BaseModel)

OP_RESULT: dict[str, type[BaseModel]] = {op: res for op, (_req, res) in v1.OPS.items()}
LANES = {
    "audio.separate": "gpu · separate",
    "asr.transcribe": "gpu · asr",
    "asr.backcheck": "gpu · asr",
    "text.boundaries": "gpu · tts",
    "audio.diarize": "gpu · diarize",
    "audio.embed": "gpu · tts",
    "mt.context": "llm",
    "mt.translate": "llm",
    "mt.rewrite": "llm",
    "tts.synthesize": "gpu · tts",
}


@dataclass
class CallStat:
    op: str
    provider: str
    subject: str | None
    key: str
    hit: bool
    started_at: float
    dur_ms: float


CRLF, LF = bytes([13, 10]), bytes([10])


@functools.cache
def _module_hash(module: str) -> str:
    mod = sys.modules.get(module) or importlib.import_module(module)
    path = inspect.getsourcefile(mod)
    if not path:
        return "unknown"
    # Normalise line endings so a Windows checkout and a Linux pod agree on the key.
    return hash_bytes(Path(path).read_bytes().replace(CRLF, LF))[:12]


def _impl_version(m: ProviderManifest) -> str:
    """`impl_version: auto` = hash of the adapter's source file (handy for cheap providers).
    Expensive providers keep a manual version so editing a comment does not redo hours of GPU."""
    if m.impl_version == "auto" and m.impl:
        return _module_hash(m.impl.split(":")[0])
    return m.impl_version


class Ctx:
    def __init__(
        self,
        *,
        cfg: Config,
        ws: Workspace,
        router: Router,
        executor: Executor,
        tracer: Tracer,
        job_id: str,
        facts: dict[str, Any],
        log: Callable[[str], None] = print,
        fresh: bool = False,
    ):
        self.cfg = cfg
        self.ws = ws
        self.router = router
        self.executor = executor
        self.tracer = tracer
        self.job_id = job_id
        self.facts = facts
        self.log = log
        self.fresh = fresh  # benchmarks: recompute everything (results are still stored)
        self.routes: list[v1.RouteTrace] = []
        self.stats: list[CallStat] = []
        self._prompt_cache: dict[str, str] = {}

    # ------------------------------------------------------------ helpers
    def _prompt_params(self, m: ProviderManifest, op: str) -> dict[str, Any]:
        name = m.prompt_files.get(op)
        if not name:
            return {}
        if name not in self._prompt_cache:
            text = (self.cfg.prompts_dir / name).read_text(encoding="utf-8")
            self._prompt_cache[name] = hash_bytes(text.encode("utf-8"))
        return {"prompt_file": name, "prompt_hash": self._prompt_cache[name]}

    def _lookup(self, key: str) -> dict[str, Any] | None:
        if self.fresh:
            return None
        cached = self.ws.memo.get(key)
        if cached is None:
            return None
        if not all(self.ws.cas.exists(k) for k in cas_keys(cached.get("result"))):
            return None  # a referenced file was deleted: recompute
        return cached

    def _record(self, op, provider, subject, key, hit, started_at, dur_ms) -> None:
        self.stats.append(CallStat(op, provider, subject, key, hit, started_at, dur_ms))

    def params_for(self, m: ProviderManifest, op: str) -> dict[str, Any]:
        """Manifest params, then the profile's overrides for this provider, then prompt hashes."""
        return {
            **m.params,
            **self.router.profile.params.get(m.id, {}),
            **self._prompt_params(m, op),
        }

    # ------------------------------------------------------------ provider calls
    async def call(
        self,
        op: str,
        req: BaseModel | Callable[[ProviderManifest], BaseModel],
        *,
        facts: dict[str, Any] | None = None,
        subject: str | None = None,
        seed: int = 0,
    ) -> Any:
        """`req` may be a function of the chosen provider's manifest, because some request
        fields depend on its capabilities (e.g. target_ms only for "exact" TTS engines)."""
        res_type = OP_RESULT[op]
        lane = LANES.get(op, op)
        trace = self.router.route(op, {**self.facts, **(facts or {})}, subject)
        self.routes.append(trace)
        errors: list[str] = []

        for pid in trace.chain:
            m = self.cfg.providers[pid]
            req_m = req(m) if callable(req) else req
            params = self.params_for(m, op)
            key = call_key(
                op=op,
                provider=pid,
                impl_version=_impl_version(m),
                weights_rev=m.weights_rev,
                params=params,
                seed=seed,
                request=req_m,
            )
            cached = self._lookup(key)
            if cached is not None:
                self.tracer.instant(f"{op} (cache)", lane, {"provider": pid, "subject": subject})
                self._record(op, pid, subject, key, True, time.time(), 0.0)
                return res_type.model_validate(cached["result"])

            waits = (
                list(m.on_error.retry_wait_s)
                if m.on_error.usage_limit == "pause_then_retry"
                else []
            )
            res = None
            while True:
                started, t0 = time.time(), time.perf_counter()
                try:
                    with self.tracer.span(op, lane, {"provider": pid, "subject": subject}):
                        res = await self.executor.run(m, op, req_m, params)
                    break
                except ProviderError as e:
                    if e.kind == "usage_limit" and waits:
                        wait = waits.pop(0)
                        self.log(f"[{pid}] usage limit reached, pausing {wait:.0f}s then retrying")
                        await asyncio.sleep(wait)
                        continue
                    errors.append(f"{pid}: [{e.kind}] {e}")
                    self.log(f"[{pid}] {op} failed ({e.kind}): {e}")
                    break
            if res is None:
                continue
            dur_ms = (time.perf_counter() - t0) * 1000
            if not isinstance(res, res_type):
                res = res_type.model_validate(res.model_dump())
            self.ws.memo.put(
                key,
                {
                    "op": op,
                    "provider": pid,
                    "subject": subject,
                    "result": res.model_dump(mode="json"),
                },
            )
            self._record(op, pid, subject, key, False, started, dur_ms)
            return res

        raise ProviderError(
            f"All providers failed for {op} ({subject or 'job'}): " + " | ".join(errors),
            kind="crash",
        )

    # ------------------------------------------------------------ local CPU steps
    async def local(
        self,
        op: str,
        fn: Callable[..., T],
        req: BaseModel,
        res_type: type[T],
        *,
        subject: str | None = None,
        with_cas: bool = True,
    ) -> T:
        """Memoized CPU step. The key includes a hash of the function's module source, so
        editing the code of a step re-runs it (and only what depends on its output)."""
        provider = f"local:{fn.__module__}.{fn.__name__}"
        key = call_key(
            op=op,
            provider=provider,
            impl_version=_module_hash(fn.__module__),
            weights_rev=None,
            params={},
            seed=0,
            request=req,
        )
        cached = self._lookup(key)
        if cached is not None:
            self.tracer.instant(f"{op} (cache)", "cpu", {"subject": subject})
            self._record(op, provider, subject, key, True, time.time(), 0.0)
            return res_type.model_validate(cached["result"])
        started, t0 = time.time(), time.perf_counter()
        args = (req, self.ws.cas) if with_cas else (req,)
        with self.tracer.span(op, "cpu", {"subject": subject}):
            res = await asyncio.to_thread(fn, *args)
        dur_ms = (time.perf_counter() - t0) * 1000
        self.ws.memo.put(
            key,
            {
                "op": op,
                "provider": provider,
                "subject": subject,
                "result": res.model_dump(mode="json"),
            },
        )
        self._record(op, provider, subject, key, False, started, dur_ms)
        return res

    # ------------------------------------------------------------ bookkeeping
    def flush_stats(self) -> None:
        self.ws.db.add_spans(
            [
                (
                    self.job_id,
                    s.op,
                    s.provider,
                    s.subject,
                    s.key,
                    int(s.hit),
                    s.started_at,
                    s.dur_ms,
                )
                for s in self.stats
            ]
        )
