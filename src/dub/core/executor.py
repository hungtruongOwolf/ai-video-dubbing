"""Executor: runs a provider for one call.

M0 supports the `inproc` runtime: the provider class is imported from the `impl` string in
its manifest and runs in a worker thread. Each provider gets a semaphore sized by
`limits.max_concurrency`, so for example at most 4 LLM calls run at the same time.
The `http` runtime sends the same request as JSON to a worker process (`dub.worker`) that
hosts the provider in its own Python environment and reads/writes the shared CAS.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from dub.contracts.v1 import OPS
from dub.core.config import Config, ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.store.cas import CAS
from dub.worker.protocol import post_json, worker_urls


def _load_class(impl: str) -> type:
    module_name, _, cls_name = impl.partition(":")
    if not cls_name:
        raise ValueError(f"impl must look like 'module.path:ClassName', got {impl!r}")
    return getattr(importlib.import_module(module_name), cls_name)


class Executor:
    def __init__(self, cfg: Config, cas: CAS, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.cas = cas
        self.log = log
        self._instances: dict[str, Any] = {}
        self._sems: dict[str, asyncio.Semaphore] = {}

    def _instance(self, m: ProviderManifest) -> Any:
        if m.id not in self._instances:
            if m.runtime != "inproc" or not m.impl:
                raise ProviderError(
                    f"Runtime '{m.runtime}' is not supported yet", kind="unsupported"
                )
            self._instances[m.id] = _load_class(m.impl)(m)
        return self._instances[m.id]

    async def run(
        self, m: ProviderManifest, op: str, req: BaseModel, params: dict[str, Any]
    ) -> BaseModel:
        if m.runtime == "http":
            return await self._run_http(m, op, req, params)
        inst = self._instance(m)
        sem = self._sems.setdefault(m.id, asyncio.Semaphore(m.limits.max_concurrency))
        ctx = RunCtx(
            cas=self.cas, manifest=m, params=params, prompts_dir=self.cfg.prompts_dir, log=self.log
        )
        async with sem:
            return await asyncio.to_thread(inst.run, op, req, ctx)

    async def unload(self, m: ProviderManifest) -> bool:
        """Ask the provider's worker to free its model (no-op for in-process providers)."""
        url = worker_urls().get(m.env or "") if m.runtime == "http" else None
        if not url:
            return False
        try:
            reply = await asyncio.to_thread(post_json, f"{url}/v1/unload", {"provider": m.id}, 60)
        except ProviderError:
            return False
        return bool(reply.get("freed"))

    async def _run_http(
        self, m: ProviderManifest, op: str, req: BaseModel, params: dict[str, Any]
    ) -> BaseModel:
        url = worker_urls().get(m.env or "")
        if not url:
            raise ProviderError(f"worker '{m.env}' is not running", kind="unsupported")
        body = {
            "provider": m.id,
            "op": op,
            "params": params,
            "request": req.model_dump(mode="json"),
        }
        sem = self._sems.setdefault(m.id, asyncio.Semaphore(m.limits.max_concurrency))
        async with sem:
            reply = await asyncio.to_thread(post_json, f"{url}/v1/run", body)
        return OPS[op][1].model_validate(reply["result"])
