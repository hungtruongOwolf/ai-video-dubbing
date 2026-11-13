"""Executor: runs a provider for one call.

M0 supports the `inproc` runtime: the provider class is imported from the `impl` string in
its manifest and runs in a worker thread. Each provider gets a semaphore sized by
`limits.max_concurrency`, so for example at most 4 LLM calls run at the same time.
M1 adds the `http` runtime: the same request body is sent to a worker process on the GPU
pod, which reads and writes the shared CAS.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from dub.core.config import Config, ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.store.cas import CAS


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
        inst = self._instance(m)
        sem = self._sems.setdefault(m.id, asyncio.Semaphore(m.limits.max_concurrency))
        ctx = RunCtx(
            cas=self.cas, manifest=m, params=params, prompts_dir=self.cfg.prompts_dir, log=self.log
        )
        async with sem:
            return await asyncio.to_thread(inst.run, op, req, ctx)
