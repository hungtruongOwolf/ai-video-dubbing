"""The interface every provider (the adapter for one model family) implements.

A provider is a class with `__init__(manifest)` and `run(op, req, ctx) -> result`.
- `req` and `result` are models from `dub.contracts.v1`.
- `run` must be a pure function of (req, params, seed): same input, same output. That is
  what makes results cacheable and makes a resumed run byte-identical.
- Providers may only import this module, `dub.contracts` and `dub.store` from the core.
  The core never imports a provider (enforced by the import-graph test), so adding or
  swapping a model never touches the core.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from dub.core.config import ProviderManifest
from dub.store.cas import CAS


class ProviderError(RuntimeError):
    """A classified error so the core can decide to retry, fall back, or stop.

    kind: usage_limit | bad_input | unsupported | timeout | oom | crash
    retryable: the same provider may succeed if asked again (the core retries twice)
    """

    def __init__(self, message: str, *, kind: str = "crash", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


@dataclass
class RunCtx:
    cas: CAS
    manifest: ProviderManifest
    params: dict[str, Any]
    prompts_dir: Path
    log: Callable[[str], None] = field(default=print)


class Provider(Protocol):
    def __init__(self, manifest: ProviderManifest) -> None: ...

    def run(self, op: str, req: BaseModel, ctx: RunCtx) -> BaseModel: ...
