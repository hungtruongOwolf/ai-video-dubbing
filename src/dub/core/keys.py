"""Content hashing for cache keys (call keys).

Same (op, provider, version, params, seed, input) => same key. Same key means the result
already exists and nothing needs to run. This is what powers:
- resume after a crash: finished steps are cache hits;
- editing one line: only keys that depend on that line change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

KEY_BYTES = 16  # 128 bits: collisions are not a practical concern for this project


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        raise TypeError("Do not put file paths in cache keys; pass a CAS key instead")
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)  # 1.0 and 1 produce the same key
    return obj


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=KEY_BYTES).hexdigest()


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=KEY_BYTES)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def call_key(
    *,
    op: str,
    provider: str,
    impl_version: str,
    weights_rev: str | None,
    params: dict[str, Any],
    seed: int,
    request: Any,
) -> str:
    """Key of one call. Hardware and wall time are excluded, so switching GPUs keeps the cache."""
    return hash_bytes(
        canonical_json(
            {
                "op": op,
                "provider": provider,
                "impl_version": impl_version,
                "weights_rev": weights_rev,
                "params": params,
                "seed": seed,
                "request": request,
            }
        )
    )
