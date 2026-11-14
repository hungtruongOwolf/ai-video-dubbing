"""Wire protocol between the core and a worker process (HTTP/1.1 + JSON).

    GET  /v1/health  -> {"ok": true, "env": "...", "providers": [...], "loaded": [...]}
    POST /v1/load    {"provider": id}                                   -> {"ok": true, "ms": ...}
    POST /v1/run     {"provider": id, "op": op, "params": {...}, "request": {...}}
                     -> 200 {"result": {...}, "ms": ...}
                     -> 4xx/5xx {"error": {"kind": ..., "message": ..., "retryable": bool}}

Only small JSON travels over the wire. Audio stays in the shared CAS on the network volume
and is referenced by CAS key, exactly like in-process providers.

The core finds workers through the DUB_WORKERS environment variable, a JSON object
{"<env name>": "http://127.0.0.1:<port>"} set by the pod server once each worker is up.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from dub.core.provider_api import ProviderError


def worker_urls() -> dict[str, str]:
    raw = os.environ.get("DUB_WORKERS", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def post_json(url: str, body: dict[str, Any], timeout: float = 3600.0) -> dict[str, Any]:
    """POST JSON and return the decoded reply; worker errors become ProviderError."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8")).get("error", {})
        except (json.JSONDecodeError, UnicodeDecodeError):
            err = {}
        raise ProviderError(
            err.get("message") or f"worker returned HTTP {e.code}",
            kind=err.get("kind", "crash"),
            retryable=bool(err.get("retryable", False)),
        ) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        msg = f"worker unreachable at {url}: {e}"
        raise ProviderError(msg, kind="crash", retryable=True) from e


def get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
