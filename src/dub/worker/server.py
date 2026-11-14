"""A worker process: hosts the providers of one model family behind a tiny HTTP server.

One worker per Python environment ("env"), because model families need conflicting
dependencies (torch versions, CUDA libraries). On the pod, each env is its own uv venv on the
network volume, and the pod server starts one worker per env:

    <venv>/bin/python -m dub.worker --env tts --port 7103 --providers omnivoice \
        --workspace /workspace/dub/ws --config-dir /workspace/dub/repo/configs

Models load once, on first use (or with --preload), and stay resident in VRAM. One thread
runs the models (one GPU, one model run at a time); HTTP handler threads queue calls for it.
Continuous batching (M5): when the runner frees up, it takes the first queued call and every
other queued call with the same provider, op and params, up to the manifest's
`limits.max_batch`, and runs them in one `run_batch` call if the provider has one. Nothing
waits for a batch to fill: an idle worker runs a lone call at once.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dub.contracts.v1 import OPS
from dub.core.config import Config, load_config
from dub.core.provider_api import ProviderError, RunCtx
from dub.store.cas import CAS


class _Task:
    __slots__ = ("pid", "op", "params", "pkey", "req", "done", "result", "error")

    def __init__(self, pid: str, op: str, params: dict[str, Any], req: Any):
        self.pid, self.op, self.params, self.req = pid, op, params, req
        self.pkey = json.dumps(params, sort_keys=True, default=str)
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None


class WorkerState:
    def __init__(self, env: str, providers: list[str], cfg: Config, cas: CAS):
        unknown = [p for p in providers if p not in cfg.providers]
        if unknown:
            raise SystemExit(f"unknown providers for worker '{env}': {unknown}")
        self.env = env
        self.allowed = set(providers)
        self.cfg = cfg
        self.cas = cas
        self.instances: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.queue: list[_Task] = []
        self.cv = threading.Condition()
        self.batches: list[int] = []  # sizes of the batches run (health, tests)
        self._runner: threading.Thread | None = None

    # ------------------------------------------------------------ the model runner
    def submit(self, pid: str, op: str, params: dict[str, Any], req: Any) -> Any:
        """Queue one call for the runner thread and wait for its result."""
        task = _Task(pid, op, params, req)
        with self.cv:
            if self._runner is None:
                self._runner = threading.Thread(target=self._loop, daemon=True)
                self._runner.start()
            self.queue.append(task)
            self.cv.notify()
        task.done.wait()
        if task.error is not None:
            raise task.error
        return task.result

    def _take(self) -> list[_Task]:
        with self.cv:
            while not self.queue:
                self.cv.wait()
            first = self.queue.pop(0)
            limit = self.cfg.providers[first.pid].limits.max_batch
            batch, rest = [first], []
            for t in self.queue:
                same = (t.pid, t.op, t.pkey) == (first.pid, first.op, first.pkey)
                (batch if same and len(batch) < limit else rest).append(t)
            self.queue[:] = rest
            return batch

    def _loop(self) -> None:
        while True:
            batch = self._take()
            try:
                self._execute(batch)
            finally:
                for t in batch:
                    t.done.set()

    def _execute(self, batch: list[_Task]) -> None:
        try:
            self._execute_batch(batch)
        finally:
            _release_gpu_cache()

    def _execute_batch(self, batch: list[_Task]) -> None:
        first = batch[0]
        try:
            with self.lock:
                inst = self.instance(first.pid)
        except BaseException as e:  # noqa: BLE001 - a failed load fails every queued call
            for t in batch:
                t.error = e
            return
        self.batches.append(len(batch))
        if len(batch) > 1 and hasattr(inst, "run_batch"):
            try:
                results = inst.run_batch(
                    first.op, [t.req for t in batch], self.ctx(first.pid, first.params)
                )
                for t, r in zip(batch, results, strict=True):
                    t.result = r
                return
            except Exception:  # noqa: BLE001 - retry one by one: one bad call fails alone
                traceback.print_exc()
                _release_gpu_cache()  # a batch that ran out of memory may fit one by one
        for t in batch:
            try:
                t.result = inst.run(t.op, t.req, self.ctx(t.pid, t.params))
            except BaseException as e:  # noqa: BLE001 - reported to that caller
                t.error = _classify(e)

    def unload(self, pid: str) -> bool:
        """Free a model's GPU memory (the 24 GB profile frees phase-A models after phase A)."""
        with self.lock:
            inst = self.instances.pop(pid, None)
        if inst is None:
            return False
        del inst
        import gc

        gc.collect()
        _release_gpu_cache()
        return True

    def ctx(self, pid: str, params: dict[str, Any]) -> RunCtx:
        return RunCtx(
            cas=self.cas,
            manifest=self.cfg.providers[pid],
            params=params,
            prompts_dir=self.cfg.prompts_dir,
            log=lambda m: print(f"[{self.env}] {m}", file=sys.stderr, flush=True),
        )

    def instance(self, pid: str) -> Any:
        """Create (and load) a provider once. Call with self.lock held."""
        if pid not in self.instances:
            m = self.cfg.providers[pid]
            module_name, _, cls_name = (m.impl or "").partition(":")
            cls = getattr(importlib.import_module(module_name), cls_name)
            inst = cls(m)
            if hasattr(inst, "load"):
                inst.load(self.ctx(pid, dict(m.params)))
            self.instances[pid] = inst
        return self.instances[pid]


def _release_gpu_cache() -> None:
    """Give cached-but-free CUDA memory back, so the other workers on the GPU can use it."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _classify(e: BaseException) -> BaseException:
    """Out of GPU memory is transient here (other workers share the GPU): retryable."""
    if isinstance(e, ProviderError):
        return e
    text = str(e).lower()
    if "out of memory" in text or "cudaerrorinvaliddevice" in text:
        return ProviderError(f"{type(e).__name__}: {e}", kind="oom", retryable=True)
    return e


def make_handler(state: WorkerState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep stdout quiet; errors go to stderr below
            pass

        def _reply(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, kind: str, message: str, retryable: bool = False) -> None:
            self._reply(code, {"error": {"kind": kind, "message": message, "retryable": retryable}})

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        def do_GET(self):
            if self.path == "/v1/health":
                self._reply(
                    200,
                    {
                        "ok": True,
                        "env": state.env,
                        "providers": sorted(state.allowed),
                        "loaded": sorted(state.instances),
                        # batch size -> runs (continuous batching at work)
                        "batches": {
                            str(n): state.batches.count(n) for n in sorted(set(state.batches))
                        },
                    },
                )
            else:
                self._error(404, "bad_input", f"no route {self.path}")

        def do_POST(self):
            try:
                body = self._body()
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._error(400, "bad_input", "body is not JSON")
            pid = body.get("provider")
            if pid not in state.allowed:
                return self._error(404, "unsupported", f"provider {pid!r} is not hosted here")
            t0 = time.perf_counter()
            if self.path == "/v1/unload":
                return self._reply(200, {"ok": True, "freed": state.unload(pid)})
            if self.path == "/v1/load":
                try:
                    with state.lock:
                        state.instance(pid)
                except Exception as e:  # noqa: BLE001 - report every load failure to the core
                    traceback.print_exc()
                    return self._error(500, "crash", f"load failed: {e}")
                return self._reply(200, {"ok": True, "ms": (time.perf_counter() - t0) * 1000})
            if self.path != "/v1/run":
                return self._error(404, "bad_input", f"no route {self.path}")

            op = body.get("op")
            if op not in OPS:
                return self._error(400, "bad_input", f"unknown op {op!r}")
            try:
                req = OPS[op][0].model_validate(body.get("request") or {})
            except ValidationError as e:
                return self._error(400, "bad_input", f"invalid request: {e}")
            params = body.get("params") or {}
            try:
                res = state.submit(pid, op, params, req)
            except ProviderError as e:
                return self._error(500, e.kind, str(e), e.retryable)
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                print(tb, file=sys.stderr, flush=True)
                return self._error(500, "crash", f"{type(e).__name__}: {e}\n{tb[-1500:]}")
            self._reply(
                200,
                {"result": res.model_dump(mode="json"), "ms": (time.perf_counter() - t0) * 1000},
            )

    return Handler


def serve(state: WorkerState, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dub.worker")
    ap.add_argument("--env", required=True)
    ap.add_argument("--providers", required=True, help="comma-separated provider ids")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--preload", action="store_true", help="load every model before serving")
    a = ap.parse_args(argv)

    cfg = load_config(a.config_dir)
    state = WorkerState(
        a.env, [p for p in a.providers.split(",") if p], cfg, CAS(Path(a.workspace) / "cas")
    )
    if a.preload:
        for pid in sorted(p for p in state.allowed if cfg.providers[p].preload):
            t0 = time.perf_counter()
            with state.lock:
                state.instance(pid)
            print(f"[{a.env}] loaded {pid} in {time.perf_counter() - t0:.1f}s", flush=True)
    server = serve(state, a.host, a.port)
    print(f"[{a.env}] serving {sorted(state.allowed)} on {a.host}:{a.port}", flush=True)
    server.serve_forever()
    return 0
