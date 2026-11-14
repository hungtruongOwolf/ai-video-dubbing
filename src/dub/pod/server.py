"""dubd: the pipeline server on the GPU pod.

The laptop talks to it through the Runpod HTTPS proxy (https://<pod id>-8000.proxy.runpod.net).
Every route except /health needs `Authorization: Bearer <DUB_TOKEN>`; `dub up` generates the
token and keeps it in the laptop's workspace.

    GET  /health                          setup progress, worker status, running jobs
    PUT  /uploads/<id>/<n>                one part of an upload (parts stay under the proxy's
                                          100 MB request limit)
    POST /uploads/<id>/commit             {"ext": ".mka", "parts": N} -> {"key": CAS key}
    POST /jobs                            job spec -> {"job_id": ...}
    GET  /jobs/<id>                       status, log lines, result keys
    GET  /jobs/<id>/files/<name>          report.md, trace.json, units.json, cast.json, routes.jsonl
    GET  /cas/<key>                       download a file from the CAS

Idle watchdog: when no job has run and no request has arrived for DUB_IDLE_MINUTES, the pod
terminates itself through the Runpod API with the pod-scoped key Runpod injects
(RUNPOD_API_KEY, RUNPOD_POD_ID). Everything worth keeping is already on the network volume.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from fnmatch import fnmatch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dub.contracts.v1 import Edit
from dub.core.config import Config, load_config
from dub.core.ctx import OP_RESULT
from dub.core.pipeline import JobSpec, run_job
from dub.core.router import parse_overrides
from dub.ops.preview import cut_mp3
from dub.pod.envs import EnvManager
from dub.pod.gpu import GpuSampler, total_memory_mib
from dub.store import Workspace
from dub.worker.protocol import get_json

JOB_FILES = {
    "report.md",
    "trace.json",
    "units.json",
    "cast.json",
    "cast.md",
    "routes.jsonl",
    "tiles.json",
    "metrics.json",
    "segmentation.json",
    "gpu.csv",
}
_ID = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_KEY = re.compile(r"^[0-9a-f]{32}(\.[A-Za-z0-9]{1,8})?$")


def overrides_for(cfg: Config, use: list[str], fake: bool) -> list[tuple[str, list[str]]]:
    overrides = parse_overrides(use)
    if fake:
        for op in OP_RESULT:
            if not any(fnmatch(op, pat) for pat, _ in overrides):
                fp = cfg.fake_provider_for(op)
                if fp:
                    overrides.append((op, [fp]))
    return overrides


class Jobs:
    def __init__(self, cfg: Config, ws: Workspace, envs: EnvManager, touch: Callable[[], None]):
        self.cfg, self.ws, self.envs, self.touch = cfg, ws, envs, touch
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def running(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j["status"] in ("queued", "running"))

    def submit(self, body: dict[str, Any]) -> str:
        spec = JobSpec(
            audio_key=body["audio_key"],
            source_name=body.get("source_name", "input"),
            tgt_lang=body["tgt_lang"],
            src_lang=body.get("src_lang", "auto"),
            profile=body.get("profile"),
            transcript_key=body.get("transcript_key"),
            speakers_hint=body.get("speakers_hint"),
            subs_only=bool(body.get("subs_only", False)),
            edits=[Edit.model_validate(e) for e in body.get("edits", [])],
            export=bool(body.get("export", True)),
            analysis_only=bool(body.get("analysis_only", False)),
            confirm_cast=bool(body.get("confirm_cast", False)),
            fresh=bool(body.get("fresh", False)),
            params=dict(body.get("params") or {}),
        )
        overrides = overrides_for(self.cfg, list(body.get("use", [])), bool(body.get("fake")))
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[job_id] = {
                "status": "queued",
                "log": [],
                "result": None,
                "error": None,
                "created": time.time(),
            }
        threading.Thread(target=self._run, args=(job_id, spec, overrides), daemon=True).start()
        return job_id

    def _log(self, job_id: str, line: str) -> None:
        with self.lock:
            self.jobs[job_id]["log"].append(f"{time.strftime('%H:%M:%S')} {line}")
        self.touch()

    def _run(self, job_id: str, spec: JobSpec, overrides) -> None:
        job = self.jobs[job_id]
        try:
            if self.envs.started and not self.envs.settled.is_set():
                self._log(job_id, "waiting for the model workers to finish loading")
                self.envs.settled.wait()
            job["status"] = "running"
            t0 = time.perf_counter()
            gpu = GpuSampler().start()
            try:
                res = asyncio.run(
                    run_job(
                        self.cfg, self.ws, spec, overrides=overrides,
                        log=lambda m: self._log(job_id, m),
                    )
                )  # fmt: skip
            finally:
                gpu.stop()
            gpu.write(res.job_dir / "gpu.csv")
            job["result"] = {
                "job_dir": res.job_id,
                "src_lang": res.src_lang,
                "tgt_lang": res.tgt_lang,
                "dub_m4a_key": res.dub_m4a_key,
                "dub_audio_key": res.dub_audio_key,
                "subs": res.subs,
                "tiles": [t.model_dump() for t in res.tiles],
                "a48k_key": res.a48k_key,
                "warnings": res.warnings,
                "redone": res.redone,
                "seconds": round(time.perf_counter() - t0, 2),
                "calls": res.calls,
                "cache_hits": res.cache_hits,
                "gpu": gpu.summary(),
            }
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001 - report every failure to the laptop
            traceback.print_exc()
            job["error"] = f"{type(e).__name__}: {e}"
            job["status"] = "error"
        finally:
            self.touch()

    def wait(self, job_id: str, timeout_s: float, seen: int, min_s: float = 2.0) -> None:
        """Long-poll: return when the job ends, or after `min_s` if there are new log lines,
        or at the timeout. A short job therefore costs a single round trip."""
        start = time.time()
        deadline = start + min(timeout_s, 25)  # stay under proxy timeouts
        while time.time() < deadline:
            j = self.jobs.get(job_id)
            if j is None or j["status"] in ("done", "error"):
                return
            if len(j["log"]) > seen and time.time() - start >= min_s:
                return
            time.sleep(0.03)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            j = self.jobs.get(job_id)
            return None if j is None else {k: (list(v) if k == "log" else v) for k, v in j.items()}


class DubServer:
    def __init__(
        self,
        cfg: Config,
        ws: Workspace,
        envs: EnvManager,
        token: str,
        idle_minutes: float = 10.0,
        version: str = "dev",
    ):
        self.cfg, self.ws, self.envs, self.token = cfg, ws, envs, token
        self.idle_s = idle_minutes * 60
        self.version = version
        self.last_activity = time.time()
        self.started = time.time()
        self.jobs = Jobs(cfg, ws, envs, self.touch)
        self.gc_state: dict[str, Any] = {"status": "idle"}  # GC runs in the background
        self.uploads = ws.root / "uploads"

    def job_files(self, r: dict[str, Any]) -> dict[str, str]:
        """The small result files of a finished job, plus its subtitles, by name."""
        d = self.ws.root / "jobs" / r["job_dir"]
        files = {n: (d / n).read_text(encoding="utf-8") for n in JOB_FILES if (d / n).exists()}
        for name, key in r["subs"].items():
            files[f"subs:{name}"] = self.ws.cas.read_text(key)
        return files

    def touch(self) -> None:
        self.last_activity = time.time()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "version": self.version,
            "ready": self.envs.settled.is_set(),
            "envs": self.envs.snapshot(),
            "jobs_running": self.jobs.running(),
            "idle_s": round(time.time() - self.last_activity),
            "uptime_s": round(time.time() - self.started),
        }

    # ------------------------------------------------------------ uploads
    def put_part(self, uid: str, n: int, data: bytes) -> None:
        d = self.uploads / uid
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{n:05d}.part").write_bytes(data)

    def commit(self, uid: str, parts: int, ext: str) -> str:
        d = self.uploads / uid
        files = [d / f"{i:05d}.part" for i in range(parts)]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise ValueError(f"missing upload parts: {missing}")
        tmp = self.ws.cas.tmp_path(ext)
        with open(tmp, "wb") as out:
            for f in files:
                with open(f, "rb") as src:
                    shutil.copyfileobj(src, out)
        shutil.rmtree(d, ignore_errors=True)
        return self.ws.cas.put_file(tmp, ext=ext, move=True)

    # ------------------------------------------------------------ watchdog
    def _workers_busy(self) -> bool:
        """Experiments call the workers directly (not through a job): their activity counts."""
        for url in json.loads(os.environ.get("DUB_WORKERS") or "{}").values():
            idle = _worker_idle_s(url)
            if idle is not None and idle < 60:
                return True
        return False

    def watchdog(self, terminate: Callable[[], None]) -> None:
        while True:
            time.sleep(30)
            # Never idle out while a job runs or while the first boot is still installing envs
            # and downloading weights (that can take longer than the idle limit).
            installing = self.envs.started and not self.envs.settled.is_set()
            if self.jobs.running() or installing or self._workers_busy():
                self.touch()
                continue
            if time.time() - self.last_activity > self.idle_s:
                print(
                    f"[dubd] idle for {self.idle_s / 60:.0f} min: terminating the pod", flush=True
                )
                terminate()
                return


def _worker_idle_s(url: str) -> float | None:
    try:
        return float(get_json(f"{url}/v1/health", timeout=5).get("idle_s", 1e9))
    except (OSError, ValueError):
        return None


def terminate_this_pod() -> None:
    pod_id, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not key:
        print("[dubd] not on Runpod (no RUNPOD_POD_ID): nothing to terminate", flush=True)
        return
    req = urllib.request.Request(
        f"https://api.runpod.io/v2/pods/{pod_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        urllib.request.urlopen(req, timeout=30).close()
    except Exception as e:  # noqa: BLE001
        print(f"[dubd] self-terminate failed: {e}", flush=True)


def make_handler(srv: DubServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _file(self, path: Path) -> None:
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _authed(self) -> bool:
            if secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {srv.token}"):
                srv.touch()
                return True
            # Drain the body first: replying while the client is still sending resets the
            # connection on some platforms.
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.close_connection = True
            self._json(401, {"error": "missing or wrong token"})
            return False

        def do_GET(self):
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if parts == ["health"]:
                return self._json(200, srv.health())
            if not self._authed():
                return
            if parts == ["gc"]:
                return self._json(200, srv.gc_state)
            if len(parts) == 2 and parts[0] == "jobs" and _ID.match(parts[1]):
                query = self.path.partition("?")[2]
                q = {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}
                if "wait" in q:
                    srv.jobs.wait(parts[1], float(q["wait"]), int(q.get("seen", 0)))
                job = srv.jobs.get(parts[1])
                if job and job.get("result") and q.get("bundle") == "1":
                    job["files"] = srv.job_files(job["result"])  # saves one more round trip
                return self._json(200, job) if job else self._json(404, {"error": "no such job"})
            if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "files":
                job = srv.jobs.get(parts[1])
                if not job or not job.get("result") or parts[3] not in JOB_FILES:
                    return self._json(404, {"error": "no such file"})
                path = srv.ws.root / "jobs" / job["result"]["job_dir"] / parts[3]
                return self._file(path) if path.exists() else self._json(404, {"error": "missing"})
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "bundle":
                job = srv.jobs.get(parts[1])
                if not job or not job.get("result"):
                    return self._json(404, {"error": "no result"})
                return self._json(200, {"files": srv.job_files(job["result"])})
            if len(parts) == 2 and parts[0] == "cas" and _KEY.match(parts[1]):
                path = srv.ws.cas.path(parts[1])
                return self._file(path) if path.exists() else self._json(404, {"error": "missing"})
            self._json(404, {"error": f"no route {self.path}"})

        def do_HEAD(self):
            parts = [p for p in self.path.split("/") if p]
            ok = (
                secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {srv.token}")
                and len(parts) == 2
                and parts[0] == "cas"
                and _KEY.match(parts[1])
                and srv.ws.cas.exists(parts[1])
            )
            self.send_response(200 if ok else 404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _mp3(self, data: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_PUT(self):
            parts = [p for p in self.path.split("/") if p]
            if not self._authed():
                return
            if (
                len(parts) == 3
                and parts[0] == "uploads"
                and _ID.match(parts[1])
                and parts[2].isdigit()
            ):
                srv.put_part(parts[1], int(parts[2]), self._body())
                return self._json(200, {"ok": True})
            self._json(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            parts = [p for p in self.path.split("/") if p]
            if not self._authed():
                return
            try:
                body = json.loads(self._body() or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "body is not JSON"})
            try:
                if (
                    len(parts) == 3
                    and parts[0] == "uploads"
                    and parts[2] == "commit"
                    and _ID.match(parts[1])
                ):
                    ext = str(body.get("ext", ""))
                    if not re.match(r"^\.[A-Za-z0-9]{1,8}$", ext):
                        return self._json(400, {"error": "bad ext"})
                    return self._json(200, {"key": srv.commit(parts[1], int(body["parts"]), ext)})
                if parts == ["preview"]:
                    segs = [(str(k), int(st)) for k, st in body["segments"]]
                    if not all(_KEY.match(k) for k, _ in segs):
                        return self._json(400, {"error": "bad key"})
                    data = cut_mp3(srv.ws.cas, segs, int(body["from_ms"]), int(body["to_ms"]))
                    return self._mp3(data)
                if parts == ["jobs"]:
                    return self._json(200, {"job_id": srv.jobs.submit(body)})
                if parts == ["gc"]:
                    if srv.jobs.running():
                        return self._json(409, {"error": "a job is running; try again later"})
                    if srv.gc_state["status"] == "running":
                        return self._json(202, srv.gc_state)
                    # A big CAS takes minutes to sweep, longer than the proxy keeps a request
                    # open: run it in the background and let the laptop poll GET /gc.
                    min_age, dry = float(body.get("min_age_s", 600)), bool(body.get("dry_run"))
                    evict_s = body.get("evict_older_than_s")
                    srv.gc_state = {"status": "running"}

                    def sweep() -> None:
                        from dub.store.gc import collect

                        try:
                            rep = collect(srv.ws, min_age_s=min_age, dry_run=dry,
                                          evict_older_than_s=evict_s)  # fmt: skip
                            srv.gc_state = {"status": "done", "report": rep.as_dict()}
                        except Exception as e:  # noqa: BLE001 - reported to the poller
                            srv.gc_state = {"status": "error", "error": str(e)}

                    threading.Thread(target=sweep, daemon=True).start()
                    return self._json(202, srv.gc_state)
            except (KeyError, ValueError) as e:
                return self._json(400, {"error": str(e)})
            self._json(404, {"error": f"no route {self.path}"})

    return Handler


def serve(srv: DubServer, host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(srv))
    httpd.daemon_threads = True
    return httpd


def main() -> int:
    root = Path(os.environ.get("DUB_ROOT", "/workspace/dub"))
    token = os.environ.get("DUB_TOKEN", "")
    if len(token) < 16:
        raise SystemExit("DUB_TOKEN is missing or too short")
    repo = Path(os.environ.get("DUB_REPO_DIR", root / "repo"))
    cfg = load_config(repo / "configs")
    ws_root = root / "ws"
    # The SQLite index stays on the container disk: SQLite locking is unreliable on a network
    # file system, and the index can be rebuilt from CAS + memo anyway.
    ws = Workspace.open(ws_root, db_path=Path(os.environ.get("DUB_DB", "/tmp/dub.sqlite")))
    mem = total_memory_mib()
    if mem:  # the pipeline frees phase-A models on small GPUs (throughput.free_phase_a_models)
        os.environ["DUB_GPU_MEM_MIB"] = str(mem)
    envs = EnvManager(cfg, root, ws_root, repo)
    srv = DubServer(
        cfg,
        ws,
        envs,
        token,
        float(os.environ.get("DUB_IDLE_MINUTES", "10")),
        os.environ.get("DUB_REF", "dev"),
    )
    envs.start()
    threading.Thread(target=srv.watchdog, args=(terminate_this_pod,), daemon=True).start()
    port = int(os.environ.get("DUB_PORT", "8000"))
    print(f"[dubd] {srv.version} listening on :{port}", flush=True)
    serve(srv, port=port).serve_forever()
    return 0
