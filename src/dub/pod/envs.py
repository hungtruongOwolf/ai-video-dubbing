"""Worker environments on the pod: install (once, cached on the network volume) and launch.

Every http provider names an `env` in its manifest. Providers sharing an env run in one worker
process, so their models share one Python environment and one GPU lock:

    asr  faster-whisper (Whisper large-v3, PhoWhisper)        pod/envs/asr.txt
    sep  audio-separator (Mel-Band RoFormer)                  pod/envs/sep.txt
    tts  OmniVoice + SaT (sentences) + ECAPA (speakers)       pod/envs/tts.txt
    tts2 VieNeu-TTS (its transformers pin differs)            pod/envs/tts2.txt

Installing is `pod/setup_env.sh <env>`: a uv venv under /workspace/dub/envs/<env> whose files are
symlinks into the shared uv cache on the volume (one copy of torch for every env), skipped when
the requirements file has not changed since the last install. Workers start as soon as their
own env is ready and preload their models, so the first job does not pay the load time.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dub.core.config import Config
from dub.worker.protocol import get_json

BASE_PORT = 7101
READY_TIMEOUT_S = 30 * 60  # first boot downloads several GB of weights
# A worker that dies (CUDA fault, killed for memory) is started again; one that keeps dying
# is left failed rather than crash-looping. Calls in flight retry while it reloads (see
# protocol.post_json: kind "unavailable").
MAX_RESTARTS = 3
RESTART_WINDOW_S = 15 * 60
WATCH_EVERY_S = 5.0


@dataclass
class EnvInfo:
    name: str
    providers: list[str]
    port: int
    status: str = "pending"  # pending | installing | starting | ready | failed
    detail: str = ""
    started_at: float = field(default_factory=time.time)
    proc: subprocess.Popen | None = None
    restarts: list[float] = field(default_factory=list)  # when the worker was started again


class EnvManager:
    def __init__(
        self, cfg: Config, root: Path, ws_root: Path, repo: Path, log: Callable[[str], None] = print
    ):
        self.cfg = cfg
        self.root = root
        self.ws_root = ws_root
        self.repo = repo
        self.log = log
        groups: dict[str, list[str]] = {}
        for m in cfg.providers.values():
            if m.runtime == "http" and m.env:
                groups.setdefault(m.env, []).append(m.id)
        self.envs = {
            name: EnvInfo(name, sorted(pids), BASE_PORT + i)
            for i, (name, pids) in enumerate(sorted(groups.items()))
        }
        self.settled = threading.Event()
        self.started = False
        self._lock = threading.Lock()
        if not self.envs:
            self.settled.set()

    # ------------------------------------------------------------ public
    def start(self) -> None:
        self.started = True
        # every env is "coming": jobs submitted while the pod boots route to it and wait
        with self._lock:
            os.environ["DUB_WORKERS_PENDING"] = json.dumps(sorted(self.envs))
        threading.Thread(target=self._run_all, name="envs", daemon=True).start()

    def snapshot(self) -> dict[str, dict]:
        return {
            n: {
                "status": e.status,
                "detail": e.detail,
                "providers": e.providers,
                "restarts": len(e.restarts),
            }
            for n, e in self.envs.items()
        }

    def all_ready(self) -> bool:
        return all(e.status == "ready" for e in self.envs.values())

    def stop(self) -> None:
        for e in self.envs.values():
            if e.proc and e.proc.poll() is None:
                e.proc.terminate()

    # ------------------------------------------------------------ internals
    def _set(self, e: EnvInfo, status: str, detail: str = "") -> None:
        e.status, e.detail = status, detail
        if status in ("ready", "failed"):
            self._not_pending(e.name)
        if status == "failed":  # registered early (it served while loading): calls fail now
            with self._lock:
                workers = json.loads(os.environ.get("DUB_WORKERS") or "{}")
                workers.pop(e.name, None)
                os.environ["DUB_WORKERS"] = json.dumps(workers)
        self.log(f"[env {e.name}] {status} {detail}".rstrip())

    def _not_pending(self, name: str) -> None:
        with self._lock:
            pending = json.loads(os.environ.get("DUB_WORKERS_PENDING") or "[]")
            os.environ["DUB_WORKERS_PENDING"] = json.dumps([n for n in pending if n != name])

    def _run_all(self) -> None:
        # Every env at once: an env that is up to date starts its worker in seconds instead
        # of after the others' checks (the tts worker, which diarization needs, started last).
        # setup_env.sh serializes the real installs (flock), which share the uv cache.
        def one(e: EnvInfo) -> None:
            try:
                self._install(e)
                self._launch(e)
            except Exception as exc:  # noqa: BLE001 - one broken env must not stop the rest
                self._set(e, "failed", str(exc)[-500:])
                return
            self._wait_ready(e)

        waiters = [threading.Thread(target=one, args=(e,), daemon=True) for e in self.envs.values()]
        try:
            for t in waiters:
                t.start()
            for t in waiters:
                t.join()
        finally:
            self.settled.set()
        for e in self.envs.values():
            if e.status == "ready":
                threading.Thread(target=self._watch, args=(e,), daemon=True).start()

    def _watch(self, e: EnvInfo) -> None:
        """Start the worker again when its process exits, at most MAX_RESTARTS times within
        RESTART_WINDOW_S."""
        while True:
            time.sleep(WATCH_EVERY_S)
            if not self.check_worker(e):
                return

    def check_worker(self, e: EnvInfo, now: float | None = None) -> bool:
        """One supervision step; False once the worker is given up."""
        if e.proc is None or e.proc.poll() is None:
            return True
        now = time.time() if now is None else now
        code = e.proc.returncode
        e.restarts = [t for t in e.restarts if now - t < RESTART_WINDOW_S]
        if len(e.restarts) >= MAX_RESTARTS:
            self._set(e, "failed", f"worker exited ({code}) {MAX_RESTARTS} times, given up")
            return False
        e.restarts.append(now)
        self._set(e, "restarting", f"worker exited ({code}), restart {len(e.restarts)}")
        try:
            self._launch(e)
        except Exception as exc:  # noqa: BLE001 - reported, the next step gives up or retries
            self._set(e, "failed", f"restart failed: {exc}")
            return False
        self._wait_ready(e)
        return e.status == "ready"

    def _logs(self) -> Path:
        d = self.root / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _install(self, e: EnvInfo) -> None:
        self._set(e, "installing")
        log_path = self._logs() / f"{e.name}.install.log"
        with open(log_path, "w", encoding="utf-8") as f:
            p = subprocess.run(
                ["bash", str(self.repo / "pod" / "setup_env.sh"), e.name],
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if p.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
            raise RuntimeError(f"install failed (see {log_path}):\n{tail}")

    def _launch(self, e: EnvInfo) -> None:
        venv = self.root / "envs" / e.name
        env = dict(os.environ)
        # CTranslate2 and friends load cuBLAS/cuDNN from the pip "nvidia-*" packages.
        libs = sorted(str(p) for p in venv.glob("lib/python3*/site-packages/nvidia/*/lib"))
        if libs:
            env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
        env["PYTHONUNBUFFERED"] = "1"
        # Kernels compiled at run time kept on the volume: CTranslate2's wheels carry no sm_120
        # code, so the driver compiled its PTX again in every fresh container (M10 review).
        cache = self.root / "cache"
        env.setdefault("CUDA_CACHE_PATH", str(cache / "cuda"))
        env.setdefault("CUDA_CACHE_MAXSIZE", str(4 << 30))
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
        cmd = [
            str(venv / "bin" / "python"),
            "-m",
            "dub.worker",
            "--env",
            e.name,
            "--port",
            str(e.port),
            "--providers",
            ",".join(e.providers),
            "--workspace",
            str(self.ws_root),
            "--config-dir",
            str(self.cfg.root),
            "--preload",
        ]
        out = open(self._logs() / f"{e.name}.log", "a", encoding="utf-8")  # noqa: SIM115
        e.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        e.started_at = time.time()
        self._set(e, "starting", f"loading {', '.join(e.providers)}")

    def _wait_ready(self, e: EnvInfo) -> None:
        url = f"http://127.0.0.1:{e.port}"
        deadline = e.started_at + READY_TIMEOUT_S
        registered = False
        while time.time() < deadline:
            if e.proc and e.proc.poll() is not None:
                tail = (self._logs() / f"{e.name}.log").read_text(errors="replace")[-800:]
                self._set(e, "failed", f"worker exited ({e.proc.returncode}):\n{tail}")
                return
            try:
                health = get_json(f"{url}/v1/health", timeout=5)
                if not registered:
                    # the worker serves while it loads: a job's calls go to it now and wait
                    # there for their model only (ECAPA is loaded long before OmniVoice)
                    self._register(e.name, url)
                    registered = True
                wanted = {p for p in e.providers if self.cfg.providers[p].preload}
                if set(health.get("loaded", [])) >= wanted:
                    self._set(e, "ready", f"{time.time() - e.started_at:.0f}s")
                    return
            except OSError:
                pass
            time.sleep(3)
        self._set(e, "failed", "timed out waiting for the worker")

    def _register(self, name: str, url: str) -> None:
        with self._lock:
            workers = json.loads(os.environ.get("DUB_WORKERS") or "{}")
            workers[name] = url
            os.environ["DUB_WORKERS"] = json.dumps(workers)
