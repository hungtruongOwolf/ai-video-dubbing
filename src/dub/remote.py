"""Laptop side of the GPU pod: rent it, talk to dubd on it, delete it.

    dub up      create a pod on the network volume's data center (GPU list tried in order),
                wait until dubd and every model worker are ready
    dub run ... --remote
                upload the audio track, run the job on the pod, download the dubbed audio
                (AAC) and subtitles, mux locally
    dub down    terminate the pod (the network volume and its cache stay)
    dub status  pod state, price, worker status

The Runpod API key is read from RUNPOD_API_KEY (on Windows also from the user environment in the
registry, so a freshly set variable works without restarting the terminal).
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

API = "https://api.runpod.io/v2"
PART_BYTES = 32 * 1024 * 1024  # under the proxy's 100 MB request limit
USER_AGENT = "dub-cli/0.1"  # the Runpod proxy (Cloudflare) blocks Python's default agent
CAPACITY_HINTS = (
    "no longer any instances",
    "instances available",
    "no available",
    "not available",
    "insufficient",
    "capacity",
    "no gpu",
    "unavailable",
    "could not find",
)


class RemoteError(RuntimeError):
    pass


# -------------------------------------------------------------------- credentials, config


def runpod_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key and sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = str(winreg.QueryValueEx(k, "RUNPOD_API_KEY")[0])
        except OSError:
            key = ""
    if not key:
        cfg = Path.home() / ".runpod" / "config.toml"
        if cfg.exists():
            for line in cfg.read_text(encoding="utf-8").splitlines():
                name, _, value = line.partition("=")
                if name.strip().lower() in ("api_key", "apikey"):
                    key = value.strip().strip('"').strip("'")
                    break
    if not key:
        raise RemoteError("RUNPOD_API_KEY is not set (see docs/m1-gpu-pod.md)")
    return key


@dataclass
class PodConfig:
    image: str
    gpus: list[str]
    volume_id: str | None
    data_center: str | None
    min_cuda: str | None = "12.8"
    container_disk_gb: int = 30
    port: int = 8000
    idle_minutes: int = 10
    repo: str = "https://github.com/hungtruongOwolf/ai-video-dubbing.git"
    claude_secret: str | None = None

    @classmethod
    def load(cls, config_dir: Path) -> PodConfig:
        data = yaml.safe_load((config_dir / "pod.yaml").read_text(encoding="utf-8")) or {}
        local = config_dir.parent / "pod.local.yaml"
        if local.exists():
            data.update(yaml.safe_load(local.read_text(encoding="utf-8")) or {})
        return cls(**data)

    def raw_url(self, ref: str, path: str) -> str:
        slug = self.repo.removesuffix(".git").split("github.com/")[-1]
        return f"https://raw.githubusercontent.com/{slug}/{ref}/{path}"


@dataclass
class PodState:
    pod_id: str
    url: str
    token: str
    gpu: str
    ref: str
    created_at: float = field(default_factory=time.time)
    cost_per_hr: float | None = None

    @staticmethod
    def path(ws_root: Path) -> Path:
        return Path(ws_root) / "pod.json"

    def save(self, ws_root: Path) -> None:
        self.path(ws_root).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, ws_root: Path) -> PodState | None:
        p = cls.path(ws_root)
        return cls(**json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None


# -------------------------------------------------------------------- HTTP helpers


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: float = 60,
) -> tuple[int, bytes]:
    headers = {"User-Agent": USER_AGENT, **(headers or {})}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class Runpod:
    def __init__(self, key: str):
        self.key = key

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        data = json.dumps(body).encode("utf-8") if body is not None else None
        code, raw = _request(method, f"{API}{path}", headers=headers, data=data)
        text = raw.decode("utf-8", errors="replace")
        if code >= 400:
            raise RemoteError(f"Runpod API {method} {path} -> HTTP {code}: {text[:500]}")
        return json.loads(text) if text.strip() else None

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/pods", body)

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        try:
            return self._call("GET", f"/pods/{pod_id}")
        except RemoteError as e:
            if "HTTP 404" in str(e):
                return None
            raise

    def update_pod(self, pod_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("PATCH", f"/pods/{pod_id}", body)

    def delete_pod(self, pod_id: str) -> None:
        self._call("DELETE", f"/pods/{pod_id}")

    def secret_names(self) -> set[str]:
        try:
            data = self._call("GET", "/account/secrets")
        except RemoteError:
            return set()
        items = data.get("secrets", data) if isinstance(data, dict) else data
        return {str(s.get("name")) for s in items or [] if isinstance(s, dict)}


class Dubd:
    """Client for dubd on the pod."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}

    def health(self) -> dict[str, Any] | None:
        try:
            code, raw = _request("GET", f"{self.url}/health", timeout=15)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return None
        if code != 200:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _json(self, method: str, path: str, body: Any = None, timeout: float = 120) -> Any:
        headers = dict(self.headers)
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        code, raw = _request(
            method, f"{self.url}{path}", headers=headers, data=data, timeout=timeout
        )
        if code >= 400:
            raise RemoteError(f"dubd {method} {path} -> HTTP {code}: {raw[:300]!r}")
        return json.loads(raw)

    def upload(self, path: Path, ext: str, log: Callable[[str], None] = print) -> str:
        uid = uuid.uuid4().hex
        size = path.stat().st_size
        n = 0
        with open(path, "rb") as f:
            while chunk := f.read(PART_BYTES):
                for attempt in range(3):
                    code, raw = _request(
                        "PUT",
                        f"{self.url}/uploads/{uid}/{n}",
                        headers=self.headers,
                        data=chunk,
                        timeout=600,
                    )
                    if code == 200:
                        break
                    if attempt == 2:
                        raise RemoteError(f"upload part {n} failed: HTTP {code} {raw[:200]!r}")
                    time.sleep(3)
                n += 1
                log(f"[laptop] uploaded {min(n * PART_BYTES, size) / 1e6:.1f}/{size / 1e6:.1f} MB")
        return self._json("POST", f"/uploads/{uid}/commit", {"parts": n, "ext": ext})["key"]

    def has(self, key: str) -> bool:
        code, _ = _request("HEAD", f"{self.url}/cas/{key}", headers=self.headers, timeout=30)
        return code == 200

    def preview(self, segments: list[tuple[str, int]], from_ms: int, to_ms: int) -> bytes:
        body = {"segments": [[k, st] for k, st in segments], "from_ms": from_ms, "to_ms": to_ms}
        headers = {**self.headers, "Content-Type": "application/json"}
        code, raw = _request(
            "POST",
            f"{self.url}/preview",
            headers=headers,
            data=json.dumps(body).encode("utf-8"),
            timeout=60,
        )
        if code != 200:
            raise RemoteError(f"preview failed: HTTP {code} {raw[:200]!r}")
        return raw

    def submit(self, body: dict[str, Any]) -> str:
        return self._json("POST", "/jobs", body)["job_id"]

    def job(self, job_id: str, wait: float = 0, seen: int = 0) -> dict[str, Any]:
        q = f"?wait={wait}&seen={seen}" if wait else ""
        return self._json("GET", f"/jobs/{job_id}{q}", timeout=wait + 30)

    def bundle(self, job_id: str) -> dict[str, str]:
        """All small result files of a job in one round trip."""
        return self._json("GET", f"/jobs/{job_id}/bundle")["files"]

    def download(self, path_on_server: str, dest: Path) -> Path:
        req = urllib.request.Request(f"{self.url}{path_on_server}", headers=self.headers)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
            while block := resp.read(1 << 20):
                out.write(block)
        os.replace(tmp, dest)
        return dest


# -------------------------------------------------------------------- up / down / status


def pod_status_of(pod: dict[str, Any]) -> str:
    return str(pod.get("status") or pod.get("desiredStatus") or "UNKNOWN")


def git_ref(repo_root: Path) -> str:
    """The commit the pod should run: local HEAD, which must already be on GitHub."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    if not head:
        raise RemoteError("not a git checkout: cannot tell the pod which code to run")
    if not git("branch", "-r", "--contains", head):
        raise RemoteError(f"commit {head[:8]} is not pushed to GitHub yet: git push first")
    if git("status", "--porcelain", "--untracked-files=no"):
        print("[laptop] warning: uncommitted changes are NOT used by the pod", flush=True)
    return head


def pod_body(cfg: PodConfig, gpu: str, ref: str, token: str, claude_secret: bool) -> dict:
    env = {
        "DUB_TOKEN": token,
        "DUB_REF": ref,
        "DUB_REPO": cfg.repo,
        "DUB_IDLE_MINUTES": str(cfg.idle_minutes),
        "DUB_PORT": str(cfg.port),
    }
    if claude_secret and cfg.claude_secret:
        # Runpod substitutes the stored secret when the pod boots.
        env["CLAUDE_CODE_OAUTH_TOKEN"] = "{{ RUNPOD_SECRET_" + cfg.claude_secret + " }}"
    boot = f"curl -fsSL {cfg.raw_url(ref, 'pod/bootstrap.sh')} | bash"
    body: dict[str, Any] = {
        "name": f"dub-{time.strftime('%m%d-%H%M')}",
        "image": cfg.image,
        "cloud": "SECURE",
        "gpu": {"id": gpu, "count": 1},
        "disk": cfg.container_disk_gb,
        "ports": [f"{cfg.port}/http"],
        "env": env,
        "entrypoint": ["/bin/bash", "-c"],
        "cmd": [boot],
    }
    if cfg.min_cuda:
        body["gpu"]["minCudaVersion"] = cfg.min_cuda
    if cfg.volume_id:
        body["mounts"] = {"network": [{"volumeId": cfg.volume_id, "path": "/workspace"}]}
    if cfg.data_center:
        body["dataCenterIds"] = [cfg.data_center]
    return body


def pod_up(
    cfg: PodConfig,
    ws_root: Path,
    repo_root: Path,
    *,
    gpus: list[str] | None = None,
    log: Callable[[str], None] = print,
    wait_minutes: float = 45,
) -> PodState:
    api = Runpod(runpod_key())
    state = PodState.load(ws_root)
    if state and (pod := api.get_pod(state.pod_id)) and pod_status_of(pod) == "RUNNING":
        log(f"[up] reusing pod {state.pod_id} ({state.gpu})")
        ref = git_ref(repo_root)
        if ref != state.ref:
            # New code: change DUB_REF; Runpod restarts the container in place (same machine,
            # no stock risk) and the bootstrap checks out the new commit.
            has_secret = "CLAUDE_CODE_OAUTH_TOKEN" in (pod.get("env") or {})
            env = pod_body(cfg, state.gpu, ref, state.token, has_secret)["env"]
            api.update_pod(state.pod_id, {"env": env})
            log(f"[up] updating the pod's code {state.ref[:8]} -> {ref[:8]} (container restart)")
            state.ref = ref
            state.save(ws_root)
            time.sleep(15)  # let the old dubd go away before polling health
    else:
        ref = git_ref(repo_root)
        token = secrets.token_urlsafe(24)
        has_secret = bool(cfg.claude_secret) and cfg.claude_secret in api.secret_names()
        if cfg.claude_secret and not has_secret:
            log(
                f"[up] no Runpod secret '{cfg.claude_secret}': translation with claude_cli will "
                "not work on the pod (see docs/m1-gpu-pod.md)"
            )
        errors = []
        for gpu in gpus or cfg.gpus:
            try:
                pod = api.create_pod(pod_body(cfg, gpu, ref, token, has_secret))
            except RemoteError as e:
                errors.append(f"{gpu}: {e}")
                if any(h in str(e).lower() for h in CAPACITY_HINTS):
                    log(f"[up] {gpu}: no capacity, trying the next GPU")
                    continue
                raise
            state = PodState(
                pod_id=pod["id"],
                url=f"https://{pod['id']}-{cfg.port}.proxy.runpod.net",
                token=token,
                gpu=gpu,
                ref=ref,
                cost_per_hr=pod.get("cost"),
            )
            state.save(ws_root)
            log(
                f"[up] created pod {state.pod_id} on {gpu} "
                f"at ${state.cost_per_hr}/h, code {ref[:8]}"
            )
            break
        else:
            raise RemoteError("no GPU available right now:\n  " + "\n  ".join(errors))
    wait_ready(state, log, wait_minutes)
    return state


def wait_ready(state: PodState, log: Callable[[str], None], wait_minutes: float) -> None:
    dubd = Dubd(state.url, state.token)
    deadline = time.time() + wait_minutes * 60
    last = ""
    t0 = time.time()
    while time.time() < deadline:
        h = dubd.health()
        if h:
            envs = h.get("envs", {})
            line = ", ".join(f"{n}: {e['status']}" for n, e in envs.items()) or "no workers"
            if line != last:
                log(f"[up] {time.time() - t0:5.0f}s  dubd up, {line}")
                last = line
            if h.get("ready"):
                failed = {n: e for n, e in envs.items() if e["status"] != "ready"}
                if failed:
                    for n, e in failed.items():
                        log(f"[up] worker {n} {e['status']}: {e['detail']}")
                    raise RemoteError("some workers failed to start (see above)")
                log(f"[up] ready after {time.time() - t0:.0f}s: {state.url}")
                return
        elif not last:
            last = "booting"
            log("[up] waiting for the pod to boot (first boot installs envs and models)")
        time.sleep(10)
    raise RemoteError(f"pod not ready after {wait_minutes:.0f} min; check `dub status`")


def pod_down(ws_root: Path, log: Callable[[str], None] = print) -> None:
    state = PodState.load(ws_root)
    if not state:
        log("[down] no pod recorded in this workspace")
        return
    api = Runpod(runpod_key())
    if api.get_pod(state.pod_id):
        api.delete_pod(state.pod_id)
        for _ in range(10):
            if not api.get_pod(state.pod_id):
                break
            time.sleep(2)
    PodState.path(ws_root).unlink(missing_ok=True)
    log(f"[down] pod {state.pod_id} terminated (the network volume and its cache remain)")


def pod_status(ws_root: Path, log: Callable[[str], None] = print) -> None:
    state = PodState.load(ws_root)
    if not state:
        log("no pod recorded in this workspace (dub up creates one)")
        return
    pod = Runpod(runpod_key()).get_pod(state.pod_id)
    if not pod:
        log(f"pod {state.pod_id} no longer exists (idle shutdown?); dub up creates a new one")
        PodState.path(ws_root).unlink(missing_ok=True)
        return
    if state.cost_per_hr is None and pod.get("cost"):
        state.cost_per_hr = float(pod["cost"])
        state.save(ws_root)
    hours = (time.time() - state.created_at) / 3600
    cost = f", ~${hours * state.cost_per_hr:.2f} so far" if state.cost_per_hr else ""
    log(
        f"pod {state.pod_id} {pod_status_of(pod)} on {state.gpu} "
        f"(${state.cost_per_hr}/h, up {hours * 60:.0f} min{cost})"
    )
    h = Dubd(state.url, state.token).health()
    if not h:
        log("dubd: not reachable yet")
        return
    log(
        f"dubd {h.get('version', '')[:8]}: ready={h.get('ready')}, jobs={h.get('jobs_running')}, "
        f"idle {h.get('idle_s')}s"
    )
    for n, e in h.get("envs", {}).items():
        log(f"  {n:<5} {e['status']:<10} {e['detail'][:120]}")
