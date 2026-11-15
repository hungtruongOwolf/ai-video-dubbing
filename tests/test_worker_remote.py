"""M1 plumbing without a GPU: a worker over HTTP, and the laptop <-> dubd protocol end to end
(upload in parts, job, logs, downloads, mux) against a dubd running in this process."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from dub import remote
from dub.cli import main
from dub.contracts.v1 import SynthReq, SynthRes
from dub.core.config import Config
from dub.core.executor import Executor
from dub.core.provider_api import ProviderError
from dub.ops.media import probe
from dub.pod.envs import EnvManager
from dub.pod.server import DubServer, serve
from dub.store import Workspace
from dub.worker.server import WorkerState
from dub.worker.server import serve as serve_worker
from tests.conftest import requires_ffmpeg


def _http_cfg(cfg: Config) -> Config:
    """Config copy where fake_tts runs behind a worker ('fx' env) instead of in-process."""
    providers = dict(cfg.providers)
    providers["fake_tts"] = providers["fake_tts"].model_copy(
        update={"runtime": "http", "env": "fx"}
    )
    return Config(cfg.root, cfg.dub, cfg.langpacks, providers)


@pytest.fixture
def worker(cfg, ws, monkeypatch):
    state = WorkerState("fx", ["fake_tts"], cfg, ws.cas)
    httpd = serve_worker(state, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    monkeypatch.setenv("DUB_WORKERS", json.dumps({"fx": url}))
    yield url
    httpd.shutdown()


def test_executor_runs_a_provider_through_a_worker(cfg, ws, worker):
    hcfg = _http_cfg(cfg)
    ex = Executor(hcfg, ws.cas)
    req = SynthReq(unit_id="u1", text="xin chào các bạn", lang="vi", speaker="S1", target_ms=900)
    res = asyncio.run(ex.run(hcfg.providers["fake_tts"], "tts.synthesize", req, {}))
    assert isinstance(res, SynthRes) and res.dur_ms == 900 and ws.cas.exists(res.audio_key)


def test_worker_errors_are_classified(cfg, ws, worker):
    hcfg = _http_cfg(cfg)
    other = hcfg.providers["fake_tts"].model_copy(update={"id": "not_hosted"})
    with pytest.raises(ProviderError) as e:
        asyncio.run(
            Executor(hcfg, ws.cas).run(
                other,
                "tts.synthesize",
                SynthReq(unit_id="u", text="x", lang="vi", speaker="S1"),
                {},
            )
        )
    assert e.value.kind == "unsupported"


def test_router_sees_running_workers(cfg, worker):
    from dub.core.router import Router

    hcfg = _http_cfg(cfg)
    hcfg.langpacks["vi"] = hcfg.langpacks["vi"].model_copy(
        update={"chains": {"tts.synthesize": ["fake_tts"]}}
    )
    assert Router(hcfg).route("tts.synthesize", {"tgt_lang": "vi"}).chain == ["fake_tts"]


def test_unregistered_worker_is_rejected_with_reason(cfg, monkeypatch):
    from dub.core.router import Router, RoutingError

    monkeypatch.delenv("DUB_WORKERS", raising=False)
    hcfg = _http_cfg(cfg)
    hcfg.langpacks["vi"] = hcfg.langpacks["vi"].model_copy(
        update={"chains": {"tts.synthesize": ["fake_tts"]}}
    )
    with pytest.raises(RoutingError, match="not running here"):
        Router(hcfg).route("tts.synthesize", {"tgt_lang": "vi"})


def test_a_job_routes_to_a_booting_worker_and_its_calls_wait_for_it(cfg, ws, monkeypatch):
    # The laptop submits as soon as the pod answers: TTS may still be loading its models.
    from dub.core import executor as executor_mod
    from dub.core.router import Router

    monkeypatch.delenv("DUB_WORKERS", raising=False)
    monkeypatch.setenv("DUB_WORKERS_PENDING", json.dumps(["fx"]))
    monkeypatch.setattr(executor_mod, "PENDING_POLL_S", 0.01)
    hcfg = _http_cfg(cfg)
    hcfg.langpacks["vi"] = hcfg.langpacks["vi"].model_copy(
        update={"chains": {"tts.synthesize": ["fake_tts"]}}
    )
    assert Router(hcfg).route("tts.synthesize", {"tgt_lang": "vi"}).chain == ["fake_tts"]

    state = WorkerState("fx", ["fake_tts"], cfg, ws.cas)
    httpd = serve_worker(state, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"

    def worker_comes_up():  # what EnvManager does once the models are loaded
        import os
        import time

        time.sleep(0.2)
        os.environ["DUB_WORKERS"] = json.dumps({"fx": url})
        os.environ["DUB_WORKERS_PENDING"] = "[]"

    threading.Thread(target=worker_comes_up, daemon=True).start()
    logs: list[str] = []
    ex = Executor(hcfg, ws.cas, log=logs.append)
    req = SynthReq(unit_id="u1", text="xin chào", lang="vi", speaker="S1", target_ms=500)
    res = asyncio.run(ex.run(hcfg.providers["fake_tts"], "tts.synthesize", req, {}))
    assert res.dur_ms == 500 and any("waiting for the fx worker" in m for m in logs)
    httpd.shutdown()

    # a worker that failed to start is no longer pending: the call fails instead of waiting
    monkeypatch.setenv("DUB_WORKERS", "{}")
    monkeypatch.setenv("DUB_WORKERS_PENDING", "[]")
    with pytest.raises(ProviderError) as e:
        asyncio.run(
            Executor(hcfg, ws.cas).run(hcfg.providers["fake_tts"], "tts.synthesize", req, {})
        )
    assert e.value.kind == "unsupported"


def test_env_manager_marks_envs_pending_until_they_settle(cfg, tmp_path, monkeypatch):
    from dub.pod.envs import EnvInfo
    from dub.worker.protocol import pending_workers

    monkeypatch.setenv("DUB_WORKERS_PENDING", "[]")
    envs = EnvManager(cfg, tmp_path, tmp_path, tmp_path)
    envs.envs = {"asr": EnvInfo("asr", ["x"], 7101), "tts": EnvInfo("tts", ["y"], 7102)}
    monkeypatch.setattr(envs, "_run_all", lambda: None)
    envs.start()
    assert pending_workers() == {"asr", "tts"}
    envs._set(envs.envs["asr"], "ready")
    envs._set(envs.envs["tts"], "starting", "loading y")
    assert pending_workers() == {"tts"}
    envs._set(envs.envs["tts"], "failed", "boom")
    assert pending_workers() == set()


def test_run_waits_only_for_the_pod_server(monkeypatch):
    # `dub run --remote` goes on once dubd answers; `dub up` waits for every worker
    replies = iter([None, {"ready": False, "envs": {"asr": {"status": "starting", "detail": ""}}}])
    monkeypatch.setattr(remote.Dubd, "health", lambda self: next(replies))
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    state = remote.PodState(pod_id="p", url="http://x", token="t", gpu="g", ref="r")
    logs: list[str] = []
    remote.wait_ready(state, logs.append, 1, until="dubd")
    assert "workers still loading" in logs[-1]
    failed = {"ready": False, "envs": {"tts": {"status": "failed", "detail": "oom"}}}
    replies = iter([failed, {**failed, "ready": True}])
    with pytest.raises(remote.RemoteError, match="failed to start"):
        remote.wait_ready(state, logs.append, 1, until="dubd")


# -------------------------------------------------------------------- dubd + laptop client


@pytest.fixture
def dubd(cfg, tmp_path):
    ws = Workspace.open(tmp_path / "pod-ws")
    envs = EnvManager(cfg, tmp_path, ws.root, tmp_path)  # no http providers: nothing to start
    srv = DubServer(cfg, ws, envs, token="t" * 24, idle_minutes=60)
    httpd = serve(srv, host="127.0.0.1", port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", srv
    httpd.shutdown()


def test_dubd_auth_upload_and_download(dubd, tmp_path, monkeypatch):
    url, srv = dubd
    assert remote.Dubd(url, "wrong-token-xxxxxxxxxxxx").health()["ok"]  # /health is public
    with pytest.raises(remote.RemoteError, match="401"):
        remote.Dubd(url, "wrong-token-xxxxxxxxxxxx").submit({})
    client = remote.Dubd(url, "t" * 24)
    monkeypatch.setattr(remote, "PART_BYTES", 1000)  # force several parts
    src = tmp_path / "blob.bin"
    src.write_bytes(bytes(range(256)) * 20)
    key = client.upload(src, ".bin", log=lambda m: None)
    assert srv.ws.cas.read_bytes(key) == src.read_bytes()
    back = client.download(f"/cas/{key}", tmp_path / "back.bin")
    assert back.read_bytes() == src.read_bytes()


@requires_ffmpeg
def test_remote_run_end_to_end(dubd, tmp_path):
    from dub.demo import make_demo

    url, _ = dubd
    video, _ = make_demo(tmp_path / "demo")
    ws_dir = tmp_path / "laptop-ws"
    ws_dir.mkdir()
    remote.PodState(pod_id="local", url=url, token="t" * 24, gpu="none", ref="dev").save(ws_dir)
    rc = main(["run", str(video), "--to", "vi", "--fake", "--remote", "--workspace", str(ws_dir),
               "--keep-original-audio"])  # fmt: skip
    assert rc == 0
    out = video.parent / "interview.dub" / "interview.vi.mp4"
    streams = probe(out)["streams"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    assert [s["codec_name"] for s in audio] == ["aac", "aac"]
    assert sum(s["codec_type"] == "subtitle" for s in streams) == 2
    job_dirs = list((ws_dir / "jobs").iterdir())
    assert (job_dirs[0] / "report.md").exists() and (job_dirs[0] / "dub.m4a").exists()


def test_pod_body_shape():
    cfg = remote.PodConfig(
        image="img",
        gpus=["NVIDIA A40"],
        volume_id="vol1",
        data_center="EU-SE-1",
        claude_secret="claude_code_oauth_token",
    )
    body = remote.pod_body(cfg, "NVIDIA A40", "abc123", "tok", claude_secret=True)
    assert body["gpu"] == {"id": "NVIDIA A40", "count": 1, "minCudaVersion": "12.8"}
    assert body["mounts"] == {"network": [{"volumeId": "vol1", "path": "/workspace"}]}
    assert body["dataCenterIds"] == ["EU-SE-1"] and body["ports"] == ["8000/http"]
    assert body["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "{{ RUNPOD_SECRET_claude_code_oauth_token }}"
    assert (
        "raw.githubusercontent.com/hungtruongOwolf/ai-video-dubbing/abc123/pod/bootstrap.sh"
        in body["cmd"][0]
    )
    no_secret = remote.pod_body(cfg, "NVIDIA A40", "abc123", "tok", claude_secret=False)
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in no_secret["env"]


def test_pod_config_merges_local_overrides(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "pod.yaml").write_text(
        "image: img\ngpus: [A]\nvolume_id: null\ndata_center: null\n", encoding="utf-8"
    )
    (tmp_path / "pod.local.yaml").write_text(
        "volume_id: v9\ndata_center: EU-SE-1\n", encoding="utf-8"
    )
    cfg = remote.PodConfig.load(cfg_dir)
    assert (cfg.volume_id, cfg.data_center) == ("v9", "EU-SE-1")


def test_repo_pod_config_is_valid():
    cfg = remote.PodConfig.load(Path(__file__).resolve().parents[1] / "configs")
    assert cfg.gpus and cfg.image


def test_worker_batches_queued_calls_and_keeps_each_result(cfg, ws, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    from dub.providers.fake import FakeTTS

    providers = dict(cfg.providers)
    m = providers["fake_tts"]
    providers["fake_tts"] = m.model_copy(
        update={"limits": m.limits.model_copy(update={"max_batch": 4})}
    )
    bcfg = Config(cfg.root, cfg.dub, cfg.langpacks, providers)
    calls = []
    original = FakeTTS.run_batch

    def slow_batch(self, op, reqs, ctx):
        calls.append(len(reqs))
        time.sleep(0.05)
        return original(self, op, reqs, ctx)

    monkeypatch.setattr(FakeTTS, "run_batch", slow_batch)
    single = FakeTTS.op_tts_synthesize

    def slow_single(self, req, ctx):
        time.sleep(0.05)  # the first call keeps the runner busy while the others queue
        return single(self, req, ctx)

    monkeypatch.setattr(FakeTTS, "op_tts_synthesize", slow_single)
    state = WorkerState("fx", ["fake_tts"], bcfg, ws.cas)
    reqs = [SynthReq(unit_id=f"u{i}", text="xin chào", lang="vi", speaker="S1", target_ms=500 + i)
            for i in range(9)]  # fmt: skip
    with ThreadPoolExecutor(9) as pool:
        out = list(pool.map(lambda r: state.submit("fake_tts", "tts.synthesize", {}, r), reqs))
    assert [r.unit_id for r in out] == [f"u{i}" for i in range(9)]
    assert [r.dur_ms for r in out] == [500 + i for i in range(9)]
    assert sum(state.batches) == 9 and max(state.batches) > 1 and max(state.batches) <= 4
    assert calls and all(n > 1 for n in calls)
    # different params never share a batch
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda i: state.submit("fake_tts", "tts.synthesize", {"x": i % 2}, reqs[i]),
                      range(4)))  # fmt: skip


def test_worker_unloads_models_and_marks_gpu_memory_errors_retryable(cfg, ws):
    from dub.worker.server import _classify

    state = WorkerState("fx", ["fake_tts"], cfg, ws.cas)
    assert state.unload("fake_tts") is False  # not loaded yet
    state.submit("fake_tts", "tts.synthesize", {},
                 SynthReq(unit_id="u", text="xin chào", lang="vi", speaker="S1"))  # fmt: skip
    assert state.unload("fake_tts") is True and "fake_tts" not in state.instances
    e = _classify(RuntimeError("CUDA failed with error out of memory"))
    assert isinstance(e, ProviderError) and e.kind == "oom" and e.retryable
    other = RuntimeError("boom")
    assert _classify(other) is other


def test_worker_offloads_models_that_can_and_restores_them_on_the_next_call(cfg, ws, monkeypatch):
    from dub.providers.fake import FakeTTS

    moves = []
    monkeypatch.setattr(FakeTTS, "offload", lambda self: moves.append("cpu"), raising=False)
    monkeypatch.setattr(FakeTTS, "restore", lambda self: moves.append("gpu"), raising=False)
    state = WorkerState("fx", ["fake_tts"], cfg, ws.cas)
    req = SynthReq(unit_id="u", text="xin chào", lang="vi", speaker="S1")
    state.submit("fake_tts", "tts.synthesize", {}, req)
    assert state.unload("fake_tts") is True
    assert "fake_tts" in state.instances and state.offloaded == {"fake_tts"}
    assert state.unload("fake_tts") is False  # already parked
    state.submit("fake_tts", "tts.synthesize", {}, req)
    assert moves == ["cpu", "gpu"] and not state.offloaded


def test_unload_waits_for_a_running_call(cfg, ws, monkeypatch):
    # Two jobs: one frees its phase-A models while the other is still using one of them.
    import time

    from dub.providers.fake import FakeTTS

    events: list[str] = []
    real_run = FakeTTS.run

    def slow_run(self, op, req, ctx):
        events.append("run start")
        time.sleep(0.3)
        events.append("run end")
        return real_run(self, op, req, ctx)

    monkeypatch.setattr(FakeTTS, "run", slow_run)
    monkeypatch.setattr(FakeTTS, "offload", lambda self: events.append("cpu"), raising=False)
    monkeypatch.setattr(FakeTTS, "restore", lambda self: events.append("gpu"), raising=False)
    state = WorkerState("fx", ["fake_tts"], cfg, ws.cas)
    req = SynthReq(unit_id="u", text="xin chào", lang="vi", speaker="S1")
    state.submit("fake_tts", "tts.synthesize", {}, req)  # loaded
    events.clear()
    t = threading.Thread(target=state.submit, args=("fake_tts", "tts.synthesize", {}, req))
    t.start()
    while "run start" not in events:
        time.sleep(0.01)
    assert state.unload("fake_tts") is True
    t.join()
    assert events == ["run start", "run end", "cpu"]  # never moved in the middle of the call


def test_a_model_serves_while_another_one_loads(cfg, ws, monkeypatch):
    # The tts worker loaded ECAPA in 41 s and OmniVoice in 73 s: diarization must not wait
    # for OmniVoice. A call for the model being loaded waits for that one load.
    import time

    from dub.contracts.v1 import EmbedReq
    from dub.providers.fake import FakeTTS

    gate = threading.Event()
    loads: list[str] = []

    def slow_load(self, ctx):
        loads.append("tts")
        gate.wait(5)

    monkeypatch.setattr(FakeTTS, "load", slow_load, raising=False)
    state = WorkerState("fx", ["fake_tts", "fake_diarize"], cfg, ws.cas)
    assert state.preload_order() == ["fake_diarize", "fake_tts"]  # TTS engines last
    order = WorkerState("tts", ["omnivoice", "sat_12l_sm", "voice_agender", "ecapa_cluster"],
                        cfg, ws.cas).preload_order()  # fmt: skip
    assert order == ["ecapa_cluster", "voice_agender", "sat_12l_sm", "omnivoice"]  # first use
    state.preload("fake_diarize")
    t = threading.Thread(target=state.preload, args=("fake_tts",))
    t.start()
    while "fake_tts" not in state.loading:
        time.sleep(0.01)
    tone = ws.cas.put_bytes(_tone_wav(), ".wav")
    t0 = time.perf_counter()
    state.submit("fake_diarize", "audio.embed", {}, EmbedReq(audio_key=tone))
    assert time.perf_counter() - t0 < 2  # served while fake_tts still loads
    done: list = []
    req = SynthReq(unit_id="u", text="xin chào", lang="vi", speaker="S1", target_ms=400)
    caller = threading.Thread(
        target=lambda: done.append(state.submit("fake_tts", "tts.synthesize", {}, req))
    )
    caller.start()
    time.sleep(0.2)
    assert not done  # waits for the load in progress
    gate.set()
    t.join()
    caller.join()
    assert done and done[0].dur_ms == 400 and loads == ["tts"]  # loaded once, not twice


def _tone_wav() -> bytes:
    import io
    import wave

    import numpy as np

    x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(x.tobytes())
    return buf.getvalue()


def test_remote_gc_runs_in_the_background_and_reports(dubd, tmp_path, monkeypatch):
    url, srv = dubd
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    import os
    import time as _time

    orphan = srv.ws.cas.put_bytes(b"nobody points to me", ".bin")
    old = _time.time() - 60  # a fresh file's mtime can be a few ms ahead of time.time()
    os.utime(srv.ws.cas.path(orphan), (old, old))
    rep = remote.Dubd(url, "t" * 24).gc(min_age_s=0)
    assert rep["deleted"] >= 1 and not srv.ws.cas.exists(orphan)
    assert srv.gc_state["status"] == "done"


def test_direct_worker_calls_keep_the_pod_awake(cfg, ws, dubd, monkeypatch):
    """Experiments call workers directly; dubd's idle watchdog must see that activity."""
    url, srv = dubd
    state = WorkerState("fx", ["fake_tts"], cfg, ws.cas)
    httpd = serve_worker(state, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    wurl = f"http://127.0.0.1:{httpd.server_address[1]}"
    monkeypatch.setenv("DUB_WORKERS", json.dumps({"fx": wurl}))
    state.submit("fake_tts", "tts.synthesize", {},
                 SynthReq(unit_id="u", text="xin chào", lang="vi", speaker="S1"))  # fmt: skip
    assert srv._workers_busy()
    state.last_active -= 3600
    assert not srv._workers_busy()
    httpd.shutdown()


def test_a_heartbeat_file_keeps_the_pod_awake(dubd, tmp_path, monkeypatch):
    """Downloads and other work outside dubd (pod/busy.sh) touch a heartbeat file."""
    import os
    import time

    from dub.pod import server as server_mod

    _url, srv = dubd
    beat = tmp_path / "dub.busy"
    monkeypatch.setattr(server_mod, "BUSY_FILE", beat)
    assert not srv._heartbeat_busy()  # no file
    beat.touch()
    assert srv._heartbeat_busy()
    old = time.time() - server_mod.BUSY_FRESH_S - 5
    os.utime(beat, (old, old))
    assert not srv._heartbeat_busy()  # stale: the work ended without cleaning up


# -------------------------------------------------------------------- worker restarts


def test_a_down_worker_is_reported_as_unavailable(cfg, ws, monkeypatch):
    import socket

    with socket.socket() as s:  # a free port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("DUB_WORKERS", json.dumps({"fx": f"http://127.0.0.1:{port}"}))
    hcfg = _http_cfg(cfg)
    req = SynthReq(unit_id="u", text="x", lang="vi", speaker="S1")
    with pytest.raises(ProviderError) as e:
        asyncio.run(
            Executor(hcfg, ws.cas).run(hcfg.providers["fake_tts"], "tts.synthesize", req, {})
        )
    assert e.value.kind == "unavailable" and e.value.retryable


class _Proc:
    def __init__(self, code):
        self.returncode = code

    def poll(self):
        return self.returncode


def test_dubd_restarts_a_dead_worker_but_not_forever(cfg, tmp_path, monkeypatch):
    from dub.pod import envs as envs_mod
    from dub.pod.envs import EnvInfo

    mgr = EnvManager(cfg, tmp_path, tmp_path, tmp_path, log=lambda _m: None)
    e = EnvInfo("fx", ["fake_tts"], 7199, status="ready", proc=_Proc(None))
    launched = []

    def launch(env):
        launched.append(env.name)
        env.proc = _Proc(-9)  # dies again at once

    monkeypatch.setattr(mgr, "_launch", launch)
    monkeypatch.setattr(mgr, "_wait_ready", lambda env: setattr(env, "status", "ready"))
    assert mgr.check_worker(e, now=0.0) and launched == []  # alive: nothing to do
    e.proc = _Proc(-11)
    for i in range(envs_mod.MAX_RESTARTS):
        assert mgr.check_worker(e, now=float(i)) and e.status == "ready"
    assert len(launched) == envs_mod.MAX_RESTARTS
    assert not mgr.check_worker(e, now=10.0) and e.status == "failed"  # crash loop: given up
    # restarts older than the window no longer count
    e.status = "ready"
    assert mgr.check_worker(e, now=10.0 + envs_mod.RESTART_WINDOW_S)
    assert len(launched) == envs_mod.MAX_RESTARTS + 1


def test_ctx_waits_for_a_restarting_worker(cfg, ws, monkeypatch):
    from dub.core import ctx as ctx_mod
    from dub.core.ctx import Ctx

    monkeypatch.setattr(ctx_mod, "UNAVAILABLE_WAITS_S", (0.0, 0.0, 0.0))
    calls = {"n": 0}
    real_run = Executor.run

    async def flaky(self, m, op, req, params):
        calls["n"] += 1
        if calls["n"] <= 3:  # down for three attempts (more than the 2 generic retries)
            raise ProviderError("worker unreachable", kind="unavailable", retryable=True)
        return await real_run(self, m, op, req, params)

    monkeypatch.setattr(Executor, "run", flaky)
    from dub.core.router import Router
    from dub.core.trace import Tracer

    cx = Ctx(
        cfg=cfg, ws=ws, router=Router(cfg, overrides=[("tts.synthesize", ["fake_tts"])]),
        executor=Executor(cfg, ws.cas), tracer=Tracer(), job_id="j", facts={"tgt_lang": "vi"},
        log=lambda m: None,
    )  # fmt: skip
    req = SynthReq(unit_id="u", text="xin chào", lang="vi", speaker="S1", target_ms=500)
    res = asyncio.run(cx.call("tts.synthesize", req, subject="u"))
    assert isinstance(res, SynthRes) and calls["n"] == 4


def test_cheap_calls_are_served_before_queued_batches(cfg, ws):
    # A 10 ms speaker embedding waited p50 1.7 s behind TTS batches on the drama.
    from dub.contracts.v1 import EmbedReq
    from dub.worker.server import _Task

    state = WorkerState("fx", ["fake_tts", "fake_diarize"], cfg, ws.cas)
    req = SynthReq(unit_id="u", text="xin chào", lang="vi", speaker="S1")
    state.queue = [
        _Task("fake_tts", "tts.synthesize", {}, req),
        _Task("fake_tts", "tts.synthesize", {}, req),
        _Task("fake_diarize", "audio.embed", {}, EmbedReq(audio_key="k.wav")),
    ]
    first = state._take()
    assert [t.op for t in first] == ["audio.embed"]
    assert state._take()[0].op == "tts.synthesize"  # then the queue in order
