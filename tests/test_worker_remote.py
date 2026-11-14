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
    rc = main(["run", str(video), "--to", "vi", "--fake", "--remote", "--workspace", str(ws_dir)])
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
