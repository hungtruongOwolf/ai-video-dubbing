"""A run on a pod that runs older code updates the pod first (a whole round of measurements
once ran on a pod still on the commit before the fix being measured)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dub import remote, runner


@pytest.fixture
def pod(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(remote.Dubd, "health", lambda self: {"ready": True})
    monkeypatch.setattr(
        remote, "pod_up", lambda *a, **k: calls.append("up") or SimpleNamespace(ref="new")
    )
    monkeypatch.setattr(remote.PodConfig, "load", classmethod(lambda cls, d: None))
    ws = SimpleNamespace(root=tmp_path)
    opts = runner.RunOptions(video=tmp_path / "v.mp4", tgt="vi", remote=True)
    return calls, ws, opts, monkeypatch


def _state(tmp_path, ref):
    remote.PodState(pod_id="p", url="http://x", token="t" * 24, gpu="g", ref=ref).save(tmp_path)


def test_a_pod_on_older_code_is_updated_before_the_run(pod, tmp_path):
    calls, ws, opts, mp = pod
    mp.setattr(remote, "git_ref", lambda root: "new")
    _state(tmp_path, "old")
    runner._pod_in_background(opts, ws, lambda m: None).result()
    assert calls == ["up"]


def test_a_pod_on_this_code_is_used_as_it_is(pod, tmp_path):
    calls, ws, opts, mp = pod
    mp.setattr(remote, "git_ref", lambda root: "same")
    _state(tmp_path, "same")
    assert runner._pod_in_background(opts, ws, lambda m: None).result().ref == "same"
    _state(tmp_path, "dev")  # a dubd started by hand
    assert runner._pod_in_background(opts, ws, lambda m: None).result().ref == "dev"
    assert calls == []


def test_an_empty_404_from_the_proxy_is_tried_again(monkeypatch):
    answers = [(404, b""), (404, b""), (200, b'{"job_id": "j"}')]
    monkeypatch.setattr(remote, "_request", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    assert remote.Dubd("http://x", "t")._json("POST", "/jobs", {}) == {"job_id": "j"}
    # dubd's own 404 (with a JSON body) is an answer, not a proxy hiccup
    answers[:] = [(404, b'{"error": "no route /x"}'), (200, b"{}")]
    with pytest.raises(remote.RemoteError):
        remote.Dubd("http://x", "t")._json("POST", "/x", {})
