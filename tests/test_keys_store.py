from __future__ import annotations

from pathlib import Path

import pytest

from dub.contracts.v1 import Word
from dub.core.keys import call_key, canonical_json
from dub.store.cas import CAS
from dub.store.memo import Memo


def key(**over):
    base = dict(
        op="asr.transcribe",
        provider="p",
        impl_version="1",
        weights_rev=None,
        params={"a": 1, "b": 2},
        seed=0,
        request=Word(w="hi", start_ms=0, end_ms=10),
    )
    base.update(over)
    return call_key(**base)


def test_canonical_json_ignores_key_order_and_int_float():
    assert canonical_json({"a": 1, "b": [1.0, 2]}) == canonical_json({"b": [1, 2], "a": 1.0})


def test_call_key_is_stable_and_sensitive():
    assert key() == key(params={"b": 2, "a": 1})
    assert key() != key(seed=1)
    assert key() != key(params={"a": 1, "b": 3})
    assert key() != key(provider="q")
    assert key() != key(impl_version="2")
    assert key() != key(request=Word(w="hi", start_ms=0, end_ms=11))


def test_paths_are_rejected_in_keys():
    with pytest.raises(TypeError):
        key(params={"file": Path("x.wav")})


def test_cas_dedupes_and_is_content_addressed(tmp_path: Path):
    cas = CAS(tmp_path / "cas")
    k1 = cas.put_bytes(b"hello", ".txt")
    k2 = cas.put_text("hello", ".txt")
    assert k1 == k2 and k1.endswith(".txt")
    assert cas.read_text(k1) == "hello"
    src = tmp_path / "a.wav"
    src.write_bytes(b"RIFF....")
    k3 = cas.put_file(src, move=True)
    assert not src.exists() and cas.exists(k3)
    with pytest.raises(ValueError):
        cas.path("../evil")


def test_memo_roundtrip_and_corruption(tmp_path: Path):
    memo = Memo(tmp_path / "memo")
    memo.put("abcd", {"result": {"x": 1}})
    assert memo.get("abcd") == {"result": {"x": 1}}
    assert memo.get("ffff") is None
    (tmp_path / "memo" / "ab" / "abcd.json").write_text("{broken", encoding="utf-8")
    assert memo.get("abcd") is None


def test_new_default_fields_keep_old_keys():
    from pydantic import BaseModel

    class V1(BaseModel):
        a: int

    class V2(BaseModel):
        a: int
        b: list[str] = []

    assert canonical_json(V1(a=1)) == canonical_json(V2(a=1))
    assert canonical_json(V2(a=1)) != canonical_json(V2(a=1, b=["x"]))


def test_gc_deletes_only_unreferenced_old_files(tmp_path):
    import os
    import time

    from dub.store import Workspace
    from dub.store.gc import collect

    ws = Workspace.open(tmp_path / "ws")
    used = ws.cas.put_text("used", ".txt")
    junk = ws.cas.put_text("junk", ".txt")
    fresh = ws.cas.put_text("fresh junk", ".txt")
    ws.memo.put("k1", {"op": "x", "result": {"nested": [{"audio_key": used}]}})
    old = time.time() - 7200
    for k in (used, junk):
        os.utime(ws.cas.path(k), (old, old))
    assert collect(ws, dry_run=True).deleted == 1 and ws.cas.exists(junk)
    rep = collect(ws, min_age_s=3600)
    assert rep.deleted == 1 and rep.freed_bytes == 4
    assert ws.cas.exists(used) and ws.cas.exists(fresh) and not ws.cas.exists(junk)


def test_eviction_forgets_old_audio_results_but_keeps_text_results(tmp_path):
    import os
    import time

    from dub.store import Workspace
    from dub.store.gc import collect

    ws = Workspace.open(tmp_path / "ws")
    wav = ws.cas.put_bytes(b"RIFF....", ".wav")
    ws.memo.put("a" * 64, {"op": "audio.separate", "result": {"vocals_key": wav}})
    ws.memo.put("b" * 64, {"op": "mt.translate", "result": {"items": [{"target": "xin chào"}]}})
    old = time.time() - 3 * 3600
    for p in ws.memo.root.glob("*/*.json"):
        os.utime(p, (old, old))
    os.utime(ws.cas.path(wav), (old, old))
    rep = collect(ws, min_age_s=0, evict_older_than_s=2 * 3600)
    assert rep.evicted == 1 and rep.deleted == 1 and not ws.cas.exists(wav)
    assert ws.memo.get("a" * 64) is None and ws.memo.get("b" * 64) is not None


def test_pinned_library_versions_are_part_of_the_provider_version(tmp_path):
    from dub.core.config import ProviderManifest
    from dub.core.ctx import _env_pins, _impl_version

    envs = tmp_path / "envs"
    envs.mkdir()
    (envs / "sep.txt").write_text(
        "# comment\naudio-separator[gpu]==0.48.0\naudioread  # unpinned\ntorch==2.8.0\n",
        encoding="utf-8",
    )
    m = ProviderManifest(
        id="sep", op=["audio.separate"], runtime="http", env="sep", impl_version="2",
        libs=["audio_separator", "torch"],
    )  # fmt: skip
    assert _impl_version(m) == "2"  # no env files (laptop tests): the manual version only
    v1 = _impl_version(m, envs)
    assert v1.startswith("2+") and v1 == _impl_version(m, envs)
    (envs / "sep.txt").write_text("audio-separator[gpu]==0.49.0\ntorch==2.8.0\n", encoding="utf-8")
    _env_pins.cache_clear()
    assert _impl_version(m, envs) != v1  # an upgrade changes every key of the provider
    other = m.model_copy(update={"libs": ["torch"]})
    (envs / "sep.txt").write_text("audio-separator[gpu]==0.50.0\ntorch==2.8.0\n", encoding="utf-8")
    _env_pins.cache_clear()
    t1 = _impl_version(other, envs)
    (envs / "sep.txt").write_text("audio-separator[gpu]==0.51.0\ntorch==2.8.0\n", encoding="utf-8")
    _env_pins.cache_clear()
    assert _impl_version(other, envs) == t1  # libraries it does not name do not matter


def test_repo_providers_name_pinned_libraries():
    from pathlib import Path

    from dub.core.config import load_config
    from dub.core.ctx import _env_pins, _pkg

    cfg = load_config()
    envs = Path(cfg.root).parent / "pod" / "envs"
    for m in cfg.providers.values():
        for lib in m.libs:
            spec = _env_pins(envs / f"{m.env}.txt").get(_pkg(lib), "")
            assert "==" in spec, f"{m.id}: {lib} is not pinned in pod/envs/{m.env}.txt"
