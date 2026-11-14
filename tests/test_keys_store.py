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
