"""Sentence punctuation moved from a punctuating ASR onto one that does not punctuate."""

from __future__ import annotations

from dub.contracts.v1 import Word
from dub.providers.asr_whisper import transfer_punct


def ws(text: str, step: int = 300) -> list[Word]:
    return [Word(w=t, start_ms=i * step, end_ms=i * step + 250, conf=0.9)
            for i, t in enumerate(text.split())]  # fmt: skip


def test_marks_move_where_both_heard_the_same_words():
    ours = ws("tất cả mọi người đều sinh ra bình đẳng tạo hóa cho họ quyền sống")
    donor = ws("Tất cả mọi người đều sinh ra bình đẳng. Tạo hoá cho họ quyền sống!")
    out = [w.w for w in transfer_punct(ours, donor)]
    assert out[8] == "đẳng." and out[-1] == "sống!" and out[0] == "tất"


def test_no_mark_where_the_words_differ():
    ours = ws("năm một nghìn chín trăm bốn mươi lăm nước ta độc lập")
    donor = ws("Năm 1945. Nước ta độc lập.")  # the number as digits: no match there
    out = [w.w for w in transfer_punct(ours, donor)]
    assert out[-1] == "lập." and all(not w.endswith(".") for w in out[:-1])


def test_long_input_is_aligned_window_by_window():
    ours = ws(" ".join(["một câu ngắn thôi"] * 200), step=500)
    donor = ws(" ".join(["Một câu ngắn thôi."] * 200), step=500)
    out = transfer_punct(ours, donor, window_ms=60_000)
    assert sum(w.w.endswith(".") for w in out) >= 190


class _FakeCT2:
    def __init__(self):
        self.model_is_loaded = True
        self.loads = 0

    def unload_model(self, to_cpu=False):
        self.model_is_loaded = False

    def load_model(self):
        self.model_is_loaded = True
        self.loads += 1


class _FakeWhisper:
    def __init__(self):
        self.model = _FakeCT2()


def test_providers_with_the_same_weights_share_one_model(monkeypatch):
    """whisper_large_v3 and the punctuation donor of phowhisper_punct hold one copy of
    Whisper large-v3; when one parks it in CPU memory, the other brings it back."""
    from dub.providers import asr_whisper as aw

    monkeypatch.setattr(aw, "_MODELS", {})
    made = []

    def fake_load(self, ctx):
        key = (ctx.params["model"], "cuda", "float16")
        if key not in aw._MODELS:
            aw._MODELS[key] = _FakeWhisper()
            made.append(key)
        self.model = aw._MODELS[key]

    monkeypatch.setattr(aw.FasterWhisperASR, "load", fake_load)

    class Ctx:
        params = {"model": "Systran/faster-whisper-large-v3"}

    a, b = aw.FasterWhisperASR(None), aw.FasterWhisperASR(None)
    a._ready(Ctx())
    b._ready(Ctx())
    assert a.model is b.model and len(made) == 1
    a.offload()
    assert not b.model.model.model_is_loaded
    b._ready(Ctx())  # b finds it parked and brings it back
    assert b.model.model.model_is_loaded and b.model.model.loads == 1
    a.restore()  # already back: no second load
    assert a.model.model.loads == 1
