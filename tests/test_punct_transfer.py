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
