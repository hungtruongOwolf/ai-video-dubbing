"""Burned-in subtitles correct the transcript and mark what is not dialogue (M8)."""

from __future__ import annotations

from dub.contracts.v1 import HardsubCue, HardsubReq, Unit, Word
from dub.ops.hardsubs import correct, uncaptioned


def _words(pieces: list[str], start: int = 0, step: int = 300) -> list[Word]:
    return [Word(w=p, start_ms=start + i * step, end_ms=start + i * step + 250)
            for i, p in enumerate(pieces)]  # fmt: skip


def _cue(a: int, b: int, text: str, conf: float = 0.95) -> HardsubCue:
    return HardsubCue(start_ms=a, end_ms=b, text=text, conf=conf)


def test_homophones_are_corrected_and_times_kept():
    # Whisper: 城东前府 / 夜里长不长 ; the subtitles: 城东钱府 / 夜里涨不涨
    words = _words(["城东", "前府，", "夜里", "长不长"])
    cues = [_cue(0, 600, "城东钱府"), _cue(600, 1200, "夜里涨不涨")]
    fix = correct(HardsubReq(lang="zh", words=words, cues=cues))
    assert [w.w for w in fix.words] == ["城东", "钱府，", "夜里", "涨不涨"]  # punctuation kept
    assert [w.start_ms for w in fix.words] == [w.start_ms for w in words]
    assert fix.corrected == 3 and fix.coverage > 0.9


def test_dropped_characters_are_added_and_extra_ones_kept():
    # the ASR dropped 哭 and said 那个 that the subtitle leaves out
    words = _words(["那个", "娃饿的", "整夜"])
    cues = [_cue(0, 900, "娃饿的整夜哭")]
    fix = correct(HardsubReq(lang="zh", words=words, cues=cues))
    assert "".join(w.w for w in fix.words) == "那个娃饿的整夜哭"


def test_unsure_cues_correct_nothing():
    words = _words(["城东", "前府"])
    fix = correct(HardsubReq(lang="zh", words=words, cues=[_cue(0, 600, "城东钱府", 0.5)]))
    assert [w.w for w in fix.words] == ["城东", "前府"] and fix.corrected == 0


def test_lines_without_subtitles_are_not_dialogue():
    def unit(i, a, b):
        return Unit(id=f"u{i}", speaker="S1", sentence_ids=[], start_ms=a, end_ms=b, text="x",
                    n_words=1)  # fmt: skip

    units = [unit(0, 0, 2000), unit(1, 2500, 4000), unit(2, 5000, 7000)]
    cues = [_cue(0, 2000, "一二三"), _cue(5000, 7000, "四五六")]  # nothing over u1 (a song)
    assert uncaptioned(units, cues, coverage=0.9) == ["u1"]
    assert uncaptioned(units, cues, coverage=0.4) == []  # subtitles that skip lines: no verdict
