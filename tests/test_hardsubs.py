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


def test_a_cue_the_asr_missed_becomes_its_own_word():
    # The ASR heard the invitation and the next line, not the short refusal between them
    # (the Nailong cartoon: 了解一下呀 / 没兴趣 / 奶龙，大爷年纪大了).
    words = [Word(w=w, start_ms=a, end_ms=b, speaker="S1")
             for w, a, b in [("了解", 8500, 9000), ("一下", 9000, 9600), ("呀", 9600, 10200),
                             ("奶龙", 12300, 12700), ("大爷", 13000, 13400)]]  # fmt: skip
    cues = [_cue(8500, 11250, "健身舞蹈要不要了解一下呀"), _cue(11250, 12250, "没兴趣", 0.986),
            _cue(12250, 12750, "奶龙"), _cue(13000, 14000, "大爷年纪大了")]  # fmt: skip
    fix = correct(HardsubReq(lang="zh", words=words, cues=cues))
    refusal = [w for w in fix.words if w.w == "没兴趣"]
    assert len(refusal) == 1 and fix.inserted == 1
    assert (refusal[0].start_ms, refusal[0].end_ms) == (11250, 12250)  # timed by its cue
    assert refusal[0].speaker == "S1"  # the nearest word's label, for the attribution to fix
    assert [w.start_ms for w in fix.words] == sorted(w.start_ms for w in fix.words)
    assert "".join(w.w for w in fix.words if w.start_ms < 11000) == "了解一下呀"  # not glued on


def test_a_missed_cue_needs_a_sure_reading_and_silence_under_it():
    words = [Word(w="你好", start_ms=0, end_ms=500), Word(w="再见", start_ms=3000, end_ms=3500)]
    one_char = [_cue(0, 500, "你好"), _cue(1000, 2000, "糖"), _cue(3000, 3500, "再见")]
    assert correct(HardsubReq(lang="zh", words=words, cues=one_char)).inserted == 0
    unsure = [_cue(0, 500, "你好"), _cue(1000, 2000, "没兴趣", 0.9), _cue(3000, 3500, "再见")]
    assert correct(HardsubReq(lang="zh", words=words, cues=unsure)).inserted == 0
    # the ASR heard words there (written differently): the cue is no missed line
    heard = words + [Word(w="嗯嗯", start_ms=1000, end_ms=1900)]
    sure = [_cue(0, 500, "你好"), _cue(1000, 2000, "没兴趣"), _cue(3000, 3500, "再见")]
    fix = correct(HardsubReq(lang="zh", words=sorted(heard, key=lambda w: w.start_ms), cues=sure))
    assert fix.inserted == 0
    assert correct(HardsubReq(lang="zh", words=words, cues=sure)).inserted == 1


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


def test_band_cut_to_the_text():
    # A wide band with a short line in the middle: recognition is fed only the text columns
    # (fed the whole band it read "没兴趣" as "兴").
    import numpy as np
    from PIL import Image

    from dub.hardsubs_ocr import _same, text_crop

    band = np.full((40, 600), 30, dtype=np.uint8)
    band[10:30, 250:330:4] = 250  # thin bright strokes: letter edges
    crop = text_crop(Image.fromarray(band))
    # edges between columns 249 and 327, plus half the band height (20) on each side
    assert crop.size == (347 - 229, 40)
    empty = Image.fromarray(np.full((40, 600), 30, dtype=np.uint8))
    assert text_crop(empty).size == (600, 40)  # nothing to cut to: the whole band
    assert _same("健身舞蹈要不要了解一下呀，", "健身舞蹈要不要了解一下呀")
    assert not _same("他不吃", "不吃")
