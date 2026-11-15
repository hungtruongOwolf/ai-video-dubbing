"""Laughter, cries, screams between the dubbed lines stay in the dub; gaps with words do not."""

from __future__ import annotations

from dub.core.config import load_config
from dub.ops.nonverbal import gaps, subtract, wordless


def test_vocal_gaps_outside_lines_and_cues():
    # the cartoon: a scream continues past its 240 ms line, a laugh with no line at all
    regions = [(38520, 40600), (57400, 60700)]
    lines = [(38520, 38760), (52800, 57750)]
    assert gaps(regions, lines) == [(38910, 40600), (57900, 60700)]
    # a gap shorter than the minimum is left out; the line's padding is respected
    assert gaps([(1000, 1500)], [(900, 1200)]) == []
    # two pieces of one sound split by a short pause merge; pieces split by a line do not
    assert gaps([(0, 1000), (1100, 2000)], []) == [(0, 2000)]
    assert gaps([(0, 1000), (1100, 2000)], [(1000, 1100)], pad_ms=0) == [(0, 1000), (1100, 2000)]


def test_subtract():
    assert subtract([(0, 10)], [(2, 4), (6, 8)]) == [(0, 2), (4, 6), (8, 10)]
    assert subtract([(0, 10)], [(0, 10)]) == []


def test_only_wordless_gaps_are_kept():
    zh = load_config(None).langpack("zh")
    assert wordless("", zh.is_interjection)
    assert wordless("哈哈哈", zh.is_interjection) and wordless("啊！", zh.is_interjection)
    # an answer the ASR missed would leak Chinese into the dub
    assert not wordless("你瘦了", zh.is_interjection)
    assert not wordless("浓", zh.is_interjection)
