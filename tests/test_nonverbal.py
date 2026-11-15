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


def test_a_kept_line_covers_its_real_sound():
    # The cartoon: the ASR squeezed the song into 40-340 ms; its subtitles run 1.5-4.75 s; the
    # next dubbed line starts at 6.02 s. Cut at the slot, the song went silent at 0.3-3.8 s.
    from dub.ops.nonverbal import original_extents, share

    lines = [("song", 40, 340, "扭扭大肚肚舞步萌"), ("dudu", 3780, 4940, "嘟嘟"),
             ("line", 6020, 8300, "大爷 你切什么呀")]  # fmt: skip
    cues = [(1500, 3000, "扭扭大肚肚"), (3250, 4750, "舞步萌嘟嘟"), (6000, 6750, "大爷"),
            (6750, 8500, "你切什么呀")]  # fmt: skip
    regions = [(0, 2900), (3050, 4950), (5150, 5500), (6000, 8300)]
    ext = original_extents(lines, {"song", "dudu"}, cues, regions)
    assert ext["song"][0] == 0 and ext["song"][1] >= 4750  # the whole song
    assert ext["dudu"][0] >= ext["song"][1]  # never twice over the same span
    assert ext["dudu"][1] <= 6020 - 150  # never into the dubbed line
    assert "line" not in ext
    # a sound 200 ms after the song (a word of the next line?) is not pulled in: the
    # non-verbal step judges it, with its check for words
    assert max(b for _, b in ext.values()) < 5150
    assert share("舞步萌嘟嘟", "扭扭大肚肚舞步萌") == 0.6
