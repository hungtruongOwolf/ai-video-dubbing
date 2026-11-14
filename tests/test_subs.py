from __future__ import annotations

from dub.contracts.v1 import SubItem, SubsReq, Word
from dub.ops.align import cer, char_times
from dub.ops.script import RLM, ZWSP, break_tokens, display_len, is_rtl
from dub.ops.subs import balanced_lines, build_cues, to_srt


def _req(lang, text, start=0, end=5000, words=(), **kw):
    item = SubItem(start_ms=start, end_ms=end, text=text, words=list(words))
    return SubsReq(lang=lang, items=[item], linger_ms=0, min_cue_ms=0, **kw)


def test_balanced_lines():
    tokens = break_tokens(
        "Good evening everyone, and welcome back to our little kitchen show.", "en"
    )
    lines = balanced_lines(tokens, 42, "en", 2)
    assert lines and len(lines) == 2 and all(len(x) <= 42 for x in lines)
    assert abs(len(lines[0]) - len(lines[1])) < 12
    assert balanced_lines(["x" * 50], 42, "en", 2) is None


def test_long_line_becomes_balanced_cues_without_orphans():
    text = " ".join(["word"] * 40)
    cues = build_cues(_req("en", text, end=8000))
    assert len(cues) >= 2
    sizes = [len(c.text) for c in cues]
    assert min(sizes) > 0.5 * max(sizes)  # no tiny orphan cue
    assert cues[0].start_ms == 0 and cues[-1].end_ms == 8000
    for c in cues:
        assert all(len(line) <= 42 for line in c.text.split("\n"))
        assert len(c.text.split("\n")) <= 2


def test_chinese_breaks_between_phrases():
    text = "我们今天要做一道非常简单的鱼汤，这是我奶奶的做法，大家一定要试试看。"
    cues = build_cues(_req("zh", text, cpl=16))
    lines = [line for c in cues for line in c.text.split("\n")]
    assert all(display_len(line, "zh") <= 16 for line in lines)
    assert "".join(lines) == text
    assert not any(line.startswith(("，", "。")) for line in lines)
    # "奶奶" (grandma) is one BudouX phrase: never split across lines.
    pairs = zip(lines, lines[1:], strict=False)
    assert not any(a.endswith("奶") and b.startswith("奶") for a, b in pairs)


def test_japanese_kinsoku():
    tokens = break_tokens("今日は「明日」も晴れるでしょう、たぶん。ちょっと待ってください。", "ja")
    assert (
        "".join(tokens).replace(" ", "")
        == "今日は「明日」も晴れるでしょう、たぶん。ちょっと待ってください。"
    )
    for t in tokens:
        assert t[0] not in "、。」ょっ"  # no piece (so no line) starts with these
        assert t.rstrip()[-1] not in "「"  # and none ends with an opening bracket


def test_thai_breaks_only_at_word_boundaries():
    text = "วันนี้อากาศดีมาก ฉันจะไปเที่ยวทะเลกับครอบครัวของฉันในวันหยุดสุดสัปดาห์นี้"
    tokens = break_tokens(text, "th")
    assert "".join(tokens) == text
    assert "วันนี้" in tokens[0]  # BudouX alone would split this compound ("today")
    cues = build_cues(_req("th", text, cpl=20))
    for c in cues:
        for line in c.text.split("\n"):
            assert display_len(line, "th") <= 20
    assert ZWSP in "".join(c.text for c in cues)  # break hints for players that re-wrap


def test_display_width_rules():
    assert display_len("ที่นี่", "th") == 2  # tone marks and vowels above take no width
    assert display_len("日本語ABC", "ja") == 4.5  # half-width counts half in CJK
    assert display_len("Tiếng Việt", "vi") == 10


def test_rtl_lines_get_a_direction_mark():
    text = "مرحبا بكم في برنامجنا اليوم، سنتحدث عن الطبخ"
    assert is_rtl(text) and not is_rtl("hello مرحبا world again")
    cues = build_cues(_req("ar", text, cpl=42))
    assert all(line.startswith(RLM) for c in cues for line in c.text.split("\n"))


def test_cue_boundaries_follow_the_voice():
    # The voice says the first sentence fast and pauses long before the second one.
    text = "Hello there, my friend. This is a much longer second sentence that follows."
    words, t = [], 0
    for i, w in enumerate(text.split()):
        dur = 150 if i < 4 else 400
        words.append(Word(w=w, start_ms=t, end_ms=t + dur))
        t += dur + (1200 if i == 3 else 50)
    cues = build_cues(_req("en", text, end=t, words=words, cpl=30, max_lines=1))
    assert cues[0].text.startswith("Hello") and cues[1].text.startswith("This")
    # Boundary sits in the 1.2 s pause after "friend." (ends at 750 ms), not at the
    # character share (~30% of 5.6 s = 1.7 s).
    assert 750 <= cues[0].end_ms <= 1950 and cues[1].start_ms <= 1950
    # Without words the boundary falls by length instead.
    plain = build_cues(_req("en", text, end=t, cpl=30, max_lines=1))
    assert plain[1].start_ms > cues[1].start_ms


def test_cues_linger_but_never_overlap_and_srt_format():
    items = [
        SubItem(start_ms=0, end_ms=1000, text="one"),
        SubItem(start_ms=1200, end_ms=3000, text="two"),
    ]
    cues = build_cues(SubsReq(lang="en", items=items))
    assert cues[0].end_ms == 1200 - 84  # lingered into the gap, kept 2 frames
    assert cues[1].end_ms == 3500
    srt = to_srt(cues)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,116\none\n")


def test_char_times_and_cer():
    words = [Word(w="xin", start_ms=0, end_ms=200), Word(w="chao", start_ms=300, end_ms=600)]
    t = char_times("Xin chào!", words)
    assert t is not None and t[0][0] == 0 and t[4][0] >= 300 and t[-1][0] == t[-2][1] == 600
    assert cer("Xin chào!", "xin chào") == 0.0
    assert cer("Xin chào", "xin chao") > 0  # a wrong tone is an error
    assert cer("abc", "") == 1.0
