from __future__ import annotations

from dub.contracts.v1 import Cue, SubsReq
from dub.ops.subs import balanced_lines, build_cues, to_srt


def test_balanced_lines():
    tokens = "Good evening everyone, and welcome back to our little kitchen show.".split()
    lines = balanced_lines(tokens, 42, " ", 2)
    assert lines and len(lines) == 2 and all(len(x) <= 42 for x in lines)
    assert abs(len(lines[0]) - len(lines[1])) < 12
    assert balanced_lines(["x" * 50], 42, " ", 2) is None


def test_long_line_becomes_balanced_cues_without_orphans():
    text = " ".join(["word"] * 40)
    cues = build_cues(SubsReq(lang="en", items=[Cue(start_ms=0, end_ms=8000, text=text)]))
    assert len(cues) >= 2
    sizes = [len(c.text) for c in cues]
    assert min(sizes) > 0.5 * max(sizes)  # no tiny orphan cue
    assert cues[0].start_ms == 0 and cues[-1].end_ms == 8000
    for c in cues:
        assert all(len(line) <= 42 for line in c.text.split("\n"))
        assert len(c.text.split("\n")) <= 2


def test_cjk_wraps_by_character():
    text = "我们今天要做一道非常简单的鱼汤这是我奶奶的做法"
    cues = build_cues(
        SubsReq(lang="zh", items=[Cue(start_ms=0, end_ms=5000, text=text)], cpl=16, joiner="")
    )
    assert all(len(line) <= 16 for c in cues for line in c.text.split("\n"))
    assert "".join(c.text.replace("\n", "") for c in cues) == text


def test_cues_never_overlap_and_srt_format():
    items = [Cue(start_ms=0, end_ms=2000, text="one"), Cue(start_ms=1500, end_ms=3000, text="two")]
    cues = build_cues(SubsReq(lang="en", items=items))
    assert cues[0].end_ms <= cues[1].start_ms
    srt = to_srt(cues)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\none\n")
