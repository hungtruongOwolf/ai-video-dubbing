"""Who says each line: the script read by an LLM, checked against the voices (M8).

The diarizer clusters 1.5 s windows of speaker embeddings. It separates a man from a woman,
but on a zh drama it gave one voice to seven people: ECAPA's mean cosine between two lines of
the same woman (0.32) was hardly above that between two different women (0.23-0.31), and a
hiring interview of one-second questions and answers ("How many months?" "Three." "Boy or
girl?" "Boy.") never reached a cluster of its own. Only 24% of the speaker changes got a
change of voice (experiments/phrase_speakers.py, hand labels of every subtitle cue).

A reader tells them apart from the words: a question and its answer are two people, nobody
calls himself "Steward Sun", the employer states the rules. So:

1. `lines`: the dialogue cut into phrases: the burned-in subtitle cues when the subtitles
   carry the dialogue (one cue is almost always one speaker), plus the speech no cue covers;
   otherwise the transcript cut at pauses, sentence ends and diarizer turn changes.
2. `text.speakers` (an LLM) names the characters and says who speaks each line, seeing the
   diarizer's voice under each line and the gender of each voice (weak evidence).
3. `fuse` checks the answer against the voices and turns it into speaker turns: characters
   become S1, S2... in order of first appearance. When too many lines would give a character
   a voice of the other gender (both sure), the LLM misread the scene and the diarizer's
   turns are kept.
"""

from __future__ import annotations

from dub.contracts.v1 import (
    Attribution,
    HardsubCue,
    SpeakerLine,
    SpeakerVoice,
    Turn,
    Word,
)

LINE_PAUSE_MS = 300  # a silence this long ends a phrase
LINE_MAX_MS = 8000  # a longer phrase is cut at its longest inner pause
CUE_PAD_MS = 300  # a word this close to a cue belongs to it
UNCAPTIONED_MIN = 0.7  # share of a speech piece outside the cues that makes it a line
GENDER_SURE = 0.8  # voice gender confidence above which a mismatch counts
MAX_GENDER_CONFLICT = 0.15  # share of lines; above it the attribution is not used
_ENDS = set(".?!。？！…")


def voice_under(t0: int, t1: int, turns: list[Turn]) -> str:
    """The diarizer speaker overlapping [t0, t1) the most (the nearest turn if none)."""
    best, ov = None, 0
    for t in turns:
        o = min(t1, t.end_ms) - max(t0, t.start_ms)
        if o > ov:
            best, ov = t.speaker, o
    if best is None and turns:
        best = min(turns, key=lambda t: max(t.start_ms - t1, t0 - t.end_ms)).speaker
    return best or "S1"


def _pieces(words: list[Word], turns: list[Turn], joiner: str) -> list[list[Word]]:
    """The transcript cut at pauses, sentence ends and changes of diarizer voice."""
    out: list[list[Word]] = []
    prev_voice = None
    for w in words:
        voice = voice_under(w.start_ms, w.end_ms, turns)
        if (
            not out
            or w.start_ms - out[-1][-1].end_ms >= LINE_PAUSE_MS
            or out[-1][-1].w.rstrip()[-1:] in _ENDS
            or voice != prev_voice
        ):
            out.append([w])
        else:
            out[-1].append(w)
        prev_voice = voice
    final: list[list[Word]] = []
    for p in out:  # cut over-long pieces at their longest pause
        stack = [p]
        while stack:
            q = stack.pop(0)
            if q[-1].end_ms - q[0].start_ms <= LINE_MAX_MS or len(q) < 2:
                final.append(q)
                continue
            k = max(range(1, len(q)), key=lambda i: q[i].start_ms - q[i - 1].end_ms)
            stack[:0] = [q[:k], q[k:]]
    return final


def _text(ws: list[Word], joiner: str) -> str:
    if joiner:
        return joiner.join(w.w.strip() for w in ws)
    return "".join((" " if w.space_before and i else "") + w.w.strip() for i, w in enumerate(ws))


def _overlap(w: Word, c: HardsubCue) -> int:
    return max(0, min(w.end_ms, c.end_ms + CUE_PAD_MS) - max(w.start_ms, c.start_ms - CUE_PAD_MS))


def lines(
    words: list[Word],
    turns: list[Turn],
    cues: list[HardsubCue] | None = None,
    joiner: str = " ",
    min_cue_conf: float = 0.6,
) -> list[SpeakerLine]:
    """The phrases of the dialogue, in time order (see the module doc)."""
    pieces = _pieces(words, turns, joiner)
    out: list[SpeakerLine] = []
    cues = [c for c in cues or [] if c.conf >= min_cue_conf and c.text.strip()]
    if cues:
        for c in cues:
            if out and out[-1].text == c.text and c.start_ms - out[-1].end_ms <= CUE_PAD_MS:
                out[-1] = out[-1].model_copy(update={"end_ms": c.end_ms})
                continue
            out.append(SpeakerLine(start_ms=c.start_ms, end_ms=c.end_ms, text=c.text,
                                   voice=voice_under(c.start_ms, c.end_ms, turns)))  # fmt: skip
        for p in pieces:
            t0, t1 = p[0].start_ms, p[-1].end_ms
            covered = sum(_overlap(w, c) for w in p for c in cues)
            spoken = sum(max(1, w.end_ms - w.start_ms) for w in p)
            if 1 - min(1.0, covered / spoken) >= UNCAPTIONED_MIN:
                voice = voice_under(t0, t1, turns)
                out.append(SpeakerLine(start_ms=t0, end_ms=t1, text=_text(p, joiner), voice=voice,
                                       captioned=False))  # fmt: skip
    else:
        for p in pieces:
            t0, t1 = p[0].start_ms, p[-1].end_ms
            out.append(SpeakerLine(start_ms=t0, end_ms=t1, text=_text(p, joiner),
                                   voice=voice_under(t0, t1, turns)))  # fmt: skip
    return sorted(out, key=lambda ln: (ln.start_ms, ln.end_ms))


def gender_conflicts(
    lines_: list[SpeakerLine], attr: Attribution, voices: list[SpeakerVoice]
) -> list[int]:
    """Lines given to a character of the other gender than their voice (both sure)."""
    vg = {v.speaker: v.gender for v in voices if v.gender_conf >= GENDER_SURE}
    cg = {c.id: c.gender for c in attr.characters}
    out = []
    for i, (ln, who) in enumerate(zip(lines_, attr.speaker, strict=True)):
        a, b = vg.get(ln.voice, "unknown"), cg.get(who, "unknown")
        if "unknown" not in (a, b) and a != b:
            out.append(i)
    return out


def fuse(
    lines_: list[SpeakerLine], attr: Attribution, voices: list[SpeakerVoice]
) -> tuple[list[Turn] | None, dict]:
    """Speaker turns from the attribution, or None when it contradicts the voices."""
    stats: dict = {"lines": len(lines_), "characters": len({*attr.speaker})}
    if not lines_ or len(attr.speaker) != len(lines_):
        stats["rejected"] = "line count"
        return None, stats
    conflicts = gender_conflicts(lines_, attr, voices)
    stats["gender_conflicts"] = len(conflicts)
    if len(conflicts) > MAX_GENDER_CONFLICT * len(lines_):
        stats["rejected"] = "gender"
        return None, stats
    names: dict[str, str] = {}
    for who in attr.speaker:  # S1, S2... in order of first appearance
        names.setdefault(who, f"S{len(names) + 1}")
    turns: list[Turn] = []
    for ln, who in zip(lines_, attr.speaker, strict=True):
        spk = names[who]
        if turns and turns[-1].speaker == spk and ln.start_ms - turns[-1].end_ms < 1000:
            turns[-1] = turns[-1].model_copy(update={"end_ms": max(turns[-1].end_ms, ln.end_ms)})
        else:
            turns.append(Turn(speaker=spk, start_ms=ln.start_ms, end_ms=ln.end_ms))
    voice_of: dict[str, dict[str, int]] = {}
    for ln, who in zip(lines_, attr.speaker, strict=True):
        d = voice_of.setdefault(names[who], {})
        d[ln.voice] = d.get(ln.voice, 0) + ln.end_ms - ln.start_ms
    # lines whose speaker differs from the diarizer's (after the best voice -> character match)
    main = {spk: max(d, key=d.get) for spk, d in voice_of.items()}
    stats["moved"] = sum(
        1 for ln, who in zip(lines_, attr.speaker, strict=True) if main[names[who]] != ln.voice
    )
    stats["unsure"] = sum(1 for s in attr.sure if not s)
    stats["names"] = {names[c.id]: c.who for c in attr.characters if c.id in names}
    stats["kinds"] = {names[c.id]: c.kind for c in attr.characters
                      if c.id in names and c.kind != "person"}  # fmt: skip
    return turns, stats
