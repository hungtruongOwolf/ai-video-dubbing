"""Burned-in subtitles as a second opinion on the transcript (M8).

Chinese short dramas carry their script in the picture. Read by OCR on the laptop
(`dub.hardsubs_ocr`), the cues are what the actors say, written by the producer: on a TikTok
drama Whisper's transcript differed from them by 8.8% of the characters, mostly homophones
that change the meaning (城东前府 for 城东钱府, 乃至为 for 奶只喂).

`correct`: the cue text and the ASR text are aligned character by character, window by
window along the timeline. Where they differ, the cue wins:
- a substituted run of characters takes the cue's characters (the ASR word keeps its time);
- characters only the cue has (the ASR dropped them, at most 3) join the previous word;
- a whole cue the ASR did not hear at all (a short answer under another voice: "没兴趣")
  becomes a word of its own, timed by the cue, so that it can be another person's line;
- characters only the ASR has stay: subtitles often leave out fillers and repetitions.
Cues read less surely than `min_conf` correct nothing. Punctuation the ASR wrote stays.

`uncaptioned`: when the subtitles carry the dialogue (most ASR words fall inside a cue), a
unit with almost no subtitle over it is not dialogue: a song in the background, a crowd.
The pipeline keeps its original audio instead of dubbing it.
"""

from __future__ import annotations

import difflib
import unicodedata

from dub.contracts.v1 import HardsubCue, HardsubFix, HardsubReq, Unit, Word

WINDOW_MS = 60_000
MAX_INSERT = 3
INSERT_CONF = 0.95  # adding characters needs a surer reading than replacing them
WHOLE_CUE_MIN_CHARS = 2  # a one-character cue missing from the ASR is too often a misread
COVER_PAD_MS = 300  # a word this close to a cue counts as captioned
CAPTIONED_MIN = 0.7  # coverage above which the subtitles are taken to carry the dialogue
UNIT_MIN_COVER = 0.2  # a unit under this share of captioned time is not dialogue
ALIGN_MAX_GAP_MS = 1000  # a word aligned to a cue lies at most this far outside its time


def _content(text: str) -> list[int]:
    """Positions of the letters and digits of `text` (what is compared)."""
    return [i for i, ch in enumerate(text) if ch.isalnum()]


def _fold(ch: str) -> str:
    return unicodedata.normalize("NFKC", ch).lower()


def _covered(t0: int, t1: int, cues: list[HardsubCue]) -> int:
    """Milliseconds of [t0, t1) inside some cue (padded)."""
    total = 0
    for c in cues:
        a, b = max(t0, c.start_ms - COVER_PAD_MS), min(t1, c.end_ms + COVER_PAD_MS)
        total += max(0, b - a)
    return min(total, max(0, t1 - t0))


def correct(req: HardsubReq) -> HardsubFix:
    words = [w.model_copy() for w in req.words]
    cues = [c for c in req.cues if c.conf >= req.min_conf and _content(c.text)]
    texts = [list(w.w) for w in words]
    changed = 0
    extra: list[Word] = []  # whole cues the ASR missed
    if words and cues:
        end = max(words[-1].end_ms, cues[-1].end_ms)
        for ws in range(0, end + 1, WINDOW_MS):
            we = ws + WINDOW_MS
            asr: list[tuple[int, int]] = []  # (word index, char index in the word)
            for i, w in enumerate(words):
                if ws <= w.start_ms < we:
                    asr += [(i, k) for k in _content(w.w)]
            ocr: list[str] = []
            sure: list[bool] = []
            owner: list[int] = []  # the cue of each OCR character
            for ci, c in enumerate(cues):
                if ws <= (c.start_ms + c.end_ms) // 2 < we:
                    ks = _content(c.text)
                    ocr += [c.text[k] for k in ks]
                    sure += [c.conf >= INSERT_CONF] * len(ks)
                    owner += [ci] * len(ks)
            if not asr or not ocr:
                continue
            a_seq = [_fold(texts[i][k]) for i, k in asr]
            o_seq = [_fold(ch) for ch in ocr]
            sm = difflib.SequenceMatcher(None, o_seq, a_seq, autojunk=False)
            for tag, o1, o2, a1, a2 in sm.get_opcodes():
                if tag == "replace":
                    new = ocr[o1:o2]
                    span = asr[a1:a2]
                    if len(new) == len(span):
                        for (i, k), ch in zip(span, new, strict=True):
                            texts[i][k] = ch
                    else:  # different lengths: the cue's run goes to the first ASR char
                        (i0, k0), rest = span[0], span[1:]
                        texts[i0][k0] = "".join(new)
                        for i, k in rest:
                            texts[i][k] = ""
                    changed += max(len(new), len(span))
                elif tag == "delete":
                    # only the cue has them; one run per cue: a whole cue becomes a word of
                    # its own, a few characters join the ASR word before them
                    o = o1
                    while o < o2:
                        ci, run = owner[o], o
                        while run < o2 and owner[run] == ci:
                            run += 1
                        n = run - o
                        whole = n == len(_content(cues[ci].text))
                        if (whole and n >= WHOLE_CUE_MIN_CHARS and all(sure[o:run])
                                and not _heard(cues[ci], words)):  # fmt: skip
                            extra.append(_cue_word(cues[ci], words))
                            changed += n
                        elif n <= MAX_INSERT and all(sure[o:run]):
                            # (a transition frame can add a stray character: sure ones only)
                            if a1 > 0:
                                i, k = asr[a1 - 1]
                                texts[i][k] = texts[i][k] + "".join(ocr[o:run])
                            else:
                                i, k = asr[0]
                                texts[i][k] = "".join(ocr[o:run]) + texts[i][k]
                            changed += n
                        o = run
    out: list[Word] = []
    for w, t in zip(words, texts, strict=True):
        text = "".join(t)
        if _content(text):
            out.append(w.model_copy(update={"w": text}))
    out = sorted(out + extra, key=lambda w: (w.start_ms, w.end_ms))
    spoken = sum(max(0, w.end_ms - w.start_ms) for w in req.words) or 1
    inside = sum(_covered(w.start_ms, w.end_ms, cues) for w in req.words)
    return HardsubFix(words=out, corrected=changed, coverage=round(inside / spoken, 3),
                      inserted=len(extra))  # fmt: skip


def _heard(cue: HardsubCue, words: list[Word]) -> bool:
    """Most of the cue's time already holds ASR words: the ASR heard something there (words
    the character match put elsewhere), so the cue is not a line it missed."""
    span = max(1, cue.end_ms - cue.start_ms)
    inside = sum(max(0, min(w.end_ms, cue.end_ms) - max(w.start_ms, cue.start_ms)) for w in words)
    return inside > 0.5 * span


def _cue_word(cue: HardsubCue, words: list[Word]) -> Word:
    """A missed cue as one word timed by the cue; its voice label is the nearest word's."""
    mid = (cue.start_ms + cue.end_ms) // 2
    near = min(words, key=lambda w: abs((w.start_ms + w.end_ms) // 2 - mid), default=None)
    return Word(w=cue.text, start_ms=cue.start_ms, end_ms=cue.end_ms, conf=cue.conf,
                speaker=near.speaker if near else None, space_before=True)  # fmt: skip


def uncaptioned(units: list[Unit], cues: list[HardsubCue], coverage: float) -> list[str]:
    """Units that are not dialogue: no subtitle over them although the subtitles carry the
    dialogue everywhere else (`coverage` from `correct`)."""
    if coverage < CAPTIONED_MIN or not cues:
        return []
    return [
        u.id
        for u in units
        if u.end_ms > u.start_ms
        and _covered(u.start_ms, u.end_ms, cues) / (u.end_ms - u.start_ms) < UNIT_MIN_COVER
    ]


def align_words(
    words: list[Word], cues: list[HardsubCue], min_conf: float = 0.6
) -> list[int | None]:
    """The cue each word is said in, by aligning the characters (not the times).

    Cue times come from frames sampled at 4 fps and from when the producer switched the
    subtitle: a few hundred ms off the speech. Split by time, a reply took the first character
    of the next speaker's line ("...回来? 主" for "主家许我..."). The characters of the words and
    of the cues are aligned window by window (as in `correct`); a word takes the cue most of
    its matched characters belong to, None when none matched.
    """
    out: list[int | None] = [None] * len(words)
    idx = [i for i, c in enumerate(cues) if c.conf >= min_conf and _content(c.text)]
    if not words or not idx:
        return out
    end = max(words[-1].end_ms, cues[idx[-1]].end_ms)
    for ws in range(0, end + 1, WINDOW_MS):
        we = ws + WINDOW_MS
        asr: list[int] = []  # word index of each content char
        a_seq: list[str] = []
        for i, w in enumerate(words):
            if ws <= w.start_ms < we:
                for k in _content(w.w):
                    asr.append(i)
                    a_seq.append(_fold(w.w[k]))
        owner: list[int] = []  # cue index of each content char
        o_seq: list[str] = []
        for j in idx:
            c = cues[j]
            if ws <= (c.start_ms + c.end_ms) // 2 < we:
                for k in _content(c.text):
                    owner.append(j)
                    o_seq.append(_fold(c.text[k]))
        if not asr or not owner:
            continue
        votes: dict[int, dict[int, int]] = {}
        sm = difflib.SequenceMatcher(None, o_seq, a_seq, autojunk=False)
        for tag, o1, o2, a1, a2 in sm.get_opcodes():
            if tag not in ("equal", "replace"):
                continue
            for t in range(a2 - a1):  # replace runs of other lengths: spread proportionally
                o = o1 + (t * (o2 - o1)) // max(1, a2 - a1)
                v = votes.setdefault(asr[a1 + t], {})
                v[owner[o]] = v.get(owner[o], 0) + 1
        for i, v in votes.items():
            j = max(v, key=lambda j: (v[j], -j))
            w, c = words[i], cues[j]
            # a substitution far from the cue's time pairs unrelated characters (a song
            # against the next subtitle): not an alignment
            if (
                c.start_ms - ALIGN_MAX_GAP_MS <= w.start_ms
                and w.end_ms <= c.end_ms + ALIGN_MAX_GAP_MS
            ):
                out[i] = j
    return out
