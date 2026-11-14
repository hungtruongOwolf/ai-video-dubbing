"""Voice hints for the cast sheet: scores -> SpeakerVoice, and the clean spans picked."""

from __future__ import annotations

from dub.contracts.v1 import Unit
from dub.core.pipeline import _voice_spans
from dub.providers.voice_agender import age_band, to_voice


def test_scores_become_a_weak_hint():
    v = to_voice("S1", 0.72, 0.05, 0.90, 0.05, min_conf=0.7)
    assert (v.gender, v.age_band) == ("male", "senior") and v.gender_conf > 0.9
    unsure = to_voice("S2", 0.35, 0.40, 0.45, 0.15, min_conf=0.7)
    assert unsure.gender == "unknown" and unsure.age_band == "adult"
    assert age_band(9, 0.2) == "child" and age_band(30, 0.6) == "child"
    assert age_band(19, 0.1) == "young"


def test_voice_spans_take_the_longest_clean_lines_per_speaker():
    def u(i, spk, start, dur):
        return Unit(id=f"u{i}", speaker=spk, sentence_ids=["s"], start_ms=start,
                    end_ms=start + dur, text="x", n_words=1)  # fmt: skip

    units = [u(1, "S1", 0, 1500), u(2, "S1", 2000, 6000), u(3, "S2", 9000, 3000),
             u(4, "S2", 13000, 16000), u(5, "S1", 30000, 4000)]  # fmt: skip
    spans, speakers = _voice_spans(units, {"u5": 0.5}, per_speaker=5)
    assert spans == [(2000, 8000), (9000, 12000)] and speakers == ["S1", "S2"]
