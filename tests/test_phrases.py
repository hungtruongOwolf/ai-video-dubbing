"""Phrase timing: a dubbed line pauses where its actor pauses (dub.ops.phrases)."""

from __future__ import annotations

import numpy as np

from dub.contracts.v1 import PhraseReq
from dub.ops import phrases
from dub.ops.audio_io import read_wav, write_wav
from dub.store.cas import CAS
from tests.conftest import requires_ffmpeg


def test_the_take_pauses_where_the_actor_pauses():
    # the actor: 1 s, a 500 ms pause, 1 s; the take pauses at the same share of its speech
    p = phrases.plan([(0, 1000), (1500, 2500)], [(20, 1000), (1150, 2100)], end_max_ms=2600)
    assert p is not None and p.matched == 1
    first, second = p.pieces
    assert (first.out0, first.out1) == (20, 1000)
    assert second.out0 == 1500 and abs(second.out1 - 2500) <= 2  # slowed a little to fill
    assert 0.87 <= second.rate <= 1.15


def test_a_long_pause_the_actor_speaks_through_is_shortened():
    p = phrases.plan([(0, 2000)], [(0, 900), (1300, 2000)], end_max_ms=2100)
    assert p is not None and p.shortened == 1 and p.matched == 0
    first, second = p.pieces
    assert second.out0 - first.out1 == 100  # 400 ms of silence -> 100 ms
    assert abs(first.rate - 0.87) < 0.01  # slowed down, but no more than the floor


def test_nothing_changes_without_pauses_to_match_or_shorten():
    assert phrases.plan([(0, 2000)], [(0, 2000)], end_max_ms=2100) is None
    assert phrases.plan([(0, 2000)], [(0, 900), (1000, 2000)], end_max_ms=2100) is None
    # a stop consonant's 50 ms of silence is not a cut point
    assert phrases.plan([(0, 900), (1400, 2000)], [(0, 1000), (1050, 2000)],
                        end_max_ms=2100) is None  # fmt: skip


def test_a_pause_far_from_the_takes_silences_stays_unmatched():
    # the actor pauses at 20% of the speech, the take only at 80%: no match, no change
    assert phrases.plan([(0, 400), (900, 2500)], [(0, 1600), (1700, 2000)],
                        end_max_ms=2600) is None  # fmt: skip


def test_the_line_never_ends_later_than_allowed():
    # the second phrase of the take is far longer than the actor's: re-timed, it would overrun
    assert phrases.plan([(0, 1000), (1500, 1800)], [(0, 1000), (1100, 3000)],
                        end_max_ms=3050) is None  # fmt: skip


def test_matching_is_monotonic_and_prefers_the_closest_silence():
    pairs = phrases.match([0.3, 0.7], [0.1, 0.32, 0.5, 0.69], [200, 200, 200, 200],
                          max_shift=0.15)  # fmt: skip
    assert pairs == [(0, 1), (1, 3)]


def test_the_back_checks_words_follow_the_pieces():
    pieces = [(20, 1000, 20, 1000), (1150, 2100, 1500, 2500)]
    assert phrases.warp(pieces, 500) == 500
    assert phrases.warp(pieces, 1100) == 1500  # in the cut silence: where the piece starts
    assert 1500 < phrases.warp(pieces, 1600) < 2000
    assert phrases.warp(pieces, 2100) == 2500


@requires_ffmpeg
def test_align_moves_the_second_phrase_to_the_actors(tmp_path):
    sr = 48000
    t = np.arange(sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    take = np.concatenate([tone[: sr * 98 // 100], np.zeros(sr * 15 // 100, np.float32),
                           tone[: sr * 95 // 100]])  # fmt: skip
    cas = CAS(tmp_path / "cas")
    src = tmp_path / "take.wav"
    write_wav(src, take, sr)
    key = cas.put_file(src)
    res = phrases.align(
        PhraseReq(audio_key=key, actor=[(0, 1000), (1500, 2500)], end_max_ms=2600), cas
    )
    assert res.matched == 1 and res.audio_key != key
    out, _ = read_wav(cas.path(res.audio_key))
    env = np.abs(out[:, 0])
    ms = lambda a, b: env[a * sr // 1000 : b * sr // 1000].max()  # noqa: E731
    assert ms(1100, 1450) < 0.01  # silent through the actor's pause
    assert ms(1550, 2400) > 0.1  # speaking with the actor again
    assert abs(res.dur_ms - 2500) < 60
