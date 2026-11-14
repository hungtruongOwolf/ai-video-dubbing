from __future__ import annotations

import numpy as np

from dub.ops.diarize import absorb_small, purity, turns_from_windows, windows


def test_windows_cover_speech_and_skip_blips():
    w = windows([(0, 4000), (5000, 5300), (6000, 6900)])
    assert w[0] == (0, 1500) and w[-2][1] == 4000 and w[-1] == (6000, 6900)
    assert all(not (s >= 5000 and e <= 5300) for s, e in w)  # 300 ms blip skipped


def test_two_speakers_become_turns_in_order_of_appearance():
    regions = [(0, 3000), (3500, 7000)]
    wins = windows(regions)
    labels = [7 if s < 3000 else 3 for s, _ in wins]
    turns = turns_from_windows(wins, labels, regions)
    assert [(t.speaker, t.start_ms, t.end_ms) for t in turns] == [
        ("S1", 0, 3000),
        ("S2", 3500, 7000),
    ]


def test_short_flip_inside_a_turn_is_absorbed():
    regions = [(0, 9000)]
    wins = windows(regions)
    labels = [2 if 4000 <= (s + e) / 2 < 4300 else 1 for s, e in wins]
    turns = turns_from_windows(wins, labels, regions)
    assert len(turns) == 1 and turns[0].speaker == "S1"


def test_tiny_cluster_is_folded_into_the_nearest_speaker():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=8), rng.normal(size=8)
    embs = np.stack([a] * 20 + [b] * 20 + [a * 0.9 + b * 0.1])
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    labels = np.array([1] * 20 + [2] * 20 + [3])
    wins = [(i * 750, i * 750 + 1500) for i in range(41)]
    out = absorb_small(embs, labels, wins)
    assert set(out.tolist()) == {1, 2} and out[-1] == 1


def test_purity():
    from dub.contracts.v1 import Turn

    turns = [
        Turn(speaker="S1", start_ms=0, end_ms=3000),
        Turn(speaker="S2", start_ms=3000, end_ms=4000),
    ]
    assert purity(0, 4000, "S1", turns) == 0.75
    assert purity(0, 2000, "S1", turns) == 1.0
