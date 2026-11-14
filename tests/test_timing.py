from __future__ import annotations

from dub.contracts.v1 import Translation
from dub.core.config import TimingCfg
from dub.ops.timing import (
    Neighbor,
    count_units,
    exact_target_ms,
    fit_unit,
    pick_variant,
)

T = TimingCfg()


def test_count_units():
    assert count_units("Chào buổi sáng, chú.", "syllable") == 4
    assert count_units("beautiful cake", "syllable") == 4  # beau-ti-ful + cake
    assert count_units("你好世界", "char") == 4
    assert count_units("one two three", "word") == 3


def test_pick_variant_prefers_target_when_it_fits():
    tr = Translation(unit_id="u", short="s", target="t", long="l")
    lengths = {"s": 800, "t": 1000, "l": 1300}
    assert pick_variant(tr, 1020, lengths.get, 0.08)[0] == "target"
    assert pick_variant(tr, 1350, lengths.get, 0.08)[0] == "long"
    assert pick_variant(tr, 850, lengths.get, 0.08)[0] == "short"
    # nothing fits: shortest
    assert pick_variant(tr, 500, lengths.get, 0.08)[0] == "short"
    # everything is short: the longest that fits
    assert pick_variant(tr, 5000, lengths.get, 0.08)[0] == "long"


def test_exact_target_stays_inside_safe_ratio():
    assert exact_target_ms(1000, 1000, (0.92, 1.08)) == 1000
    assert exact_target_ms(500, 1000, (0.92, 1.08)) == round(1000 / 1.08)
    assert exact_target_ms(5000, 1000, (0.92, 1.08)) == round(1000 / 0.92)


def far(nat=1000, slot=1000, gap=2000, same=False):
    return Neighbor(natural_ms=nat, slot_ms=slot, gap_ms=gap, same_speaker=same)


def test_short_clip_starts_on_the_slot():
    f = fit_unit(
        slot_start=1000, slot_end=3000, natural_ms=1500, clip_ms=1500, prev=far(), next_=far(), t=T
    )
    assert (f.start_ms, f.end_ms, f.stretch, f.flags) == (1000, 2500, 1.0, [])


def test_borrows_silence_after_before_stretching():
    f = fit_unit(
        slot_start=1000, slot_end=3000, natural_ms=2400, clip_ms=2400, prev=far(), next_=far(), t=T
    )
    assert f.stretch == 1.0 and f.start_ms == 1000 and f.end_ms == 3400


def test_stretch_when_no_room_and_flags():
    tight = far(gap=0)
    f = fit_unit(
        slot_start=1000, slot_end=3000, natural_ms=2300, clip_ms=2300, prev=tight, next_=tight, t=T
    )
    assert f.end_ms - f.start_ms == 2000 and f.stretch > 1.1 and "fast" in f.flags
    g = fit_unit(
        slot_start=1000, slot_end=3000, natural_ms=3000, clip_ms=3000, prev=tight, next_=tight, t=T
    )
    assert "overflow" in g.flags and abs(g.rate - T.rate_cap) < 0.01


def test_same_speaker_neighbour_limits_borrowing():
    other = fit_unit(
        slot_start=1000,
        slot_end=3000,
        natural_ms=3000,
        clip_ms=3000,
        prev=far(gap=0),
        next_=far(gap=600, same=False),
        t=T,
    )
    same = fit_unit(
        slot_start=1000,
        slot_end=3000,
        natural_ms=3000,
        clip_ms=3000,
        prev=far(gap=0),
        next_=far(gap=600, same=True),
        t=T,
    )
    assert other.room_after_ms > same.room_after_ms


def test_bounded_ripple_is_local():
    """Placement of a unit depends only on its neighbours' natural length and slot, never on
    how the neighbours were placed: changing unit i-2 cannot move unit i."""
    a = fit_unit(
        slot_start=5000,
        slot_end=7000,
        natural_ms=2600,
        clip_ms=2600,
        prev=far(nat=1500, gap=400),
        next_=far(gap=400),
        t=T,
    )
    b = fit_unit(
        slot_start=5000,
        slot_end=7000,
        natural_ms=2600,
        clip_ms=2600,
        prev=far(nat=1500, gap=400),
        next_=far(gap=400),
        t=T,
    )
    assert a == b
    hungry = fit_unit(
        slot_start=5000,
        slot_end=7000,
        natural_ms=2600,
        clip_ms=2600,
        prev=far(nat=1500, gap=400),
        next_=far(nat=2000, slot=1000, gap=400),
        t=T,
    )
    assert hungry.room_after_ms <= a.room_after_ms  # a needy neighbour keeps its half


def test_strict_mode_halves_limits():
    s = T.model_copy(update={"strict": True})
    f = fit_unit(
        slot_start=1000, slot_end=3000, natural_ms=2400, clip_ms=2400, prev=far(), next_=None, t=s
    )
    assert f.room_after_ms == T.late_end_ms // 2


def test_short_lines_get_a_floor_long_lines_stay_linear():
    from dub.ops.timing import predict_ms

    assert predict_ms("Trai.", "syllable", 4.7) == 213
    assert predict_ms("Trai.", "syllable", 4.7, 300, 150) == 450  # 300 + 1 x 150
    long = "Hôm nay trời rất đẹp, chúng ta cùng đi dạo nhé."  # 11 syllables
    assert predict_ms(long, "syllable", 4.7, 300, 150) == predict_ms(long, "syllable", 4.7)
