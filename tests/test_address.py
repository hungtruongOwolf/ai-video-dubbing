from __future__ import annotations

from dub.contracts.v1 import AddressTerms, CastSheet, Relation, Translation, Unit
from dub.ops.address import check_line, consistency

GENERIC = {"tôi", "tao", "tớ", "mày", "cậu", "bạn"}


def _rel(listener: str, self_t: str, other_t: str, since: int = 0) -> Relation:
    terms = AddressTerms(self_term=self_t, other_term=other_t)
    return Relation(speaker="S3", listener=listener, since_ms=since, address={"vi": terms})


def _cast() -> CastSheet:
    return CastSheet(
        relations=[
            _rel("S1", "cháu", "ông"),
            _rel("S2", "con", "mẹ"),
            _rel("S2", "tôi", "bà", 60000),
        ]
    )


def _unit(i: int, start: int = 0) -> Unit:
    return Unit(id=f"u{i}", speaker="S3", sentence_ids=[], start_ms=start, end_ms=start + 2000,
                text="x", n_words=1)  # fmt: skip


def _tr(i, text, listener, self_t=None, other_t=None) -> Translation:
    return Translation(unit_id=f"u{i}", short=text, target=text, long=text, listener=listener,
                       self_term=self_t, other_term=other_t)  # fmt: skip


def test_line_check_against_the_cast_sheet():
    cast = _cast()
    ok = _tr(1, "Ông ơi, cháu đói rồi.", "S1", "cháu", "ông")
    assert check_line(_unit(1), ok.target, ok, cast, "vi", GENERIC) == []
    wrong = _tr(2, "Ông ơi, tôi đói rồi.", "S1", "tôi", "ông")
    issues = check_line(_unit(2), wrong.target, wrong, cast, "vi", GENERIC)
    assert any("self term 'tôi'" in i for i in issues) and any("generic" in i for i in issues)
    # after since_ms the relation changed (formal): "tôi" is right there
    later = _tr(3, "Thưa bà, tôi hiểu rồi.", "S2", "tôi", "bà")
    assert check_line(_unit(3, 70000), later.target, later, cast, "vi", GENERIC) == []


def test_whole_script_consistency_flags_the_odd_line_out():
    units = [_unit(i, i * 3000) for i in range(5)]
    trs = {f"u{i}": _tr(i, "...", "S1", "cháu", "ông") for i in range(4)}
    trs["u4"] = _tr(4, "...", "S1", "con", "bố")
    trs["u3"] = _tr(3, "...", "S1", None, "ông")  # a dropped pronoun is not a conflict
    out = consistency(units, trs, _cast())
    assert list(out) == ["u4"] and "3 of 5" in out["u4"]
