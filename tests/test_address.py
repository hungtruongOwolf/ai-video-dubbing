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


def test_question_answered_by_the_next_speaker_sets_the_listener():
    from dub.ops.address import listener_issues

    u1 = Unit(id="a", speaker="S3", sentence_ids=[], start_ms=0, end_ms=2000,
              text="Can Tom listen to the story too?", n_words=7)  # fmt: skip
    u2 = Unit(id="b", speaker="S2", sentence_ids=[], start_ms=2300, end_ms=4000,
              text="Only if he finishes first.", n_words=5)  # fmt: skip
    trs = {"a": _tr(0, "Ông ơi...", "S1", "cháu", "ông")}
    trs["a"] = trs["a"].model_copy(update={"unit_id": "a"})
    out = listener_issues([u1, u2], trs)
    assert "addressed to S2" in out["a"]


def test_generic_pronoun_without_a_known_listener():
    family = CastSheet(relations=_cast().relations[:2])  # kinship terms only
    tr = _tr(9, "Ông đã hứa thì tôi luôn giữ lời.", None)
    assert check_line(_unit(9), tr.target, tr, family, "vi", GENERIC)
    # with a formal relation that uses "tôi", the same line is not flagged without a listener
    assert check_line(_unit(9), tr.target, tr, _cast(), "vi", GENERIC) == []


def test_alternatives_and_notes_in_the_cast_sheet():
    from dub.ops.address import matches

    assert matches("cháu", "cháu / cháu yêu") and matches("cháu yêu", "cháu")
    assert matches("con", "con (Linda)") and not matches("bố", "con (Linda)")


def test_whole_script_consistency_flags_the_odd_line_out():
    units = [_unit(i, i * 3000) for i in range(5)]
    trs = {f"u{i}": _tr(i, "...", "S1", "cháu", "ông") for i in range(4)}
    trs["u4"] = _tr(4, "...", "S1", "con", "bố")
    trs["u3"] = _tr(3, "...", "S1", None, "ông")  # a dropped pronoun is not a conflict
    out = consistency(units, trs, _cast())
    assert list(out) == ["u4"] and "3 of 5" in out["u4"]


def test_a_stranger_and_a_name_are_not_address_errors():
    # The husband says "tôi" to the gatekeeper (no relation in the cast sheet): right.
    family = CastSheet(relations=_cast().relations[:2])
    stranger = _tr(10, "Tôi tìm vợ tôi.", "S7", "tôi", "anh")
    assert check_line(_unit(10), stranger.target, stranger, family, "vi", GENERIC) == []
    # Addressing someone by name is a valid vocative: the fix loop had replaced "A Duyên" by
    # "em" and lost the name.
    named = _tr(11, "Ông ơi, cháu đói rồi.", "S1", "cháu", "A Duyên")
    assert check_line(_unit(11), named.target, named, family, "vi", GENERIC) == []
    # a wrong kinship term is still an error
    wrong = _tr(12, "Bà ơi, cháu đói rồi.", "S1", "cháu", "bà")
    assert check_line(_unit(12), wrong.target, wrong, family, "vi", GENERIC)
