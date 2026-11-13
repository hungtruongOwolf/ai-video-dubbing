from __future__ import annotations

import pytest

from dub.core.config import ProviderManifest, RouteRule
from dub.core.router import Router, RoutingError, parse_overrides


def test_override_wins_and_is_traced(cfg):
    r = Router(cfg, overrides=parse_overrides(["mt.*=fake_mt"]))
    tr = r.route("mt.translate", {"src_lang": "en", "tgt_lang": "vi"})
    assert tr.source == "override" and tr.chain == ["fake_mt"]


def test_missing_real_models_give_a_helpful_error(cfg):
    r = Router(cfg)
    with pytest.raises(RoutingError) as e:
        r.route("asr.transcribe", {"src_lang": "vi", "tgt_lang": "en"})
    msg = str(e.value)
    assert "phowhisper_large_ct2" in msg and "--fake" in msg


def test_override_of_wrong_op_is_rejected(cfg):
    r = Router(cfg, overrides=parse_overrides(["tts.synthesize=fake_mt"]))
    with pytest.raises(RoutingError, match="does not serve op"):
        r.route("tts.synthesize", {"tgt_lang": "vi"})


def _cfg_with_tts(cfg):
    """A copy of the config with two fake TTS engines in the vi chain."""
    c = type(cfg)(cfg.root, cfg.dub.model_copy(deep=True), dict(cfg.langpacks), dict(cfg.providers))
    c.langpacks["vi"] = cfg.langpacks["vi"].model_copy(
        update={"chains": {"tts.synthesize": ["fake_tts", "fake_tts_exact"]}}
    )
    return c


def test_langpack_chain_and_rule_require(cfg):
    c = _cfg_with_tts(cfg)
    r = Router(c, profile="final")
    long_unit = r.route("tts.synthesize", {"tgt_lang": "vi", "unit.slot_ms": 5000})
    assert long_unit.source == "langpack"
    assert long_unit.chain == ["fake_tts", "fake_tts_exact"]

    # Rule "short-line": slots < 1.3 s need an exact-duration engine.
    short = r.route("tts.synthesize", {"tgt_lang": "vi", "unit.slot_ms": 900})
    assert short.rule_id == "short-line"
    assert short.chain == ["fake_tts_exact"]
    assert any(x.provider == "fake_tts" and "duration_control" in x.reason for x in short.rejected)


def test_draft_profile_keeps_only_the_incumbent(cfg):
    c = _cfg_with_tts(cfg)
    tr = Router(c, profile="draft").route("tts.synthesize", {"tgt_lang": "vi"})
    assert tr.chain == ["fake_tts"]


def test_capability_filter_language_and_status(cfg):
    c = _cfg_with_tts(cfg)
    c.providers["fake_tts_exact"] = c.providers["fake_tts_exact"].model_copy(
        update={"caps": c.providers["fake_tts_exact"].caps.model_copy(update={"langs": ["en"]})}
    )
    c.providers["fake_tts"] = c.providers["fake_tts"].model_copy(update={"status": "candidate"})
    with pytest.raises(RoutingError) as e:
        Router(c, profile="final").route("tts.synthesize", {"tgt_lang": "vi"})
    assert "candidate" in str(e.value) and "language vi" in str(e.value)


def test_first_matching_rule_wins(cfg):
    c = _cfg_with_tts(cfg)
    c.dub.routes["tts.synthesize"] = [
        RouteRule(id="a", when={"unit.words": {"lte": 3}}, use=["fake_tts_exact"]),
        RouteRule(id="b", when={"unit.words": {"lte": 10}}, use=["fake_tts"]),
    ]
    r = Router(c, profile="final")
    assert r.route("tts.synthesize", {"tgt_lang": "vi", "unit.words": 2}).rule_id == "a"
    assert r.route("tts.synthesize", {"tgt_lang": "vi", "unit.words": 5}).rule_id == "b"
    assert r.route("tts.synthesize", {"tgt_lang": "vi", "unit.words": 50}).source == "langpack"


def test_unknown_runtime_is_unavailable(cfg):
    c = _cfg_with_tts(cfg)
    c.providers["remote"] = ProviderManifest(id="remote", op=["tts.synthesize"], runtime="http")
    c.langpacks["vi"] = c.langpacks["vi"].model_copy(
        update={"chains": {"tts.synthesize": ["remote", "fake_tts"]}}
    )
    tr = Router(c, profile="final").route("tts.synthesize", {"tgt_lang": "vi"})
    assert tr.chain == ["fake_tts"]
    assert tr.rejected[0].provider == "remote" and "M1" in tr.rejected[0].reason
