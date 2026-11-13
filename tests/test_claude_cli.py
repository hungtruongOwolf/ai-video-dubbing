"""The Claude CLI provider, tested with a fake `claude` process (no network, no quota)."""

from __future__ import annotations

import json
import subprocess

import pytest

from dub.contracts.v1 import (
    AddressTerms,
    CastSheet,
    CastSpeaker,
    ContextReq,
    ContextSheet,
    Relation,
    TranslateReq,
    Unit,
)
from dub.core.provider_api import ProviderError, RunCtx
from dub.providers.claude_cli import ClaudeCLI, LLMContext, inline_schema, translate_prompt


def unit(uid, spk, text, start):
    return Unit(
        id=uid,
        speaker=spk,
        sentence_ids=["s"],
        start_ms=start,
        end_ms=start + 2000,
        text=text,
        n_words=len(text.split()),
    )


@pytest.fixture
def ctx(cfg, ws):
    return RunCtx(
        cas=ws.cas,
        manifest=cfg.providers["claude_cli"],
        params=cfg.providers["claude_cli"].params,
        prompts_dir=cfg.prompts_dir,
        log=lambda m: None,
    )


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def install(reply: dict | None = None, *, code=0, stdout=None):
        def runner(cmd, stdin, env, timeout):
            seen.append({"cmd": cmd, "stdin": stdin, "env": env})
            out = stdout if stdout is not None else json.dumps(reply)
            return subprocess.CompletedProcess(cmd, code, out, "")

        monkeypatch.setattr(ClaudeCLI, "runner", staticmethod(runner))
        return seen

    return install


def ok(structured):
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 2,
        "result": json.dumps(structured),
        "structured_output": structured,
    }


def test_schema_has_no_refs():
    s = json.dumps(inline_schema(LLMContext))
    assert "$ref" not in s and "$defs" not in s


def test_context_pass_builds_a_cast_sheet(cfg, ctx, calls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")
    seen = calls(
        ok(
            {
                "summary": "A cooking show.",
                "glossary": [{"source": "porridge", "target": "cháo"}],
                "speakers": [
                    {
                        "id": "S1",
                        "name": "Lan",
                        "gender": "unknown",
                        "gender_confidence": 0.4,
                        "age_band": "adult",
                        "role": "host",
                        "style": "warm",
                        "evidence": ["voice only"],
                    },
                    {
                        "id": "S2",
                        "name": "Tu",
                        "gender": "male",
                        "gender_confidence": 0.95,
                        "age_band": "senior",
                        "role": "guest",
                        "style": None,
                        "evidence": ["'Mr. Tu'"],
                    },
                ],
                "relations": [
                    {
                        "speaker": "S1",
                        "listener": "S2",
                        "relation": "host to older guest",
                        "formality": "polite",
                        "self_term": "cháu",
                        "other_term": "chú",
                        "since_ms": 0,
                        "confidence": 0.8,
                        "evidence": ["sir"],
                    }
                ],
                "audience": {"formality": "polite", "address_term": "các bạn"},
                "mentioned": [{"name": "mother", "gender": "female", "note": None}],
                "needs_confirmation": ["Is S1 female?"],
            }
        )
    )
    req = ContextReq(src_lang="en", tgt_lang="vi", units=[unit("u1", "S1", "Welcome, Mr. Tu.", 0)])
    sheet = ClaudeCLI(cfg.providers["claude_cli"]).run("mt.context", req, ctx)

    assert isinstance(sheet, ContextSheet)
    assert sheet.cast.relations[0].address["vi"] == AddressTerms(self_term="cháu", other_term="chú")
    assert sheet.cast.audience.address == {"vi": "các bạn"}
    assert sheet.flags == ["Is S1 female?"]

    call = seen[0]
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert "--bare" not in call["cmd"]
    assert call["cmd"][call["cmd"].index("--tools") + 1] == ""
    assert "Welcome, Mr. Tu." in call["stdin"]
    assert "script supervisor" in call["cmd"][call["cmd"].index("--system-prompt") + 1]


def test_translate_drops_unknown_ids_and_passes_cast_rows(cfg, ctx, calls):
    calls(
        ok(
            {
                "lines": [
                    {
                        "unit_id": "u1",
                        "listener": "S2",
                        "self_term": "cháu",
                        "other_term": "chú",
                        "target": "Chào chú Tư.",
                        "short": "Chào chú.",
                        "long": "Cháu chào chú Tư ạ.",
                        "note": None,
                    },
                    {
                        "unit_id": "zzz",
                        "listener": None,
                        "self_term": None,
                        "other_term": None,
                        "target": "x",
                        "short": "x",
                        "long": "x",
                        "note": None,
                    },
                ]
            }
        )
    )
    sheet = ContextSheet(
        cast=CastSheet(
            speakers=[CastSpeaker(id="S1"), CastSpeaker(id="S2", gender="male")],
            relations=[
                Relation(
                    speaker="S1",
                    listener="S2",
                    address={"vi": AddressTerms(self_term="cháu", other_term="chú")},
                )
            ],
        )
    )
    req = TranslateReq(
        src_lang="en",
        tgt_lang="vi",
        units=[unit("u1", "S1", "Hello Mr. Tu.", 0)],
        context=sheet,
        units_per_s=5.22,
    )
    payload = json.loads(translate_prompt(req))
    assert payload["relations"][0]["self_term"] == "cháu"
    assert payload["lines"][0]["target_units"] == round(2.0 * 5.22)

    batch = ClaudeCLI(cfg.providers["claude_cli"]).run("mt.translate", req, ctx)
    assert [t.unit_id for t in batch.items] == ["u1"]
    assert batch.items[0].other_term == "chú"


@pytest.mark.parametrize(
    "stdout,kind",
    [
        (
            json.dumps(
                {"is_error": True, "subtype": "error", "result": "Claude usage limit reached"}
            ),
            "usage_limit",
        ),
        (json.dumps({"is_error": True, "subtype": "error", "result": "Not logged in"}), "auth"),
        (
            json.dumps({"is_error": False, "subtype": "success", "result": "no json here"}),
            "bad_output",
        ),
        ("garbage", "crash"),
    ],
)
def test_errors_are_classified(cfg, ctx, calls, stdout, kind):
    calls(stdout=stdout, code=0 if kind == "bad_output" else 1)
    req = ContextReq(src_lang="en", tgt_lang="vi", units=[unit("u1", "S1", "Hi.", 0)])
    with pytest.raises(ProviderError) as e:
        ClaudeCLI(cfg.providers["claude_cli"]).run("mt.context", req, ctx)
    assert e.value.kind == kind
