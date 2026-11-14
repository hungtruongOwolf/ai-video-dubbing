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
                        "characters": 1,
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
                        "characters": 1,
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


def test_llm_sentence_boundaries_in_windows(cfg, ctx, calls):
    from dub.contracts.v1 import BoundaryReq
    from dub.providers.claude_cli import boundary_windows

    assert boundary_windows(10, core=4, context=1) == [(0, 0, 4, 5), (3, 4, 8, 9), (7, 8, 10, 10)]
    words = [f"w{i}" for i in range(10)]
    seen = calls(ok({"ends": [1, 3, 5, 9]}))  # every window answers the same: out-of-window
    prm = {**ctx.params, "window_words": 4, "context_words": 1, "parallel_windows": 1}
    run_ctx = RunCtx(cas=ctx.cas, manifest=ctx.manifest, params=prm, prompts_dir=ctx.prompts_dir,
                     log=ctx.log)  # fmt: skip
    req = BoundaryReq(lang="en", words=words, pauses_ms=[0, 900] + [0] * 8)
    res = ClaudeCLI(cfg.providers["claude_cli"]).run("text.boundaries", req, run_ctx)
    ends = [i for i, p in enumerate(res.probs) if p > 0.5]
    assert ends == [1, 3, 5, 9]  # indices outside a window's core are ignored there
    assert len(seen) == 3
    first = seen[0]["stdin"]
    assert "w1{1} <pause 0.9 s>" in first and "[DECIDE]" in first and "Context after:" in first


def test_an_incomplete_translation_is_a_retryable_error(cfg, ctx, calls):
    calls(ok({"lines": [{"unit_id": "u1", "listener": None, "self_term": None,
                         "other_term": None, "target": "Xin chào.", "short": "Chào.",
                         "long": "Xin chào bạn.", "note": None}]}))  # fmt: skip
    req = TranslateReq(
        src_lang="en", tgt_lang="vi",
        units=[unit("u1", "S1", "Hello.", 0), unit("u2", "S1", "Bye.", 3000)],
        context=ContextSheet(),
    )  # fmt: skip
    with pytest.raises(ProviderError) as e:
        ClaudeCLI(cfg.providers["claude_cli"]).run("mt.translate", req, ctx)
    assert e.value.kind == "bad_output" and e.value.retryable and "u2" in str(e.value)


def test_speakers_in_windows_keep_character_ids(cfg, ctx, monkeypatch):
    """Windows go in order and each one sees the characters named before; a line the answer
    leaves out takes the previous speaker on its voice."""
    from dub.contracts.v1 import SpeakerLine, SpeakersReq, SpeakerVoice

    texts = ["How many months?", "Three.", "Boy or girl?", "Boy.", "Where's the money from?"]
    lines = [SpeakerLine(start_ms=i * 1000, end_ms=i * 1000 + 800, text=t,
                         voice="S2" if i == 4 else "S1") for i, t in enumerate(texts)]  # fmt: skip
    answers = iter([
        {"characters": [{"id": "P1", "who": "the employer", "gender": "female"},
                        {"id": "P2", "who": "the wet nurse", "gender": "female"}],
         "lines": [{"line": 0, "speaker": "P1", "sure": True},
                   {"line": 1, "speaker": "P2", "sure": True},
                   {"line": 2, "speaker": "P1", "sure": True}]},
        {"characters": [{"id": "P3", "who": "her husband", "gender": "male"}],
         "lines": [{"line": 4, "speaker": "P3", "sure": True}]},  # line 3 left out
    ])  # fmt: skip
    seen = []

    def runner(cmd, stdin, env, timeout):
        seen.append(stdin)
        return subprocess.CompletedProcess(cmd, 0, json.dumps(ok(next(answers))), "")

    monkeypatch.setattr(ClaudeCLI, "runner", staticmethod(runner))
    prm = {**ctx.params, "window_lines": 3, "context_lines": 1}
    run_ctx = RunCtx(cas=ctx.cas, manifest=ctx.manifest, params=prm,
                     prompts_dir=ctx.prompts_dir, log=ctx.log)  # fmt: skip
    req = SpeakersReq(
        lang="en",
        lines=lines,
        voices=[SpeakerVoice(speaker="S1", gender="female", gender_conf=0.9)],
    )
    res = ClaudeCLI(cfg.providers["claude_cli"]).run("text.speakers", req, run_ctx)
    assert res.speaker == ["P1", "P2", "P1", "P1", "P3"]
    assert res.sure == [True, True, True, False, True]
    assert [c.id for c in res.characters] == ["P1", "P2", "P3"]
    assert "0 [00:00.0] V1: How many months?" in seen[0]
    assert "- V1: sounds female" in seen[0] and "- none yet" in seen[0]
    assert "- P2: the wet nurse (female)" in seen[1] and "Context before:" in seen[1]
