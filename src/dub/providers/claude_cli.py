"""Claude via the Claude Code CLI (`claude -p`), using your Claude subscription instead of an
API key.

Each call runs:

    claude -p --model sonnet --max-turns 2 --output-format json --tools "" \
           --strict-mcp-config --no-session-persistence \
           --system-prompt <prompt file> --json-schema <schema>      (user prompt on stdin)

and reads the validated JSON from the `structured_output` field of the result.

Notes
- Authentication: on the laptop, your normal `claude` login is used. On a pod, create a
  token once on the laptop with `claude setup-token` and set CLAUDE_CODE_OAUTH_TOKEN.
- ANTHROPIC_API_KEY is removed from the child environment: if it were set, `claude` would
  bill the API key instead of the subscription.
- `--bare` is NOT used: it skips the subscription login.
- `--tools ""` disables every tool, so the model can only answer.
- Personal use only: the subscription must not power an app used by other people. Switch the
  mt.* route to an API provider if you ever share the app.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from dub.contracts.v1 import (
    AddressTerms,
    Audience,
    Boundaries,
    BoundaryReq,
    CastSheet,
    CastSpeaker,
    ContextReq,
    ContextSheet,
    GlossaryEntry,
    Mentioned,
    Relation,
    RewriteReq,
    RewriteRes,
    TranslateReq,
    Translation,
    TranslationBatch,
    Unit,
)
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx

Gender = Literal["female", "male", "unknown"]
AgeBand = Literal["child", "young", "adult", "senior", "unknown"]
Formality = Literal["formal", "polite", "casual", "intimate", "unknown"]

# ---------------------------------------------------------------- LLM-facing schemas
# Flat, all-required models: structured output is most reliable without optional fields
# and free-form maps. They are converted to the /v1 contracts after validation.


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMSpeaker(_S):
    id: str
    name: str | None
    gender: Gender
    gender_confidence: float
    age_band: AgeBand
    role: str | None
    style: str | None
    evidence: list[str]


class LLMRelation(_S):
    speaker: str
    listener: str
    relation: str | None
    formality: Formality
    self_term: str | None
    other_term: str | None
    since_ms: int
    confidence: float
    evidence: list[str]


class LLMAudience(_S):
    formality: Formality
    address_term: str | None


class LLMMentioned(_S):
    name: str
    gender: Gender
    note: str | None


class LLMGloss(_S):
    source: str
    target: str


class LLMContext(_S):
    summary: str
    glossary: list[LLMGloss]
    speakers: list[LLMSpeaker]
    relations: list[LLMRelation]
    audience: LLMAudience
    mentioned: list[LLMMentioned]
    needs_confirmation: list[str]


class LLMLine(_S):
    unit_id: str
    listener: str | None
    self_term: str | None
    other_term: str | None
    target: str
    short: str
    long: str
    note: str | None


class LLMTranslation(_S):
    lines: list[LLMLine]


class LLMRewrite(_S):
    text: str


class LLMBoundaries(_S):
    ends: list[int]


def inline_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema with every $ref inlined (no $defs), for the --json-schema flag."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


# ---------------------------------------------------------------- prompt payloads


def _clock(ms: int) -> str:
    m, s = divmod(ms / 1000, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _line(u: Unit) -> str:
    return f"[{_clock(u.start_ms)}] {u.speaker}: {u.text}"


def context_prompt(req: ContextReq) -> str:
    voices = [
        f"- {v.speaker}: voice sounds {v.gender} (confidence {v.gender_conf:.2f}), "
        f"age band {v.age_band}"
        for v in req.voices
    ]
    return "\n".join(
        [
            f"Source language: {req.src_lang}",
            f"Target language: {req.tgt_lang}",
            "",
            "Voice hints from an audio model (weak evidence):",
            *(voices or ["- none"]),
            "",
            "Transcript:",
            *(_line(u) for u in req.units),
        ]
    )


def translate_prompt(req: TranslateReq) -> str:
    unit_name = {"syllable": "syllables", "char": "characters", "word": "words"}[req.speech_unit]
    payload = {
        "source_language": req.src_lang,
        "target_language": req.tgt_lang,
        "length_unit": unit_name,
        "summary": req.context.summary,
        "glossary": [g.model_dump() for g in req.context.glossary],
        "cast": [s.model_dump(exclude={"locked"}) for s in req.context.cast.speakers],
        "relations": [
            {
                "speaker": r.speaker,
                "listener": r.listener,
                "relation": r.relation,
                "formality": r.formality,
                "since": _clock(r.since_ms),
                **(r.address[req.tgt_lang].model_dump() if req.tgt_lang in r.address else {}),
            }
            for r in req.context.cast.relations
        ],
        "audience": {
            "formality": req.context.cast.audience.formality,
            "address_term": req.context.cast.audience.address.get(req.tgt_lang),
        },
        "mentioned": [m.model_dump() for m in req.context.cast.mentioned],
        "before": [_line(u) for u in req.before],
        "lines": [
            {
                "unit_id": u.id,
                "time": _clock(u.start_ms),
                "speaker": u.speaker,
                "seconds": round(u.slot_ms / 1000, 2),
                "target_units": max(1, round(u.slot_ms / 1000 * req.units_per_s)),
                "text": u.text,
                **({"fix": req.feedback[u.id]} if u.id in req.feedback else {}),
            }
            for u in req.units
        ],
        "after": [_line(u) for u in req.after],
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


def rewrite_prompt(req: RewriteReq) -> str:
    return json.dumps(req.model_dump(), ensure_ascii=False, indent=1)


def boundary_windows(n: int, core: int, context: int) -> list[tuple[int, int, int, int]]:
    """(lo, a, b, hi): decide for words a..b-1, show lo..hi-1 (context on both sides)."""
    out = []
    for a in range(0, n, core):
        b = min(n, a + core)
        out.append((max(0, a - context), a, b, min(n, b + context)))
    return out


def boundaries_prompt(req: BoundaryReq, lo: int, a: int, b: int, hi: int) -> str:
    def tok(i: int) -> str:
        p = req.pauses_ms[i] if i < len(req.pauses_ms) else 0
        pause = f" <pause {p / 1000:.1f} s>" if p >= 500 else ""
        return f"{req.words[i].strip()}{{{i}}}{pause}"

    def run(i: int, j: int) -> str:
        return " ".join(tok(k) for k in range(i, j))

    parts = [f"Language: {req.lang}"]
    if lo < a:
        parts.append("Context before:\n" + run(lo, a))
    parts.append("[DECIDE]\n" + run(a, b) + "\n[/DECIDE]")
    if b < hi:
        parts.append("Context after:\n" + run(b, hi))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- conversions


def to_context_sheet(out: LLMContext, tgt: str) -> ContextSheet:
    return ContextSheet(
        summary=out.summary,
        glossary=[GlossaryEntry(source=g.source, target=g.target) for g in out.glossary],
        cast=CastSheet(
            speakers=[
                CastSpeaker(
                    id=s.id,
                    name=s.name,
                    gender=s.gender,
                    conf=s.gender_confidence,
                    age_band=s.age_band,
                    role=s.role,
                    style=s.style,
                    evidence=s.evidence,
                )
                for s in out.speakers
            ],
            relations=[
                Relation(
                    speaker=r.speaker,
                    listener=r.listener,
                    relation=r.relation,
                    formality=r.formality,
                    since_ms=r.since_ms,
                    conf=r.confidence,
                    evidence=r.evidence,
                    address={tgt: AddressTerms(self_term=r.self_term, other_term=r.other_term)},
                )
                for r in out.relations
            ],
            audience=Audience(
                formality=out.audience.formality,
                address={tgt: out.audience.address_term} if out.audience.address_term else {},
            ),
            mentioned=[Mentioned(name=m.name, gender=m.gender, note=m.note) for m in out.mentioned],
        ),
        flags=out.needs_confirmation,
    )


def to_batch(out: LLMTranslation, req: TranslateReq) -> TranslationBatch:
    wanted = {u.id for u in req.units}
    items = [
        Translation(
            unit_id=ln.unit_id,
            short=ln.short,
            target=ln.target,
            long=ln.long,
            listener=ln.listener,
            self_term=ln.self_term,
            other_term=ln.other_term,
            note=ln.note,
        )
        for ln in out.lines
        if ln.unit_id in wanted
    ]
    return TranslationBatch(items=items)


# ---------------------------------------------------------------- the provider

Runner = Callable[[list[str], str, dict[str, str], float], subprocess.CompletedProcess]

_USAGE = re.compile(r"usage limit|limit reached|rate.?limit|\b429\b|quota|overloaded", re.I)
_AUTH = re.compile(r"not logged in|please run /login|invalid api key|authentication|401", re.I)


def default_runner(cmd: list[str], stdin: str, env: dict[str, str], timeout: float):
    return subprocess.run(
        cmd,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
    )


class ClaudeCLI:
    runner: Runner = staticmethod(default_runner)

    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest
        self._warned = False

    # -- public entry point
    def run(self, op: str, req: BaseModel, ctx: RunCtx) -> BaseModel:
        system = (ctx.prompts_dir / self.manifest.prompt_files[op]).read_text(encoding="utf-8")
        if op == "mt.context":
            out = self.invoke(system, context_prompt(req), LLMContext, ctx)
            return to_context_sheet(out, req.tgt_lang)
        if op == "mt.translate":
            out = self.invoke(system, translate_prompt(req), LLMTranslation, ctx)
            return to_batch(out, req)
        if op == "mt.rewrite":
            out = self.invoke(system, rewrite_prompt(req), LLMRewrite, ctx)
            return RewriteRes(text=out.text)
        if op == "text.boundaries":
            return self.boundaries(system, req, ctx)
        raise ProviderError(f"claude_cli does not implement {op}", kind="unsupported")

    def boundaries(self, system: str, req: BoundaryReq, ctx: RunCtx) -> Boundaries:
        """Sentence ends chosen by the LLM, as boundary probabilities (0.95 / 0.05). Long
        transcripts go in windows (core words decided, context shown), several at a time."""
        from concurrent.futures import ThreadPoolExecutor

        n = len(req.words)
        if n == 0:
            return Boundaries(probs=[])
        p = ctx.params
        wins = boundary_windows(n, int(p.get("window_words", 400)), int(p.get("context_words", 40)))

        def one(w: tuple[int, int, int, int]) -> list[int]:
            lo, a, b, hi = w
            out = self.invoke(system, boundaries_prompt(req, lo, a, b, hi), LLMBoundaries, ctx)
            return [i for i in out.ends if a <= i < b]

        with ThreadPoolExecutor(max_workers=int(p.get("parallel_windows", 4))) as pool:
            ends = {i for found in pool.map(one, wins) for i in found}
        ends.add(n - 1)
        return Boundaries(probs=[0.95 if i in ends else 0.05 for i in range(n)])

    # -- one CLI call
    def build_cmd(self, system: str, schema: dict[str, Any], params: dict[str, Any]) -> list[str]:
        exe = shutil.which("claude") or "claude"
        return [
            exe,
            "-p",
            "--model",
            str(params.get("model", "sonnet")),
            "--max-turns",
            str(params.get("max_turns", 2)),
            "--output-format",
            "json",
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--system-prompt",
            system,
            "--json-schema",
            json.dumps(schema, separators=(",", ":")),
        ]

    def child_env(self, ctx: RunCtx) -> dict[str, str]:
        env = dict(os.environ)
        if env.pop("ANTHROPIC_API_KEY", None) and not self._warned:
            ctx.log("[claude_cli] ignoring ANTHROPIC_API_KEY so the subscription is used")
            self._warned = True
        return env

    def invoke(self, system: str, prompt: str, schema_model: type[BaseModel], ctx: RunCtx):
        cmd = self.build_cmd(system, inline_schema(schema_model), ctx.params)
        timeout = float(ctx.params.get("timeout_s", 600))
        try:
            p = type(self).runner(cmd, prompt, self.child_env(ctx), timeout)
        except subprocess.TimeoutExpired as e:
            raise ProviderError(
                f"claude timed out after {timeout:.0f}s", kind="timeout", retryable=True
            ) from e
        except FileNotFoundError as e:
            raise ProviderError("the 'claude' CLI is not installed", kind="unsupported") from e

        text = (p.stdout or "") + "\n" + (p.stderr or "")
        data: dict[str, Any] = {}
        try:
            data = json.loads(p.stdout) if p.stdout.strip() else {}
        except json.JSONDecodeError:
            data = {}
        if p.returncode != 0 or data.get("is_error") or data.get("subtype", "success") != "success":
            detail = str(data.get("result") or text).strip()[-800:]
            if _USAGE.search(detail):
                raise ProviderError(f"usage limit: {detail}", kind="usage_limit", retryable=True)
            if _AUTH.search(detail):
                raise ProviderError(
                    f"not logged in ({detail}). Run `claude` once to log in, or on a pod set "
                    "CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`.",
                    kind="auth",
                )
            raise ProviderError(f"claude failed (exit {p.returncode}): {detail}", kind="crash")

        raw = data.get("structured_output")
        if raw is None and isinstance(data.get("result"), str):
            try:
                raw = json.loads(data["result"])
            except json.JSONDecodeError:
                raw = None
        if raw is None:
            raise ProviderError(
                "claude returned no structured output", kind="bad_output", retryable=True
            )
        try:
            return schema_model.model_validate(raw)
        except ValidationError as e:
            raise ProviderError(
                f"structured output failed validation: {e}", kind="bad_output", retryable=True
            ) from e
