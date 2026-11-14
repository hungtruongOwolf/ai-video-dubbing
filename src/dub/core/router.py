"""Routing: pick the provider (model) for each call of an op.

Priority (first match wins):
  1. command-line overrides:  --use mt.*=claude_cli,   --fake
  2. rules in dub.yaml `routes:` (first match), evaluated on job/unit facts
  3. the chain in the langpack of the language (source or target, depending on the op)
  4. `defaults:` in dub.yaml
Every candidate then goes through the capability filter. Rejected candidates are recorded
with a reason. The result is a RouteTrace that explains "why this line used this model".
"""

from __future__ import annotations

import importlib.util
import shutil
from fnmatch import fnmatch
from typing import Any

from dub.contracts.v1 import Rejection, RouteTrace
from dub.core.config import CHAIN_FALLBACK, SOURCE_SIDE_OPS, Config, ProviderManifest, RouteRule
from dub.worker.protocol import worker_urls


class RoutingError(RuntimeError):
    pass


_OPS = {
    "lt": lambda v, x: v < x,
    "lte": lambda v, x: v <= x,
    "gt": lambda v, x: v > x,
    "gte": lambda v, x: v >= x,
    "eq": lambda v, x: v == x,
    "ne": lambda v, x: v != x,
    "in": lambda v, x: v in x,
}


def _cond_ok(cond: Any, value: Any) -> bool:
    if isinstance(cond, dict):
        for name, arg in cond.items():
            if name not in _OPS:
                raise ValueError(f"Unsupported operator in route rule: {name}")
            try:
                if not _OPS[name](value, arg):
                    return False
            except TypeError:
                return False
        return True
    if isinstance(cond, list):
        return value in cond
    return value == cond


def rule_matches(rule: RouteRule, facts: dict[str, Any]) -> bool:
    return all(k in facts and _cond_ok(c, facts[k]) for k, c in rule.when.items())


def unavailable_reason(m: ProviderManifest) -> str | None:
    """Can this provider run on this machine at all (ignoring op and language)?"""
    for b in m.requires_bin:
        if shutil.which(b) is None:
            return f"program '{b}' not found on PATH"
    if m.runtime == "inproc":
        if not m.impl:
            return "manifest has no 'impl'"
        module = m.impl.split(":")[0]
        try:
            found = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            found = False
        if not found:
            return f"module '{module}' is not installed"
    elif m.runtime == "http":
        if not m.env:
            return "http provider without 'env'"
        if m.env not in worker_urls():
            return f"worker '{m.env}' is not running here (GPU pod only: dub run --remote)"
    return None


class Router:
    def __init__(
        self,
        cfg: Config,
        *,
        profile: str | None = None,
        overrides: list[tuple[str, list[str]]] | None = None,
    ):
        self.cfg = cfg
        self.profile_name = profile or cfg.dub.profile
        self.profile = cfg.profile(self.profile_name)
        self.overrides = overrides or []
        self._avail: dict[str, str | None] = {}

    # ------------------------------------------------------------ filters
    def _available(self, pid: str) -> str | None:
        if pid not in self._avail:
            self._avail[pid] = unavailable_reason(self.cfg.providers[pid])
        return self._avail[pid]

    def reject_reason(
        self, pid: str, op: str, facts: dict[str, Any], require: dict[str, Any]
    ) -> str | None:
        m = self.cfg.providers.get(pid)
        if m is None:
            return "no manifest in configs/providers (not installed in this version)"
        if op not in m.op:
            return f"does not serve op {op}"
        if m.status == "candidate" and not facts.get("experiment"):
            return "status is candidate (experiments only)"
        lang = facts.get("src_lang" if op in SOURCE_SIDE_OPS else "tgt_lang")
        if m.caps.langs != "*" and lang not in (None, "auto") and lang not in m.caps.langs:
            return f"does not support language {lang}"
        slot = facts.get("unit.slot_ms")
        if m.caps.max_out_s and slot and slot > m.caps.max_out_s * 1000:
            return f"unit is {slot / 1000:.1f}s > max_out_s {m.caps.max_out_s}s"
        for k, want in require.items():
            have = getattr(m.caps, k, None)
            if have != want:
                return f"rule requires {k}={want}, provider has {have}"
        return self._available(pid)

    # ------------------------------------------------------------ main
    def route(self, op: str, facts: dict[str, Any], subject: str | None = None) -> RouteTrace:
        rejected: list[Rejection] = []
        require: dict[str, Any] = {}
        rule_id: str | None = None

        override = next((p for pat, p in self.overrides if fnmatch(op, pat)), None)
        if override is not None:
            source, candidates = "override", list(override)
            rule_id = "cli"
        else:
            rule = next(
                (r for r in self.cfg.dub.routes.get(op, []) if rule_matches(r, facts)), None
            )
            if rule:
                rule_id, require = rule.id, dict(rule.require)
            if rule and rule.use:
                source, candidates = "rule", list(rule.use)
            else:
                side = "src_lang" if op in SOURCE_SIDE_OPS else "tgt_lang"
                lp = self.cfg.langpack(str(facts.get(side, "default")))
                lp_chain = lp.chains.get(op) or lp.chains.get(CHAIN_FALLBACK.get(op, ""))
                if lp_chain:
                    source, candidates = "langpack", list(lp_chain)
                else:
                    source, candidates = "default", list(self._defaults(op))

        chain = self._filter(candidates, op, facts, require, rejected)

        # The langpack has a chain but nothing in it can run: try the defaults, and say so
        # in the trace.
        if not chain and source == "langpack":
            extra = [p for p in self._defaults(op) if p not in candidates]
            chain = self._filter(extra, op, facts, require, rejected)
            if chain:
                source = "default"

        if op == "tts.synthesize" and self.profile.tts_chain == "incumbent_only":
            chain = chain[:1]

        if not chain:
            why = "; ".join(f"{r.provider}: {r.reason}" for r in rejected) or "no candidates"
            hint = "" if override is not None else " (to try it without real models, add --fake)"
            raise RoutingError(f"No runnable provider for {op}: {why}{hint}")

        return RouteTrace(
            op=op, subject=subject, source=source, rule_id=rule_id, chain=chain, rejected=rejected
        )

    def _defaults(self, op: str) -> list[str]:
        d = self.cfg.dub.defaults
        return list(d.get(op) or d.get(CHAIN_FALLBACK.get(op, ""), []))

    def _filter(self, candidates, op, facts, require, rejected) -> list[str]:
        out = []
        for pid in candidates:
            why = self.reject_reason(pid, op, facts, require)
            if why:
                rejected.append(Rejection(provider=pid, reason=why))
            else:
                out.append(pid)
        return out


def parse_overrides(items: list[str]) -> list[tuple[str, list[str]]]:
    """'mt.*=claude_cli' or 'tts.synthesize=a,b' -> [(pattern, [providers])]."""
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"--use expects OP=PROVIDER, got {it!r}")
        pat, provs = it.split("=", 1)
        out.append((pat.strip(), [p.strip() for p in provs.split(",") if p.strip()]))
    return out
