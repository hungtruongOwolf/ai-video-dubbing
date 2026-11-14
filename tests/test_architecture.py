"""Fitness tests for the architecture rules."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from dub.contracts.v1 import ALL_MODELS
from dub.core.config import Langpack

SRC = Path(__file__).resolve().parents[1] / "src" / "dub"
CORE_PACKAGES = ["core", "store", "contracts", "ops"]


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_core_never_imports_providers():
    """Adding or swapping a model must never require touching the core."""
    offenders = []
    for pkg in CORE_PACKAGES:
        for f in (SRC / pkg).rglob("*.py"):
            bad = [m for m in imports_of(f) if m.startswith("dub.providers")]
            if bad:
                offenders.append(f"{f.relative_to(SRC)} imports {bad}")
    assert not offenders, offenders


def test_every_manifest_points_to_a_real_class(cfg):
    for m in cfg.providers.values():
        if m.runtime == "inproc":
            module, _, cls = m.impl.partition(":")
            assert hasattr(importlib.import_module(module), cls), m.id
        for name in m.prompt_files.values():
            assert (cfg.prompts_dir / name).exists(), name


def test_every_fake_op_is_covered(cfg):
    from dub.core.ctx import OP_RESULT

    for op in OP_RESULT:
        assert cfg.fake_provider_for(op), f"no fake provider for {op}"


def test_langpacks_are_complete(cfg):
    for lang in ("en", "vi", "zh", "ja", "th", "es"):
        lp = cfg.langpack(lang)
        assert isinstance(lp, Langpack) and lp.lang == lang and lp.units_per_s > 0
    assert cfg.langpack("xx").lang == "xx"  # unknown languages fall back to "default"


def test_contracts_export_json_schema():
    for model in ALL_MODELS:
        assert model.model_json_schema()["type"] == "object"


def test_every_model_op_the_pipeline_calls_is_registered():
    """cx.call looks the result type up in contracts OPS: an op missing there fails at run
    time only (text.speakers did, on the pod)."""
    import re
    from pathlib import Path

    from dub.contracts.v1 import OPS
    from dub.core.config import load_config

    src = (Path(__file__).parents[1] / "src/dub/core/pipeline.py").read_text(encoding="utf-8")
    called = set(re.findall(r'cx\.call\(\s*"([a-z_]+\.[a-z_]+)"', src))
    assert "text.speakers" in called
    assert called <= set(OPS), called - set(OPS)
    cfg = load_config()
    for m in cfg.providers.values():  # every op a manifest declares has a contract
        assert set(m.op) <= set(OPS) | {"asr.align"}, (m.id, set(m.op) - set(OPS))
