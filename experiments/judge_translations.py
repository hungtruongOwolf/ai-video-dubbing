"""Which run translated better? A blind ranking of the dubbed lines of several jobs.

The same video dubbed by runs that differ in one thing (the LLM model, a prompt). Lines whose
source text is the same in every run are compared: a judge LLM sees the scene (the cast sheet
of the first run, the two source lines before), the source line and the voiced Vietnamese
lines in a shuffled order (A, B, C...), and ranks them on meaning, natural speech and forms
of address. Several judges, to see a judge's preference for its own model's wording.

Laptop (needs the `claude` CLI logged in):
    uv run python experiments/judge_translations.py NAME=JOB_DIR NAME=JOB_DIR ... \\
        --judges opus,fable --out results.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BATCH = 20
SYSTEM = """You judge the Vietnamese dubbing of a Chinese video. For every item you get the
scene, the source line, who says it to whom, and several Vietnamese lines (A, B, C...) that
were voiced for it. Rank them, best first, on:
1. meaning: does it say what the source line says, in this scene (no added or lost facts,
   no literal translation of an idiom or of a dialect word)?
2. natural spoken Vietnamese: would a Vietnamese actor say it like that (no repeated or
   missing words, no clipped phrase)?
3. forms of address: the right pronouns and kinship terms between these two people.
Length differences do not matter by themselves. Rank every label of every item."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "ranking": {"type": "array", "items": {"type": "string"}},
                    "worst_problem": {"type": "string"},
                },
                "required": ["id", "ranking", "worst_problem"],
            },
        }
    },
    "required": ["items"],
}


def load(job: Path) -> tuple[list[dict], dict]:
    units = json.loads((job / "units.json").read_text(encoding="utf-8"))
    cast = json.loads((job / "cast.json").read_text(encoding="utf-8"))
    return units, cast


def aligned(runs: dict[str, list[dict]]) -> list[dict]:
    """Lines with the same source text (and about the same time) in every run, dubbed in all."""
    names = list(runs)
    first = runs[names[0]]
    out = []
    for i, x in enumerate(first):
        u = x["unit"]
        row = {"id": u["id"], "src": u["text"], "start_ms": u["start_ms"], "i": i,
               "speaker": u["speaker"], "lines": {}}  # fmt: skip
        for name in names:
            match = next(
                (y for y in runs[name] if y["unit"]["text"] == u["text"]
                 and abs(y["unit"]["start_ms"] - u["start_ms"]) < 1500),
                None,
            )  # fmt: skip
            if match is None or match["placement"]["variant"] == "original":
                break
            row["lines"][name] = match["placement"]["text"]
        else:
            out.append(row)
    return out


def scene(cast: dict) -> str:
    who = "; ".join(
        f"{s['id']}: {s.get('role') or s.get('name') or '?'} ({s.get('gender', '?')})"
        for s in cast.get("cast", {}).get("speakers", [])
    )
    return f"Summary: {cast.get('summary', '')}\nPeople: {who}"


def prompt_for(batch: list[dict], first: list[dict], cast: dict, rng_seed: str):
    items, keys = [], {}
    for row in batch:
        names = list(row["lines"])
        order = names[:]
        random.Random(f"{rng_seed}:{row['id']}").shuffle(order)
        labels = [chr(ord("A") + k) for k in range(len(order))]
        keys[row["id"]] = dict(zip(labels, order, strict=True))
        before = " / ".join(first[j]["unit"]["text"] for j in range(max(0, row["i"] - 2), row["i"]))
        cand = "\n".join(
            f"  {lab}: {row['lines'][n]}" for lab, n in zip(labels, order, strict=True)
        )
        items.append(f"id {row['id']} (speaker {row['speaker']})\n  before: {before}\n"
                     f"  source: {row['src']}\n{cand}")  # fmt: skip
    return scene(cast) + "\n\n" + "\n\n".join(items), keys


def call(model: str, prompt: str) -> dict:
    exe = shutil.which("claude") or "claude"
    cmd = [exe, "-p", "--model", model, "--max-turns", "2", "--output-format", "json",
           "--tools", "", "--strict-mcp-config", "--no-session-persistence",
           "--system-prompt", SYSTEM, "--json-schema", json.dumps(SCHEMA)]  # fmt: skip
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                       timeout=900)  # fmt: skip
    data = json.loads(p.stdout)
    raw = data.get("structured_output")
    if raw is None:
        raw = json.loads(data["result"])
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=JOB_DIR")
    ap.add_argument("--judges", default="opus,fable")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    jobs = dict(r.split("=", 1) for r in a.runs)
    loaded = {n: load(Path(p)) for n, p in jobs.items()}
    runs = {n: v[0] for n, v in loaded.items()}
    first_name = next(iter(runs))
    rows = aligned(runs)
    differ = [r for r in rows if len(set(r["lines"].values())) > 1]
    print(f"{len(rows)} lines in every run, {len(differ)} worded differently", flush=True)
    batches = [differ[k : k + BATCH] for k in range(0, len(differ), BATCH)]
    tasks = [(j, b) for j in a.judges.split(",") for b in range(len(batches))]

    def work(task):
        judge, b = task
        prompt, keys = prompt_for(batches[b], runs[first_name], loaded[first_name][1], judge)
        res = call(judge, prompt)
        out = []
        for it in res.get("items", []):
            key = keys.get(it["id"])
            if not key or sorted(it["ranking"]) != sorted(key):
                continue
            out.append({"judge": judge, "id": it["id"],
                        "ranking": [key[lab] for lab in it["ranking"]],
                        "problem": it.get("worst_problem", "")})  # fmt: skip
        return out

    with ThreadPoolExecutor(max_workers=6) as pool:
        verdicts = [v for vs in pool.map(work, tasks) for v in vs]
    report: dict = {"lines": len(rows), "differ": len(differ), "judges": {}}
    for judge in a.judges.split(","):
        vs = [v for v in verdicts if v["judge"] == judge]
        first = Counter(v["ranking"][0] for v in vs)
        rank = defaultdict(list)
        for v in vs:
            for k, n in enumerate(v["ranking"]):
                rank[n].append(k + 1)
        pair = {}
        names = list(runs)
        for x in names:
            for y in names:
                if x < y:
                    xw = sum(v["ranking"].index(x) < v["ranking"].index(y) for v in vs)
                    pair[f"{x}>{y}"] = round(xw / max(1, len(vs)), 3)
        report["judges"][judge] = {
            "judged": len(vs),
            "best": dict(first),
            "mean_rank": {n: round(sum(r) / len(r), 2) for n, r in rank.items()},
            "pairwise": pair,
        }
        print(judge, json.dumps(report["judges"][judge]), flush=True)
    if a.out:
        a.out.write_text(json.dumps({**report, "verdicts": verdicts}, ensure_ascii=False,
                                    indent=1), encoding="utf-8")  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
