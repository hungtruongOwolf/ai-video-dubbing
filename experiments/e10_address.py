"""Experiments E10 (forms of address) and E8 at line level, on a dubbed run of the drama.

Input: the job folder of `dub run drama.mp4 --to vi --from en --remote` and drama.json.
Every true line is matched to the dubbed units that overlap it in time.

- Speaker swaps: found speakers are mapped to the characters by majority; a line whose units
  carry another character's voice is a swap.
- Forms of address (vi): a line is correct when it contains no generic pronoun ("tôi", "bạn",
  "cậu"...) and every kinship word in it is one the speaker may use: the expected self/other
  terms for this listener, or how the speaker names the other characters (third person, e.g.
  the mother telling her son to help "ông"). Lines without any such word count as correct
  (Vietnamese often drops pronouns). The full table is written for a manual review.

Usage: uv run python experiments/e10_address.py JOB_DIR
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import data_dir, save_json  # noqa: E402

GENERIC = {"tôi", "tao", "tớ", "mình", "mày", "cậu", "bạn", "ta"}
KIN = {"ông", "bà", "bố", "ba", "cha", "mẹ", "má", "con", "cháu", "anh", "chị", "em", "cô",
       "chú", "bác", "dì"}  # fmt: skip


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"\w+", unicodedata.normalize("NFC", text))]


def main() -> None:
    job = Path(sys.argv[1])
    drama = json.loads((Path(__file__).parent / "drama.json").read_text(encoding="utf-8"))
    truth = json.loads((data_dir() / "drama" / "truth.json").read_text(encoding="utf-8"))
    rows = json.loads((job / "units.json").read_text(encoding="utf-8"))
    table = drama["address_vi"]

    # units overlapping each true line
    per_line = []
    for t in truth:
        us = [
            r for r in rows
            if min(r["unit"]["end_ms"], t["offset_ms"]) - max(r["unit"]["start_ms"], t["onset_ms"])
            > 0.3 * (r["unit"]["end_ms"] - r["unit"]["start_ms"])
        ]  # fmt: skip
        per_line.append(us)
    # map found speakers -> characters by majority over lines
    votes: dict[str, Counter] = {}
    for t, us in zip(truth, per_line, strict=True):
        for r in us:
            votes.setdefault(r["unit"]["speaker"], Counter())[t["speaker"]] += 1
    mapping = {spk: c.most_common(1)[0][0] for spk, c in votes.items()}

    out, swaps, addr_ok, n_addr = [], 0, 0, 0
    for t, us in zip(truth, per_line, strict=True):
        spk, to = t["speaker"], t["to"]
        voices = {mapping.get(r["unit"]["speaker"]) for r in us}
        swap = bool(us) and voices != {spk}
        swaps += swap
        text = " ".join(
            (r["placement"] or {}).get("text") or (r["translation"] or {}).get("target", "")
            for r in us
        )
        toks = _tokens(text)
        exp = table[f"{spk}>{to}"]
        allowed = set(exp["self"]) | set(exp["other"])
        for key, v in table.items():  # how this speaker names the other characters
            if key.startswith(f"{spk}>"):
                allowed |= set(v["other"])
        generic = sorted({w for w in toks if w in GENERIC})
        wrong = sorted({w for w in toks if w in KIN and w not in allowed})
        ok = not generic and not wrong
        n_addr += 1
        addr_ok += ok
        out.append({
            "line": t["line"], "speaker": spk, "to": to, "en": t["text"], "vi": text,
            "units": [r["unit"]["id"] for r in us],
            "voice_ok": not swap, "address_ok": ok, "generic": generic, "unexpected_kin": wrong,
            "declared": [((r["translation"] or {}).get("listener"),
                          (r["translation"] or {}).get("self_term"),
                          (r["translation"] or {}).get("other_term")) for r in us],
            "flag": [r.get("address_issue") for r in us if r.get("address_issue")],
        })  # fmt: skip
    res = {
        "lines": len(truth),
        "speaker_mapping": mapping,
        "speaker_swaps": swaps,
        "address_correct": round(addr_ok / max(n_addr, 1), 3),
        "rows": out,
    }
    save_json(data_dir() / "results" / f"e10_{job.name}.json", res)
    print(f"lines {len(truth)}, speaker swaps {swaps}, address correct {addr_ok}/{n_addr} "
          f"({res['address_correct']:.1%}), mapping {mapping}")  # fmt: skip
    for r in out:
        mark = ("" if r["voice_ok"] else " SWAP") + ("" if r["address_ok"] else " ADDR")
        print(f"{r['line']:2d} {r['speaker']:>7}>{r['to']:<7}{mark:<10} {r['vi'][:90]}")


if __name__ == "__main__":
    main()
