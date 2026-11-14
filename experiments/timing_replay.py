"""Replay the timing fit of a finished job with other settings (no GPU, no TTS).

`fit_unit` is a pure function of the slot, the clip length, the predicted natural length and
the two neighbours, so a job's `units.json` is enough to ask "what if": every line keeps its
synthesised clip, only the placement and the stretch change. Reported per setting:
speed-up p50/p95 and max (natural length / final length), lines over the p95 limit, overflow
(even the cap could not fit), and how late the dub ends after the original line (p95).

Usage: uv run python experiments/timing_replay.py JOB_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dub.core.config import load_config  # noqa: E402
from dub.ops.timing import Neighbor, fit_unit  # noqa: E402

VARIANTS = {
    "late 500 (now)": {},
    "late 800": {"late_end_ms": 800},
    "late 1200": {"late_end_ms": 1200},
    "late 1200, early 300": {"late_end_ms": 1200, "early_start_ms": 300},
}


def replay(rows: list[dict], t) -> dict:
    rows = sorted(rows, key=lambda r: r["unit"]["start_ms"])
    rates, late, flags = [], [], {"fast": 0, "overflow": 0}

    def nb(u, o, gap) -> Neighbor:
        return Neighbor(
            natural_ms=o["placement"]["natural_ms"],
            slot_ms=o["unit"]["end_ms"] - o["unit"]["start_ms"],
            gap_ms=max(0, gap),
            same_speaker=o["unit"]["speaker"] == u["speaker"],
        )

    for i, r in enumerate(rows):
        u, p = r["unit"], r["placement"]
        if not p:
            continue
        prev = nb(u, rows[i - 1], u["start_ms"] - rows[i - 1]["unit"]["end_ms"]) if i else None
        nxt = (nb(u, rows[i + 1], rows[i + 1]["unit"]["start_ms"] - u["end_ms"])
               if i + 1 < len(rows) else None)  # fmt: skip
        f = fit_unit(slot_start=u["start_ms"], slot_end=u["end_ms"], natural_ms=p["natural_ms"],
                     clip_ms=p["clip_ms"], prev=prev, next_=nxt, t=t)  # fmt: skip
        rates.append(f.rate)
        late.append(max(0, f.end_ms - u["end_ms"]))
        for k in flags:
            flags[k] += k in f.flags
    rates_a, late_a = np.array(rates), np.array(late)
    return {
        "lines": len(rates),
        "speedup_p50": round(float(np.percentile(rates_a, 50)), 3),
        "speedup_p95": round(float(np.percentile(rates_a, 95)), 3),
        "speedup_max": round(float(rates_a.max()), 3),
        "fast": flags["fast"],
        "overflow": flags["overflow"],
        "late_end_ms_p95": int(np.percentile(late_a, 95)),
        "late_end_ms_max": int(late_a.max()),
    }


def main() -> None:
    job = Path(sys.argv[1])
    rows = json.loads((job / "units.json").read_text(encoding="utf-8"))
    base = load_config().dub.timing
    out = {}
    for name, change in VARIANTS.items():
        out[name] = replay(rows, base.model_copy(update=change))
        print(f"{name:<24} {out[name]}")
    (job / "timing_replay.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
