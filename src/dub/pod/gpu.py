"""GPU utilisation samples while a job runs (milestone M5: "the GPU is busy >= 70% of the time").

`nvidia-smi` reports, every `interval_ms`, the share of the last interval during which a kernel
was running (utilization.gpu) and the memory in use. The samples are written next to the job
(`gpu.csv`) and lined up with the trace by wall clock (`dub bench speed`).

Without nvidia-smi (laptop, CI) the sampler records nothing and never fails a job.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path


def total_memory_mib() -> int | None:
    """Memory of the first GPU, or None without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout  # fmt: skip
        return int(float(out.splitlines()[0]))
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None


class GpuSampler:
    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, float, float]] = []  # unix time, util %, memory MiB
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> GpuSampler:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return self
        self._proc = subprocess.Popen(
            [exe, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
             f"--loop-ms={self.interval_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )  # fmt: skip
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            try:
                util, mem = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue
            self.samples.append((time.time(), util, mem))

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}
        utils = [u for _, u, _ in self.samples]
        return {
            "gpu_util_mean": round(sum(utils) / len(utils), 1),
            "gpu_mem_max_mib": max(m for _, _, m in self.samples),
            "gpu_samples": len(self.samples),
        }

    def write(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("unix_s,util_pct,mem_mib\n")
            for t, u, m in self.samples:
                f.write(f"{t:.3f},{u:.0f},{m:.0f}\n")
