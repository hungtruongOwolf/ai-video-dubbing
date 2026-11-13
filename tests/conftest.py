from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dub.core.config import Config, load_config
from dub.store import Workspace

REPO = Path(__file__).resolve().parents[1]

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(REPO / "configs")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace.open(tmp_path / "ws")
