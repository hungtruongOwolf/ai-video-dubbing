"""CAS (content-addressed store): files named by the hash of their content.

- A key looks like `<hash>.<ext>`, e.g. `9b1c...e2.wav`. Same content => same key.
- Files live at `cas/<first 2 chars>/<key>` so no directory gets too large.
- Writes are atomic: write to a temp file, then rename. If the process dies midway there is
  never a half-written file under a real key.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from dub.core.keys import hash_bytes, hash_file


class CAS:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._tmp = self.root / "tmp"
        self._tmp.mkdir(exist_ok=True)

    # ------------------------------------------------------------ paths
    def path(self, key: str) -> Path:
        if not key or "/" in key or "\\" in key or key.startswith("."):
            raise ValueError(f"Invalid CAS key: {key!r}")
        return self.root / key[:2] / key

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def tmp_path(self, suffix: str) -> Path:
        """A temp path on the same disk as the CAS. Write output there, then call put_file."""
        fd, name = tempfile.mkstemp(suffix=suffix, dir=self._tmp)
        os.close(fd)
        return Path(name)

    # ------------------------------------------------------------ write
    def put_bytes(self, data: bytes, ext: str) -> str:
        key = f"{hash_bytes(data)}{_dot(ext)}"
        dest = self.path(key)
        if not dest.exists():
            tmp = self.tmp_path(_dot(ext))
            tmp.write_bytes(data)
            self._commit(tmp, dest)
        return key

    def put_text(self, text: str, ext: str = ".txt") -> str:
        return self.put_bytes(text.encode("utf-8"), ext)

    def put_file(self, src: Path, ext: str | None = None, move: bool = False) -> str:
        src = Path(src)
        ext = _dot(ext if ext is not None else src.suffix)
        key = f"{hash_file(src)}{ext}"
        dest = self.path(key)
        if dest.exists():
            if move:
                src.unlink(missing_ok=True)
            return key
        if move and _same_drive(src, self.root):
            self._commit(src, dest)
        else:
            tmp = self.tmp_path(ext)
            shutil.copyfile(src, tmp)
            self._commit(tmp, dest)
            if move:
                src.unlink(missing_ok=True)
        return key

    def _commit(self, tmp: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(tmp, dest)
        except PermissionError:
            # Windows: the destination may be open for reading by another process. The
            # content is identical (same hash), so keep the existing file and drop ours.
            if dest.exists():
                tmp.unlink(missing_ok=True)
            else:
                raise

    # ------------------------------------------------------------ read
    def read_bytes(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def read_text(self, key: str) -> str:
        return self.path(key).read_text(encoding="utf-8")


def _dot(ext: str) -> str:
    if not ext:
        return ""
    return ext if ext.startswith(".") else f".{ext}"


def _same_drive(a: Path, b: Path) -> bool:
    return Path(a).resolve().anchor.lower() == Path(b).resolve().anchor.lower()
