"""SHA-256 manifest for data/raw/. Raw pulls are write-once; the offline pipeline refuses
to run if a raw file no longer matches the checksum it was recorded with."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from .config import ROOT, Paths


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_manifest(manifest: Path | None = None) -> dict[str, str]:
    manifest = manifest or Paths().manifest
    out: dict[str, str] = {}
    if not manifest.exists():
        return out
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        out[rel.strip()] = digest
    return out


def write_atomic(path: Path, data: bytes) -> None:
    """temp file + rename, so a crash never leaves a half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def record(paths_to_add: list[Path], manifest: Path | None = None) -> None:
    """Add or update entries for the given raw files."""
    manifest = manifest or Paths().manifest
    entries = read_manifest(manifest)
    for p in paths_to_add:
        entries[_rel(p)] = sha256(p)
    body = "# sha256  path (relative to repo root). Written by a3bx3.manifest.record.\n"
    body += "".join(f"{d}  {r}\n" for r, d in sorted(entries.items()))
    write_atomic(manifest, body.encode("utf-8"))


def verify(manifest: Path | None = None) -> list[str]:
    """Return a list of problems (empty = every recorded raw file matches)."""
    problems = []
    entries = read_manifest(manifest)
    if not entries:
        return ["manifest is empty or missing"]
    for rel, digest in entries.items():
        p = ROOT / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif sha256(p) != digest:
            problems.append(f"checksum mismatch: {rel}")
    return problems
