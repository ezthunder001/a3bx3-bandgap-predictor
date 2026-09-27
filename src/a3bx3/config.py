"""Repository paths and the v2 config loader. Every path is resolved from the repo root,
so the pipeline behaves the same from any working directory."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "v2.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Path | None = None) -> dict[str, Any]:
    return _load(str(path or CONFIG_PATH))


class Paths:
    """Output locations. `out_root` lets the determinism test rebuild into a temp dir
    while still reading the committed raw data."""

    def __init__(self, out_root: Path | None = None):
        cfg = load_config()
        self.raw = ROOT / cfg["data"]["raw_dir"]
        self.literature = ROOT / cfg["data"]["literature_dir"]
        self.manifest = ROOT / cfg["data"]["manifest"]
        base = Path(out_root) if out_root else ROOT
        self.interim = base / "data" / "interim"
        self.processed = base / "data" / "processed"
        self.reports = base / "reports"
        self.figures = self.reports / "figures"

    def ensure(self) -> "Paths":
        for p in (self.interim, self.processed, self.reports, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self
