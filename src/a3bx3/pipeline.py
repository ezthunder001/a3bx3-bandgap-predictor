"""The offline pipeline as one call, for the determinism test and for scripting:
data -> features -> train -> evaluate (-> report). Reads committed data/raw/ only."""
from __future__ import annotations

from pathlib import Path

from . import curate, evaluate, features
from .config import Paths
from .manifest import verify


def run_offline(out_root: Path | None = None, with_report: bool = False, n_jobs: int = -1) -> Paths:
    problems = verify()
    if problems:
        raise RuntimeError("data/raw does not match MANIFEST.sha256: " + "; ".join(problems))
    paths = Paths(out_root).ensure()
    curate.curate(paths)
    features.build(paths)
    evaluate.train(paths, n_jobs=n_jobs)
    evaluate.evaluate(paths)
    if with_report:
        evaluate.report(paths)
    return paths
