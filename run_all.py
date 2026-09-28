"""Cross-platform stand-in for the Makefile (Windows boxes often have no `make`).

    python run_all.py               # = make all   (offline: data -> features -> train -> evaluate -> report)
    python run_all.py test          # = make test
    python run_all.py fetch         # = make fetch (network; MP needs MP_API_KEY)
    python run_all.py data features # any subset, run in the order given

Steps: fetch | data | features | train | evaluate | report | uncertainty | explain | test | all
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from a3bx3 import utf8_stdio  # noqa: E402

OFFLINE = ["data", "features", "train", "evaluate", "report", "uncertainty", "explain"]


def step(name: str) -> int:
    if name == "fetch":
        from a3bx3.fetch import jarvis, mp, oqmd
        rc = 0
        for mod in (oqmd, jarvis, mp):
            r = mod.main([])
            print(f"fetch {mod.__name__.rsplit('.', 1)[-1]}: exit {r}")
            rc = rc or r
        return rc
    if name == "data":
        from a3bx3 import curate
        return curate.main([])
    if name == "features":
        from a3bx3 import features
        return features.main([])
    if name in ("train", "evaluate", "report"):
        from a3bx3 import evaluate
        return evaluate.main([name])
    if name == "uncertainty":
        from a3bx3 import uncertainty
        return uncertainty.main([])
    if name == "explain":
        from a3bx3 import explain
        return explain.main([])
    if name == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT)
    raise SystemExit(f"unknown step {name!r}; choose from fetch data features train evaluate report uncertainty explain test all")


def main(argv: list[str]) -> int:
    utf8_stdio()
    steps = argv or ["all"]
    expanded = [s for a in steps for s in (OFFLINE if a == "all" else [a])]
    for s in expanded:
        print(f"== {s} ==", flush=True)
        rc = step(s)
        if rc:
            print(f"step {s} failed with exit {rc}", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
