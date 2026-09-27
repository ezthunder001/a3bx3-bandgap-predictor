"""JARVIS-DFT pull (keyless, network). `python -m a3bx3.fetch.jarvis`.

Wraps the v1 build_a3bx3.py selection rule (3:1:3 with a pnictogen B, a halide X, no
f-electron A-site, OptB88vdW gap > 0.1 eV) and writes the raw records, with the
jarvis-tools dataset name and package version, to data/raw/jarvis_a3bx3.json.

JARVIS gaps are OptB88vdW (and TBmBJ), NOT PBE. They are never pooled into the PBE
target; they are kept for the delta model and for cross-source comparison only.
Exit codes: 0 ok · 3 jarvis-tools missing · 4 download/parse failed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from .. import utf8_stdio
from ..config import Paths, load_config
from ..manifest import record, write_atomic

HAL = {"F", "Cl", "Br", "I"}
PNB = {"P", "As", "Sb", "Bi"}
A_EXCL = {"Pr", "La", "Ce", "Nd", "Sm", "Eu", "Gd", "Yb"}   # same exclusion as v1


def parse(f: str) -> dict:
    out: dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        if el:
            out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def select(entry: dict) -> dict | None:
    """v1 selection rule, unchanged. Returns a raw record or None."""
    f = entry.get("formula")
    if not f:
        return None
    c = parse(f)
    if len(c) != 3:
        return None
    g = np.gcd.reduce(list(c.values()))
    norm = {k: v // g for k, v in c.items()}
    threes = [k for k, v in norm.items() if v == 3]
    ones = [k for k, v in norm.items() if v == 1]
    if len(threes) != 2 or len(ones) != 1:
        return None
    B = ones[0]
    Xs = [t for t in threes if t in HAL]
    As = [t for t in threes if t not in HAL]
    if B not in PNB or len(Xs) != 1 or len(As) != 1 or As[0] in A_EXCL:
        return None
    try:
        eg = float(entry.get("optb88vdw_bandgap"))
    except (TypeError, ValueError):
        return None
    if eg <= 0.10:
        return None
    keep = ["jid", "formula", "spg_number", "crys", "optb88vdw_bandgap", "mbj_bandgap",
            "ehull", "formation_energy_peratom", "density", "elastic_tensor"]
    rec = {k: entry.get(k) for k in keep}
    rec.update(A=As[0], B=B, X=Xs[0])
    rec["abc"] = (entry.get("atoms") or {}).get("abc")
    return rec


def fetch(out_dir: Path | None = None) -> dict:
    try:
        from jarvis.db.figshare import data, get_db_info
        import jarvis
    except ImportError as e:
        raise ImportError("jarvis-tools is not installed: pip install jarvis-tools") from e
    name = load_config()["data"]["jarvis"]["dataset"]
    info = get_db_info()[name]          # [url, file tag, message, reference]
    d = data(name)
    rows = sorted((r for r in (select(e) for e in d) if r), key=lambda r: str(r["jid"]))
    payload = {"source": "JARVIS-DFT", "doi": "10.1038/s41524-020-00440-1",
               "dataset": name, "dataset_file": info[1], "dataset_url": info[0],
               "jarvis_tools_version": getattr(jarvis, "__version__", None),
               "n_dataset": len(d), "functional": "OptB88vdW (optb88vdw_bandgap); TBmBJ (mbj_bandgap)",
               "n": len(rows), "records": rows}
    p = Path(out_dir or Paths().raw) / "jarvis_a3bx3.json"
    write_atomic(p, (json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n").encode("utf-8"))
    record([p])
    return {"dataset": name, "file": info[1], "n": len(rows), "n_dataset": len(d)}


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    try:
        info = fetch()
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"ERROR: JARVIS fetch failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 4
    print(f"JARVIS {info['dataset']} ({info['file']}): {info['n']} A3BX3 family rows of {info['n_dataset']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
