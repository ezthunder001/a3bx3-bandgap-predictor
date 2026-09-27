"""Materials Project pull (needs network + MP_API_KEY). Run with `make fetch` or
`python -m a3bx3.fetch.mp`.

What it writes (never run without a key; nothing is faked):
    data/raw/mp_a3bx3.json    the A3BX3 entries (anonymous formula AB3C3 in MP's
                              count-sorted convention) with one pnictogen + one halogen
    data/raw/mp_context.json  every ternary containing a pnictogen and a halogen
                              (the ~800-row T3 context set; never pooled into family metrics)
Both files carry the MP database version and the query that produced them, and are
added to data/raw/MANIFEST.sha256.

Exit codes: 0 ok · 2 MP_API_KEY missing · 3 mp-api not installed · 4 query failed.
An outage exits non-zero; it is never written out as an empty result.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .. import utf8_stdio
from ..config import Paths, load_config
from ..manifest import record, write_atomic

EXIT_NO_KEY = 2
EXIT_NO_CLIENT = 3
EXIT_QUERY_FAILED = 4

FIELDS = ["material_id", "formula_pretty", "elements", "nelements", "band_gap",
          "is_gap_direct", "energy_above_hull", "formation_energy_per_atom",
          "is_stable", "symmetry", "volume", "density", "nsites", "theoretical"]


class MissingKeyError(RuntimeError):
    pass


def get_api_key(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    key = (env.get("MP_API_KEY") or "").strip()
    if not key:
        raise MissingKeyError(
            "MP_API_KEY is not set. Create a free key at https://next-gen.materialsproject.org/api, "
            "put it in this repo's git-ignored .env (see .env.example) or export it, then rerun "
            "`make fetch`. The offline pipeline (`make all`) does not need it.")
    return key


def _load_dotenv() -> None:
    """Read MP_API_KEY from the repo's own .env if present (never from any other .env)."""
    from ..config import ROOT
    p = ROOT / ".env"
    if not p.exists() or os.environ.get("MP_API_KEY"):
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("MP_API_KEY="):
            os.environ["MP_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def is_family_member(elements: list[str], pnictogens: set[str], halogens: set[str]) -> bool:
    els = set(elements)
    return len(els) == 3 and len(els & pnictogens) == 1 and len(els & halogens) == 1


def _doc_to_dict(doc) -> dict:
    d = doc.model_dump() if hasattr(doc, "model_dump") else dict(doc)
    out = {}
    for k in FIELDS:
        v = d.get(k)
        if k == "elements" and v is not None:
            v = [str(e) for e in v]
        elif k == "symmetry" and v is not None:
            v = {"crystal_system": str(v.get("crystal_system")),
                 "symbol": v.get("symbol"), "number": v.get("number")}
        elif k == "material_id" and v is not None:
            v = str(v)
        out[k] = v
    return out


def fetch(out_dir: Path | None = None) -> dict:
    cfg = load_config()["data"]["mp"]
    pn, hal = set(cfg["pnictogens"]), set(cfg["halogens"])
    key = get_api_key()
    try:
        from mp_api.client import MPRester
    except ImportError as e:  # pragma: no cover - environment problem
        raise ImportError("mp-api is not installed: pip install mp-api") from e

    out_dir = out_dir or Paths().raw
    with MPRester(key) as mpr:
        db_version = mpr.get_database_version()
        fam_docs = mpr.materials.summary.search(formula=cfg["anonymous_formula"], fields=FIELDS)
        family = sorted((_doc_to_dict(d) for d in fam_docs
                         if is_family_member([str(e) for e in d.elements], pn, hal)),
                        key=lambda r: r["material_id"])
        context: dict[str, dict] = {}
        for p in sorted(pn):
            for h in sorted(hal):
                for d in mpr.materials.summary.search(elements=[p, h], num_elements=3,
                                                      fields=FIELDS):
                    dd = _doc_to_dict(d)
                    context[dd["material_id"]] = dd
    ctx = [context[k] for k in sorted(context)]

    written = []
    for name, rows, query in (
            ("mp_a3bx3.json", family,
             {"formula": cfg["anonymous_formula"], "post_filter": "1 pnictogen + 1 halogen"}),
            ("mp_context.json", ctx,
             {"elements": "each (pnictogen, halogen) pair", "num_elements": 3})):
        payload = {"source": "Materials Project", "doi": "10.1063/1.4812323",
                   "database_version": db_version, "query": query,
                   "pnictogens": sorted(pn), "halogens": sorted(hal),
                   "n": len(rows), "records": rows}
        p = Path(out_dir) / name
        write_atomic(p, (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8"))
        written.append(p)
    record(written)
    return {"database_version": db_version, "n_family": len(family), "n_context": len(ctx)}


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    _load_dotenv()
    try:
        info = fetch()
    except MissingKeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_NO_KEY
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_NO_CLIENT
    except Exception as e:  # network / API failure: report, never write an empty result
        print(f"ERROR: Materials Project query failed: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_QUERY_FAILED
    print(f"MP database {info['database_version']}: {info['n_family']} A3BX3 entries, "
          f"{info['n_context']} context entries. Record db_version in configs/v2.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
