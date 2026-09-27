"""OQMD pull through its OPTIMADE endpoint (keyless, network). `python -m a3bx3.fetch.oqmd`.

Query: anonymous formula A3B3C, three elements, one alkaline-earth, one pnictogen, one
halogen. Writes every returned structure (all pages) to data/raw/oqmd_a3bx3.json with the
provider's reported implementation version and the retrieval date.

OQMD gaps are PBE (qmpy), but they are NOT pooled into the family target in v2.0: they
are kept for cross-source comparison (curate.py writes data/interim/cross_source_oqmd.csv).
Note: the OQMD set contains Sr3BiBr3, a locked-holdout formula. Any future pooling must go
through curate's holdout guard.

Exit codes: 0 ok · 4 HTTP/network failure (an outage is never written as zero results).
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from .. import utf8_stdio
from ..config import Paths, load_config
from ..manifest import record, write_atomic

FILTER = ('chemical_formula_anonymous="A3B3C" AND nelements=3 '
          'AND elements HAS ANY "Mg","Ca","Sr","Ba" '
          'AND elements HAS ANY "P","As","Sb","Bi" '
          'AND elements HAS ANY "F","Cl","Br","I"')
KEEP = ["chemical_formula_reduced", "elements", "nsites", "space_group_it_number",
        "_oqmd_entry_id", "_oqmd_calculation_id", "_oqmd_icsd_id", "_oqmd_band_gap",
        "_oqmd_delta_e", "_oqmd_stability", "_oqmd_spacegroup", "_oqmd_volume",
        "lattice_vectors"]


def _get(url: str, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "a3bx3-bandgap-predictor/2"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch(out_dir: Path | None = None) -> dict:
    endpoint = load_config()["data"]["oqmd"]["endpoint"]
    url = endpoint + "?" + urllib.parse.urlencode({"filter": FILTER, "page_limit": 100})
    records, meta = [], None
    while url:
        page = _get(url)
        meta = meta or page.get("meta", {})
        for d in page.get("data") or []:
            a = d.get("attributes") or {}
            rec = {"id": d.get("id")}
            rec.update({k: a.get(k) for k in KEEP})
            records.append(rec)
        url = ((page.get("links") or {}).get("next")) or None
        if isinstance(url, dict):
            url = url.get("href")
    records.sort(key=lambda r: str(r["id"]))
    impl = (meta or {}).get("implementation") or {}
    payload = {"source": "OQMD via OPTIMADE", "doi": "10.1007/s11837-013-0755-4",
               "endpoint": endpoint, "filter": FILTER, "functional": "PBE (qmpy)",
               "implementation": {"name": impl.get("name"), "version": impl.get("version")},
               "api_version": (meta or {}).get("api_version"),
               "retrieved": _dt.date.today().isoformat(),
               "n": len(records), "records": records}
    p = Path(out_dir or Paths().raw) / "oqmd_a3bx3.json"
    write_atomic(p, (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    record([p])
    return {"n": len(records), "n_formulas": len({r["chemical_formula_reduced"] for r in records})}


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    try:
        info = fetch()
    except Exception as e:
        print(f"ERROR: OQMD OPTIMADE query failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 4
    print(f"OQMD: {info['n']} structures, {info['n_formulas']} unique formulas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
