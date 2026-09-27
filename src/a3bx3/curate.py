"""Curate raw literature rows into the family training set and the locked holdout.

Steps (all offline, all logged):
1. Load every CSV under data/raw/literature (both schemas, see fetch/literature.py).
2. Check each row's formula against its A/B/X columns (must be A3 B1 X3).
3. Trainable = functional == target_functional ("PBE"), or "PBE+SOC" when the config's
   include_soc flag is on. Every other row goes to data/interim/excluded.csv with a reason.
4. Dedupe by canonical formula. When a formula has more than one trainable row, one is
   picked by this rule, in order:
       (a) same functional as the target (plain PBE beats PBE+SOC)
       (b) most recent year
       (c) carries a structure (lattice_a_A present)
       (d) first in file order (deterministic tie-break)
   If the rows differ by more than conflict_threshold_eV, every row is written to
   data/interim/conflict_log.csv with the chosen one marked. Values are never averaged.
5. Split out the locked holdout (config holdout_formulas). It is written to its own file
   and nothing downstream may train on it (tests enforce this).
6. If data/raw/oqmd_a3bx3.json exists, write a cross-source comparison (not used to train).
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

from . import utf8_stdio
from .config import Paths, load_config
from .fetch.literature import load_all
from .manifest import verify, write_atomic

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
TRUNCATED_ELSEVIER = re.compile(r"^10\.1016/j\.[a-z]+\.\d{4}$")

FAMILY_COLUMNS = ["formula", "A", "B", "X", "Eg_eV", "functional", "soc", "doi", "doi_status",
                  "year", "source_file", "source_row"]


def canonical(A: str, B: str, X: str) -> str:
    return f"{A}3{B}{X}3"


def _counts(formula: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def doi_status(doi: str | None) -> str:
    """'ok' for a well-formed DOI, otherwise a short description of the problem.
    Well-formed is not the same as verified: resolution is checked by the harvest step."""
    doi = (doi or "").strip()
    if not doi:
        return "missing"
    if TRUNCATED_ELSEVIER.match(doi):
        return "truncated DOI"
    if DOI_RE.match(doi):
        return "ok"
    return "not a DOI (citation string)"


def _pick(rows: list[dict], target: str) -> dict:
    return sorted(rows, key=lambda r: (
        0 if r["functional"] == target else 1,
        -(r["year"] or 0),
        0 if r.get("lattice_a_A") is not None else 1,
        r["source_file"], r["source_row"]))[0]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    write_atomic(path, buf.getvalue().encode("utf-8"))


def curate(paths: Paths | None = None, literature_dir: Path | None = None,
           include_soc: bool | None = None, holdout_formulas: list[str] | None = None) -> dict:
    """Overrides exist for tests only; the pipeline always runs from configs/v2.yaml."""
    cfg = load_config()["data"]
    paths = (paths or Paths()).ensure()
    target = cfg["target_functional"]
    soc = cfg["include_soc"] if include_soc is None else include_soc
    allowed = {target} | ({f"{target}+SOC"} if soc else set())
    holdout_set = set(cfg["holdout_formulas"] if holdout_formulas is None else holdout_formulas)

    raw = load_all(literature_dir or paths.literature)
    excluded, candidates = [], []
    for r in raw:
        reason = r["parse_error"]
        if not reason:
            expect = {r["A"]: 3, r["B"]: 1, r["X"]: 3}
            if _counts(r["formula"]) != expect:
                reason = f"formula {r['formula']} does not match A3BX3 with A={r['A']} B={r['B']} X={r['X']}"
        if not reason and r["functional"] not in allowed:
            reason = f"functional {r['functional']} not in trainable set {sorted(allowed)}"
        if reason:
            excluded.append({**r, "reason": reason})
            continue
        r = dict(r)
        r["formula"] = canonical(r["A"], r["B"], r["X"])
        r["doi_status"] = doi_status(r["doi"])
        candidates.append(r)

    # v1 marks the holdout rows with TARGET in the note; the config list must agree.
    v1_targets = {canonical(r["A"], r["B"], r["X"]) for r in candidates
                  if r["schema"] == "v1" and "TARGET" in (r["note"] or "")}
    if v1_targets and v1_targets != holdout_set:
        raise ValueError(f"holdout mismatch: config {sorted(holdout_set)} vs v1 TARGET {sorted(v1_targets)}")

    by_formula: dict[str, list[dict]] = {}
    for r in candidates:
        by_formula.setdefault(r["formula"], []).append(r)

    chosen, conflicts, dedupes = [], [], []
    for f, rows in by_formula.items():   # dict preserves first-appearance (file) order
        pick = _pick(rows, target)
        chosen.append(pick)
        if len(rows) > 1:
            gaps = [x["Eg_eV"] for x in rows]
            spread = max(gaps) - min(gaps)
            log = conflicts if spread > cfg["conflict_threshold_eV"] else dedupes
            for x in rows:
                log.append({**x, "chosen": x is pick, "spread_eV": round(spread, 4)})

    family = [r for r in chosen if r["formula"] not in holdout_set]
    holdout = [r for r in chosen if r["formula"] in holdout_set]
    missing_holdout = holdout_set - {r["formula"] for r in holdout}
    if missing_holdout:
        raise ValueError(f"holdout formulas absent from data: {sorted(missing_holdout)}")

    log_cols = FAMILY_COLUMNS + ["chosen", "spread_eV", "note"]
    _write_csv(paths.processed / "family.csv", family, FAMILY_COLUMNS)
    _write_csv(paths.processed / "holdout_sr3bix3.csv", holdout, FAMILY_COLUMNS)
    _write_csv(paths.interim / "excluded.csv", excluded,
               ["formula", "A", "B", "X", "Eg_eV", "functional", "soc", "doi", "source_file",
                "source_row", "reason"])
    _write_csv(paths.interim / "conflict_log.csv", conflicts, log_cols)
    _write_csv(paths.interim / "dedupe_log.csv", dedupes, log_cols)

    cross = cross_source_oqmd(paths, family + holdout)
    summary = {"n_raw": len(raw), "n_excluded": len(excluded), "n_family": len(family),
               "n_holdout": len(holdout), "n_conflicts": len({c["formula"] for c in conflicts}),
               "n_duplicates_within_threshold": len({d["formula"] for d in dedupes}),
               "soc_counts": {s: sum(1 for r in family if r["soc"] == s)
                              for s in ("yes", "no", "unknown")},
               "doi_status_counts": {s: sum(1 for r in family + holdout if r["doi_status"] == s)
                                     for s in sorted({r["doi_status"] for r in family + holdout})},
               "oqmd_overlap": len(cross)}
    write_atomic(paths.interim / "curate_summary.json",
                 (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return summary


def cross_source_oqmd(paths: Paths, rows: list[dict]) -> list[dict]:
    """Literature PBE vs OQMD PBE for overlapping formulas (Pm-3m, lowest hull distance).
    Written for the data card; never used to train."""
    p = paths.raw / "oqmd_a3bx3.json"
    if not p.exists():
        return []
    recs = json.loads(p.read_text(encoding="utf-8"))["records"]
    best: dict[frozenset, dict] = {}
    for r in recs:
        if r.get("_oqmd_spacegroup") != "Pm-3m" or r.get("_oqmd_band_gap") is None:
            continue
        key = frozenset(_counts(r["chemical_formula_reduced"]).items())
        if key not in best or (r["_oqmd_stability"] or 9) < (best[key]["_oqmd_stability"] or 9):
            best[key] = r
    out = []
    for r in rows:
        key = frozenset({r["A"]: 3, r["B"]: 1, r["X"]: 3}.items())
        if key in best:
            o = best[key]
            out.append({"formula": r["formula"], "literature_Eg_eV": r["Eg_eV"],
                        "oqmd_Eg_eV": round(float(o["_oqmd_band_gap"]), 4),
                        "diff_eV": round(float(o["_oqmd_band_gap"]) - r["Eg_eV"], 4),
                        "oqmd_id": o["id"], "oqmd_stability_eV": o["_oqmd_stability"],
                        "holdout": r["formula"] in set(load_config()["data"]["holdout_formulas"])})
    _write_csv(paths.interim / "cross_source_oqmd.csv", out,
               ["formula", "literature_Eg_eV", "oqmd_Eg_eV", "diff_eV", "oqmd_id",
                "oqmd_stability_eV", "holdout"])
    return out


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    problems = verify()
    if problems:
        print("ERROR: data/raw does not match MANIFEST.sha256:\n  " + "\n  ".join(problems),
              file=sys.stderr)
        return 1
    s = curate()
    print(f"curate: {s['n_family']} family rows, {s['n_holdout']} locked holdout, "
          f"{s['n_excluded']} excluded, {s['n_conflicts']} conflicts, "
          f"{s['oqmd_overlap']} OQMD overlaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
