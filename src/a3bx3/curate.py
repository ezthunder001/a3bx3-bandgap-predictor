"""Curate raw literature rows into labelled datasets and the locked holdout.

All steps are offline and logged under data/interim/:

1. Load every literature CSV under data/raw/literature (v1 + harvest schemas; see
   fetch/literature.py). Raw files are never edited.
2. Apply data/raw/literature/corrections.csv to rows of the v1 file (config
   `corrections_apply_to`). Each correction must match exactly one row, or curate stops.
   Actions:
       relabel / fix_doi / correct_value   change one field (old -> new)
       flag              keep the row in primary, add the status as a flag
       sensitivity_only  keep the row out of primary; it enters `plus_unverified` only
       exclude           the row is used nowhere
   -> corrections_applied.csv
3. Check each row: the formula must be A3B1X3 with its A/B/X columns; functional must be
   PBE (no-SOC pool) or PBE+SOC (SOC pool); anything else (HSE, mBJ, GW, bare "GGA",
   unstated) is excluded with a reason -> excluded.csv.
4. Locked holdout: the v1 reference rows for Sr3BiX3 (note TARGET). Every other row for a
   holdout formula, from any source, is excluded.
5. The same number from the same source is counted once (the v1 file and the harvest often
   quote the same paper) -> dedupe_log.csv.
6. Datasets (config `datasets`): the label of a formula is the MEDIAN of its values in that
   dataset's pool. Values are never picked by hand. Spread (max - min) is logged; spread >
   conflict_threshold_eV is flagged high_conflict and kept -> conflict_log.csv.
       primary          PBE without SOC, row class primary (preprints included)
       verified_only    primary minus preprint rows (arXiv-only sources)
       plus_unverified  primary plus sensitivity_only rows
       soc              PBE+SOC, row class primary; a separate target, never pooled
   -> data/processed/family_<name>.csv
7. If data/raw/oqmd_a3bx3.json exists, a cross-source comparison is written (not used to train).
"""
from __future__ import annotations

import csv
import io
import json
import re
import statistics
import sys
from pathlib import Path

from . import utf8_stdio
from .config import Paths, load_config
from .fetch.literature import load_all, load_corrections
from .manifest import verify, write_atomic

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
TRUNCATED_ELSEVIER = re.compile(r"^10\.1016/j\.[a-z]+\.\d{4}$")

DATASET_COLUMNS = ["formula", "A", "B", "X", "Eg_eV", "n_values", "spread_eV", "soc", "flags",
                   "values", "sources"]
ROW_LOG_COLUMNS = ["formula", "A", "B", "X", "Eg_eV", "functional", "soc", "code", "doi",
                   "arxiv_id", "source_id", "preprint", "status", "use", "source_file",
                   "source_row"]
HOLDOUT_COLUMNS = ["formula", "A", "B", "X", "Eg_eV", "functional", "soc", "doi", "doi_status",
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
    Well-formed is not the same as verified: resolution was checked by the harvest step."""
    doi = (doi or "").strip()
    if not doi:
        return "missing"
    if TRUNCATED_ELSEVIER.match(doi):
        return "truncated DOI"
    if DOI_RE.match(doi):
        return "ok"
    return "not a DOI (citation string)"


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    write_atomic(path, buf.getvalue().encode("utf-8"))


def apply_corrections(rows: list[dict], corrections: list[dict], target_file: str) -> list[dict]:
    """Mutates rows in place; returns the applied-corrections log."""
    for r in rows:
        r.setdefault("use", "primary")
        r["flags"] = []
    log = []
    for c in corrections:
        field, old, new, action = c["field"], c["old"], c["new"], c["action"]

        def current(r):
            v = r.get(field)
            if field == "status" and v == "ok":
                return ""
            if field == "Eg_eV" and v is not None:
                return repr(float(v))
            return "" if v is None else str(v)

        want_old = repr(float(old)) if field == "Eg_eV" and old else old
        hits = [r for r in rows if r["source_file"] == target_file and r["formula"] == c["formula"]
                and current(r) == want_old]
        if len(hits) != 1:
            raise ValueError(f"correction {c['formula']}.{field} {old!r}->{new!r} matched "
                             f"{len(hits)} rows of {target_file} (need exactly 1)")
        r = hits[0]
        if action in {"relabel", "fix_doi", "correct_value"}:
            r[field] = float(new) if field == "Eg_eV" else new
            if field == "doi":
                from .fetch.literature import set_source_id
                set_source_id(r)
        elif action == "flag":
            r["flags"].append(new)
        elif action in {"sensitivity_only", "exclude"}:
            r["status"] = new
            r["use"] = action
        else:
            raise ValueError(f"unknown correction action {action!r}")
        log.append({**c, "source_file": r["source_file"], "source_row": r["source_row"]})
    return log


def _aggregate(rows: list[dict], dataset: str, soc_set: bool, thr: float) -> tuple[list[dict], list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["formula"], []).append(r)
    out, conflicts = [], []
    for f in sorted(by, key=lambda k: (by[k][0]["B"], by[k][0]["A"], by[k][0]["X"])):
        rs = sorted(by[f], key=lambda r: (r["Eg_eV"], r["source_id"] or ""))
        vals = [r["Eg_eV"] for r in rs]
        spread = round(max(vals) - min(vals), 4)
        flags = set()
        for r in rs:
            flags.update(r["flags"])
        if spread > thr:
            flags.add("high_conflict")
        if not soc_set and any(r["soc"] != "no" for r in rs):
            flags.add("soc_unknown")
        if all(r["preprint"] for r in rs):
            flags.add("preprint_only")
        elif any(r["preprint"] for r in rs):
            flags.add("preprint_in_median")
        if any(r["use"] == "sensitivity_only" for r in rs):
            flags.add("source_unverified")
        rec = {"formula": f, "A": rs[0]["A"], "B": rs[0]["B"], "X": rs[0]["X"],
               "Eg_eV": round(statistics.median(vals), 4), "n_values": len(vals),
               "spread_eV": spread,
               "soc": "yes" if soc_set else ("no" if all(r["soc"] == "no" for r in rs) else "unknown"),
               "flags": ";".join(sorted(flags)),
               "values": ";".join(f"{r['Eg_eV']}" for r in rs),
               "sources": ";".join(r["source_id"] or "?" for r in rs)}
        out.append(rec)
        if len(vals) > 1:
            conflicts.append({"dataset": dataset, **rec, "high_conflict": spread > thr})
    return out, conflicts


def curate(paths: Paths | None = None, literature_dir: Path | None = None,
           holdout_formulas: list[str] | None = None) -> dict:
    """Overrides exist for tests only; the pipeline always runs from configs/v2.yaml."""
    cfg = load_config()["data"]
    paths = (paths or Paths()).ensure()
    lit = Path(literature_dir or paths.literature)
    holdout_set = set(cfg["holdout_formulas"] if holdout_formulas is None else holdout_formulas)
    pbe, soc_f = cfg["target_functional"], cfg["soc_functional"]
    thr = cfg["conflict_threshold_eV"]

    raw = load_all(lit)
    applied = apply_corrections(raw, load_corrections(lit), cfg["corrections_apply_to"])

    excluded, pools, holdout = [], {pbe: [], soc_f: []}, []
    for r in raw:
        reason = r["parse_error"]
        if not reason:
            if _counts(r["formula"]) != {r["A"]: 3, r["B"]: 1, r["X"]: 3}:
                reason = f"formula does not match A3BX3 with A={r['A']} B={r['B']} X={r['X']}"
        if not reason:
            r["formula"] = canonical(r["A"], r["B"], r["X"])
        if not reason and r["use"] == "exclude":
            reason = f"correction: {r['status']}"
        if not reason and r["formula"] in holdout_set:
            if r["schema"] == "v1" and "TARGET" in (r["note"] or "") and r["functional"] == pbe:
                r["doi_status"] = doi_status(r["doi"])
                holdout.append(r)
                continue
            reason = "locked holdout formula; only the v1 reference row is kept, and never trained on"
        if not reason and r["functional"] not in pools:
            reason = f"functional {r['functional']} is not a training label (PBE or PBE+SOC only)"
        if reason:
            excluded.append({**r, "reason": reason})
            continue
        pools[r["functional"]].append(r)

    # The same number from the same source counts once.
    dedupes = []
    for fn, rows in pools.items():
        seen, kept = {}, []
        for r in rows:
            key = (r["formula"], r["source_id"], round(r["Eg_eV"], 4))
            if key in seen:
                dedupes.append({**r, "duplicate_of": f"{seen[key]['source_file']}:{seen[key]['source_row']}"})
                continue
            seen[key] = r
            kept.append(r)
        pools[fn] = kept

    if {r["formula"] for r in holdout} != holdout_set:
        raise ValueError(f"holdout rows found {sorted(r['formula'] for r in holdout)}, "
                         f"expected {sorted(holdout_set)}")

    datasets, conflicts, summary_ds = {}, [], {}
    for name, spec in cfg["datasets"].items():
        rows = [r for r in pools[spec["functional"]] if r["use"] in spec["use"]
                and not (spec["drop_preprint"] and r["preprint"])]
        agg, conf = _aggregate(rows, name, spec["functional"] == soc_f, thr)
        if holdout_set & {a["formula"] for a in agg}:
            raise AssertionError(f"{name}: holdout formula reached a dataset")
        datasets[name] = agg
        conflicts += conf
        _write_csv(paths.processed / f"family_{name}.csv", agg, DATASET_COLUMNS)
        summary_ds[name] = {
            "n_formulas": len(agg), "n_values": len(rows),
            "n_high_conflict": sum("high_conflict" in a["flags"] for a in agg),
            "n_multi_value": sum(a["n_values"] > 1 for a in agg),
            "n_preprint_only": sum("preprint_only" in a["flags"] for a in agg),
            "n_soc_unknown": sum("soc_unknown" in a["flags"] for a in agg),
            "by_A": {k: sum(a["A"] == k for a in agg) for k in ("Mg", "Ca", "Sr", "Ba")},
            "by_B": {k: sum(a["B"] == k for a in agg) for k in ("P", "As", "Sb", "Bi")},
            "by_X": {k: sum(a["X"] == k for a in agg) for k in ("F", "Cl", "Br", "I")}}

    _write_csv(paths.processed / "holdout_sr3bix3.csv", holdout, HOLDOUT_COLUMNS)
    used = [r for fn in pools for r in pools[fn]]
    _write_csv(paths.interim / "label_rows.csv", used, ROW_LOG_COLUMNS)
    _write_csv(paths.interim / "excluded.csv", excluded, ROW_LOG_COLUMNS + ["reason"])
    _write_csv(paths.interim / "dedupe_log.csv", dedupes, ROW_LOG_COLUMNS + ["duplicate_of"])
    _write_csv(paths.interim / "conflict_log.csv", conflicts,
               ["dataset", "formula", "Eg_eV", "n_values", "spread_eV", "high_conflict",
                "values", "sources", "flags"])
    _write_csv(paths.interim / "corrections_applied.csv", applied,
               ["formula", "field", "old", "new", "action", "source_file", "source_row", "doi",
                "evidence"])
    cross = cross_source_oqmd(paths, datasets["primary"], holdout)
    k1 = cfg["k1_min_primary_formulas"]
    summary = {"n_raw_rows": len(raw), "n_excluded_rows": len(excluded),
               "n_corrections": len(applied), "n_holdout": len(holdout),
               "n_duplicate_rows_dropped": len(dedupes), "datasets": summary_ds,
               "k1": {"threshold": k1, "primary_formulas": summary_ds["primary"]["n_formulas"],
                      "pass": summary_ds["primary"]["n_formulas"] >= k1},
               "oqmd_overlap": len(cross)}
    write_atomic(paths.interim / "curate_summary.json",
                 (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return summary


def cross_source_oqmd(paths: Paths, rows: list[dict], holdout: list[dict]) -> list[dict]:
    """Literature (primary label or holdout reference) vs OQMD PBE for overlapping formulas
    (Pm-3m, lowest hull distance). Written for the data card; never used to train."""
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
    for r, is_hold in [(x, False) for x in rows] + [(x, True) for x in holdout]:
        key = frozenset({r["A"]: 3, r["B"]: 1, r["X"]: 3}.items())
        if key in best:
            o = best[key]
            out.append({"formula": r["formula"], "literature_Eg_eV": r["Eg_eV"],
                        "oqmd_Eg_eV": round(float(o["_oqmd_band_gap"]), 4),
                        "diff_eV": round(float(o["_oqmd_band_gap"]) - r["Eg_eV"], 4),
                        "oqmd_id": o["id"], "oqmd_stability_eV": o["_oqmd_stability"],
                        "holdout": is_hold})
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
    ds = ", ".join(f"{k} {v['n_formulas']}" for k, v in s["datasets"].items())
    print(f"curate: {ds} formulas; {s['n_holdout']} locked holdout; "
          f"{s['n_excluded_rows']} rows excluded; {s['n_corrections']} corrections; "
          f"K1 {'PASS' if s['k1']['pass'] else 'FAIL'} "
          f"({s['k1']['primary_formulas']} vs >= {s['k1']['threshold']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
