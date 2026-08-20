"""
Step 1b — Bulk data retrieval from JARVIS-DFT (NIST), self-consistent method.

Why JARVIS over scraped papers:
  Every entry is computed with ONE uniform methodology (OptB88vdW relaxation,
  OptB88vdW + TBmBJ band gaps, DFPT elastic tensor). This removes the cross-source
  label noise that caps the hand-curated PBE model (~0.16 eV MAE floor).

What this pulls:
  Full dft_3d dataset (~80k materials, downloaded once & cached by jarvis-tools),
  then filters to ABX3 perovskite-stoichiometry halides/oxides and extracts:
    - optb88vdw_bandgap (eV)   — the "PBE-like" GGA gap
    - mbj_bandgap (eV)         — TBmBJ, near-experimental accuracy
    - elastic_tensor           — full Voigt 6x6 -> C11, C12, C44 (cubic)
    - bulk_modulus_kv, shear_modulus_gv (Voigt-averaged, GPa)
    - spg_number, crystal_system, formula, jid

Output:
    data/jarvis_perovskites_raw.csv   — every ABX3 hit, all columns
    data/jarvis_perovskites_clean.csv — cubic (spg 221) subset with valid targets

Run:
    .venv\\Scripts\\python.exe fetch_jarvis.py
"""
from __future__ import annotations

import sys
import re
import csv
from pathlib import Path
from collections import Counter

# UTF-8 console (Windows cp1252 crashes on glyphs)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HALIDES = {"F", "Cl", "Br", "I"}
CHALCOGENS = {"O", "S", "Se"}          # oxide/chalcogenide perovskites (optional)
ANIONS = HALIDES | CHALCOGENS

# ── Domain definition: optoelectronic halide perovskite ──────────────────────
# B-site = p-block metal / metalloid (semiconductor-forming). Excludes 3d TM
# fluorides (magnetic/correlated, OptB88 fails) that merely match ABX3 formula.
BSITE_OK = {"Pb", "Sn", "Ge", "Bi", "Sb", "In", "Tl", "Ga"}
# A-site = large electropositive cation. Excludes TM/N/C/H/O false matches.
ASITE_OK = {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
            "Tl", "In", "Ag", "Au"}

# JARVIS sentinel for "not computed"
NA_VALUES = {"na", "", None, "None"}


def parse_formula(formula: str) -> dict[str, int] | None:
    """Parse a formula like 'CsPbI3' or 'Cs1Pb1I3' -> {'Cs':1,'Pb':1,'I':3}."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts: dict[str, int] = {}
    for el, num in tokens:
        if not el:
            continue
        counts[el] = counts.get(el, 0) + (int(num) if num else 1)
    return counts or None


def is_ABX3(counts: dict[str, int]) -> tuple[str, str, str] | None:
    """
    Return (A, B, X) if composition reduces to ABX3 with an anion on X.
    Accepts both 1-1-3 (e.g. CsPbI3) cells; rejects everything else.
    """
    if len(counts) != 3:
        return None
    # normalise by gcd so 2-2-6 also reduces to 1-1-3
    vals = list(counts.values())
    g = np.gcd.reduce(vals)
    norm = {el: n // g for el, n in counts.items()}
    # need exactly one element with count 3 and two with count 1
    threes = [el for el, n in norm.items() if n == 3]
    ones = [el for el, n in norm.items() if n == 1]
    if len(threes) != 1 or len(ones) != 2:
        return None
    X = threes[0]
    if X not in ANIONS:
        return None
    # B = higher-electronegativity / smaller cation is ambiguous from formula alone;
    # we keep both cations and let downstream featurizer assign by radius.
    A, B = ones[0], ones[1]
    return A, B, X


def _num(v):
    """Coerce JARVIS value to float or NaN."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.nan
    if v in NA_VALUES:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def elastic_from_tensor(et) -> tuple[float, float, float]:
    """Extract C11, C12, C44 (GPa) from a JARVIS Voigt 6x6 elastic tensor."""
    if not isinstance(et, (list, tuple, np.ndarray)):
        return (np.nan, np.nan, np.nan)
    try:
        M = np.array(et, dtype=float)
        if M.shape != (6, 6):
            return (np.nan, np.nan, np.nan)
        # cubic averaging for robustness
        c11 = np.mean([M[0, 0], M[1, 1], M[2, 2]])
        c12 = np.mean([M[0, 1], M[0, 2], M[1, 2]])
        c44 = np.mean([M[3, 3], M[4, 4], M[5, 5]])
        return (round(c11, 3), round(c12, 3), round(c44, 3))
    except Exception:
        return (np.nan, np.nan, np.nan)


def main() -> None:
    print("Loading JARVIS-DFT dft_3d (first run downloads ~ a few hundred MB) ...")
    from jarvis.db.figshare import data as jarvis_data

    d = jarvis_data("dft_3d")
    print(f"  loaded {len(d):,} JARVIS-DFT entries")

    rows = []
    for e in d:
        formula = e.get("formula")
        if not formula:
            continue
        counts = parse_formula(formula)
        if not counts:
            continue
        abx = is_ABX3(counts)
        if not abx:
            continue
        A, B, X = abx

        c11, c12, c44 = elastic_from_tensor(e.get("elastic_tensor"))

        # structural (relaxed cell) features
        atoms = e.get("atoms") or {}
        abc = atoms.get("abc") or [np.nan, np.nan, np.nan]
        ang = atoms.get("angles") or [np.nan, np.nan, np.nan]
        nat = len(atoms.get("elements") or [])
        la, lb, lc = (float(abc[0]), float(abc[1]), float(abc[2])) if len(abc) == 3 else (np.nan,)*3
        al, be, ga = (float(ang[0]), float(ang[1]), float(ang[2])) if len(ang) == 3 else (np.nan,)*3
        vol = la * lb * lc if not any(np.isnan([la, lb, lc])) else np.nan

        rows.append({
            "jid":            e.get("jid"),
            "formula":        formula,
            "A_raw":          A,
            "B_raw":          B,
            "X":              X,
            "anion_class":    "halide" if X in HALIDES else "chalcogen",
            "spg_number":     e.get("spg_number"),
            "crystal_system": e.get("crys"),
            "lattice_a":      round(la, 4) if not np.isnan(la) else np.nan,
            "lattice_b":      round(lb, 4) if not np.isnan(lb) else np.nan,
            "lattice_c":      round(lc, 4) if not np.isnan(lc) else np.nan,
            "alpha":          round(al, 3) if not np.isnan(al) else np.nan,
            "beta":           round(be, 3) if not np.isnan(be) else np.nan,
            "gamma":          round(ga, 3) if not np.isnan(ga) else np.nan,
            "cell_vol":       round(vol, 3) if not np.isnan(vol) else np.nan,
            "n_atoms":        nat,
            "density":        _num(e.get("density")),
            "Eg_optb88_eV":   _num(e.get("optb88vdw_bandgap")),
            "Eg_mbj_eV":      _num(e.get("mbj_bandgap")),
            "C11_GPa":        c11,
            "C12_GPa":        c12,
            "C44_GPa":        c44,
            "B_voigt_GPa":    _num(e.get("bulk_modulus_kv")),
            "G_voigt_GPa":    _num(e.get("shear_modulus_gv")),
        })

    cols = ["jid", "formula", "A_raw", "B_raw", "X", "anion_class", "spg_number",
            "crystal_system", "lattice_a", "lattice_b", "lattice_c",
            "alpha", "beta", "gamma", "cell_vol", "n_atoms", "density",
            "Eg_optb88_eV", "Eg_mbj_eV",
            "C11_GPa", "C12_GPa", "C44_GPa", "B_voigt_GPa", "G_voigt_GPa"]

    def _w(val):
        return "" if (isinstance(val, float) and np.isnan(val)) else val

    raw_path = DATA_DIR / "jarvis_perovskites_raw.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: _w(r[k]) for k in cols})
    print(f"\nABX3 hits: {len(rows)}  ->  {raw_path}")

    def notna(v):
        return not (isinstance(v, float) and np.isnan(v))

    hal = [r for r in rows if r["anion_class"] == "halide"]

    # ── Coverage report ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  COVERAGE REPORT  (ABX3 stoichiometry hits)")
    print("=" * 64)
    print("  by anion class:")
    for k, v in Counter(r["anion_class"] for r in rows).most_common():
        print(f"    {k:<12} {v}")
    print("\n  halide subset by crystal system:")
    for k, v in Counter(r["crystal_system"] for r in hal).most_common():
        print(f"    {str(k):<14} {v}")

    has_eg   = sum(notna(r["Eg_optb88_eV"]) for r in rows)
    has_mbj  = sum(notna(r["Eg_mbj_eV"]) for r in rows)
    has_elas = sum(notna(r["C11_GPa"]) for r in rows)
    hal_cubic = sum(r["spg_number"] == 221 for r in hal)
    hal_eg_el = sum(notna(r["Eg_optb88_eV"]) and notna(r["C11_GPa"]) for r in hal)
    print(f"\n  with OptB88 band gap : {has_eg}")
    print(f"  with mBJ band gap    : {has_mbj}")
    print(f"  with elastic tensor  : {has_elas}")
    print(f"  halide + cubic(221)  : {hal_cubic}")
    print(f"  halide + Eg + elastic: {hal_eg_el}")

    # ── Domain modelling subset: optoelectronic halide perovskite ────────────
    # B-site in BSITE_OK, A-site in ASITE_OK, semiconducting (Eg>0.1), all phases.
    domain = []
    for r in hal:
        a, b = r["A_raw"], r["B_raw"]
        # assign which cation is the B-site (p-block metal)
        if b in BSITE_OK and a in ASITE_OK:
            A, B = a, b
        elif a in BSITE_OK and b in ASITE_OK:
            A, B = b, a
        else:
            continue
        eg = r["Eg_optb88_eV"]
        if not notna(eg) or eg <= 0.10:
            continue
        r2 = dict(r)
        r2["A"], r2["B"] = A, B
        domain.append(r2)

    dom_cols = ["jid", "formula", "A", "B", "X", "spg_number", "crystal_system",
                "lattice_a", "lattice_b", "lattice_c", "alpha", "beta", "gamma",
                "cell_vol", "n_atoms", "density",
                "Eg_optb88_eV", "Eg_mbj_eV",
                "C11_GPa", "C12_GPa", "C44_GPa", "B_voigt_GPa", "G_voigt_GPa"]
    dom_path = DATA_DIR / "jarvis_domain_model.csv"
    with open(dom_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=dom_cols)
        wr.writeheader()
        for r in domain:
            wr.writerow({k: _w(r.get(k, "")) for k in dom_cols})

    n_elas = sum(notna(r["C11_GPa"]) for r in domain)
    print(f"\n  DOMAIN modelling set (p-block B, clean A, Eg>0.1, all phases): {len(domain)}")
    print(f"    -> {dom_path}")
    print(f"    with elastic constants: {n_elas}")
    print(f"    B-site: {dict(Counter(r['B'] for r in domain).most_common())}")
    print(f"    A-site: {dict(Counter(r['A'] for r in domain).most_common())}")
    print(f"    X-site: {dict(Counter(r['X'] for r in domain).most_common())}")
    print(f"    phases: {dict(Counter(r['crystal_system'] for r in domain).most_common())}")
    print("=" * 64)


if __name__ == "__main__":
    main()
