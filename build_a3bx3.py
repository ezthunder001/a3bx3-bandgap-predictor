"""
Build the A3BX3 pnictogen-halide family dataset — the CORRECT structural class
for supporting the Sr3BiX3 (X=I,Br,Cl) paper.

Sr3BiX3 is a 3:1:3 'inverse / antiperovskite-derivative', NOT a 1:1:3 ABX3
perovskite. This script harvests the same family so an ML model can actually
predict Sr3BiX3 instead of extrapolating from the wrong chemistry.

Source A — JARVIS-DFT: all A3-B-X3 (3:1:3) with B = pnictogen (P/As/Sb/Bi),
           X = halide, semiconducting (Eg>0.1). Uniform OptB88vdW method.
Source B — literature-confirmed values (hand-entered with DOIs below), the
           same family from recent DFT papers.

Output:
  data/a3bx3_jarvis.csv       JARVIS family subset (structure + gap + elastic)
  data/a3bx3_literature.csv   literature-confirmed family points (with DOI)

Run:
    .venv\\Scripts\\python.exe build_a3bx3.py
"""
from __future__ import annotations

import sys
import re
import csv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

HAL = {"F", "Cl", "Br", "I"}
PNB = {"P", "As", "Sb", "Bi"}          # pnictogen B-site (Sr3BiX3 has B=Bi)
# exclude f-electron / rare-earth A-sites (metallic, OptB88 unreliable)
A_EXCL = {"Pr", "La", "Ce", "Nd", "Sm", "Eu", "Gd", "Yb"}


def parse(f: str) -> dict:
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        if el:
            out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def cij(et):
    if not isinstance(et, list):
        return ("", "", "")
    M = np.array(et, dtype=float)
    if M.shape != (6, 6):
        return ("", "", "")
    return (round(np.mean([M[0, 0], M[1, 1], M[2, 2]]), 2),
            round(np.mean([M[0, 1], M[0, 2], M[1, 2]]), 2),
            round(np.mean([M[3, 3], M[4, 4], M[5, 5]]), 2))


def main():
    from jarvis.db.figshare import data
    d = data("dft_3d")

    rows = []
    for e in d:
        f = e.get("formula")
        if not f:
            continue
        c = parse(f)
        if len(c) != 3:
            continue
        g = np.gcd.reduce(list(c.values()))
        norm = {k: v // g for k, v in c.items()}
        threes = [k for k, v in norm.items() if v == 3]
        ones = [k for k, v in norm.items() if v == 1]
        if len(threes) != 2 or len(ones) != 1:
            continue
        B = ones[0]
        Xs = [t for t in threes if t in HAL]
        As = [t for t in threes if t not in HAL]
        if B not in PNB or len(Xs) != 1 or len(As) != 1:
            continue
        A = As[0]
        if A in A_EXCL:
            continue
        try:
            eg = float(e.get("optb88vdw_bandgap"))
        except (TypeError, ValueError):
            continue
        if eg <= 0.10:
            continue
        atoms = e.get("atoms") or {}
        abc = atoms.get("abc") or [np.nan]*3
        c11, c12, c44 = cij(e.get("elastic_tensor"))
        rows.append({
            "formula": f, "A": A, "B": B, "X": Xs[0],
            "spg_number": e.get("spg_number"), "crystal_system": e.get("crys"),
            "lattice_a": round(float(abc[0]), 4) if len(abc) == 3 else "",
            "cell_vol": round(float(abc[0])*float(abc[1])*float(abc[2]), 2) if len(abc) == 3 else "",
            "density": e.get("density"),
            "Eg_optb88_eV": round(eg, 4),
            "C11_GPa": c11, "C12_GPa": c12, "C44_GPa": c44,
            "source": "JARVIS-DFT", "doi": "10.1016/j.commatsci.2025.114063",
        })

    cols = ["formula", "A", "B", "X", "spg_number", "crystal_system",
            "lattice_a", "cell_vol", "density", "Eg_optb88_eV",
            "C11_GPa", "C12_GPa", "C44_GPa", "source", "doi"]
    with open(DATA / "a3bx3_jarvis.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"JARVIS A3BX3 pnictogen-halide family: {len(rows)} compounds")
    for r in sorted(rows, key=lambda x: (x["B"], x["A"], x["X"])):
        el = "elastic" if r["C11_GPa"] != "" else "       "
        print(f"  {r['formula']:<12} {r['crystal_system']:<12} Eg={r['Eg_optb88_eV']:<7} {el}")

    # ── Source B: literature-confirmed family points (PBE/GGA, with DOI) ──────
    # These are the SAME A3BX3 pnictogen-halide family, from recent DFT papers
    # already verified during the literature search. Bandgaps in eV; Cij in GPa.
    LIT = [
        # name, A, B, X, Eg_PBE, C11, C12, C44, year, doi, note
        # ── Bi-halide family (closest to Sr3BiX3 target) ────────────────────
        ("Ca3BiI3",  "Ca", "Bi", "I",  1.29,  "", "", "", 2026, "NJC 2026,50,522 Islam", "A3BiX3; Table1"),
        ("Ca3BiBr3", "Ca", "Bi", "Br", 1.60,  "", "", "", 2026, "NJC 2026,50,522 Islam", "A3BiX3; Table1"),
        ("Ca3BiCl3", "Ca", "Bi", "Cl", 1.76,  "", "", "", 2026, "NJC 2026,50,522 Islam", "A3BiX3; Table1"),
        ("Mg3BiI3",  "Mg", "Bi", "I",  0.867, 65.09, 21.42, 15.81, 2025, "10.1039/D4RA08680C", "Khandaker 2025 RSC Adv"),
        ("Mg3BiBr3", "Mg", "Bi", "Br", 1.626, 76.36, 18.58, 22.49, 2025, "10.1039/D4RA08680C", "Khandaker 2025 RSC Adv"),
        # ── As / P / Sb halide family (same structure, brackets Bi) ─────────
        ("Ba3AsI3",  "Ba", "As", "I",  0.834, 56.31, 6.58,  8.01,  2023, "10.1016/j.heliyon.2023.e21675", "Heliyon 2023"),
        ("Ca3PI3",   "Ca", "P",  "I",  1.4909, 55.62, 35.32, 17.27, 2024, "10.1016/j.heliyon.2024.e29144", "Ca3PX3 GGA-PBE+elastic"),
        ("Ca3PBr3",  "Ca", "P",  "Br", 1.9502, 61.52, 40.51, 21.53, 2024, "10.1016/j.heliyon.2024.e29144", "Ca3PX3 GGA-PBE+elastic"),
        ("Ca3PCl3",  "Ca", "P",  "Cl", 2.2058, 67.35, 43.83, 24.78, 2024, "10.1016/j.heliyon.2024.e29144", "Ca3PX3 GGA-PBE+elastic"),
        ("Ca3SbF3",  "Ca", "Sb", "F",  2.24,  "", "", "", 2024, "10.1007/s11082-024-07968-2", "Ca3SbX3 GGA-PBE"),
        ("Ca3SbCl3", "Ca", "Sb", "Cl", 1.83,  "", "", "", 2024, "10.1007/s11082-024-07968-2", "Ca3SbX3 GGA-PBE"),
        ("Ca3SbBr3", "Ca", "Sb", "Br", 1.67,  "", "", "", 2024, "10.1007/s11082-024-07968-2", "Ca3SbX3 GGA-PBE"),
        ("Ca3SbI3",  "Ca", "Sb", "I",  1.35,  "", "", "", 2024, "10.1007/s11082-024-07968-2", "Ca3SbX3 GGA-PBE"),
        ("Ca3AsCl3", "Ca", "As", "Cl", 1.742, "", "", "", 2024, "10.1016/j.mtcomm.2024", "Ca3AsCl3 GGA"),
        ("Ca3AsI3",  "Ca", "As", "I",  1.58,  "", "", "", 2025, "10.1134/S1063782625603371", "Ca3AsX3 GGA"),
        # ── Sr-containing family (closest A-site to Sr3BiX3 target) ─────────
        ("Sr3PCl3",  "Sr", "P",  "Cl", 1.70,  "", "", "", 2024, "10.1007/s11082-024-07388-2", "Sr3PX3 GGA ambient"),
        ("Sr3PBr3",  "Sr", "P",  "Br", 1.55,  "", "", "", 2024, "10.1007/s11082-024-07388-2", "Sr3PX3 GGA ambient"),
        ("Ba3AsCl3", "Ba", "As", "Cl", 0.942, "", "", "", 2025, "10.1088/1402-4896/adee54", "Ba3AsCl3 GGA"),
        ("Sr3AsCl3", "Sr", "As", "Cl", 1.70,  75.07, 10.03, 15.21, 2024, "10.1016/j.heliyon.2024.e35855", "Sr3BCl3 GGA+elastic"),
        ("Sr3SbCl3", "Sr", "Sb", "Cl", 1.72,  65.29, 8.40,  12.57, 2024, "10.1016/j.heliyon.2024.e35855", "Sr3BCl3 GGA+elastic"),
        ("Mg3PCl3",  "Mg", "P",  "Cl", 2.297, "", "", "", 2025, "10.1039/D5RA01185J", "Mg3PX3 GGA"),
        ("Mg3PBr3",  "Mg", "P",  "Br", 1.506, "", "", "", 2025, "10.1039/D5RA01185J", "Mg3PX3 GGA"),
        ("Ba3SbI3",  "Ba", "Sb", "I",  0.78,  "", "", "", 2024, "10.1007/s10853-024-10487-w", "Ba3MI3 GGA"),
        # validation TARGETS (the paper's own DFT — withheld from training)
        ("Sr3BiI3",  "Sr", "Bi", "I",  1.324, 60.324, 9.446, "", 2026, "NJC 2026,50,522 Islam", "TARGET"),
        ("Sr3BiBr3", "Sr", "Bi", "Br", 1.512, "", 10.40, "", 2026, "NJC 2026,50,522 Islam", "TARGET"),
        ("Sr3BiCl3", "Sr", "Bi", "Cl", 1.731, "", 11.51, "", 2026, "NJC 2026,50,522 Islam", "TARGET"),
    ]
    lit_cols = ["formula", "A", "B", "X", "Eg_PBE_eV", "C11_GPa", "C12_GPa",
                "C44_GPa", "year", "doi", "note"]
    with open(DATA / "a3bx3_literature.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp); w.writerow(lit_cols)
        for r in LIT:
            w.writerow(r)
    n_train = sum(1 for r in LIT if "TARGET" not in r[-1])
    print(f"\nLiterature family points: {len(LIT)} ({n_train} trainable + "
          f"{len(LIT)-n_train} Sr3BiX3 validation targets)")
    print(f"  -> data/a3bx3_jarvis.csv, data/a3bx3_literature.csv")

    print("\n" + "=" * 60)
    print("  FAMILY-FOCUSED DATASET so far")
    print("=" * 60)
    print(f"  JARVIS A3BX3 (uniform method): {len(rows)}")
    print(f"  Literature A3BX3 (with DOI):   {n_train} trainable")
    print(f"  Combined same-family anchors:  {len(rows) + n_train}")
    print(f"  Still sparse: Sr/Ba-Bi specifically -> candidates for your DFT")
    print("=" * 60)


if __name__ == "__main__":
    main()
