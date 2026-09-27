"""Feature sets. Both are row-wise and deterministic (no statistics are fitted across rows),
so computing them before the CV split leaks nothing. Anything that IS fitted — scaling,
variance filtering, feature selection — lives inside the model Pipelines in models.py.

physics9  the nine v1 crystal-chemistry descriptors (ionic radii, electronegativities,
          differences, radius ratio). The element tables are copied verbatim from v1
          train_a3bx3_family.py; tests/test_v2_pipeline.py asserts they still match.
magpie    matminer ElementProperty.from_preset("magpie") on the composition
          (Ward et al. 2016, doi 10.1038/npjcompumats.2016.28). Element values come from
          the data files shipped inside matminer, never typed by hand.
"""
from __future__ import annotations

import csv
import io
import sys
import warnings

import numpy as np

from . import utf8_stdio
from .config import Paths
from .manifest import write_atomic

# ── v1 element tables (copied from train_a3bx3_family.py, unchanged) ──────────
RAD_A = {"Mg": 0.89, "Ca": 1.34, "Sr": 1.44, "Ba": 1.61}
CHI_A = {"Mg": 1.31, "Ca": 1.00, "Sr": 0.95, "Ba": 0.89}
RAD_B = {"P": 1.07, "As": 1.19, "Sb": 1.39, "Bi": 1.48}
CHI_B = {"P": 2.19, "As": 2.18, "Sb": 2.05, "Bi": 2.02}
RAD_X = {"F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20}
CHI_X = {"F": 3.98, "Cl": 3.16, "Br": 2.96, "I": 2.66}

PHYSICS9 = ["rA", "rB", "rX", "chiA", "chiB", "chiX", "dchi_BX", "dchi_AX", "rB_over_rX"]
CHIX_INDEX = PHYSICS9.index("chiX")   # used by the HalideRule baseline


def physics9(A: str, B: str, X: str) -> list[float]:
    rA, rB, rX = RAD_A[A], RAD_B[B], RAD_X[X]
    cA, cB, cX = CHI_A[A], CHI_B[B], CHI_X[X]
    return [rA, rB, rX, cA, cB, cX, cX - cB, cX - cA, rB / rX]


_MAGPIE = None


def _magpie_featurizer():
    global _MAGPIE
    if _MAGPIE is None:
        from matminer.featurizers.composition import ElementProperty
        _MAGPIE = ElementProperty.from_preset("magpie", impute_nan=False)
    return _MAGPIE


def magpie_labels() -> list[str]:
    return list(_magpie_featurizer().feature_labels())


def magpie(A: str, B: str, X: str) -> list[float]:
    from pymatgen.core import Composition
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [float(v) for v in _magpie_featurizer().featurize(Composition({A: 3, B: 1, X: 3}))]


def featurize_rows(rows: list[dict], feature_set: str) -> tuple[np.ndarray, list[str]]:
    if feature_set == "physics9":
        return np.array([physics9(r["A"], r["B"], r["X"]) for r in rows], float), list(PHYSICS9)
    if feature_set == "magpie":
        M = np.array([magpie(r["A"], r["B"], r["X"]) for r in rows], float)
        return M, magpie_labels()
    raise ValueError(f"unknown feature set {feature_set!r}")


def read_csv(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["Eg_eV"] = float(r["Eg_eV"])
    return rows


def build(paths: Paths | None = None) -> dict:
    """Write data/processed/features_<set>.csv for every dataset formula + holdout (for inspection;
    the evaluator recomputes from the same functions so the two cannot drift)."""
    from .config import load_config
    paths = (paths or Paths()).ensure()
    seen, rows = set(), []
    for name in load_config()["data"]["datasets"]:
        for r in read_csv(paths.processed / f"family_{name}.csv"):
            if r["formula"] not in seen:
                seen.add(r["formula"])
                rows.append(r)
    rows = sorted(rows, key=lambda r: r["formula"]) + read_csv(paths.processed / "holdout_sr3bix3.csv")
    out = {}
    for fs in ("physics9", "magpie"):
        M, names = featurize_rows(rows, fs)
        if np.isnan(M).any():
            raise ValueError(f"{fs}: NaN features for {[r['formula'] for r, m in zip(rows, M) if np.isnan(m).any()]}")
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["formula"] + names)
        for r, m in zip(rows, M):
            w.writerow([r["formula"]] + [repr(round(float(v), 10)) for v in m])
        write_atomic(paths.processed / f"features_{fs}.csv", buf.getvalue().encode("utf-8"))
        out[fs] = M.shape
    return out


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    shapes = build()
    print("features: " + ", ".join(f"{k} {v[0]}x{v[1]}" for k, v in shapes.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
