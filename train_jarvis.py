"""
Step 2 — Train band-gap & elastic models on the JARVIS-DFT domain dataset.

Input : data/jarvis_domain_model.csv   (103 semiconducting halide perovskites,
        uniform OptB88vdW methodology, all phases, structural features included)

Why this is more reliable than the 32-point hand-curated PBE model:
  - 3x more data, single consistent DFT method (no cross-paper label noise)
  - Real relaxed-structure features (lattice, density, spg) capture the phase
    effect that the idealized-cubic model could not.

Models:
  Band gap  : GradientBoosting + GPR, 5-fold CV (n=103)
  Elastic   : RandomForest multi-output C11/C12/C44, 5-fold CV (n=50)

Outputs:
  data/jarvis_bandgap_cv.csv     per-compound CV predictions
  data/jarvis_elastic_cv.csv     per-compound CV predictions
  data/jarvis_model_metrics.json summary metrics
  data/jarvis_bandgap_parity.png / jarvis_elastic_parity.png

Run:
    .venv\\Scripts\\python.exe train_jarvis.py
"""
from __future__ import annotations

import sys
import csv
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_DIR = Path(__file__).parent / "data"
SRC = DATA_DIR / "jarvis_domain_model.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Element property tables (Shannon ionic radii Å / Pauling χ / mass / Z / valence)
# Tl & In appear on BOTH sites with different oxidation states -> separate radii.
# ─────────────────────────────────────────────────────────────────────────────
RADIUS_A = {  # large cation, ~12-coord effective ionic radius
    "Li": 1.06, "Na": 1.39, "K": 1.64, "Rb": 1.72, "Cs": 1.88,
    "Be": 0.45, "Mg": 0.89, "Ca": 1.34, "Sr": 1.44, "Ba": 1.61,
    "Tl": 1.70, "In": 1.40, "Ag": 1.28, "Au": 1.37,
}
RADIUS_B = {  # 6-coord ionic radius (+2 for group14, +3 for group13/15)
    "Pb": 1.19, "Sn": 1.18, "Ge": 0.73, "Bi": 1.03, "Sb": 0.76,
    "In": 0.80, "Tl": 0.885, "Ga": 0.62,
}
RADIUS_X = {"F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20}

ELEC = {
    "Li": 0.98, "Na": 0.93, "K": 0.82, "Rb": 0.82, "Cs": 0.79,
    "Be": 1.57, "Mg": 1.31, "Ca": 1.00, "Sr": 0.95, "Ba": 0.89,
    "Tl": 1.62, "In": 1.78, "Ag": 1.93, "Au": 2.54,
    "Pb": 2.33, "Sn": 1.96, "Ge": 2.01, "Bi": 2.02, "Sb": 2.05, "Ga": 1.81,
    "F": 3.98, "Cl": 3.16, "Br": 2.96, "I": 2.66,
}
MASS = {
    "Li": 6.94, "Na": 22.99, "K": 39.10, "Rb": 85.47, "Cs": 132.91,
    "Be": 9.01, "Mg": 24.31, "Ca": 40.08, "Sr": 87.62, "Ba": 137.33,
    "Tl": 204.38, "In": 114.82, "Ag": 107.87, "Au": 196.97,
    "Pb": 207.2, "Sn": 118.71, "Ge": 72.63, "Bi": 208.98, "Sb": 121.76, "Ga": 69.72,
    "F": 19.00, "Cl": 35.45, "Br": 79.90, "I": 126.90,
}
ZNUM = {
    "Li": 3, "Na": 11, "K": 19, "Rb": 37, "Cs": 55,
    "Be": 4, "Mg": 12, "Ca": 20, "Sr": 38, "Ba": 56,
    "Tl": 81, "In": 49, "Ag": 47, "Au": 79,
    "Pb": 82, "Sn": 50, "Ge": 32, "Bi": 83, "Sb": 51, "Ga": 31,
    "F": 9, "Cl": 17, "Br": 35, "I": 53,
}
VALENCE = {  # nominal valence electrons (group-based)
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    "Tl": 3, "In": 3, "Ga": 3, "Ag": 1, "Au": 1,
    "Ge": 4, "Sn": 4, "Pb": 4, "Sb": 5, "Bi": 5,
    "F": 7, "Cl": 7, "Br": 7, "I": 7,
}

FEATURE_NAMES = [
    "rA", "rB", "rX", "tolerance_t", "octahedral_mu",
    "chiA", "chiB", "chiX", "dchi_BX", "dchi_AX",
    "massA", "massB", "massX", "ZA", "ZB", "ZX",
    "valA", "valB",
    "lattice_a", "lattice_b", "lattice_c", "cell_vol", "density",
    "spg_number", "phase_distortion",
]


def _f(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def featurize(row: dict) -> np.ndarray | None:
    A, B, X = row["A"], row["B"], row["X"]
    if A not in RADIUS_A or B not in RADIUS_B or X not in RADIUS_X:
        return None
    rA, rB, rX = RADIUS_A[A], RADIUS_B[B], RADIUS_X[X]
    t = (rA + rX) / (np.sqrt(2) * (rB + rX))
    mu = rB / rX
    chiA, chiB, chiX = ELEC[A], ELEC[B], ELEC[X]

    la, lb, lc = _f(row["lattice_a"]), _f(row["lattice_b"]), _f(row["lattice_c"])
    vol, dens = _f(row["cell_vol"]), _f(row["density"])
    spg = _f(row["spg_number"], 0)
    # phase distortion proxy: anisotropy of the relaxed cell (0 = cubic)
    if not any(np.isnan([la, lb, lc])) and la > 0:
        distortion = (np.std([la, lb, lc]) / np.mean([la, lb, lc]))
    else:
        distortion = np.nan

    return np.array([
        rA, rB, rX, t, mu,
        chiA, chiB, chiX, chiB - chiX, chiA - chiX,
        MASS[A], MASS[B], MASS[X], ZNUM[A], ZNUM[B], ZNUM[X],
        VALENCE[A], VALENCE[B],
        la, lb, lc, vol, dens, spg, distortion,
    ])


def load_rows() -> list[dict]:
    with open(SRC, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_gpr(n_feat: int) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e4)) *
        Matern(length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e3), nu=2.5) +
        WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 1.0))
    )
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                    normalize_y=True, random_state=42)


def cv_metrics(y, pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y, pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "R2": round(float(r2_score(y, pred)), 4),
        "n": int(len(y)),
    }


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    rows = load_rows()
    feats, names, eg, elastic = [], [], [], []
    for r in rows:
        x = featurize(r)
        if x is None or np.isnan(x[:18]).any():   # require all compositional feats
            continue
        feats.append(x)
        names.append(r["formula"])
        eg.append(_f(r["Eg_optb88_eV"]))
        c11, c12, c44 = _f(r["C11_GPa"]), _f(r["C12_GPa"]), _f(r["C44_GPa"])
        elastic.append((c11, c12, c44))

    X_all = np.array(feats)
    y_eg = np.array(eg)
    # impute missing structural cols (rare) with column medians
    col_med = np.nanmedian(X_all, axis=0)
    inds = np.where(np.isnan(X_all))
    X_all[inds] = np.take(col_med, inds[1])

    groups = np.array(names)  # group polymorphs of the same formula together
    print(f"Featurized {len(names)} compounds x {X_all.shape[1]} features "
          f"({len(set(names))} unique formulas)")

    metrics = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)

    # ── BAND GAP ──────────────────────────────────────────────────────────────
    # Primary CV is GroupKFold on formula: the domain set contains polymorphs
    # (55 duplicate formulas), and random KFold puts polymorphs of the same
    # compound in train and test simultaneously, inflating the metrics.
    print("\n[BAND GAP]  5-fold GroupKFold by formula (OptB88vdW target)")
    models = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=None, random_state=42, n_jobs=-1),
        "GPR": make_pipeline(StandardScaler(), make_gpr(X_all.shape[1])),
    }
    eg_results, eg_leaky = {}, {}
    best_pred = None
    best_name, best_r2 = None, -1e9
    for mname, model in models.items():
        pred = cross_val_predict(model, X_all, y_eg, cv=gkf, groups=groups)
        m = cv_metrics(y_eg, pred)
        eg_results[mname] = m
        # random-KFold metrics kept for reference (these are leakage-inflated)
        pred_kf = cross_val_predict(model, X_all, y_eg, cv=kf)
        eg_leaky[mname] = cv_metrics(y_eg, pred_kf)
        print(f"  {mname:18} MAE={m['MAE']:.3f} eV  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}"
              f"   (random-KFold leaky: MAE={eg_leaky[mname]['MAE']:.3f} R2={eg_leaky[mname]['R2']:.3f})")
        if m["R2"] > best_r2:
            best_r2, best_name, best_pred = m["R2"], mname, pred
    metrics["bandgap"] = {"target": "optb88vdw_bandgap_eV",
                          "cv": "GroupKFold(5) by formula",
                          "models": eg_results,
                          "models_random_kfold_leaky": eg_leaky,
                          "best": best_name}

    with open(DATA_DIR / "jarvis_bandgap_cv.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["formula", "Eg_DFT_eV", "Eg_CV_pred_eV", "residual"])
        for n, a, p in zip(names, y_eg, best_pred):
            wr.writerow([n, round(a, 4), round(p, 4), round(p - a, 4)])

    # ── ELASTIC ───────────────────────────────────────────────────────────────
    print("\n[ELASTIC]  5-fold GroupKFold by formula (C11/C12/C44, multi-output RF)")
    has_el = np.array([not np.isnan(e[0]) for e in elastic])
    Xe = X_all[has_el]
    Ye = np.array([elastic[i] for i in range(len(elastic)) if has_el[i]])
    el_names = [names[i] for i in range(len(names)) if has_el[i]]
    el_groups = np.array(el_names)
    print(f"  elastic subset: {len(Ye)} compounds ({len(set(el_names))} unique formulas)")
    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    gkf_e = GroupKFold(n_splits=5)
    pred_e = cross_val_predict(rf, Xe, Ye, cv=gkf_e, groups=el_groups)
    el_metrics = {}
    for j, comp in enumerate(["C11", "C12", "C44"]):
        m = cv_metrics(Ye[:, j], pred_e[:, j])
        el_metrics[comp] = m
        print(f"  {comp}: MAE={m['MAE']:.2f} GPa  RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}")
    metrics["elastic"] = {"target": "Cij_GPa", "cv": "GroupKFold(5) by formula",
                          "models": {"RandomForest": el_metrics},
                          "n": int(len(Ye))}

    with open(DATA_DIR / "jarvis_elastic_cv.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["formula", "C11_DFT", "C11_pred", "C12_DFT", "C12_pred", "C44_DFT", "C44_pred"])
        for n, a, p in zip(el_names, Ye, pred_e):
            wr.writerow([n, round(a[0], 2), round(p[0], 2), round(a[1], 2),
                         round(p[1], 2), round(a[2], 2), round(p[2], 2)])

    # ── FIGURES ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].scatter(y_eg, best_pred, s=45, alpha=0.75, edgecolors="k", linewidths=0.4, color="#2196F3")
    lim = [0, max(y_eg.max(), best_pred.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("JARVIS OptB88vdW gap (eV)"); ax[0].set_ylabel("5-fold CV prediction (eV)")
    bm = eg_results[best_name]
    ax[0].set_title(f"Band gap — {best_name}  (n={bm['n']})\nMAE={bm['MAE']} eV  R²={bm['R2']}")

    for j, comp in enumerate(["C11", "C12", "C44"]):
        ax[1].scatter(Ye[:, j], pred_e[:, j], s=40, alpha=0.7, edgecolors="k",
                      linewidths=0.3, label=comp)
    lim2 = [0, Ye.max() * 1.05]
    ax[1].plot(lim2, lim2, "k--", lw=1, alpha=0.5)
    ax[1].set_xlim(lim2); ax[1].set_ylim(lim2)
    ax[1].set_xlabel("JARVIS DFT (GPa)"); ax[1].set_ylabel("5-fold CV prediction (GPa)")
    em = el_metrics["C11"]
    ax[1].set_title(f"Elastic constants — RF  (n={metrics['elastic']['n']})\nC11 R²={em['R2']}")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "jarvis_parity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(DATA_DIR / "jarvis_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 62)
    print("  SUMMARY vs hand-curated 32-point PBE model (MAE 0.16 eV)")
    print("=" * 62)
    print(f"  JARVIS band gap best: {best_name}  MAE={eg_results[best_name]['MAE']} eV"
          f"  R2={eg_results[best_name]['R2']}  (n={eg_results[best_name]['n']})")
    print(f"  JARVIS elastic C11: MAE={el_metrics['C11']['MAE']} GPa"
          f"  R2={el_metrics['C11']['R2']}  (n={metrics['elastic']['n']})")
    print(f"\n  Saved: jarvis_parity.png, jarvis_model_metrics.json,")
    print(f"         jarvis_bandgap_cv.csv, jarvis_elastic_cv.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
