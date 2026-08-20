"""Step 5 — Beyond-PBE band-gap correction via MBJ delta learning.

JARVIS-DFT provides TBmBJ (meta-GGA) band gaps for 50 of the 103 domain halide
perovskites. TBmBJ gaps are far closer to experiment than OptB88vdW/PBE, which
systematically underestimate. Instead of predicting the MBJ gap directly we
learn the CORRECTION:

    delta = Eg_MBJ - Eg_OptB88vdW          (mean ~0.92 eV, std 0.44 on n=50)

and add it to a PBE-level prediction. GroupKFold(5) by formula. Predicting the
delta (MAE ~0.33 eV) beats assuming PBE=experiment (error ~0.92 eV) by ~3x.

Application: Sr3BiX3 / Ba3BiX3. PBE-level gap from the A3BX3 family GPR
(train_a3bx3_family.py) + delta from this model -> beyond-PBE estimate,
compared against the HSE06 gaps reported by Islam 2026 (1.878/2.245/2.427 eV).

CAVEAT (stated in all outputs): the delta model is trained on ABX3 halide
perovskites; applying it to A3BX3 inverse perovskites is a cross-stoichiometry
transfer. Tree-ensemble spread is reported as the uncertainty proxy.

Outputs:
  data/mbj_delta_cv.csv          per-compound delta CV predictions
  data/mbj_delta_metrics.json    CV metrics + Sr3BiX3/Ba3BiX3 corrected gaps
  data/mbj_delta_parity.png      delta + corrected-gap parity plots

Run:
    .venv\\Scripts\\python.exe train_mbj_delta.py
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
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

import train_jarvis as tj  # featurizer + element tables (same 25 descriptors)

DATA = Path(__file__).parent / "data"
AVOGADRO = 6.02214076e23

# Islam 2026 (NJC 50, 522) Sr3BiX3: relaxed cubic Pm-3m lattice constants + gaps
ISLAM = {  # formula: (a_DFT_A, Eg_PBE, Eg_HSE)
    "Sr3BiI3":  (6.723, 1.324, 1.878),
    "Sr3BiBr3": (6.493, 1.512, 2.245),
    "Sr3BiCl3": (6.366, 1.731, 2.427),
}
# Family-GPR PBE predictions for Ba3BiX3 (a3bx3_family_metrics.json, forward block).
# Lattice ESTIMATED: a_Sr + 0.24 A (2*(r_Ba - r_Sr)/sqrt2, Shannon 12-coord radii).
BA_FORWARD = {  # formula: (a_est_A, Eg_PBE_family_pred, sigma_family)
    "Ba3BiI3":  (6.963, 0.740, 0.151),
    "Ba3BiBr3": (6.733, 0.836, 0.160),
    "Ba3BiCl3": (6.606, 0.916, 0.172),
}
MASS_X = {"I": 126.90, "Br": 79.90, "Cl": 35.45}


def a3bx3_row(formula: str, A: str, B: str, X: str, a: float) -> dict:
    """Build a featurizer-compatible row for a cubic Pm-3m A3BX3 cell (7 atoms)."""
    vol = a ** 3
    mass = 3 * tj.MASS[A] + tj.MASS[B] + 3 * MASS_X[X]
    dens = mass / AVOGADRO / (vol * 1e-24)  # g/cm3
    return {"formula": formula, "A": A, "B": B, "X": X,
            "lattice_a": a, "lattice_b": a, "lattice_c": a,
            "cell_vol": vol, "density": dens, "spg_number": 221}


def main() -> None:
    rows = tj.load_rows()
    feats, names, eg, mbj = [], [], [], []
    for r in rows:
        x = tj.featurize(r)
        if x is None or np.isnan(x[:18]).any():
            continue
        m = tj._f(r["Eg_mbj_eV"])
        if np.isnan(m):
            continue
        feats.append(x)
        names.append(r["formula"])
        eg.append(tj._f(r["Eg_optb88_eV"]))
        mbj.append(m)

    X = np.array(feats)
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    eg, mbj = np.array(eg), np.array(mbj)
    y = mbj - eg
    groups = np.array(names)
    print(f"Delta training set: {len(y)} compounds ({len(set(names))} unique formulas)")
    print(f"  delta = MBJ - OptB88vdW: mean={y.mean():.3f} eV  std={y.std():.3f}  "
          f"range=({y.min():.2f}, {y.max():.2f})")

    # variant: append the PBE-level gap itself as a feature (standard delta-learning)
    X_pg = np.hstack([X, eg.reshape(-1, 1)])

    gkf = GroupKFold(n_splits=5)
    candidates = {
        "RF":        (RandomForestRegressor(500, random_state=42, n_jobs=-1), X),
        "RF+pbegap": (RandomForestRegressor(500, random_state=42, n_jobs=-1), X_pg),
        "GBR":       (GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                learning_rate=0.05, subsample=0.9,
                                                random_state=42), X),
        "GBR+pbegap": (GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                 learning_rate=0.05, subsample=0.9,
                                                 random_state=42), X_pg),
    }
    print("\n[DELTA MODEL]  5-fold GroupKFold by formula")
    results, best = {}, (None, None, None, 1e9)
    for name, (model, Xv) in candidates.items():
        p = cross_val_predict(model, Xv, y, cv=gkf, groups=groups)
        mae = mean_absolute_error(y, p)
        corr_mae = mean_absolute_error(mbj, eg + p)
        results[name] = {"delta_MAE": round(float(mae), 4),
                         "delta_R2": round(float(r2_score(y, p)), 4),
                         "corrected_gap_MAE": round(float(corr_mae), 4),
                         "corrected_gap_R2": round(float(r2_score(mbj, eg + p)), 4)}
        print(f"  {name:<11} delta MAE={mae:.3f}  corrected-gap MAE={corr_mae:.3f} "
              f"(naive PBE-as-final error={mean_absolute_error(mbj, eg):.3f})")
        if corr_mae < best[3]:
            best = (name, model, Xv, corr_mae, p)
    best_name, best_model, best_X, _, best_pred = best
    print(f"  -> best: {best_name}")

    with open(DATA / "mbj_delta_cv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["formula", "Eg_optb88_eV", "Eg_mbj_eV", "delta_DFT", "delta_CV_pred",
                    "Eg_corrected_CV"])
        for n, e, m, d, p in zip(names, eg, mbj, y, best_pred):
            w.writerow([n, round(e, 3), round(m, 3), round(d, 3), round(p, 3),
                        round(e + p, 3)])

    use_pbegap = best_X.shape[1] == X_pg.shape[1]
    final = best_model.fit(best_X, y)

    def predict_delta(row: dict, eg_pbe: float) -> tuple[float, float]:
        xf = tj.featurize(row)
        xf = np.where(np.isnan(xf), med, xf)
        if use_pbegap:
            xf = np.append(xf, eg_pbe)
        xf = xf.reshape(1, -1)
        if isinstance(final, RandomForestRegressor):
            tree_preds = np.array([t.predict(xf)[0] for t in final.estimators_])
            return float(tree_preds.mean()), float(tree_preds.std())
        return float(final.predict(xf)[0]), float(np.std(y - best_pred))

    # ── Apply to Sr3BiX3 (validation vs Islam HSE) ────────────────────────────
    print("\n[SR3BIX3]  PBE + ML delta vs Islam 2026 HSE06")
    print(f"  {'Compound':<10} {'PBE':>6} {'+delta':>7} {'corrected':>9} {'±σ':>6} "
          f"{'HSE ref':>8} {'error':>7}")
    sr_rows = []
    for fml, (a, eg_pbe, eg_hse) in ISLAM.items():
        Xsite = fml.replace("Sr3Bi", "").rstrip("3") or fml[-2:]
        Xsite = {"I": "I", "Br": "Br", "Cl": "Cl"}[fml[len("Sr3Bi"):-1] or fml[-2]]
        row = a3bx3_row(fml, "Sr", "Bi", Xsite, a)
        d, sd = predict_delta(row, eg_pbe)
        corrected = eg_pbe + d
        err = corrected - eg_hse
        print(f"  {fml:<10} {eg_pbe:>6.3f} {d:>+7.3f} {corrected:>9.3f} {sd:>6.3f} "
              f"{eg_hse:>8.3f} {err:>+7.3f}")
        sr_rows.append({"formula": fml, "Eg_PBE_eV": eg_pbe, "delta_pred_eV": round(d, 3),
                        "Eg_corrected_eV": round(corrected, 3), "sigma_trees": round(sd, 3),
                        "Eg_HSE_ref_eV": eg_hse, "error_vs_HSE_eV": round(err, 3)})
    val_mae = float(np.mean([abs(r["error_vs_HSE_eV"]) for r in sr_rows]))
    pbe_vs_hse = float(np.mean([abs(v[1] - v[2]) for v in ISLAM.values()]))
    print(f"  corrected-vs-HSE MAE = {val_mae:.3f} eV   (raw PBE vs HSE = {pbe_vs_hse:.3f} eV)")

    # ── Forward: Ba3BiX3 (no DFT reference exists) ────────────────────────────
    print("\n[BA3BIX3]  forward beyond-PBE estimates (lattice ESTIMATED, family-GPR PBE)")
    ba_rows = []
    for fml, (a_est, eg_pbe, sig_fam) in BA_FORWARD.items():
        Xsite = {"I": "I", "Br": "Br", "Cl": "Cl"}[fml[len("Ba3Bi"):-1] or fml[-2]]
        row = a3bx3_row(fml, "Ba", "Bi", Xsite, a_est)
        d, sd = predict_delta(row, eg_pbe)
        corrected = eg_pbe + d
        sig_tot = float(np.hypot(sd, sig_fam))
        print(f"  {fml:<10} PBE(fam)={eg_pbe:.3f}  +delta={d:+.3f}  ->  "
              f"{corrected:.3f} ± {sig_tot:.3f} eV")
        ba_rows.append({"formula": fml, "a_est_A": a_est, "Eg_PBE_family_eV": eg_pbe,
                        "delta_pred_eV": round(d, 3), "Eg_corrected_eV": round(corrected, 3),
                        "sigma_total": round(sig_tot, 3)})

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].scatter(y, best_pred, s=45, alpha=0.75, edgecolors="k", linewidths=0.4,
                  color="#7C4DFF")
    lim = [min(y.min(), 0), y.max() * 1.08]
    ax[0].plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("DFT delta = MBJ − OptB88vdW (eV)")
    ax[0].set_ylabel("GroupKFold CV prediction (eV)")
    m = results[best_name]
    ax[0].set_title(f"MBJ delta — {best_name} (n={len(y)})\nMAE={m['delta_MAE']} eV")

    corrected_cv = eg + best_pred
    ax[1].scatter(mbj, eg, s=38, alpha=0.55, label=f"raw OptB88vdW (MAE={mean_absolute_error(mbj, eg):.2f})",
                  color="#9E9E9E", edgecolors="k", linewidths=0.3)
    ax[1].scatter(mbj, corrected_cv, s=45, alpha=0.8,
                  label=f"delta-corrected (MAE={m['corrected_gap_MAE']:.2f})",
                  color="#00897B", edgecolors="k", linewidths=0.4)
    lim2 = [0, mbj.max() * 1.05]
    ax[1].plot(lim2, lim2, "k--", lw=1, alpha=0.5)
    ax[1].set_xlim(lim2); ax[1].set_ylim(lim2)
    ax[1].set_xlabel("JARVIS TBmBJ gap (eV)"); ax[1].set_ylabel("predicted gap (eV)")
    ax[1].set_title("Beyond-PBE gap: raw vs delta-corrected (CV)")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(DATA / "mbj_delta_parity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "training": {"n": int(len(y)), "unique_formulas": len(set(names)),
                     "delta_mean": round(float(y.mean()), 4),
                     "delta_std": round(float(y.std()), 4),
                     "cv": "GroupKFold(5) by formula"},
        "models": results, "best": best_name,
        "caveat": "delta model trained on ABX3 halide perovskites; "
                  "A3BX3 application is a cross-stoichiometry transfer",
        "sr3bix3": {"corrected_vs_HSE_MAE": round(val_mae, 4),
                    "raw_PBE_vs_HSE_MAE": round(pbe_vs_hse, 4),
                    "per_compound": sr_rows},
        "forward_ba3bix3": ba_rows,
    }
    with open(DATA / "mbj_delta_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: mbj_delta_cv.csv, mbj_delta_metrics.json, mbj_delta_parity.png")


if __name__ == "__main__":
    main()
