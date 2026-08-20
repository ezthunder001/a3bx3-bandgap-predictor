"""
Step 4 — Train the A3BX3 family band-gap model and VALIDATE against the
held-out Sr3BiX3 (X=I,Br,Cl) targets from the Islam 2026 paper.

This is the payoff of the family-focused strategy. Unlike the original mixed
ABX3 model (which extrapolated to Sr3BiX3 with ~0.9 eV error), the training set
here is the SAME structural family (A3BX3 inverse perovskite, pnictogen B,
halide X), and it contains BOTH Sr (via Sr3PX3) and Bi (via Ca3BiX3, Mg3BiX3) —
so Sr3BiX3 is an INTERPOLATION.

Data : data/a3bx3_literature.csv  (PBE-consistent; 23 trainable + 3 Sr3BiX3 targets)
Target: GGA-PBE band gap (matches the paper's functional)

Outputs:
  data/a3bx3_family_loo.csv      LOO-CV predictions
  data/a3bx3_family_validation.csv  Sr3BiX3 predicted vs paper DFT
  data/a3bx3_family_metrics.json

Run:
    .venv\\Scripts\\python.exe train_a3bx3_family.py
"""
from __future__ import annotations

import sys
import csv
import json
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

DATA = Path(__file__).parent / "data"
SRC = DATA / "a3bx3_literature.csv"

# ── element properties (A = alkaline earth, B = pnictogen, X = halide) ────────
RAD_A = {"Mg": 0.89, "Ca": 1.34, "Sr": 1.44, "Ba": 1.61}          # 12-coord 2+
CHI_A = {"Mg": 1.31, "Ca": 1.00, "Sr": 0.95, "Ba": 0.89}
Z_A   = {"Mg": 12, "Ca": 20, "Sr": 38, "Ba": 56}
M_A   = {"Mg": 24.31, "Ca": 40.08, "Sr": 87.62, "Ba": 137.33}

RAD_B = {"P": 1.07, "As": 1.19, "Sb": 1.39, "Bi": 1.48}           # covalent
CHI_B = {"P": 2.19, "As": 2.18, "Sb": 2.05, "Bi": 2.02}
Z_B   = {"P": 15, "As": 33, "Sb": 51, "Bi": 83}
M_B   = {"P": 30.97, "As": 74.92, "Sb": 121.76, "Bi": 208.98}

RAD_X = {"F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20}            # ionic 1-
CHI_X = {"F": 3.98, "Cl": 3.16, "Br": 2.96, "I": 2.66}
Z_X   = {"F": 9, "Cl": 17, "Br": 35, "I": 53}
M_X   = {"F": 19.0, "Cl": 35.45, "Br": 79.9, "I": 126.9}

FEATURE_NAMES = ["rA", "rB", "rX", "chiA", "chiB", "chiX",
                 "dchi_BX", "dchi_AX", "rB_over_rX"]


def featurize(A, B, X):
    rA, rB, rX = RAD_A[A], RAD_B[B], RAD_X[X]
    cA, cB, cX = CHI_A[A], CHI_B[B], CHI_X[X]
    return np.array([rA, rB, rX, cA, cB, cX, cX - cB, cX - cA, rB / rX])


def make_gpr(nf):
    k = (ConstantKernel(1.0, (1e-2, 1e4)) *
         Matern(length_scale=np.ones(nf), length_scale_bounds=(1e-2, 1e3), nu=2.5) +
         WhiteKernel(0.02, (1e-4, 0.5)))
    return GaussianProcessRegressor(kernel=k, n_restarts_optimizer=8,
                                    normalize_y=True, random_state=42)


def main():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    train, targets = [], []
    for r in rows:
        rec = (r["formula"], r["A"], r["B"], r["X"], float(r["Eg_PBE_eV"]))
        if "TARGET" in r["note"]:
            targets.append(rec)
        else:
            train.append(rec)

    X = np.array([featurize(a, b, x) for _, a, b, x, _ in train])
    y = np.array([eg for *_, eg in train])
    names = [n for n, *_ in train]
    print(f"Family training set: {len(train)} PBE A3BX3 compounds, {X.shape[1]} features")
    print(f"  A-sites: {sorted(set(a for _,a,_,_,_ in train))}")
    print(f"  B-sites: {sorted(set(b for _,_,b,_,_ in train))}")
    print(f"  -> Sr present: {'Sr' in [a for _,a,_,_,_ in train]}, "
          f"Bi present: {'Bi' in [b for _,_,b,_,_ in train]}")

    # ── LOO cross-validation ─────────────────────────────────────────────────
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = make_pipeline(StandardScaler(), make_gpr(X.shape[1]))
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"\n[LOO-CV]  MAE = {mae:.3f} eV   R² = {r2:.3f}   (n={len(y)})")

    with open(DATA / "a3bx3_family_loo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["formula", "Eg_PBE", "Eg_LOO_pred", "residual"])
        for n, a, p in zip(names, y, preds):
            w.writerow([n, round(a, 3), round(p, 3), round(p - a, 3)])

    # ── Leave-one-B-group-out: hold out ALL compounds of one pnictogen at a ───
    # time. Harsher than LOO and closer to the real task (Sr3BiX3 sits in a
    # thinly-anchored Bi corner), so it is the honest extrapolation error bar.
    b_col = [b for _, _, b, _, _ in train]
    lobo_abs, lobo_by_b = [], {}
    for Bel in sorted(set(b_col)):
        te = [i for i, b in enumerate(b_col) if b == Bel]
        tr = [i for i in range(len(y)) if i not in te]
        m = make_pipeline(StandardScaler(), make_gpr(X.shape[1]))
        m.fit(X[tr], y[tr])
        errs = np.abs(m.predict(X[te]) - y[te])
        lobo_abs += list(errs)
        lobo_by_b[Bel] = round(float(errs.mean()), 4)
    lobo_mae = float(np.mean(lobo_abs))
    print(f"[LOBO-CV] leave-one-B-family-out MAE = {lobo_mae:.3f} eV  "
          f"(per B: {lobo_by_b})")

    # ── Fit on all training, predict held-out Sr3BiX3 ────────────────────────
    model = make_pipeline(StandardScaler(), make_gpr(X.shape[1]))
    model.fit(X, y)
    gpr = model.named_steps["gaussianprocessregressor"]
    scaler = model.named_steps["standardscaler"]

    print("\n[VALIDATION]  Held-out Sr3BiX3 vs Islam 2026 DFT (PBE)")
    print(f"  {'Compound':<10} {'ML pred':>9} {'±σ':>7} {'paper DFT':>10} {'error':>8}")
    val_rows = []
    for name, A, B, Xs, dft in targets:
        xf = featurize(A, B, Xs).reshape(1, -1)
        mu, sd = gpr.predict(scaler.transform(xf), return_std=True)
        mu, sd = float(mu[0]), float(sd[0])
        err = mu - dft
        flag = "OK" if abs(err) <= max(2*sd, 0.25) else "off"
        print(f"  {name:<10} {mu:>9.3f} {sd:>7.3f} {dft:>10.3f} {err:>+8.3f}  {flag}")
        val_rows.append({"formula": name, "ML_pred_eV": round(mu, 3),
                         "sigma": round(sd, 3), "paper_DFT_eV": dft,
                         "error_eV": round(err, 3)})

    with open(DATA / "a3bx3_family_validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["formula", "ML_pred_eV", "sigma",
                                          "paper_DFT_eV", "error_eV"])
        w.writeheader(); w.writerows(val_rows)

    val_mae = np.mean([abs(v["error_eV"]) for v in val_rows])
    metrics = {"loo": {"MAE": round(mae, 4), "R2": round(r2, 4), "n": len(y)},
               "lobo": {"MAE": round(lobo_mae, 4), "per_B": lobo_by_b,
                        "cv": "leave-one-B-family-out"},
               "sr3bix3_validation": {"MAE": round(float(val_mae), 4),
                                      "per_compound": val_rows}}

    # ═════════════════════════════════════════════════════════════════════════
    # ELASTIC family model — predict Sr3BiX3 C11/C12 vs Islam DFT
    # ═════════════════════════════════════════════════════════════════════════
    from sklearn.ensemble import RandomForestRegressor

    def load_elastic():
        pts = []  # (name, A, B, X, C11, C12, C44)
        for r in rows:  # literature CSV (PBE)
            if r["A"] in RAD_A and r["B"] in RAD_B and r["C11_GPa"]:
                try:
                    pts.append((r["formula"], r["A"], r["B"], r["X"],
                                float(r["C11_GPa"]), float(r["C12_GPa"]), float(r["C44_GPa"])))
                except ValueError:
                    pass
        jpath = DATA / "a3bx3_jarvis.csv"
        if jpath.exists():
            for r in csv.DictReader(jpath.open(encoding="utf-8")):
                if r["A"] in RAD_A and r["B"] in RAD_B and r.get("C11_GPa"):
                    try:
                        pts.append((r["formula"], r["A"], r["B"], r["X"],
                                    float(r["C11_GPa"]), float(r["C12_GPa"]), float(r["C44_GPa"])))
                    except ValueError:
                        pass
        return pts

    epts = load_elastic()
    print(f"\n[ELASTIC]  family points with C11/C12/C44: {len(epts)}")
    print(f"  {'compound':<11} {'C11':>7}{'C12':>7}{'C44':>7}")
    for n, *_ , c11, c12, c44 in [(p[0],)+p[1:] for p in epts]:
        print(f"  {n:<11} {c11:>7.1f}{c12:>7.1f}{c44:>7.1f}")

    Xe = np.array([featurize(a, b, x) for _, a, b, x, *_ in epts])
    Ye = np.array([[c11, c12, c44] for *_, c11, c12, c44 in [(p[0],)+p[1:] for p in epts]])

    el_metrics = {}
    if len(epts) >= 8:
        loo_e = LeaveOneOut(); pe = np.zeros_like(Ye)
        for tr, te in loo_e.split(Xe):
            rf = RandomForestRegressor(n_estimators=300, random_state=42)
            rf.fit(Xe[tr], Ye[tr]); pe[te] = rf.predict(Xe[te])
        for j, comp in enumerate(["C11", "C12", "C44"]):
            el_metrics[comp] = {"MAE": round(float(mean_absolute_error(Ye[:, j], pe[:, j])), 2),
                                "n": len(epts)}
        print(f"  LOO MAE (GPa): C11={el_metrics['C11']['MAE']} "
              f"C12={el_metrics['C12']['MAE']} C44={el_metrics['C44']['MAE']}")

        # predict Sr3BiX3 elastic, compare to Islam reported (partial)
        rf_full = RandomForestRegressor(n_estimators=400, random_state=42).fit(Xe, Ye)
        ISLAM_EL = {"Sr3BiI3": (60.324, 9.446, None), "Sr3BiBr3": (None, 10.40, None),
                    "Sr3BiCl3": (None, 11.51, None)}

        def moduli(c11, c12, c44):
            B = (c11 + 2*c12) / 3.0
            G = (c11 - c12 + 3*c44) / 5.0           # Voigt
            E = 9*B*G / (3*B + G)
            nu = (3*B - 2*G) / (2*(3*B + G))
            return round(B, 1), round(G, 1), round(E, 1), round(nu, 3)

        print(f"\n  Sr3BiX3 elastic: ML pred vs Islam DFT (GPa) + derived moduli")
        el_val = []
        for name, (A, B, Xs) in [("Sr3BiI3", ("Sr","Bi","I")), ("Sr3BiBr3", ("Sr","Bi","Br")),
                                  ("Sr3BiCl3", ("Sr","Bi","Cl"))]:
            p = rf_full.predict(featurize(A, B, Xs).reshape(1, -1))[0]
            ref = ISLAM_EL[name]
            Bk, Gk, Ek, nuk = moduli(p[0], p[1], p[2])
            print(f"    {name:<10} C11={p[0]:5.1f}(DFT {ref[0]}) "
                  f"C12={p[1]:5.1f}(DFT {ref[1]}) C44={p[2]:5.1f}  B={Bk} G={Gk} E={Ek}")
            el_val.append({"formula": name, "C11_pred": round(float(p[0]), 1),
                           "C12_pred": round(float(p[1]), 1), "C44_pred": round(float(p[2]), 1),
                           "C11_DFT": ref[0], "C12_DFT": ref[1],
                           "B_GPa": Bk, "G_GPa": Gk, "E_GPa": Ek, "poisson": nuk})
        el_metrics["sr3bix3"] = el_val

    metrics["elastic_family"] = el_metrics

    # ── FORWARD PREDICTIONS: Ba3BiX3 (no DFT exists — pure ML) ────────────────
    print("\n[FORWARD]  Ba3BiX3 — pure ML, no DFT reference exists")
    print(f"  {'Compound':<10} {'Eg (eV)':>9} {'±σ':>6}  {'C11':>6}{'C12':>6}{'C44':>6}")
    fwd = []
    rf_full = RandomForestRegressor(n_estimators=400, random_state=42).fit(Xe, Ye) if len(epts) >= 8 else None
    for name, (A, B, Xs) in [("Ba3BiI3", ("Ba","Bi","I")), ("Ba3BiBr3", ("Ba","Bi","Br")),
                              ("Ba3BiCl3", ("Ba","Bi","Cl"))]:
        xf = featurize(A, B, Xs).reshape(1, -1)
        mu, sd = gpr.predict(scaler.transform(xf), return_std=True)
        row = {"formula": name, "Eg_pred_eV": round(float(mu[0]), 3), "sigma": round(float(sd[0]), 3)}
        if rf_full is not None:
            pe = rf_full.predict(xf)[0]
            row.update({"C11": round(float(pe[0]), 1), "C12": round(float(pe[1]), 1),
                        "C44": round(float(pe[2]), 1)})
            print(f"  {name:<10} {mu[0]:>9.3f} {sd[0]:>6.3f}  {pe[0]:>6.1f}{pe[1]:>6.1f}{pe[2]:>6.1f}")
        fwd.append(row)
    metrics["forward_ba3bix3"] = fwd

    json.dump(metrics, open(DATA / "a3bx3_family_metrics.json", "w"), indent=2)

    print("\n" + "=" * 60)
    print("  FAMILY MODEL vs ORIGINAL MIXED MODEL on Sr3BiX3")
    print("=" * 60)
    print(f"  Original 32-pt mixed ABX3 : ~0.85 eV error (all outside 2σ)")
    print(f"  Family A3BX3 model (n={len(y)})  : {val_mae:.3f} eV mean error")
    print(f"  Paper's own 3-point ML    : ~0.08 eV (but overfit/circular)")
    print("=" * 60)


if __name__ == "__main__":
    main()
