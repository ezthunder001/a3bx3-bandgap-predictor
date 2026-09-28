"""Week 7: what the models learned, checked against pre-registered physics expectations.

Per outer fold (LOBO and GKF x10 on the primary set; the same folds as metrics_v2):
  * fit the model on the training fold (tree models: nested grid as in models.py)
  * SHAP on that fold's HELD-OUT rows, background = seeded sample of TRAINING rows only
      - gpr_physics9: shap.KernelExplainer, nsamples >= 2^9 - 2 -> exact enumeration
      - rf_magpie / xgb_magpie: shap.TreeExplainer, interventional, same background rule
  * permutation importance on the held-out rows (model-agnostic cross-check); for the GPR
    also site-group permutation (A = rA,chiA; B = rB,chiB; X = rX,chiX)
  * element-substitution effects on the held-out rows: re-predict each row with one site's
    element swapped (features rebuilt from the new composition) -> mean pairwise difference

Expectations and the verdict rule are fixed in reports/explain_expectations.md, which was
committed before this module existed; its commit hash and content hash are recorded.
Outputs: reports/explain_v2.json, reports/explain_v2.md, reports/explain_folds.csv (fold
bookkeeping: background and test rows), reports/figures/explain_*.png
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import warnings
from importlib.metadata import version as _pkg_version

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from . import utf8_stdio
from .config import ROOT, Paths, load_config
from .evaluate import load_dataset
from .features import PHYSICS9, featurize_rows, magpie_labels
from .manifest import sha256_text, write_atomic
from .models import fit, model_specs, unwrap
from .splits import assert_holdout_excluded, check_folds, make_splits

R = 4
SITES = {"A": ["Mg", "Ca", "Sr", "Ba"], "B": ["P", "As", "Sb", "Bi"], "X": ["F", "Cl", "Br", "I"]}
GROUPS = {"A": ["rA", "chiA"], "B": ["rB", "chiB"], "X": ["rX", "chiX"]}

# (site, element_1, element_2, expected sign of gap(e1) - gap(e2), role)
PAIRS = [
    ("X", "F", "Cl", +1, "E1"), ("X", "Cl", "Br", +1, "E1"), ("X", "Br", "I", +1, "E1"),
    ("B", "P", "Bi", +1, "E2 decisive"), ("B", "P", "As", +1, "E2"), ("B", "As", "Sb", +1, "E2"),
    ("B", "Sb", "Bi", +1, "E2"),
    ("A", "Ca", "Ba", +1, "E3 decisive"), ("A", "Ca", "Sr", +1, "E3"), ("A", "Sr", "Ba", +1, "E3"),
    ("A", "Mg", "Ca", 0, "E3 reported"),
]
# (feature, expected slope sign or 0 = reported, expectation id)
SLOPES = [("chiX", +1, "E1"), ("rX", -1, "E1"), ("chiB", +1, "E2"), ("rB", -1, "E2"),
          ("rA", -1, "E3"), ("chiA", 0, "E3 reported"), ("dchi_BX", +1, "E4"),
          ("dchi_AX", 0, "E5"), ("rB_over_rX", 0, "E5")]


def _r(x, d=R):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def _feats(A, B, X, fs, cache):
    key = (A, B, X, fs)
    if key not in cache:
        cache[key] = featurize_rows([{"A": A, "B": B, "X": X}], fs)[0][0]
    return cache[key]


def _final(est):
    return est.best_estimator_ if isinstance(est, GridSearchCV) else est


def _shap_values(name, pipe, bg, Xte, kernel_nsamples):
    import shap
    if name.startswith("gpr"):
        ex = shap.KernelExplainer(pipe.predict, bg)
        return np.asarray(ex.shap_values(Xte, nsamples=kernel_nsamples, silent=True))
    last = pipe.steps[-1][1]
    ex = shap.TreeExplainer(last, data=bg, feature_perturbation="interventional")
    return np.asarray(ex.shap_values(Xte, check_additivity=False))


def _perm(pipe, X, y, cols_groups, repeats, seed):
    """Increase in MAE when the given column groups are permuted (same row permutation for
    all columns of a group). Held-out rows only."""
    rng = np.random.RandomState(seed)
    base = np.mean(np.abs(pipe.predict(X) - y))
    out = []
    for cols in cols_groups:
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            p = rng.permutation(len(X))
            Xp[:, cols] = X[p][:, cols]
            inc.append(np.mean(np.abs(pipe.predict(Xp) - y)) - base)
        out.append(float(np.mean(inc)))
    return np.array(out)


def fold_explain(cfg, spec, rows, X, y, groups, tr, te, seed, fnames):
    ec = cfg["explain"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        est = spec.build(X.shape[1])
        fit(spec, est, X[tr], y[tr], groups[tr])
        pipe = _final(est)
        rng = np.random.RandomState(seed)
        nb = min(ec["background_n"], len(tr))
        bg_idx = np.sort(rng.choice(tr, size=nb, replace=False))
        sv = _shap_values(spec.name, pipe, X[bg_idx], X[te], ec["kernel_nsamples"])
        perm = _perm(pipe, X[te], y[te], [[j] for j in range(X.shape[1])],
                     ec["permutation_repeats"], seed)
        gperm = None
        if spec.feature_set == "physics9":
            gperm = _perm(pipe, X[te], y[te],
                          [[PHYSICS9.index(f) for f in GROUPS[g]] for g in ("A", "B", "X")],
                          ec["permutation_repeats"], seed + 1)
        cache = {}
        subs = {}
        for site, e1, e2, _, _ in PAIRS:
            d = []
            for i in te:
                a = {k: rows[i][k] for k in ("A", "B", "X")}
                a1, a2 = dict(a, **{site: e1}), dict(a, **{site: e2})
                f1 = _feats(a1["A"], a1["B"], a1["X"], spec.feature_set, cache)
                f2 = _feats(a2["A"], a2["B"], a2["X"], spec.feature_set, cache)
                p = pipe.predict(np.vstack([f1, f2]))
                d.append(p[0] - p[1])
            subs[f"{site}:{e1}-{e2}"] = float(np.mean(d))
    return {"shap": sv, "perm": perm, "gperm": gperm, "subs": subs,
            "bg": [int(i) for i in bg_idx], "te": [int(i) for i in te], "n_train": len(tr)}


def _expectations_meta(cfg) -> dict:
    p = ROOT / cfg["explain"]["expectations_file"]
    try:
        h = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(p.relative_to(ROOT))],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        h = None
    return {"file": cfg["explain"]["expectations_file"], "sha256_lf": sha256_text(p),
            "commit": h or None}


def _verdict(vals, sign):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
    if len(vals) == 0:
        return {"verdict": "UNCLEAR", "n": 0}
    m, lo, hi = vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    out = {"mean": _r(m), "p2_5": _r(lo), "p97_5": _r(hi), "n": int(len(vals)),
           "share_expected_sign": _r(np.mean(np.sign(vals) == sign), 3) if sign else None}
    if sign == 0:
        out["verdict"] = "REPORTED"
    elif np.sign(m) == sign and (lo > 0 or hi < 0):
        out["verdict"] = "CONFIRMED"
    elif np.sign(m) == -sign and (lo > 0 or hi < 0):
        out["verdict"] = "CONTRADICTED"
    else:
        out["verdict"] = "UNCLEAR"
    return out


def _slope(x, s):
    if np.std(x) < 1e-12 or len(x) < 3:
        return np.nan
    return float(np.polyfit(x, s, 1)[0])


def run(paths: Paths | None = None, n_jobs: int = -1) -> dict:
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    ec = cfg["explain"]
    rows = load_dataset(paths, ec["dataset"])
    assert_holdout_excluded([r["formula"] for r in rows], set(cfg["data"]["holdout_formulas"]))
    y = np.array([r["Eg_eV"] for r in rows])
    groups = np.array([r["formula"] for r in rows])
    mats = {fs: featurize_rows(rows, fs)[0] for fs in ("physics9", "magpie")}
    names = {"physics9": list(PHYSICS9), "magpie": magpie_labels()}
    specs = {s.name: s for s in model_specs(cfg)}
    splits = make_splits(rows, cfg["splits"], cfg["seed"])
    tasks, keys = [], []
    for sname in ec["splits"]:
        s = splits[sname]
        check_folds(rows, s.folds, set(cfg["data"]["holdout_formulas"]))
        for fi, (tr, te) in enumerate(s.folds):
            for m in ec["models"]:
                sp = specs[m]
                tasks.append(delayed(fold_explain)(cfg, sp, rows, mats[sp.feature_set], y, groups,
                                                   tr, te, cfg["seed"] + 1000 * fi + 7,
                                                   names[sp.feature_set]))
                keys.append((sname, fi, m))
    res = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    R_ = {k: r for k, r in zip(keys, res)}

    out = {"_readme": ("SHAP / permutation importance / element-substitution effects on HELD-OUT "
                       "rows of each outer fold; SHAP backgrounds are training-fold rows only. "
                       "Verdicts follow the pre-registered rule over the 50 GKF fold-models."),
           "dataset": ec["dataset"], "expectations": _expectations_meta(cfg),
           "models": {}, "environment": {p: _pkg_version(p) for p in
                                         ("numpy", "scikit-learn", "shap", "xgboost", "matminer")}}
    book = []
    pooled = {}
    for m in ec["models"]:
        fs = specs[m].feature_set
        fn = names[fs]
        Xm = mats[fs]
        mo = {"feature_set": fs, "explainer": "KernelExplainer (exact)" if m.startswith("gpr")
              else "TreeExplainer (interventional)", "splits": {}}
        for sname in ec["splits"]:
            folds = [(fi, R_[(sname, fi, m)]) for fi in range(len(splits[sname].folds))]
            mabs = np.array([np.abs(r["shap"]).mean(0) for _, r in folds])      # folds x features
            perm = np.array([r["perm"] for _, r in folds])
            mean_abs = mabs.mean(0)
            order = np.argsort(-mean_abs, kind="stable")
            top = [fn[j] for j in order[:10]]
            ranks = np.argsort(np.argsort(-mabs, axis=1, kind="stable"), axis=1, kind="stable")
            taus = [kendalltau(mabs[a], mabs[b])[0] for a in range(len(folds))
                    for b in range(a + 1, len(folds))]
            topk = order[:10]
            taus_top = [kendalltau(mabs[a, topk], mabs[b, topk])[0] for a in range(len(folds))
                        for b in range(a + 1, len(folds))]
            perm_mean = perm.mean(0)
            so = {
                "n_folds": len(folds),
                "mean_abs_shap": {fn[j]: {"mean": _r(mean_abs[j]), "p2_5": _r(np.percentile(mabs[:, j], 2.5)),
                                          "p97_5": _r(np.percentile(mabs[:, j], 97.5)),
                                          "mean_rank": _r(ranks[:, j].mean() + 1, 2)}
                                  for j in order[:15]},
                "top10": top,
                "rank_stability": {"kendall_tau_all_features": {"mean": _r(np.nanmean(taus), 3),
                                                                "min": _r(np.nanmin(taus), 3)},
                                   "kendall_tau_top10": {"mean": _r(np.nanmean(taus_top), 3),
                                                         "min": _r(np.nanmin(taus_top), 3)},
                                   "top3_same_as_aggregate_share": _r(np.mean(
                                       [set(np.argsort(-row)[:3]) == set(order[:3]) for row in mabs]), 3)},
                "permutation_importance_top10": {fn[j]: _r(perm_mean[j])
                                                 for j in np.argsort(-perm_mean, kind="stable")[:10]},
                "shap_vs_permutation_kendall_tau": _r(kendalltau(mean_abs, perm_mean)[0], 3),
                "substitution": {f"{site}:{e1}-{e2}": {**_verdict([r["subs"][f"{site}:{e1}-{e2}"] for _, r in folds],
                                                                 sign if sname == "gkf" else 0),
                                                       "expected_sign": sign, "role": role}
                                 for site, e1, e2, sign, role in PAIRS},
            }
            if sname == "lobo":
                for k, v in so["substitution"].items():
                    vals = [r["subs"][k] for _, r in folds]
                    v.update({"min": _r(min(vals)), "max": _r(max(vals)),
                              "verdict": "SUPPLEMENT (4 folds, no verdict)"})
            if fs == "physics9":
                slopes = {}
                for f, sign, eid in SLOPES:
                    j = PHYSICS9.index(f)
                    vals = [_slope(Xm[r["te"], j], r["shap"][:, j]) for _, r in folds]
                    v = _verdict(vals, sign if sname == "gkf" else 0)
                    v.update({"expected_sign": sign, "expectation": eid,
                              "n_folds_constant_feature": int(sum(not np.isfinite(x) for x in vals))})
                    slopes[f] = v
                so["shap_slopes"] = slopes
                gs = {g: np.array([np.abs(r["shap"][:, [PHYSICS9.index(f) for f in GROUPS[g]]]).sum(1).mean()
                                   for _, r in folds]) for g in GROUPS}
                gp = np.array([r["gperm"] for _, r in folds])  # folds x (A,B,X)
                sg = 1 if sname == "gkf" else 0
                so["site_groups"] = {
                    "shap_A_minus_B": _verdict(gs["A"] - gs["B"], sg),
                    "shap_X_minus_B": _verdict(gs["X"] - gs["B"], sg),
                    "perm_A_minus_B": _verdict(gp[:, 0] - gp[:, 1], sg),
                    "perm_X_minus_B": _verdict(gp[:, 2] - gp[:, 1], sg),
                    "mean_group_abs_shap": {g: _r(v.mean()) for g, v in gs.items()},
                    "mean_group_permutation": {g: _r(gp[:, k].mean()) for k, g in enumerate("ABX")}}
            mo["splits"][sname] = so
            if sname == "gkf":
                sel = [(fi, r) for fi, r in folds if fi < cfg["splits"]["gkf"]["n_splits"]]  # repeat 0
                pooled[m] = (np.vstack([r["shap"] for _, r in sel]),
                             np.vstack([Xm[r["te"]] for _, r in sel]),
                             [rows[i]["B"] for _, r in sel for i in r["te"]], fn, mabs, order)
            for fi, r in folds:
                for i in r["te"]:
                    book.append({"model": m, "split": sname, "fold": fi, "role": "test",
                                 "formula": rows[i]["formula"]})
                for i in r["bg"]:
                    book.append({"model": m, "split": sname, "fold": fi, "role": "background",
                                 "formula": rows[i]["formula"]})
        out["models"][m] = mo
    out["physics_check"] = physics_verdicts(out)
    write_atomic(paths.reports / "explain_v2.json",
                 (json.dumps(out, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["model", "split", "fold", "role", "formula"], lineterminator="\n")
    w.writeheader()
    w.writerows(book)
    write_atomic(paths.reports / "explain_folds.csv", buf.getvalue().encode("utf-8"))
    _figures(paths, out, pooled)
    _markdown(paths, out)
    return out


def physics_verdicts(out) -> dict:
    """Roll the GKF results up into one verdict per pre-registered expectation, per model."""
    res = {}
    for m, mo in out["models"].items():
        g = mo["splits"]["gkf"]
        sub = g["substitution"]

        def comb(keys):
            v = [sub[k]["verdict"] for k in keys]
            if any(x == "CONTRADICTED" for x in v):
                return "CONTRADICTED"
            return "CONFIRMED" if all(x == "CONFIRMED" for x in v) else "UNCLEAR"
        r = {"E1 halide I<Br<Cl<F": {"verdict": comb(["X:F-Cl", "X:Cl-Br", "X:Br-I"]),
                                     "pairs": {k: sub[k]["verdict"] for k in ("X:F-Cl", "X:Cl-Br", "X:Br-I")}},
             "E2 pnictogen P>...>Bi (decisive P-Bi)": {"verdict": sub["B:P-Bi"]["verdict"],
                                                       "pairs": {k: sub[k]["verdict"] for k in
                                                                 ("B:P-As", "B:As-Sb", "B:Sb-Bi", "B:P-Bi")}},
             "E3 A-site Ca>Sr>Ba (decisive Ca-Ba)": {"verdict": sub["A:Ca-Ba"]["verdict"],
                                                     "pairs": {k: sub[k]["verdict"] for k in
                                                               ("A:Ca-Sr", "A:Sr-Ba", "A:Ca-Ba")},
                                                     "Mg-Ca (reported)": sub["A:Mg-Ca"]["mean"]}}
        if "shap_slopes" in g:
            s = g["shap_slopes"]
            r["E1 secondary SHAP slopes"] = {"chiX>0": s["chiX"]["verdict"], "rX<0": s["rX"]["verdict"]}
            r["E2 secondary SHAP slopes"] = {"chiB>0": s["chiB"]["verdict"], "rB<0": s["rB"]["verdict"]}
            r["E3 secondary SHAP slopes"] = {"rA<0": s["rA"]["verdict"], "chiA (reported)": s["chiA"]["mean"]}
            r["E4 dchi_BX slope>0"] = {"verdict": s["dchi_BX"]["verdict"]}
            sg = g["site_groups"]
            v = [sg["shap_A_minus_B"]["verdict"], sg["shap_X_minus_B"]["verdict"]]
            r["E6 A and X groups > B group (SHAP)"] = {
                "verdict": "CONTRADICTED" if "CONTRADICTED" in v else ("CONFIRMED" if v == ["CONFIRMED"] * 2 else "UNCLEAR"),
                "A-B": sg["shap_A_minus_B"]["verdict"], "X-B": sg["shap_X_minus_B"]["verdict"],
                "permutation cross-check": {"A-B": sg["perm_A_minus_B"]["verdict"],
                                            "X-B": sg["perm_X_minus_B"]["verdict"]}}
        res[m] = r
    return res


def _figures(paths, out, pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    for m, (sv, X, Bs, fn, mabs, order) in pooled.items():
        plt.figure()
        shap.summary_plot(sv, X, feature_names=fn, max_display=10, show=False, plot_size=(6.5, 4.2))
        plt.title(f"{m}: SHAP on held-out rows (GKF repeat 0)", fontsize=9)
        plt.tight_layout()
        plt.savefig(paths.figures / f"explain_beeswarm_{m}.png", dpi=150, metadata={"Software": None})
        plt.close("all")

        k = order[:10]
        fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
        mean = mabs[:, k].mean(0)
        lo = np.clip(mean - np.percentile(mabs[:, k], 2.5, axis=0), 0, None)
        hi = np.clip(np.percentile(mabs[:, k], 97.5, axis=0) - mean, 0, None)
        ax.barh(np.arange(len(k))[::-1], mean, xerr=[lo, hi], color="#2166ac", capsize=2)
        ax.set_yticks(np.arange(len(k))[::-1])
        ax.set_yticklabels([fn[j] for j in k], fontsize=7)
        ax.set_xlabel("mean |SHAP| on held-out rows (eV); bars = 2.5–97.5 % across 50 GKF folds")
        ax.set_title(f"{m}", fontsize=9)
        fig.tight_layout()
        fig.savefig(paths.figures / f"explain_bar_{m}.png", metadata={"Software": None})
        plt.close(fig)

    sv, X, Bs, fn, mabs, order = pooled["gpr_physics9"]
    ranks = np.argsort(np.argsort(-mabs, axis=1, kind="stable"), axis=1, kind="stable") + 1
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)
    ax.boxplot([ranks[:, j] for j in order], tick_labels=[fn[j] for j in order])
    ax.set_ylabel("rank of mean |SHAP| in fold (1 = top)")
    ax.set_title("gpr_physics9: rank stability across 50 GKF folds", fontsize=9)
    ax.invert_yaxis()
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.figures / "explain_rank_stability_gpr_physics9.png", metadata={"Software": None})
    plt.close(fig)

    colors = {"P": "#1b7837", "As": "#762a83", "Sb": "#e08214", "Bi": "#2166ac"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=150)
    for ax, j in zip(axes, order[:3]):
        for b in colors:
            idx = [i for i, bb in enumerate(Bs) if bb == b]
            ax.scatter(X[idx, j], sv[idx, j], s=14, color=colors[b], label=f"B={b}")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_xlabel(fn[j])
        ax.set_ylabel("SHAP (eV)")
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle("gpr_physics9 dependence, top-3 features, held-out rows (GKF repeat 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths.figures / "explain_dependence_gpr_physics9.png", metadata={"Software": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=150)
    keys = [f"{s}:{a}-{b}" for s, a, b, _, _ in PAIRS]
    ms = list(out["models"])
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        sub = out["models"][m]["splits"]["gkf"]["substitution"]
        mean = [sub[k]["mean"] for k in keys]
        lo = [max(0.0, sub[k]["mean"] - sub[k]["p2_5"]) for k in keys]
        hi = [max(0.0, sub[k]["p97_5"] - sub[k]["mean"]) for k in keys]
        ax.bar(np.arange(len(keys)) + i * w, mean, w, yerr=[lo, hi], capsize=1.5, label=m)
    ax.axhline(0, color="#555", lw=0.6)
    ax.set_xticks(np.arange(len(keys)) + 0.4 - w / 2)
    ax.set_xticklabels(keys, rotation=30, fontsize=7)
    ax.set_ylabel("Δ predicted gap (eV)")
    ax.set_title("Element-substitution effect on held-out rows (50 GKF folds; 2.5–97.5 %)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "explain_substitution.png", metadata={"Software": None})
    plt.close(fig)


def _markdown(paths, out):
    e = out["expectations"]
    L = ["# explain_v2 — generated by `a3bx3.explain`; do not edit by hand", "", out["_readme"], "",
         f"Pre-registered expectations: `{e['file']}` (commit `{e['commit']}`, sha256 {e['sha256_lf'][:12]}…).", "",
         "## Physics check (verdicts over 50 GKF fold-models)", ""]
    for m, r in out["physics_check"].items():
        L += [f"### {m}", "", "| expectation | verdict | detail |", "|---|---|---|"]
        for k, v in r.items():
            verdict = v.get("verdict", "—") if isinstance(v, dict) else v
            detail = {kk: vv for kk, vv in v.items() if kk != "verdict"} if isinstance(v, dict) else ""
            L.append(f"| {k} | {verdict} | {json.dumps(detail)} |")
        L.append("")
    L += ["## Element-substitution effects (eV, mean [2.5, 97.5 %] over GKF folds; LOBO min–max)", "",
          "| pair | expected | " + " | ".join(out["models"]) + " |", "|---|---|" + "---|" * len(out["models"])]
    for s, a, b, sign, role in PAIRS:
        k = f"{s}:{a}-{b}"
        cells = []
        for m in out["models"]:
            g = out["models"][m]["splits"]["gkf"]["substitution"][k]
            lb = out["models"][m]["splits"]["lobo"]["substitution"][k]
            cells.append(f"{g['mean']:+.3f} [{g['p2_5']:+.3f}, {g['p97_5']:+.3f}] {g['verdict']}; "
                         f"LOBO {lb['min']:+.2f}…{lb['max']:+.2f}")
        L.append(f"| {k} ({role}) | {'+' if sign > 0 else 'none'} | " + " | ".join(cells) + " |")
    L.append("")
    for m, mo in out["models"].items():
        L += [f"## {m} ({mo['explainer']})", ""]
        for s, so in mo["splits"].items():
            rs = so["rank_stability"]
            L += [f"### {s} ({so['n_folds']} folds)", "",
                  f"Rank stability: Kendall τ all features mean {rs['kendall_tau_all_features']['mean']} "
                  f"(min {rs['kendall_tau_all_features']['min']}); top-10 mean {rs['kendall_tau_top10']['mean']}; "
                  f"top-3 set equal to aggregate in {rs['top3_same_as_aggregate_share']} of folds. "
                  f"SHAP vs permutation-importance τ = {so['shap_vs_permutation_kendall_tau']}.", "",
                  "| feature | mean |SHAP| [2.5, 97.5 %] | mean rank |", "|---|---|---|"]
            for f, v in list(so["mean_abs_shap"].items())[:8]:
                L.append(f"| {f} | {v['mean']:.3f} [{v['p2_5']:.3f}, {v['p97_5']:.3f}] | {v['mean_rank']} |")
            L += ["", "Permutation importance (ΔMAE, eV) top: " +
                  ", ".join(f"{k} {v:.3f}" for k, v in list(so["permutation_importance_top10"].items())[:5]), ""]
            if "shap_slopes" in so:
                L += ["| feature | SHAP slope mean [2.5, 97.5 %] | expected | verdict | folds w/ constant feature |",
                      "|---|---|---|---|---|"]
                for f, v in so["shap_slopes"].items():
                    L.append(f"| {f} | {v.get('mean')} [{v.get('p2_5')}, {v.get('p97_5')}] | "
                             f"{v['expected_sign']} | {v['verdict']} | {v['n_folds_constant_feature']} |")
                sg = so["site_groups"]
                L += ["", f"Site groups mean Σ|SHAP|: {sg['mean_group_abs_shap']}; permutation: "
                      f"{sg['mean_group_permutation']}", ""]
    write_atomic(paths.reports / "explain_v2.md", "\n".join(L).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    out = run()
    for m, r in out["physics_check"].items():
        print(m, {k: (v["verdict"] if isinstance(v, dict) and "verdict" in v else v) for k, v in r.items()})
        print("   top5:", out["models"][m]["splits"]["gkf"]["top10"][:5],
              "tau:", out["models"][m]["splits"]["gkf"]["rank_stability"]["kendall_tau_all_features"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
