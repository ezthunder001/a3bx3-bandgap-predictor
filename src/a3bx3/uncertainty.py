"""Week 6: predictive intervals and their held-out coverage.

For every dataset x outer split x outer fold (the same folds as metrics_v2), everything below
is fitted on the outer TRAINING fold only; the outer test fold is predicted once and scored.

Methods (interval at nominal level p):
    gpr_sigma            mu +/- z_p * sigma, the GPR's own predictive sd (no recalibration)
    gpr_sigma_scaled     mu +/- z_p * c * sigma; c = RMS of inner-CV standardised residuals
                         r / sigma, from an inner GroupKFold on the training fold only
    <model>_cvplus       CV+ (Barber et al. 2021) via MAPIE CrossConformalRegressor,
                         method="plus", inner GroupKFold(5) by formula on the training fold
    <model>_jackknife_plus   the same with LeaveOneGroupOut (jackknife+)
  conformal models: gpr_physics9 and the named baselines ridge_physics9 and mean, so widths
  are compared at (nominally) equal coverage.

Guarantees (stated in the report, not assumed): CV+/jackknife+ give marginal coverage
>= 1 - 2*alpha (typically ~ 1 - alpha) only if test and training rows are exchangeable. That
holds for the shuffled grouped KFold (and approximately for LOO); it is VIOLATED by LOBO/LOAO,
where the test fold is a whole chemistry absent from training. Coverage there is reported as
measured, with no guarantee.

Also: K2 kill checkpoint from the committed metrics_v2 out-of-fold predictions, and the
locked Sr3BiX3 holdout intervals from models fit on the full primary set.
Outputs: reports/uncertainty_v2.json, reports/uncertainty_v2.md,
reports/uncertainty_folds.csv (fold bookkeeping), reports/figures/uncertainty_*.png
"""
from __future__ import annotations

import csv
import io
import json
import sys
import warnings
from importlib.metadata import version as _pkg_version

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from . import utf8_stdio
from .config import Paths, load_config
from .evaluate import load_dataset, _read_oof
from .features import featurize_rows, read_csv
from .manifest import write_atomic
from .models import model_specs
from .splits import assert_holdout_excluded, check_folds, make_splits

R = 4


def _r(x, d=R):
    return None if x is None else round(float(x), d)


def method_names(cfg) -> list[str]:
    out = ["gpr_sigma", "gpr_sigma_scaled"]
    for m in cfg["uncertainty"]["conformal_models"]:
        out += [f"{m}_cvplus", f"{m}_jackknife_plus"]
    return out


def _specs(cfg):
    return {s.name: s for s in model_specs(cfg)}


def inner_splitter(cfg, seed):
    return GroupKFold(cfg["uncertainty"]["inner_n_splits"], shuffle=True, random_state=seed)


def sigma_scale(cfg, X, y, groups, seed) -> tuple[float, list[int]]:
    """c = sqrt(mean((r_i / sigma_i)^2)) over inner-CV out-of-fold predictions of the GPR.
    Uses ONLY the rows passed in (the outer training fold). Returns c and the row indices
    (relative to the rows passed) that entered the fit, for fold bookkeeping."""
    spec = _specs(cfg)["gpr_physics9"]
    z, used = [], []
    for itr, ite in inner_splitter(cfg, seed).split(X, y, groups):
        est = spec.build(X.shape[1]).fit(X[itr], y[itr])
        mu, sd = est.predict(X[ite], return_std=True)
        z += list((y[ite] - mu) / sd)
        used += list(ite)
    return float(np.sqrt(np.mean(np.square(z)))), sorted(set(int(i) for i in used))


def _conformal(cfg, model, X, y, groups, Xte, levels, kind, seed):
    from mapie.regression import CrossConformalRegressor
    spec = _specs(cfg)[model]
    cv = inner_splitter(cfg, seed) if kind == "cvplus" else LeaveOneGroupOut()
    reg = CrossConformalRegressor(spec.build(X.shape[1]), confidence_level=list(levels),
                                  conformity_score="absolute", method="plus", cv=cv,
                                  n_jobs=1, random_state=seed)
    reg.fit_conformalize(X, y, groups=groups)
    point, iv = reg.predict_interval(Xte)
    return np.asarray(point), np.asarray(iv)        # iv: (n_test, 2, n_levels)


def fold_intervals(cfg, X, y, groups, tr, te, levels, seed) -> dict:
    """All methods for one outer fold. Only X[tr], y[tr] are used to fit anything."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        Xtr, ytr, gtr, Xte = X[tr], y[tr], groups[tr], X[te]
        gpr = _specs(cfg)["gpr_physics9"].build(X.shape[1]).fit(Xtr, ytr)
        mu, sd = gpr.predict(Xte, return_std=True)
        c, used = sigma_scale(cfg, Xtr, ytr, gtr, seed)
        z = norm.ppf((1 + np.asarray(levels)) / 2)
        out = {"gpr_sigma": (mu[:, None] - z * sd[:, None], mu[:, None] + z * sd[:, None]),
               "gpr_sigma_scaled": (mu[:, None] - z * c * sd[:, None],
                                    mu[:, None] + z * c * sd[:, None])}
        for m in cfg["uncertainty"]["conformal_models"]:
            for kind in ("cvplus", "jackknife_plus"):
                _, iv = _conformal(cfg, m, Xtr, ytr, gtr, Xte, levels, kind, seed)
                out[f"{m}_{kind}"] = (iv[:, 0, :], iv[:, 1, :])
    return {"intervals": out, "mu": mu, "sigma": sd, "c": c,
            "calib_rows": [int(tr[i]) for i in used], "n_train": len(tr)}


def _plan(cfg, paths):
    u = cfg["uncertainty"]
    levels = u["curve_levels"]
    holdout = set(cfg["data"]["holdout_formulas"])
    tasks, keys, data = [], [], {}
    for ds in u["datasets"]:
        rows = load_dataset(paths, ds)
        assert_holdout_excluded([r["formula"] for r in rows], holdout)
        X = featurize_rows(rows, "physics9")[0]
        y = np.array([r["Eg_eV"] for r in rows])
        groups = np.array([r["formula"] for r in rows])
        splits = make_splits(rows, cfg["splits"], cfg["seed"])
        data[ds] = (rows, y)
        for sname in u["splits"]:
            s = splits[sname]
            check_folds(rows, s.folds, holdout)
            per_rep = len(s.folds) // s.n_repeats
            for fi, (tr, te) in enumerate(s.folds):
                tasks.append(delayed(fold_intervals)(cfg, X, y, groups, tr, te, levels,
                                                     cfg["seed"] + fi))
                keys.append((ds, sname, fi, fi // per_rep, tr, te))
    return tasks, keys, data, levels


def _coverage_block(y, lo, hi, formulas_idx, n_rows, boot, levels, report_levels):
    """lo/hi: (n_pred, n_levels). Coverage and width per level; bootstrap CI over rows
    (each row's coverage averaged over its repeats first)."""
    cov = (y[:, None] >= lo) & (y[:, None] <= hi)
    width = hi - lo
    out = {}
    for j, p in enumerate(levels):
        per_row = np.zeros(n_rows)
        cnt = np.zeros(n_rows)
        np.add.at(per_row, formulas_idx, cov[:, j])
        np.add.at(cnt, formulas_idx, 1)
        per_row /= cnt
        b = per_row[boot].mean(1)
        finite = np.isfinite(width[:, j])
        e = {"coverage": _r(cov[:, j].mean()), "mean_width": _r(width[finite, j].mean()),
             "median_width": _r(np.median(width[finite, j])),
             "n_unbounded": int((~finite).sum())}
        if p in report_levels:
            e["coverage_CI95"] = [_r(np.percentile(b, 2.5)), _r(np.percentile(b, 97.5))]
        out[f"{p:.2f}"] = e
    return out


def run(paths: Paths | None = None, n_jobs: int = -1) -> dict:
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    u = cfg["uncertainty"]
    tasks, keys, data, levels = _plan(cfg, paths)
    results = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

    methods = method_names(cfg)
    rng = np.random.RandomState(cfg["seed"])
    boots = {ds: rng.randint(0, len(data[ds][0]), size=(u["bootstrap_n"], len(data[ds][0])))
             for ds in u["datasets"]}
    book, pit, cov_out, cfac, iv90 = [], {}, {}, {}, []
    j90 = levels.index(0.9)
    for ds in u["datasets"]:
        rows, y = data[ds]
        cov_out[ds] = {}
        for sname in u["splits"]:
            sel = [(k, r) for k, r in zip(keys, results) if k[0] == ds and k[1] == sname]
            te_all = np.concatenate([k[5] for k, _ in sel])
            yt = y[te_all]
            per_m = {}
            for m in methods:
                lo = np.vstack([r["intervals"][m][0] for _, r in sel])
                hi = np.vstack([r["intervals"][m][1] for _, r in sel])
                per_m[m] = _coverage_block(yt, lo, hi, te_all, len(rows), boots[ds], levels,
                                           u["report_levels"])
                if m in ("gpr_sigma", "gpr_sigma_scaled", "gpr_physics9_cvplus",
                         "ridge_physics9_cvplus", "mean_cvplus"):
                    for (k, r) in sel:
                        lo_m, hi_m = r["intervals"][m]
                        for jj, i in enumerate(k[5]):
                            iv90.append({"dataset": ds, "split": sname, "fold": k[2],
                                         "repeat": k[3], "method": m,
                                         "formula": rows[i]["formula"], "y_true": y[i],
                                         "lower_90": _r(lo_m[jj, j90], 5),
                                         "upper_90": _r(hi_m[jj, j90], 5)})
            mu = np.concatenate([r["mu"] for _, r in sel])
            sd = np.concatenate([r["sigma"] for _, r in sel])
            cs = np.array([r["c"] for _, r in sel])
            cvec = np.concatenate([np.full(len(k[5]), r["c"]) for k, r in sel])
            pit[(ds, sname)] = (norm.cdf((yt - mu) / sd), norm.cdf((yt - mu) / (cvec * sd)))
            cfac[(ds, sname)] = cs
            cov_out[ds][sname] = {
                "n_folds": len(sel), "n_predictions": int(len(te_all)),
                "min_n_train": int(min(r["n_train"] for _, r in sel)),
                "sigma_scale_c": {"mean": _r(cs.mean()), "min": _r(cs.min()), "max": _r(cs.max())},
                "methods": per_m}
            for k, r in sel:
                calib = {rows[i]["formula"] for i in r["calib_rows"]}
                for i in k[5]:
                    book.append({"dataset": ds, "split": sname, "fold": k[2], "role": "test",
                                 "formula": rows[i]["formula"]})
                for f in sorted(calib):
                    book.append({"dataset": ds, "split": sname, "fold": k[2], "role": "calibration",
                                 "formula": f})

    exch = {"lobo": "violated (whole B-family held out)",
            "loao": "violated (whole A-site held out)",
            "gkf": "holds: shuffled grouped KFold, test formulas a random subset",
            "loo": "approximately holds: each formula left out once"}
    out = {
        "_readme": ("Held-out coverage of predictive intervals. Every interval is fitted on the outer "
                    "training fold only (inner grouped CV); the outer test fold is never used to "
                    "calibrate. Same outer folds as metrics_v2. Coverage = share of held-out "
                    "predictions inside the interval; width in eV."),
        "levels": levels, "report_levels": u["report_levels"],
        "methods": {
            "gpr_sigma": "GPR mu +/- z*sigma (no recalibration)",
            "gpr_sigma_scaled": "GPR mu +/- z*c*sigma, c = RMS inner-CV standardised residual (training fold only)",
            **{f"{m}_cvplus": f"CV+ around {m} (MAPIE CrossConformalRegressor, GroupKFold(5) inner)"
               for m in u["conformal_models"]},
            **{f"{m}_jackknife_plus": f"jackknife+ around {m} (MAPIE, LeaveOneGroupOut inner)"
               for m in u["conformal_models"]}},
        "exchangeability": exch,
        "guarantee": ("CV+ / jackknife+: marginal coverage >= 1 - 2 alpha under exchangeability "
                      "(Barber et al., Ann. Stat. 2021). No guarantee on lobo/loao; conditional "
                      "(per-compound) coverage is never guaranteed."),
        "coverage": cov_out,
        "k2": k2(cfg, paths),
        "sr3bix3_holdout": holdout_intervals(cfg, paths),
        "environment": {p: _pkg_version(p) for p in
                        ("numpy", "scipy", "scikit-learn", "mapie", "xgboost", "matminer")},
    }
    write_atomic(paths.reports / "uncertainty_v2.json",
                 (json.dumps(out, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _write_csv(paths.reports / "uncertainty_folds.csv", book,
               ["dataset", "split", "fold", "role", "formula"])
    _write_csv(paths.reports / "uncertainty_intervals_90.csv", iv90,
               ["dataset", "split", "fold", "repeat", "method", "formula", "y_true",
                "lower_90", "upper_90"])
    _figures(paths, out, pit, cfac)
    _markdown(paths, out)
    return out


def _write_csv(path, rows, cols):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    write_atomic(path, buf.getvalue().encode("utf-8"))


# ── K2 ───────────────────────────────────────────────────────────────────────
def k2(cfg, paths) -> dict:
    """From the committed metrics_v2 OOF predictions: skill of the best model vs the mean
    baseline and vs the best baseline on the K2 split, with paired bootstrap CIs."""
    kc = cfg["uncertainty"]["k2"]
    oof = _read_oof(paths)
    specs = model_specs(cfg)
    roles = {s.name: s.role for s in specs}
    rng = np.random.RandomState(cfg["seed"] + 7)
    out = {"rule": (f"PASS iff the best model's {kc['split']} MAE is >= {kc['min_skill']:.0%} below "
                    "the reference baseline AND the paired 95 % CI of the MAE difference excludes 0. "
                    "Design wording: reference = mean baseline. Stricter: reference = best baseline."),
           "datasets": {}}
    for ds in kc["datasets"]:
        fam = load_dataset(paths, ds)
        formulas = [r["formula"] for r in fam]
        y = np.array([r["Eg_eV"] for r in fam])
        err = {}
        for m in roles:
            P = {r["formula"]: r["y_pred"] for r in oof if r["dataset"] == ds
                 and r["split"] == kc["split"] and r["model"] == m}
            err[m] = np.abs(np.array([P[f] for f in formulas]) - y)
        boot = rng.randint(0, len(fam), size=(cfg["uncertainty"]["bootstrap_n"], len(fam)))
        best_model = min((m for m in roles if roles[m] == "model"), key=lambda m: (err[m].mean(), m))
        best_base = min((m for m in roles if roles[m] == "baseline"), key=lambda m: (err[m].mean(), m))
        res = {"best_model": best_model, "best_model_MAE": _r(err[best_model].mean()),
               "best_baseline": best_base}
        for label, ref in (("vs_mean", "mean"), ("vs_best_baseline", best_base)):
            a, b = err[best_model][boot].mean(1), err[ref][boot].mean(1)
            skill_b = 1 - a / b
            diff = a - b
            skill = 1 - err[best_model].mean() / err[ref].mean()
            ci = [_r(np.percentile(diff, 2.5)), _r(np.percentile(diff, 97.5))]
            res[label] = {"reference": ref, "reference_MAE": _r(err[ref].mean()),
                          "skill": _r(skill, 3),
                          "skill_CI95": [_r(np.percentile(skill_b, 2.5), 3), _r(np.percentile(skill_b, 97.5), 3)],
                          "MAE_diff_CI95": ci,
                          "pass": bool(skill >= kc["min_skill"] and ci[1] < 0)}
        res["verdict"] = "PASS" if res["vs_mean"]["pass"] and res["vs_best_baseline"]["pass"] else (
            "PASS vs mean only" if res["vs_mean"]["pass"] else "FAIL")
        out["datasets"][ds] = res
    return out


# ── holdout ──────────────────────────────────────────────────────────────────
def holdout_intervals(cfg, paths) -> dict:
    ds = cfg["uncertainty"]["holdout_dataset"]
    fam = load_dataset(paths, ds)
    hold = read_csv(paths.processed / "holdout_sr3bix3.csv")
    assert_holdout_excluded([r["formula"] for r in fam], set(cfg["data"]["holdout_formulas"]))
    X = featurize_rows(fam, "physics9")[0]
    Xh = featurize_rows(hold, "physics9")[0]
    y = np.array([r["Eg_eV"] for r in fam])
    groups = np.array([r["formula"] for r in fam])
    levels = cfg["uncertainty"]["report_levels"]
    idx = np.arange(len(fam))
    r = fold_intervals(cfg, np.vstack([X, Xh]), np.concatenate([y, np.full(len(hold), np.nan)]),
                       np.concatenate([groups, [h["formula"] for h in hold]]),
                       idx, np.arange(len(fam), len(fam) + len(hold)), levels, cfg["seed"])
    out = {"dataset": ds, "sigma_scale_c": _r(r["c"]),
           "literature_Sr3BiI3_PBE": {"target (Islam 2026)": 1.324, "CASTEP 10.1039/d5ra07289a": 1.164,
                                      "WIEN2k 10.1088/1361-6641/ada17e": 1.30},
           "compounds": {}}
    for j, h in enumerate(hold):
        e = {"target": h["Eg_eV"], "gpr_mu": _r(r["mu"][j]), "gpr_sigma": _r(r["sigma"][j]),
             "intervals": {}}
        for m, (lo, hi) in r["intervals"].items():
            e["intervals"][m] = {f"{p:.2f}": {"lower": _r(lo[j, k]), "upper": _r(hi[j, k]),
                                              "contains_target": bool(lo[j, k] <= h["Eg_eV"] <= hi[j, k])}
                                 for k, p in enumerate(levels)}
        out["compounds"][h["formula"]] = e
    out["inside_90_count"] = {m: sum(out["compounds"][f]["intervals"][m]["0.90"]["contains_target"]
                                     for f in out["compounds"]) for m in method_names(cfg)}
    lit = out["compounds"].get("Sr3BiI3")
    if lit:
        out["Sr3BiI3_literature_inside_90"] = {
            m: {name: bool(lit["intervals"][m]["0.90"]["lower"] <= v <= lit["intervals"][m]["0.90"]["upper"])
                for name, v in out["literature_Sr3BiI3_PBE"].items()} for m in method_names(cfg)}
    return out


# ── figures + markdown ───────────────────────────────────────────────────────
def _figures(paths, out, pit, cfac):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = out["levels"]
    show = ["gpr_sigma", "gpr_sigma_scaled", "gpr_physics9_cvplus", "gpr_physics9_jackknife_plus",
            "ridge_physics9_cvplus", "mean_cvplus"]
    for ds in out["coverage"]:
        splits = list(out["coverage"][ds])
        fig, axes = plt.subplots(1, len(splits), figsize=(3.2 * len(splits), 3.3), dpi=150, sharey=True)
        for ax, s in zip(axes, splits):
            ax.plot([0, 1], [0, 1], color="#555", lw=0.8, ls="--")
            for m in show:
                cov = [out["coverage"][ds][s]["methods"][m][f"{p:.2f}"]["coverage"] for p in levels]
                ax.plot(levels, cov, marker="o", ms=2.5, lw=1, label=m)
            ax.set_title(f"{ds} · {s}", fontsize=8.5)
            ax.set_xlabel("nominal coverage")
        axes[0].set_ylabel("held-out coverage")
        axes[0].legend(fontsize=5.5, frameon=False)
        fig.suptitle("Reliability of predictive intervals (outer test folds only)", fontsize=9)
        fig.tight_layout()
        fig.savefig(paths.figures / f"uncertainty_reliability_{ds}.png", metadata={"Software": None})
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), dpi=150, sharex=True)
    ds = "primary"
    for col, s in enumerate(["lobo", "loao", "gkf", "loo"]):
        raw, sc = pit[(ds, s)]
        for row, (vals, lab) in enumerate(((raw, "raw sigma"), (sc, "scaled c*sigma"))):
            axes[row, col].hist(vals, bins=np.linspace(0, 1, 11), color="#2166ac" if row == 0 else "#1b7837",
                                edgecolor="white")
            axes[row, col].set_title(f"{s} · {lab}", fontsize=8)
    fig.suptitle("PIT of GPR predictions, primary (flat = calibrated; U-shape = sigma too small)", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths.figures / "uncertainty_pit_primary.png", metadata={"Software": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
    for m in show:
        xs, ys = [], []
        for s in ["lobo", "loao", "gkf", "loo"]:
            e = out["coverage"]["primary"][s]["methods"][m]["0.90"]
            xs.append(e["mean_width"]); ys.append(e["coverage"])
        ax.plot(xs, ys, marker="o", lw=0.6, label=m)
        for x, y_, s in zip(xs, ys, ["lobo", "loao", "gkf", "loo"]):
            ax.annotate(s, (x, y_), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0.9, color="#555", lw=0.8, ls="--")
    ax.set_xlabel("mean 90 % interval width (eV)")
    ax.set_ylabel("held-out coverage at nominal 90 %")
    ax.set_title("Sharpness vs coverage, primary", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "uncertainty_width_vs_coverage_primary.png", metadata={"Software": None})
    plt.close(fig)


def _markdown(paths, out):
    L = ["# uncertainty_v2 — generated by `a3bx3.uncertainty`; do not edit by hand", "",
         out["_readme"], "", f"Guarantee: {out['guarantee']}", "",
         "Exchangeability per split: " + "; ".join(f"**{k}** {v}" for k, v in out["exchangeability"].items()), ""]
    for ds, D in out["coverage"].items():
        L += [f"## {ds}", ""]
        for s, S in D.items():
            c = S["sigma_scale_c"]
            L += [f"### {s} ({S['n_folds']} folds, {S['n_predictions']} predictions, min n_train "
                  f"{S['min_n_train']}; sigma scale c mean {c['mean']:.2f} [{c['min']:.2f}–{c['max']:.2f}])", "",
                  "| method | cov@68 | cov@90 [95 % CI] | cov@95 | width@68 | width@90 | width@95 |",
                  "|---|---|---|---|---|---|---|"]
            for m, e in S["methods"].items():
                a, b, cc = e["0.68"], e["0.90"], e["0.95"]
                L.append(f"| {m} | {a['coverage']:.2f} | {b['coverage']:.2f} [{b['coverage_CI95'][0]:.2f}, "
                         f"{b['coverage_CI95'][1]:.2f}] | {cc['coverage']:.2f} | {a['mean_width']:.3f} | "
                         f"{b['mean_width']:.3f} | {cc['mean_width']:.3f} |")
            L.append("")
    L += ["## K2 kill checkpoint", "", out["k2"]["rule"], "",
          "| dataset | best model (MAE) | reference | ref MAE | skill [95 % CI] | MAE diff CI | pass |",
          "|---|---|---|---|---|---|---|"]
    for ds, k in out["k2"]["datasets"].items():
        for lab in ("vs_mean", "vs_best_baseline"):
            v = k[lab]
            L.append(f"| {ds} | {k['best_model']} ({k['best_model_MAE']:.3f}) | {v['reference']} | "
                     f"{v['reference_MAE']:.3f} | {v['skill']:.3f} [{v['skill_CI95'][0]:.3f}, {v['skill_CI95'][1]:.3f}] | "
                     f"[{v['MAE_diff_CI95'][0]:+.3f}, {v['MAE_diff_CI95'][1]:+.3f}] | {v['pass']} |")
        L.append(f"| {ds} | **verdict: {k['verdict']}** | | | | | |")
    h = out["sr3bix3_holdout"]
    L += ["", f"## Sr3BiX3 locked holdout (models fit on full {h['dataset']}; c = {h['sigma_scale_c']:.2f})", "",
          "| compound | target | GPR mu ± sigma | method | 90 % interval | contains target |", "|---|---|---|---|---|---|"]
    for f, e in h["compounds"].items():
        for m, iv in e["intervals"].items():
            v = iv["0.90"]
            L.append(f"| {f} | {e['target']:.3f} | {e['gpr_mu']:.3f} ± {e['gpr_sigma']:.3f} | {m} | "
                     f"[{v['lower']:.3f}, {v['upper']:.3f}] | {v['contains_target']} |")
    L += ["", "Sr3BiI3 literature PBE values inside the 90 % interval: " +
          "; ".join(f"{m}: " + ", ".join(f"{k.split()[0]} {'yes' if ok else 'no'}" for k, ok in d.items())
                    for m, d in h.get("Sr3BiI3_literature_inside_90", {}).items()), ""]
    write_atomic(paths.reports / "uncertainty_v2.md", "\n".join(L).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    out = run()
    for ds, D in out["coverage"].items():
        for s, S in D.items():
            g = S["methods"]
            print(f"[{ds:<13} {s:<4}] cov@90 raw {g['gpr_sigma']['0.90']['coverage']:.2f} "
                  f"scaled {g['gpr_sigma_scaled']['0.90']['coverage']:.2f} "
                  f"gprCV+ {g['gpr_physics9_cvplus']['0.90']['coverage']:.2f} "
                  f"(w {g['gpr_physics9_cvplus']['0.90']['mean_width']:.3f}) "
                  f"ridgeCV+ {g['ridge_physics9_cvplus']['0.90']['coverage']:.2f} "
                  f"(w {g['ridge_physics9_cvplus']['0.90']['mean_width']:.3f})")
    for ds, k in out["k2"]["datasets"].items():
        print(f"K2 {ds}: {k['verdict']}  vs mean {k['vs_mean']['skill']}  vs best baseline "
              f"({k['best_baseline']}) {k['vs_best_baseline']['skill']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
