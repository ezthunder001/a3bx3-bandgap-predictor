"""Train (out-of-fold), evaluate, report.

    train     every model and baseline on every outer fold of every split, same folds for
              all -> reports/oof_predictions.csv
    evaluate  OOF metrics (MAE / RMSE / R2), bootstrap 95 % CI on MAE, paired bootstrap CI on
              the MAE difference to the mean and ridge baselines, per-B and per-A error, GPR
              sigma coverage; then the ONE locked-holdout score per model (fit on the whole
              family, predict Sr3BiX3) -> reports/metrics_v2.json
    report    parity plot + per-family error figure + reports/metrics_v2.md

Every number is rounded and every key sorted, so two runs give byte-identical JSON.
Nothing in the model-selection path reads the holdout: the grids are chosen by inner
grouped CV inside each outer training fold, never by a holdout score.
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import utf8_stdio
from .config import ROOT, Paths, load_config
from .features import featurize_rows, read_csv
from .manifest import sha256, write_atomic
from .models import ModelSpec, fit, model_specs, predict
from .splits import assert_holdout_excluded, check_folds, make_splits

SPLIT_ORDER = ["lobo", "loao", "gkf", "loo"]
BASELINE_REFS = ["mean", "ridge_physics9"]
R = 4  # decimals in the JSON


def _r(x, d=R):
    return None if x is None else round(float(x), d)


def _load(paths: Paths):
    fam = read_csv(paths.processed / "family.csv")
    return fam


def _feature_mats(rows):
    return {fs: featurize_rows(rows, fs)[0] for fs in ("physics9", "magpie")}


def _fold_task(spec: ModelSpec, X, y, groups, tr, te):
    # GPR length-scales hitting their bound (an irrelevant descriptor) is expected at n~20;
    # the warning is noise here, not a failure. Everything else still surfaces.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        est = spec.build(X.shape[1])
        fit(spec, est, X[tr], y[tr], groups[tr])
        mu, sd = predict(est, X[te], want_std=True)
    return mu, sd


def train(paths: Paths | None = None, n_jobs: int = -1) -> list[dict]:
    """Out-of-fold predictions for every (split, fold, model)."""
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    rows = _load(paths)
    holdout = set(cfg["data"]["holdout_formulas"])
    assert_holdout_excluded([r["formula"] for r in rows], holdout)
    y = np.array([r["Eg_eV"] for r in rows])
    groups = np.array([r["formula"] for r in rows])
    mats = _feature_mats(rows)
    splits = make_splits(rows, cfg["splits"], cfg["seed"])
    specs = model_specs(cfg)

    tasks, keys = [], []
    for sname in SPLIT_ORDER:
        s = splits[sname]
        check_folds(rows, s.folds, holdout)
        for fi, (tr, te) in enumerate(s.folds):
            for spec in specs:
                tasks.append(delayed(_fold_task)(spec, mats[spec.feature_set], y, groups, tr, te))
                keys.append((sname, fi, spec.name, te))
    results = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

    out = []
    n_splits_per_rep = {k: len(splits[k].folds) // splits[k].n_repeats for k in splits}
    for (sname, fi, mname, te), (mu, sd) in zip(keys, results):
        rep = fi // n_splits_per_rep[sname]
        for j, i in enumerate(te):
            out.append({"split": sname, "repeat": rep, "fold": fi, "model": mname,
                        "formula": rows[i]["formula"], "A": rows[i]["A"], "B": rows[i]["B"],
                        "X": rows[i]["X"], "y_true": rows[i]["Eg_eV"],
                        "y_pred": _r(mu[j], 6), "sigma": None if sd is None else _r(sd[j], 6)})
    out.sort(key=lambda r: (SPLIT_ORDER.index(r["split"]), r["model"], r["repeat"], r["formula"]))
    cols = ["split", "repeat", "fold", "model", "formula", "A", "B", "X", "y_true", "y_pred", "sigma"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in out:
        w.writerow({k: ("" if r[k] is None else r[k]) for k in cols})
    write_atomic(paths.reports / "oof_predictions.csv", buf.getvalue().encode("utf-8"))
    return out


def _read_oof(paths: Paths) -> list[dict]:
    with open(paths.reports / "oof_predictions.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["repeat"] = int(r["repeat"])
        r["y_true"] = float(r["y_true"])
        r["y_pred"] = float(r["y_pred"])
        r["sigma"] = float(r["sigma"]) if r["sigma"] else None
    return rows


def _group_mae(formulas, err, key_of):
    out = {}
    for k in sorted({key_of[f] for f in formulas}):
        m = [e for f, e in zip(formulas, err) if key_of[f] == k]
        out[k] = _r(np.mean(m))
    return out


def evaluate(paths: Paths | None = None, final: bool = True) -> dict:
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    ev = cfg["evaluate"]
    fam = _load(paths)
    formulas = [r["formula"] for r in fam]
    key_A = {r["formula"]: r["A"] for r in fam}
    key_B = {r["formula"]: r["B"] for r in fam}
    y_of = {r["formula"]: r["Eg_eV"] for r in fam}
    oof = _read_oof(paths)
    specs = model_specs(cfg)
    names = [s.name for s in specs]
    rng = np.random.RandomState(cfg["seed"])
    lo, hi = (1 - ev["ci"]) / 2 * 100, (1 + ev["ci"]) / 2 * 100
    boot = rng.randint(0, len(formulas), size=(ev["bootstrap_n"], len(formulas)))

    splits_out = {}
    for sname in SPLIT_ORDER:
        by_model: dict[str, dict[int, dict[str, float]]] = {}
        sig: dict[str, dict[str, float]] = {}
        for r in oof:
            if r["split"] == sname:
                by_model.setdefault(r["model"], {}).setdefault(r["repeat"], {})[r["formula"]] = r["y_pred"]
                if r["sigma"] is not None and r["repeat"] == 0:
                    sig.setdefault(r["model"], {})[r["formula"]] = r["sigma"]
        yv = np.array([y_of[f] for f in formulas])
        row_err, entries = {}, {}
        for m in names:
            reps = by_model[m]
            P = np.array([[reps[k][f] for f in formulas] for k in sorted(reps)])  # repeats x n
            E = np.abs(P - yv)
            row_err[m] = E.mean(0)
            rep_mae = E.mean(1)
            e = {"MAE": _r(rep_mae.mean()),
                 "RMSE": _r(np.mean([np.sqrt(mean_squared_error(yv, p)) for p in P])),
                 "R2": _r(np.mean([r2_score(yv, p) for p in P])),
                 "MAE_CI95": [_r(np.percentile(row_err[m][boot].mean(1), lo)),
                              _r(np.percentile(row_err[m][boot].mean(1), hi))],
                 "per_B": _group_mae(formulas, row_err[m], key_B),
                 "per_A": _group_mae(formulas, row_err[m], key_A),
                 "n_rows": len(formulas), "n_repeats": len(P),
                 "role": next(s.role for s in specs if s.name == m),
                 "feature_set": next(s.feature_set for s in specs if s.name == m)}
            if len(P) > 1:
                e["MAE_repeat_sd"] = _r(rep_mae.std(ddof=1))
            if m in sig:
                sd = np.array([sig[m][f] for f in formulas])
                res = np.abs(P[0] - yv)
                e["sigma_coverage"] = {"within_1sigma": _r(np.mean(res <= sd)),
                                       "within_2sigma": _r(np.mean(res <= 2 * sd)),
                                       "nominal": [0.6827, 0.9545]}
            entries[m] = e
        for m in names:
            for b in BASELINE_REFS:
                if m == b:
                    continue
                d = (row_err[m] - row_err[b])[boot].mean(1)
                entries[m].setdefault("vs_baseline", {})[b] = {
                    "MAE_diff": _r(row_err[m].mean() - row_err[b].mean()),
                    "MAE_diff_CI95": [_r(np.percentile(d, lo)), _r(np.percentile(d, hi))],
                    "skill": _r(1 - row_err[m].mean() / row_err[b].mean(), 3)}
        best_base = min((m for m in names if entries[m]["role"] == "baseline"),
                        key=lambda m: (entries[m]["MAE"], m))
        splits_out[sname] = {"models": entries, "best_baseline": best_base,
                             "best_model": min((m for m in names if entries[m]["role"] == "model"),
                                               key=lambda m: (entries[m]["MAE"], m))}

    holdout = final_holdout(paths, cfg) if final else None

    v1 = json.loads((ROOT / "data" / "a3bx3_family_metrics.json").read_text(encoding="utf-8"))
    g = "gpr_physics9"
    v1_check = {
        "loo": {"v1": v1["loo"]["MAE"], "v2": splits_out["loo"]["models"][g]["MAE"],
                "abs_diff": _r(abs(v1["loo"]["MAE"] - splits_out["loo"]["models"][g]["MAE"]))},
        "lobo": {"v1": v1["lobo"]["MAE"], "v2": splits_out["lobo"]["models"][g]["MAE"],
                 "abs_diff": _r(abs(v1["lobo"]["MAE"] - splits_out["lobo"]["models"][g]["MAE"]))},
        "sr3bix3": {"v1": v1["sr3bix3_validation"]["MAE"],
                    "v2": None if holdout is None else holdout["models"][g]["MAE"]},
    }

    metrics = {
        "_readme": ("Every model score sits beside every named baseline on the SAME folds. "
                    "Numbers are out-of-fold (held-out); nothing here is in-sample. "
                    "Headline = leave-one-B-family-out. The Sr3BiX3 holdout is scored once, "
                    "fit on the whole family, and used for no choice."),
        "version": cfg["version"],
        "target": "GGA-PBE band gap (eV)",
        "headline": {"split": ev["headline_split"], "model": ev["headline_model"]},
        "data": {"n_family": len(fam), "n_holdout": len(cfg["data"]["holdout_formulas"]),
                 "family_csv_sha256": sha256(paths.processed / "family.csv"),
                 "holdout_csv_sha256": sha256(paths.processed / "holdout_sr3bix3.csv"),
                 "config_sha256": sha256(ROOT / "configs" / "v2.yaml")},
        "splits": {k: {"description": d} for k, d in
                   (("lobo", "leave-one-B-family-out (P/As/Sb/Bi)"),
                    ("loao", "leave-one-A-out (Mg/Ca/Sr/Ba)"),
                    ("gkf", "GroupKFold(5) by formula x10 shuffled repeats; MAE = mean over repeats"),
                    ("loo", "leave-one-formula-out; v1 comparison only"))},
        "results": splits_out,
        "sr3bix3_holdout": holdout,
        "models": {s.name: {"role": s.role, "feature_set": s.feature_set,
                            "description": s.description} for s in specs},
        "bootstrap": {"n": ev["bootstrap_n"], "ci": ev["ci"], "unit": "rows (formulas)",
                      "paired": "same resample indices for every model within a split"},
        "v1_reproduction": v1_check,
        "environment": {p: _pkg_version(p) for p in
                        ("numpy", "scipy", "scikit-learn", "xgboost", "matminer", "pymatgen")},
    }
    for k in splits_out:
        metrics["splits"][k]["n_folds"] = None
    fold_counts = {}
    for r in oof:
        fold_counts.setdefault(r["split"], set()).add(int(r["fold"]))
    for k, v in fold_counts.items():
        metrics["splits"][k]["n_folds"] = len(v)
    write_atomic(paths.reports / "metrics_v2.json",
                 (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return metrics


def final_holdout(paths: Paths, cfg: dict) -> dict:
    """Fit each model on the whole family, predict the locked Sr3BiX3 rows. Read only here."""
    fam = _load(paths)
    hold = read_csv(paths.processed / "holdout_sr3bix3.csv")
    holdout = set(cfg["data"]["holdout_formulas"])
    assert_holdout_excluded([r["formula"] for r in fam], holdout)
    y = np.array([r["Eg_eV"] for r in fam])
    groups = np.array([r["formula"] for r in fam])
    yh = np.array([r["Eg_eV"] for r in hold])
    mf, mh = _feature_mats(fam), _feature_mats(hold)
    out = {}
    for spec in model_specs(cfg):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            est = spec.build(mf[spec.feature_set].shape[1])
            fit(spec, est, mf[spec.feature_set], y, groups)
            mu, sd = predict(est, mh[spec.feature_set], want_std=True)
        per = {r["formula"]: {"pred": _r(p), "dft": r["Eg_eV"], "error": _r(p - r["Eg_eV"]),
                              **({"sigma": _r(s)} if sd is not None else {})}
               for r, p, s in zip(hold, mu, sd if sd is not None else [None] * len(hold))}
        out[spec.name] = {"MAE": _r(np.mean(np.abs(mu - yh))), "per_compound": per,
                          "role": spec.role}
    return {"note": ("n = 3. Scored once per release; cannot rank models. "
                     "Reference: Islam et al., New J. Chem. 2026, 50, 522 (PBE)."),
            "models": out}


# ── report ────────────────────────────────────────────────────────────────────
B_COLORS = {"P": "#1b7837", "As": "#762a83", "Sb": "#e08214", "Bi": "#2166ac"}


def report(paths: Paths | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    paths = (paths or Paths()).ensure()
    m = json.loads((paths.reports / "metrics_v2.json").read_text(encoding="utf-8"))
    oof = _read_oof(paths)
    hs, hm = m["headline"]["split"], m["headline"]["model"]
    pts = [r for r in oof if r["split"] == hs and r["model"] == hm]
    hold = (m.get("sr3bix3_holdout") or {}).get("models", {}).get(hm, {}).get("per_compound", {})

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    for b in ["P", "As", "Sb", "Bi"]:
        s = [r for r in pts if r["B"] == b]
        ax.scatter([r["y_true"] for r in s], [r["y_pred"] for r in s], s=36, color=B_COLORS[b],
                   label=f"B = {b} (held out as a family)", edgecolor="white", linewidth=0.6)
    if hold:
        ax.scatter([v["dft"] for v in hold.values()], [v["pred"] for v in hold.values()], s=70,
                   marker="*", color="#b2182b", label="Sr3BiX3 locked holdout", zorder=5)
    lim = [0.6, 2.5]
    ax.plot(lim, lim, color="#555", lw=0.8, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("literature GGA-PBE gap (eV)")
    ax.set_ylabel(f"{hm} prediction (eV)")
    e = m["results"][hs]["models"][hm]
    base = m["results"][hs]["models"]["mean"]
    ax.set_title(f"Leave-one-B-family-out, out-of-fold\n{hm}: MAE {e['MAE']:.3f} eV "
                 f"[{e['MAE_CI95'][0]:.3f}, {e['MAE_CI95'][1]:.3f}] · mean baseline {base['MAE']:.3f}",
                 fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "parity_lobo_gpr_physics9.png", metadata={"Software": None})
    plt.close(fig)

    names = list(m["models"])
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=150)
    width = 0.8 / len(names)
    Bs = ["P", "As", "Sb", "Bi"]
    for i, n in enumerate(names):
        vals = [m["results"][hs]["models"][n]["per_B"].get(b, np.nan) for b in Bs]
        ax.bar(np.arange(len(Bs)) + i * width, vals, width, label=n,
               hatch="//" if m["models"][n]["role"] == "baseline" else None,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(np.arange(len(Bs)) + 0.4 - width / 2)
    ax.set_xticklabels([f"B = {b} held out" for b in Bs])
    ax.set_ylabel("MAE (eV)")
    ax.set_title("Per-family error, leave-one-B-family-out (hatched = baseline)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "per_family_error_lobo.png", metadata={"Software": None})
    plt.close(fig)

    lines = ["# metrics_v2 — generated by `a3bx3.evaluate report`; do not edit by hand", "",
             "Out-of-fold MAE in eV, bootstrap 95 % CI in brackets. Same folds for every row.", ""]
    for s in SPLIT_ORDER:
        res = m["results"][s]["models"]
        lines += [f"## {s}: {m['splits'][s]['description']}", "",
                  "| model | role | MAE [95 % CI] | R² | Δ vs mean [95 % CI] | Δ vs ridge_physics9 [95 % CI] |",
                  "|---|---|---|---|---|---|"]
        for n in names:
            e = res[n]
            def d(b):
                v = e.get("vs_baseline", {}).get(b)
                return "—" if not v else f"{v['MAE_diff']:+.3f} [{v['MAE_diff_CI95'][0]:+.3f}, {v['MAE_diff_CI95'][1]:+.3f}]"
            lines.append(f"| {n} | {e['role']} | {e['MAE']:.3f} [{e['MAE_CI95'][0]:.3f}, "
                         f"{e['MAE_CI95'][1]:.3f}] | {e['R2']:.3f} | {d('mean')} | {d('ridge_physics9')} |")
        lines.append("")
    if m.get("sr3bix3_holdout"):
        lines += ["## Sr3BiX3 locked holdout (n = 3, scored once; cannot rank models)", "",
                  "| model | role | MAE |", "|---|---|---|"]
        for n in names:
            h = m["sr3bix3_holdout"]["models"][n]
            lines.append(f"| {n} | {h['role']} | {h['MAE']:.3f} |")
        lines.append("")
    v = m["v1_reproduction"]
    lines += ["## v1 reproduction (gpr_physics9)", "",
              f"LOO {v['loo']['v2']:.4f} vs v1 {v['loo']['v1']:.4f}; "
              f"LOBO {v['lobo']['v2']:.4f} vs v1 {v['lobo']['v1']:.4f}.", ""]
    write_atomic(paths.reports / "metrics_v2.md", "\n".join(lines).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    argv = sys.argv[1:] if argv is None else argv
    stage = argv[0] if argv else "all"
    if stage in ("train", "all"):
        n = len(train())
        print(f"train: {n} out-of-fold predictions -> reports/oof_predictions.csv")
    if stage in ("evaluate", "all"):
        m = evaluate()
        hs, hm = m["headline"]["split"], m["headline"]["model"]
        for s in SPLIT_ORDER:
            print(f"[{s}]")
            for n, e in m["results"][s]["models"].items():
                print(f"  {n:<15} {e['role']:<8} MAE {e['MAE']:.3f} "
                      f"[{e['MAE_CI95'][0]:.3f}, {e['MAE_CI95'][1]:.3f}]  R2 {e['R2']:.3f}")
        print(f"headline {hm} / {hs}: MAE {m['results'][hs]['models'][hm]['MAE']:.4f}")
    if stage in ("report", "all"):
        report()
        print("report: reports/metrics_v2.md, reports/figures/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
