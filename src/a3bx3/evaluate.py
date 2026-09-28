"""Train (out-of-fold), evaluate, report — for every dataset built by curate.py.

    train     every model and baseline on every outer fold of every split of every dataset,
              same folds for all models -> reports/oof_predictions.csv
    evaluate  OOF metrics (MAE / RMSE / R2), bootstrap 95 % CI on MAE, paired bootstrap CI on
              the MAE difference to the mean and ridge baselines, per-B and per-A error, GPR
              sigma coverage; the ONE locked-holdout score per model for the no-SOC datasets
              (fit on the whole dataset, predict Sr3BiX3); the v1 reproduction check
              -> reports/metrics_v2.json
    report    parity plot + per-family error + dataset comparison figures, reports/metrics_v2.md

Every number is rounded and every key sorted, so two runs give byte-identical JSON.
Nothing in the model-selection path reads the holdout: grids are chosen by inner grouped CV
inside each outer training fold, never by a holdout score.
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
from sklearn.metrics import mean_squared_error, r2_score

from . import utf8_stdio
from .config import ROOT, Paths, load_config
from .features import featurize_rows, read_csv
from .fetch.literature import load_file
from .manifest import sha256_text, write_atomic
from .models import ModelSpec, fit, model_specs, predict
from .splits import assert_holdout_excluded, check_folds, make_splits

SPLIT_ORDER = ["lobo", "loao", "gkf", "loo"]
SPLIT_DESC = {"lobo": "leave-one-B-family-out (P/As/Sb/Bi)",
              "loao": "leave-one-A-out (Mg/Ca/Sr/Ba)",
              "gkf": "GroupKFold(5) by formula x10 shuffled repeats; MAE = mean over repeats",
              "loo": "leave-one-formula-out; v1 comparison only"}
DATASET_DESC = {
    "primary": "GGA-PBE without SOC (soc no/unknown), median per formula; preprint rows included",
    "verified_only": "primary minus preprint rows (arXiv 2604.01942); source_unverified never included",
    "plus_unverified": "primary plus the source_unverified v1 rows (Ca3BiX3, Ca3AsCl3)",
    "soc": "GGA-PBE+SOC, median per formula; a separate target, never pooled with no-SOC",
}
BASELINE_REFS = ["mean", "ridge_physics9"]
R = 4  # decimals in the JSON


def _r(x, d=R):
    return None if x is None else round(float(x), d)


def load_dataset(paths: Paths, name: str) -> list[dict]:
    return read_csv(paths.processed / f"family_{name}.csv")


def _feature_mats(rows):
    return {fs: featurize_rows(rows, fs)[0] for fs in ("physics9", "magpie")}


def _fold_task(spec: ModelSpec, X, y, groups, tr, te):
    # GPR length-scales hitting their bound (an irrelevant descriptor) is expected at small n;
    # that warning is noise here, not a failure. Everything else still surfaces.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        est = spec.build(X.shape[1])
        fit(spec, est, X[tr], y[tr], groups[tr])
        mu, sd = predict(est, X[te], want_std=True)
    return mu, sd


def _plan(cfg, rows, split_names, specs, holdout):
    y = np.array([r["Eg_eV"] for r in rows])
    groups = np.array([r["formula"] for r in rows])
    mats = _feature_mats(rows)
    splits = make_splits(rows, cfg["splits"], cfg["seed"])
    tasks, keys = [], []
    for sname in split_names:
        s = splits[sname]
        check_folds(rows, s.folds, holdout)
        per_rep = len(s.folds) // s.n_repeats
        for fi, (tr, te) in enumerate(s.folds):
            for spec in specs:
                tasks.append(delayed(_fold_task)(spec, mats[spec.feature_set], y, groups, tr, te))
                keys.append((sname, fi, fi // per_rep, spec.name, te))
    return tasks, keys


def train(paths: Paths | None = None, n_jobs: int = -1) -> list[dict]:
    """Out-of-fold predictions for every (dataset, split, fold, model)."""
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    holdout = set(cfg["data"]["holdout_formulas"])
    specs = model_specs(cfg)
    all_tasks, all_keys, rows_of = [], [], {}
    for ds, split_names in cfg["evaluate"]["splits_by_dataset"].items():
        rows = load_dataset(paths, ds)
        assert_holdout_excluded([r["formula"] for r in rows], holdout)
        rows_of[ds] = rows
        tasks, keys = _plan(cfg, rows, split_names, specs, holdout)
        all_tasks += tasks
        all_keys += [(ds,) + k for k in keys]
    results = Parallel(n_jobs=n_jobs, backend="loky")(all_tasks)

    out = []
    for (ds, sname, fi, rep, mname, te), (mu, sd) in zip(all_keys, results):
        rows = rows_of[ds]
        for j, i in enumerate(te):
            out.append({"dataset": ds, "split": sname, "repeat": rep, "fold": fi, "model": mname,
                        "formula": rows[i]["formula"], "A": rows[i]["A"], "B": rows[i]["B"],
                        "X": rows[i]["X"], "y_true": rows[i]["Eg_eV"],
                        "y_pred": _r(mu[j], 6), "sigma": None if sd is None else _r(sd[j], 6)})
    dorder = list(cfg["evaluate"]["splits_by_dataset"])
    out.sort(key=lambda r: (dorder.index(r["dataset"]), SPLIT_ORDER.index(r["split"]),
                            r["model"], r["repeat"], r["formula"]))
    cols = ["dataset", "split", "repeat", "fold", "model", "formula", "A", "B", "X", "y_true",
            "y_pred", "sigma"]
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
        r["fold"] = int(r["fold"])
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


def _split_metrics(fam, oof, sname, specs, boot, lo, hi) -> dict:
    formulas = [r["formula"] for r in fam]
    key_A = {r["formula"]: r["A"] for r in fam}
    key_B = {r["formula"]: r["B"] for r in fam}
    yv = np.array([r["Eg_eV"] for r in fam])
    names = [s.name for s in specs]
    by_model: dict[str, dict[int, dict[str, float]]] = {}
    sig: dict[str, dict[str, float]] = {}
    for r in oof:
        if r["split"] == sname:
            by_model.setdefault(r["model"], {}).setdefault(r["repeat"], {})[r["formula"]] = r["y_pred"]
            if r["sigma"] is not None and r["repeat"] == 0:
                sig.setdefault(r["model"], {})[r["formula"]] = r["sigma"]
    row_err, entries = {}, {}
    for m in names:
        reps = by_model[m]
        P = np.array([[reps[k][f] for f in formulas] for k in sorted(reps)])  # repeats x n
        E = np.abs(P - yv)
        row_err[m] = E.mean(0)
        rep_mae = E.mean(1)
        spec = next(s for s in specs if s.name == m)
        e = {"MAE": _r(rep_mae.mean()),
             "RMSE": _r(np.mean([np.sqrt(mean_squared_error(yv, p)) for p in P])),
             "R2": _r(np.mean([r2_score(yv, p) for p in P])),
             "MAE_CI95": [_r(np.percentile(row_err[m][boot].mean(1), lo)),
                          _r(np.percentile(row_err[m][boot].mean(1), hi))],
             "per_B": _group_mae(formulas, row_err[m], key_B),
             "per_A": _group_mae(formulas, row_err[m], key_A),
             "n_rows": len(formulas), "n_repeats": len(P),
             "role": spec.role, "feature_set": spec.feature_set}
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
    return {"models": entries,
            "best_baseline": min((m for m in names if entries[m]["role"] == "baseline"),
                                 key=lambda m: (entries[m]["MAE"], m)),
            "best_model": min((m for m in names if entries[m]["role"] == "model"),
                              key=lambda m: (entries[m]["MAE"], m)),
            "n_folds": len({r["fold"] for r in oof if r["split"] == sname})}


def evaluate(paths: Paths | None = None, final: bool = True) -> dict:
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    ev = cfg["evaluate"]
    specs = model_specs(cfg)
    oof_all = _read_oof(paths)
    lo, hi = (1 - ev["ci"]) / 2 * 100, (1 + ev["ci"]) / 2 * 100
    summary = json.loads((paths.interim / "curate_summary.json").read_text(encoding="utf-8"))

    datasets = {}
    for di, (ds, split_names) in enumerate(ev["splits_by_dataset"].items()):
        fam = load_dataset(paths, ds)
        rng = np.random.RandomState(cfg["seed"] + di)
        boot = rng.randint(0, len(fam), size=(ev["bootstrap_n"], len(fam)))
        oof = [r for r in oof_all if r["dataset"] == ds]
        datasets[ds] = {
            "description": DATASET_DESC.get(ds, ds),
            "n_formulas": len(fam),
            "curate": summary["datasets"][ds],
            "family_csv_sha256": sha256_text(paths.processed / f"family_{ds}.csv"),
            "results": {s: _split_metrics(fam, [r for r in oof if r["split"] == s], s, specs,
                                          boot, lo, hi) for s in split_names},
            "sr3bix3_holdout": (final_holdout(paths, cfg, ds)
                                if final and ds in ev["holdout_datasets"] else None),
        }

    metrics = {
        "_readme": ("Every model score sits beside every named baseline on the SAME folds. "
                    "Numbers are out-of-fold (held-out); nothing here is in-sample. "
                    "Headline = primary dataset, leave-one-B-family-out. The Sr3BiX3 holdout is "
                    "scored once per dataset, fit on the whole dataset, and used for no choice."),
        "version": cfg["version"],
        "target": "GGA-PBE band gap (eV), no SOC (the soc dataset is PBE+SOC)",
        "headline": {"dataset": ev["headline_dataset"], "split": ev["headline_split"],
                     "model": ev["headline_model"]},
        "k1": summary["k1"],
        "label_noise_floor_eV": ev["label_noise_floor_eV"],
        "data": {"n_holdout": len(cfg["data"]["holdout_formulas"]),
                 "holdout_csv_sha256": sha256_text(paths.processed / "holdout_sr3bix3.csv"),
                 "config_sha256": sha256_text(ROOT / "configs" / "v2.yaml")},
        "splits": SPLIT_DESC,
        "datasets": datasets,
        "models": {s.name: {"role": s.role, "feature_set": s.feature_set,
                            "description": s.description} for s in specs},
        "bootstrap": {"n": ev["bootstrap_n"], "ci": ev["ci"], "unit": "rows (formulas)",
                      "paired": "same resample indices for every model within a dataset"},
        "v1_reproduction": v1_reproduction(cfg),
        "environment": {p: _pkg_version(p) for p in
                        ("numpy", "scipy", "scikit-learn", "xgboost", "matminer", "pymatgen")},
    }
    write_atomic(paths.reports / "metrics_v2.json",
                 (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return metrics


def final_holdout(paths: Paths, cfg: dict, ds: str) -> dict:
    """Fit each model on the whole dataset, predict the locked Sr3BiX3 rows. Read only here."""
    fam = load_dataset(paths, ds)
    hold = read_csv(paths.processed / "holdout_sr3bix3.csv")
    assert_holdout_excluded([r["formula"] for r in fam], set(cfg["data"]["holdout_formulas"]))
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
        out[spec.name] = {"MAE": _r(np.mean(np.abs(mu - yh))),
                          "bias": _r(np.mean(mu - yh)), "per_compound": per, "role": spec.role}
    return {"note": ("n = 3. Scored once; cannot rank models. Reference: Islam et al., "
                     "New J. Chem. 2026, 50, 522, doi 10.1039/d5nj02800k (PBE). Other PBE reports "
                     "for Sr3BiI3: 1.164 (CASTEP) and 1.30 eV (WIEN2k) vs the 1.324 target."),
            "models": out}


def v1_reproduction(cfg: dict) -> dict:
    """Rerun the v2 GPR on the v1 rows EXACTLY as published (no corrections), with v1's LOO
    and LOBO folds, and compare with the committed v1 metrics. v1 is a historical record."""
    v1 = json.loads((ROOT / "data" / "a3bx3_family_metrics.json").read_text(encoding="utf-8"))
    rows = load_file(ROOT / "data" / "a3bx3_literature.csv")
    train_rows = [dict(r, formula=f"{r['A']}3{r['B']}{r['X']}3") for r in rows
                  if "TARGET" not in (r["note"] or "")]
    spec = next(s for s in model_specs(cfg) if s.name == "gpr_physics9")
    X = featurize_rows(train_rows, "physics9")[0]
    y = np.array([r["Eg_eV"] for r in train_rows])
    groups = np.array([r["formula"] for r in train_rows])
    splits = make_splits(train_rows, cfg["splits"], cfg["seed"])
    out = {"_note": ("v2 code on the uncorrected v1 file. v1 contains two HSE06 labels stored "
                     "as PBE (Mg3BiI3, Mg3BiBr3); the v2 datasets fix them. These numbers "
                     "check the code, not the data.")}
    for s, v1_mae in (("loo", v1["loo"]["MAE"]), ("lobo", v1["lobo"]["MAE"])):
        pred = np.zeros(len(y))
        for tr, te in splits[s].folds:
            pred[te] = _fold_task(spec, X, y, groups, tr, te)[0]
        v2 = _r(np.mean(np.abs(pred - y)))
        out[s] = {"v1": v1_mae, "v2": v2, "abs_diff": _r(abs(v2 - v1_mae))}
    return out


# ── report ────────────────────────────────────────────────────────────────────
B_COLORS = {"P": "#1b7837", "As": "#762a83", "Sb": "#e08214", "Bi": "#2166ac"}


def _fmt_ci(e):
    return f"{e['MAE']:.3f} [{e['MAE_CI95'][0]:.3f}, {e['MAE_CI95'][1]:.3f}]"


def report(paths: Paths | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = (paths or Paths()).ensure()
    m = json.loads((paths.reports / "metrics_v2.json").read_text(encoding="utf-8"))
    oof = _read_oof(paths)
    hd, hs, hm = m["headline"]["dataset"], m["headline"]["split"], m["headline"]["model"]
    D = m["datasets"][hd]
    pts = [r for r in oof if r["dataset"] == hd and r["split"] == hs and r["model"] == hm]
    hold = (D.get("sr3bix3_holdout") or {}).get("models", {}).get(hm, {}).get("per_compound", {})

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    for b in ["P", "As", "Sb", "Bi"]:
        s = [r for r in pts if r["B"] == b]
        ax.scatter([r["y_true"] for r in s], [r["y_pred"] for r in s], s=36, color=B_COLORS[b],
                   label=f"B = {b} (held out as a family)", edgecolor="white", linewidth=0.6)
    if hold:
        ax.scatter([v["dft"] for v in hold.values()], [v["pred"] for v in hold.values()], s=70,
                   marker="*", color="#b2182b", label="Sr3BiX3 locked holdout", zorder=5)
    allv = [r["y_true"] for r in pts] + [r["y_pred"] for r in pts]
    lim = [min(0.0, min(allv) - 0.1), max(allv) + 0.15]
    ax.plot(lim, lim, color="#555", lw=0.8, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("literature GGA-PBE gap, no SOC (eV; median per formula)")
    ax.set_ylabel(f"{hm} prediction (eV)")
    e, base = D["results"][hs]["models"][hm], D["results"][hs]["models"]["mean"]
    ax.set_title(f"{hd} (n = {D['n_formulas']}), leave-one-B-family-out, out-of-fold\n"
                 f"{hm}: MAE {_fmt_ci(e)} · mean baseline {base['MAE']:.3f}", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "parity_lobo_gpr_physics9.png", metadata={"Software": None})
    plt.close(fig)

    names = list(m["models"])
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=150)
    width = 0.8 / len(names)
    Bs = ["P", "As", "Sb", "Bi"]
    for i, n in enumerate(names):
        vals = [D["results"][hs]["models"][n]["per_B"].get(b, np.nan) for b in Bs]
        ax.bar(np.arange(len(Bs)) + i * width, vals, width, label=n,
               hatch="//" if m["models"][n]["role"] == "baseline" else None,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(np.arange(len(Bs)) + 0.4 - width / 2)
    ax.set_xticklabels([f"B = {b} held out" for b in Bs])
    ax.set_ylabel("MAE (eV)")
    ax.set_title(f"Per-family error, {hd}, leave-one-B-family-out (hatched = baseline)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "per_family_error_lobo.png", metadata={"Software": None})
    plt.close(fig)

    # dataset comparison: GPR vs the two reference baselines, per split, with CI
    dsn = [d for d in load_config()["evaluate"]["splits_by_dataset"] if d in m["datasets"]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150, sharey=True)
    for ax, s in zip(axes, ["lobo", "loao", "gkf"]):
        for j, (mod, c) in enumerate((("gpr_physics9", "#2166ac"), ("ridge_physics9", "#999"),
                                      ("mean", "#d6604d"))):
            xs = np.arange(len(dsn)) + (j - 1) * 0.25
            ys = [m["datasets"][d]["results"][s]["models"][mod]["MAE"] for d in dsn]
            lo_ = [y - m["datasets"][d]["results"][s]["models"][mod]["MAE_CI95"][0] for d, y in zip(dsn, ys)]
            hi_ = [m["datasets"][d]["results"][s]["models"][mod]["MAE_CI95"][1] - y for d, y in zip(dsn, ys)]
            ax.errorbar(xs, ys, yerr=[lo_, hi_], fmt="o", color=c, ms=4, capsize=2, label=mod)
        ax.set_xticks(np.arange(len(dsn)))
        ax.set_xticklabels(dsn, rotation=20, fontsize=7)
        ax.set_title(s, fontsize=9)
    axes[0].set_ylabel("MAE (eV), 95 % CI")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "dataset_comparison.png", metadata={"Software": None})
    plt.close(fig)

    L = ["# metrics_v2 — generated by `a3bx3.evaluate report`; do not edit by hand", "",
         "Out-of-fold MAE in eV, bootstrap 95 % CI in brackets. Same folds for every row of a "
         "table. Labels are literature GGA-PBE gaps; code-to-code spread for one formula is "
         f"{m['label_noise_floor_eV'][0]}–{m['label_noise_floor_eV'][1]} eV (label noise floor).", "",
         f"K1 checkpoint: {m['k1']['primary_formulas']} primary formulas vs ≥ {m['k1']['threshold']} "
         f"→ **{'PASS' if m['k1']['pass'] else 'FAIL'}**.", "",
         "## Summary: GPR and best model vs baselines", "",
         "| dataset | n | split | gpr_physics9 | best model | mean | ridge_physics9 | GPR − ridge [95 % CI] |",
         "|---|---|---|---|---|---|---|---|"]
    for d in dsn:
        for s in [x for x in SPLIT_ORDER if x in m["datasets"][d]["results"]]:
            r = m["datasets"][d]["results"][s]
            md = r["models"]
            bm = r["best_model"]
            v = md["gpr_physics9"]["vs_baseline"]["ridge_physics9"]
            L.append(f"| {d} | {m['datasets'][d]['n_formulas']} | {s} | {_fmt_ci(md['gpr_physics9'])} | "
                     f"{bm} {_fmt_ci(md[bm])} | {_fmt_ci(md['mean'])} | {_fmt_ci(md['ridge_physics9'])} | "
                     f"{v['MAE_diff']:+.3f} [{v['MAE_diff_CI95'][0]:+.3f}, {v['MAE_diff_CI95'][1]:+.3f}] |")
    L.append("")
    for d in dsn:
        DD = m["datasets"][d]
        L += [f"## Dataset `{d}` (n = {DD['n_formulas']}): {DD['description']}", ""]
        for s in [x for x in SPLIT_ORDER if x in DD["results"]]:
            r = DD["results"][s]
            L += [f"### {s}: {m['splits'][s]}", "",
                  "| model | role | MAE [95 % CI] | R² | Δ vs mean [95 % CI] | Δ vs ridge_physics9 [95 % CI] |",
                  "|---|---|---|---|---|---|"]
            for n in names:
                e = r["models"][n]

                def dd(b):
                    v = e.get("vs_baseline", {}).get(b)
                    return "—" if not v else f"{v['MAE_diff']:+.3f} [{v['MAE_diff_CI95'][0]:+.3f}, {v['MAE_diff_CI95'][1]:+.3f}]"
                L.append(f"| {n} | {e['role']} | {_fmt_ci(e)} | {e['R2']:.3f} | {dd('mean')} | {dd('ridge_physics9')} |")
            L.append("")
        if DD.get("sr3bix3_holdout"):
            L += ["### Sr3BiX3 locked holdout (n = 3, scored once; cannot rank models)", "",
                  "| model | role | MAE | bias |", "|---|---|---|---|"]
            for n in names:
                h = DD["sr3bix3_holdout"]["models"][n]
                L.append(f"| {n} | {h['role']} | {h['MAE']:.3f} | {h['bias']:+.3f} |")
            L.append("")
    v = m["v1_reproduction"]
    L += ["## v1 reproduction (gpr_physics9 on the uncorrected v1 file)", "",
          f"LOO {v['loo']['v2']:.4f} vs v1 {v['loo']['v1']:.4f}; "
          f"LOBO {v['lobo']['v2']:.4f} vs v1 {v['lobo']['v1']:.4f}. {v['_note']}", ""]
    write_atomic(paths.reports / "metrics_v2.md", "\n".join(L).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    argv = sys.argv[1:] if argv is None else argv
    stage = argv[0] if argv else "all"
    if stage in ("train", "all"):
        n = len(train())
        print(f"train: {n} out-of-fold predictions -> reports/oof_predictions.csv")
    if stage in ("evaluate", "all"):
        m = evaluate()
        for d, D in m["datasets"].items():
            for s, r in D["results"].items():
                md = r["models"]
                print(f"[{d:<15} {s:<4}] gpr {md['gpr_physics9']['MAE']:.3f}  "
                      f"best {r['best_model']} {md[r['best_model']]['MAE']:.3f}  "
                      f"mean {md['mean']['MAE']:.3f}  ridge {md['ridge_physics9']['MAE']:.3f}")
        print(f"K1: {m['k1']}")
    if stage in ("report", "all"):
        report()
        print("report: reports/metrics_v2.md, reports/figures/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
