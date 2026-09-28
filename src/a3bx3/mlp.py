"""Week 8: a small PyTorch MLP on the same features and the same outer folds as metrics_v2.

n ~ 39 is tiny for a neural network (the 2x64 variant has ~5k weights); that is the point of
the comparison. Everything below is fitted on the outer TRAINING fold only:
  * Pipeline(VarianceThreshold, StandardScaler, TorchMLP); the target is standardised inside
    TorchMLP.fit, i.e. inside the fold
  * a 16-point grid (width x depth x weight decay x dropout) chosen by an inner GroupKFold(5) by
    formula (GridSearchCV, one seed per fit), then the best setting refitted with 5 seeds and
    averaged
  * early stopping on an inner validation split (20 %, grouped by formula) drawn from the rows
    of that fit only; the rows used are recorded for bookkeeping

The other models are NOT recomputed: their out-of-fold predictions are read from
reports/oof_predictions.csv (same folds, same rows). The verdict follows the rule
pre-registered in reports/mlp_expectations.md.

torch is optional (requirements-torch.lock, CPU-only). Without it this module imports, but
fitting raises a clear error; tests that need torch are skipped.

Outputs: reports/mlp_v2.json, reports/mlp_v2.md, reports/mlp_folds.csv, reports/mlp_oof.csv,
reports/figures/mlp_*.png
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
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import utf8_stdio
from .config import ROOT, Paths, load_config
from .evaluate import _read_oof, load_dataset
from .features import featurize_rows
from .manifest import sha256_text, write_atomic
from .models import fit as fit_spec, model_specs
from .splits import RepeatedGroupKFold, assert_holdout_excluded, check_folds, make_splits

try:  # optional dependency
    import torch
except ImportError:  # pragma: no cover - exercised on machines without torch
    torch = None

R = 4
MLP_MODELS = {"mlp_physics9": "physics9", "mlp_magpie": "magpie"}


def torch_available() -> bool:
    return torch is not None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is not installed. Install the optional CPU build with: "
                          "pip install -r requirements-torch.lock")


def _deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class TorchMLP(RegressorMixin, BaseEstimator):
    """Tiny fully-connected regressor. fit() standardises y, early-stops on a grouped inner
    validation split of the rows it is given, and averages `n_seeds` independently trained
    networks. float64 throughout to limit numerical drift."""

    def __init__(self, hidden=16, depth=1, weight_decay=1e-3, dropout=0.0, lr=1e-2,
                 max_epochs=1000, patience=100, val_fraction=0.2, n_seeds=1, random_state=0):
        self.hidden, self.depth, self.weight_decay, self.dropout = hidden, depth, weight_decay, dropout
        self.lr, self.max_epochs, self.patience, self.val_fraction = lr, max_epochs, patience, val_fraction
        self.n_seeds, self.random_state = n_seeds, random_state

    def _net(self, nf):
        layers, d = [], nf
        for _ in range(self.depth):
            layers += [torch.nn.Linear(d, self.hidden), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            d = self.hidden
        layers.append(torch.nn.Linear(d, 1))
        return torch.nn.Sequential(*layers).double()

    def fit(self, X, y, groups=None):
        _require_torch()
        _deterministic()
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        groups = np.arange(len(y)) if groups is None else np.asarray(groups)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        ys = (y - self.y_mean_) / self.y_std_
        self.nets_, self.val_rows_, self.epochs_ = [], [], []
        for k in range(self.n_seeds):
            seed = int(self.random_state) * 1000 + k
            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=self.val_fraction,
                                            random_state=seed).split(X, ys, groups))
            torch.manual_seed(seed)
            net = self._net(X.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(ys[tr]).unsqueeze(1)
            Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(ys[va]).unsqueeze(1)
            best, best_state, best_ep, wait = np.inf, None, 0, 0
            for ep in range(self.max_epochs):
                net.train()
                opt.zero_grad()
                loss = torch.mean((net(Xt) - yt) ** 2)
                loss.backward()
                opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(torch.mean((net(Xv) - yv) ** 2))
                if v < best - 1e-9:
                    best, best_ep, wait = v, ep, 0
                    best_state = {n: p.detach().clone() for n, p in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets_.append(net)
            self.val_rows_.append([int(i) for i in va])
            self.epochs_.append(best_ep + 1)
        return self

    def predict(self, X):
        _require_torch()
        Xt = torch.from_numpy(np.asarray(X, float))
        with torch.no_grad():
            p = np.mean([n(Xt).numpy().ravel() for n in self.nets_], axis=0)
        return p * self.y_std_ + self.y_mean_


def make_pipeline(cfg) -> Pipeline:
    f = cfg["mlp"]["fixed"]
    return Pipeline([("var", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                     ("mlp", TorchMLP(lr=f["lr"], max_epochs=f["max_epochs"], patience=f["patience"],
                                      val_fraction=f["val_fraction"],
                                      n_seeds=cfg["mlp"]["inner_search_seeds"]))])


def fit_mlp(cfg, X, y, groups, seed):
    """Nested: grid by inner GroupKFold on (X, y) = the outer training fold, then refit the
    best setting with `final_seeds` seeds on the same rows. Returns the fitted pipeline."""
    _require_torch()
    mc = cfg["mlp"]
    base = make_pipeline(cfg).set_params(mlp__random_state=seed)
    gs = GridSearchCV(base, mc["grid"], cv=RepeatedGroupKFold(cfg["splits"]["inner"]["n_splits"], 1, seed),
                      scoring="neg_mean_absolute_error", refit=False, n_jobs=1)
    gs.fit(X, y, groups=groups, mlp__groups=groups)
    best = clone(base).set_params(**gs.best_params_, mlp__n_seeds=mc["final_seeds"])
    best.fit(X, y, mlp__groups=groups)
    return best, gs.best_params_


def _fold_task(cfg, X, y, groups, tr, te, seed):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        pipe, params = fit_mlp(cfg, X[tr], y[tr], groups[tr], seed)
    mlp = pipe.named_steps["mlp"]
    return {"pred": pipe.predict(X[te]), "params": params,
            "val_rows": [[int(tr[i]) for i in v] for v in mlp.val_rows_], "epochs": mlp.epochs_}


def _lc_task(cfg, name, X, y, groups, tr, te, frac, seed):
    rng = np.random.RandomState(seed)
    sub = np.sort(rng.choice(tr, size=max(10, int(round(frac * len(tr)))), replace=False)) if frac < 1 else tr
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if name.startswith("mlp"):
            pipe, _ = fit_mlp(cfg, X[sub], y[sub], groups[sub], seed)
            p = pipe.predict(X[te])
        else:
            spec = {s.name: s for s in model_specs(cfg)}[name]
            est = spec.build(X.shape[1])
            fit_spec(spec, est, X[sub], y[sub], groups[sub])
            p = est.predict(X[te])
    return {"pred": np.asarray(p), "n_train": int(len(sub)), "sub": [int(i) for i in sub]}


# ── evaluation ───────────────────────────────────────────────────────────────
def _r(x, d=R):
    return None if x is None else round(float(x), d)


def _expectations_meta(cfg) -> dict:
    p = ROOT / cfg["mlp"]["expectations_file"]
    try:
        h = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(p.relative_to(ROOT))],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        h = None
    return {"file": cfg["mlp"]["expectations_file"], "sha256_lf": sha256_text(p), "commit": h or None}


def _row_err(preds_by_rep, formulas, y_of):
    """preds_by_rep: {repeat: {formula: pred}} -> per-formula abs error averaged over repeats."""
    reps = sorted(preds_by_rep)
    E = np.array([[abs(preds_by_rep[k][f] - y_of[f]) for f in formulas] for k in reps])
    return E.mean(0), E.mean(1)


def splits_for(cfg, ds) -> list[str]:
    """MLP splits = the configured ones that metrics_v2 also has for this dataset (LOO exists
    for primary only), so every MLP row has reference rows on the same folds."""
    return [s for s in cfg["mlp"]["splits"] if s in cfg["evaluate"]["splits_by_dataset"][ds]]


def run(paths: Paths | None = None, n_jobs: int = -1) -> dict:
    _require_torch()
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    mc = cfg["mlp"]
    holdout = set(cfg["data"]["holdout_formulas"])
    tasks, keys, data = [], [], {}
    for ds in mc["datasets"]:
        rows = load_dataset(paths, ds)
        assert_holdout_excluded([r["formula"] for r in rows], holdout)
        y = np.array([r["Eg_eV"] for r in rows])
        groups = np.array([r["formula"] for r in rows])
        splits = make_splits(rows, cfg["splits"], cfg["seed"])
        data[ds] = (rows, y, groups, splits)
        for fs in mc["feature_sets"]:
            X = featurize_rows(rows, fs)[0]
            for sname in splits_for(cfg, ds):
                s = splits[sname]
                check_folds(rows, s.folds, holdout)
                per_rep = len(s.folds) // s.n_repeats
                for fi, (tr, te) in enumerate(s.folds):
                    tasks.append(delayed(_fold_task)(cfg, X, y, groups, tr, te, cfg["seed"] + fi))
                    keys.append((ds, f"mlp_{fs}", sname, fi, fi // per_rep, tr, te))
    lc = mc["learning_curve"]
    lrows, ly, lgroups, lsplits = data[lc["dataset"]]
    lX = {fs: featurize_rows(lrows, fs)[0] for fs in ("physics9", "magpie")}
    lc_keys = []
    for fi, (tr, te) in enumerate(lsplits[lc["split"]].folds):
        for frac in lc["fractions"]:
            for name in lc["models"]:
                fs = name.split("_", 1)[1]
                tasks.append(delayed(_lc_task)(cfg, name, lX[fs], ly, lgroups, tr, te, frac,
                                               cfg["seed"] + 100 * fi + int(frac * 100)))
                lc_keys.append((fi, frac, name, te))
    res = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    fold_res, lc_res = res[:len(keys)], res[len(keys):]

    # bookkeeping + MLP OOF
    book, oof_mlp, params_count = [], [], {}
    for k, r in zip(keys, fold_res):
        ds, m, sname, fi, rep, tr, te = k
        rows = data[ds][0]
        for j, i in enumerate(te):
            oof_mlp.append({"dataset": ds, "split": sname, "repeat": rep, "fold": fi, "model": m,
                            "formula": rows[i]["formula"], "y_pred": _r(r["pred"][j], 6)})
            book.append({"dataset": ds, "model": m, "split": sname, "fold": fi, "role": "test",
                         "seed": "", "formula": rows[i]["formula"]})
        for sd, v in enumerate(r["val_rows"]):
            for i in v:
                book.append({"dataset": ds, "model": m, "split": sname, "fold": fi,
                             "role": "inner_val", "seed": sd, "formula": rows[i]["formula"]})
        key = json.dumps(r["params"], sort_keys=True)
        params_count.setdefault(f"{ds}/{m}", {}).setdefault(key, 0)
        params_count[f"{ds}/{m}"][key] += 1

    out = {"_readme": ("Small PyTorch MLP on the same outer folds as metrics_v2; other models' "
                       "out-of-fold predictions are read from reports/oof_predictions.csv, not "
                       "recomputed. MAE in eV with paired bootstrap 95 % CIs over formulas. "
                       "Verdict per the rule pre-registered in reports/mlp_expectations.md."),
           "expectations": _expectations_meta(cfg),
           "mlp_spec": {"grid": mc["grid"], "fixed": mc["fixed"],
                        "inner_search_seeds": mc["inner_search_seeds"], "final_seeds": mc["final_seeds"],
                        "preprocessing": "VarianceThreshold -> StandardScaler (features); y standardised inside fit",
                        "note": "n ~ 39: the 2x64 network has more weights than rows by two orders of magnitude"},
           "chosen_params": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in params_count.items()},
           "datasets": {}, "environment": {p: _pkg_version(p) for p in
                                           ("numpy", "scikit-learn", "torch", "xgboost")}}
    oof_ref = _read_oof(paths)
    rng = np.random.RandomState(cfg["seed"] + 8)
    compare = ["mean", "halide_rule", "ridge_physics9", "ridge_magpie", "gpr_physics9",
               "rf_physics9", "rf_magpie", "xgb_physics9", "xgb_magpie"]
    for ds in mc["datasets"]:
        rows = data[ds][0]
        formulas = [r["formula"] for r in rows]
        y_of = {r["formula"]: r["Eg_eV"] for r in rows}
        boot = rng.randint(0, len(rows), size=(mc["bootstrap_n"], len(rows)))
        D = {}
        for sname in splits_for(cfg, ds):
            preds = {}
            for r in oof_ref:
                if r["dataset"] == ds and r["split"] == sname and r["model"] in compare:
                    preds.setdefault(r["model"], {}).setdefault(r["repeat"], {})[r["formula"]] = r["y_pred"]
            for r in oof_mlp:
                if r["dataset"] == ds and r["split"] == sname:
                    preds.setdefault(r["model"], {}).setdefault(r["repeat"], {})[r["formula"]] = r["y_pred"]
            err, models = {}, {}
            for m in list(MLP_MODELS) + compare:
                e, rep_mae = _row_err(preds[m], formulas, y_of)
                err[m] = e
                b = e[boot].mean(1)
                models[m] = {"MAE": _r(e.mean()), "MAE_CI95": [_r(np.percentile(b, 2.5)), _r(np.percentile(b, 97.5))]}
                if len(rep_mae) > 1:
                    models[m]["MAE_repeat_sd"] = _r(rep_mae.std(ddof=1))
            pairs = {}
            for m, others in mc["pairs"].items():
                for o in others:
                    d = (err[m] - err[o])[boot].mean(1)
                    ci = [_r(np.percentile(d, 2.5)), _r(np.percentile(d, 97.5))]
                    pairs[f"{m} vs {o}"] = {"delta_MAE": _r(err[m].mean() - err[o].mean()), "CI95": ci,
                                            "outcome": "WIN" if ci[1] < 0 else ("LOSS" if ci[0] > 0 else "TIE")}
            D[sname] = {"models": models, "pairs": pairs}
        out["datasets"][ds] = D
    out["verdict"] = verdict(out, mc)
    out["learning_curve"] = learning_curve(lc, lc_keys, lc_res, lrows, ly)
    write_atomic(paths.reports / "mlp_v2.json", (json.dumps(out, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _csv(paths.reports / "mlp_folds.csv", book, ["dataset", "model", "split", "fold", "role", "seed", "formula"])
    _csv(paths.reports / "mlp_oof.csv", oof_mlp, ["dataset", "split", "repeat", "fold", "model", "formula", "y_pred"])
    _figures(paths, out)
    _markdown(paths, out)
    return out


def verdict(out, mc) -> dict:
    D = out["datasets"][mc["verdict_dataset"]]
    sp = mc["verdict_splits"]

    def outcome(m, o, s):
        return D[s]["pairs"][f"{m} vs {o}"]["outcome"]
    xgb_of = {"mlp_physics9": "xgb_physics9", "mlp_magpie": "xgb_magpie"}
    beats_xgb = (any(all(outcome(m, xgb_of[m], s) == "WIN" for s in sp) for m in MLP_MODELS)
                 and not any(outcome(m, xgb_of[m], s) == "LOSS" for m in MLP_MODELS for s in sp))
    beats_gpr = any(all(outcome(m, "gpr_physics9", s) == "WIN" for s in sp) for m in MLP_MODELS)
    table = {f"{m} vs {o}": {s: outcome(m, o, s) for s in sp}
             for m, others in mc["pairs"].items() for o in others}
    exp = {"mlp vs gpr_physics9": {"lobo": "LOSS", "loao": "TIE or LOSS"},
           "mlp vs xgb": {"lobo": "TIE", "loao": "TIE"}, "mlp vs mean": {"lobo": "WIN"},
           "milestone": "NO / NO"}
    return {"dataset": mc["verdict_dataset"], "splits": sp,
            "MLP beats XGB": "YES" if beats_xgb else "NO",
            "MLP beats GPR": "YES" if beats_gpr else "NO",
            "outcomes": table, "pre_registered_expectation": exp}


def learning_curve(lc, keys, res, rows, y) -> dict:
    out = {"dataset": lc["dataset"], "split": lc["split"], "fractions": lc["fractions"], "models": {}}
    for name in lc["models"]:
        out["models"][name] = {}
        for frac in lc["fractions"]:
            errs, ntr, rep_mae = [], [], {}
            for (fi, fr, nm, te), r in zip(keys, res):
                if nm == name and fr == frac:
                    e = np.abs(r["pred"] - y[te])
                    errs += list(e)
                    ntr.append(r["n_train"])
                    rep_mae.setdefault(fi // 5, []).extend(e)
            rm = [np.mean(v) for _, v in sorted(rep_mae.items())]
            out["models"][name][f"{frac:.2f}"] = {"MAE": _r(np.mean(errs)), "mean_n_train": _r(np.mean(ntr), 1),
                                                   "repeat_MAE_p2_5": _r(np.percentile(rm, 2.5)),
                                                   "repeat_MAE_p97_5": _r(np.percentile(rm, 97.5))}
    return out


def _csv(path, rows, cols):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    write_atomic(path, buf.getvalue().encode("utf-8"))


def _figures(paths, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["mlp_physics9", "mlp_magpie", "xgb_physics9", "xgb_magpie", "gpr_physics9", "ridge_physics9", "mean"]
    colors = ["#b2182b", "#ef8a62", "#1b7837", "#7fbf7b", "#2166ac", "#999999", "#cccccc"]
    for ds, D in out["datasets"].items():
        fig, axes = plt.subplots(1, len(D), figsize=(3.1 * len(D), 3.4), dpi=150, sharey=False)
        for ax, (s, S) in zip(axes, D.items()):
            for i, (m, c) in enumerate(zip(show, colors)):
                e = S["models"][m]
                ax.bar(i, e["MAE"], color=c, yerr=[[max(0, e["MAE"] - e["MAE_CI95"][0])],
                                                   [max(0, e["MAE_CI95"][1] - e["MAE"])]], capsize=2)
            ax.set_xticks(range(len(show)))
            ax.set_xticklabels(show, rotation=60, fontsize=6.5)
            ax.set_title(f"{ds} · {s}", fontsize=8.5)
        axes[0].set_ylabel("held-out MAE (eV), 95 % CI")
        fig.tight_layout()
        fig.savefig(paths.figures / f"mlp_vs_models_{ds}.png", metadata={"Software": None})
        plt.close(fig)

    L = out["learning_curve"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    for name, v in L["models"].items():
        xs = [v[f"{f:.2f}"]["mean_n_train"] for f in L["fractions"]]
        ys = [v[f"{f:.2f}"]["MAE"] for f in L["fractions"]]
        lo = [y_ - v[f"{f:.2f}"]["repeat_MAE_p2_5"] for y_, f in zip(ys, L["fractions"])]
        hi = [v[f"{f:.2f}"]["repeat_MAE_p97_5"] - y_ for y_, f in zip(ys, L["fractions"])]
        ax.errorbar(xs, ys, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], marker="o", capsize=2, label=name)
    ax.set_xlabel("training rows per outer fold (subsampled from the training fold)")
    ax.set_ylabel("held-out MAE (eV)")
    ax.set_title("Learning curve, primary, GroupKFold(5)x10", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "mlp_learning_curve.png", metadata={"Software": None})
    plt.close(fig)


def _markdown(paths, out):
    e = out["expectations"]
    v = out["verdict"]
    L = ["# mlp_v2 — generated by `a3bx3.mlp`; do not edit by hand", "", out["_readme"], "",
         f"Pre-registered rule: `{e['file']}` (commit `{e['commit']}`).", "",
         f"## Milestone verdict ({v['dataset']}, splits {', '.join(v['splits'])})", "",
         f"- **MLP beats XGB: {v['MLP beats XGB']}**", f"- **MLP beats GPR: {v['MLP beats GPR']}**", "",
         "| pair | " + " | ".join(v["splits"]) + " |", "|---|" + "---|" * len(v["splits"])]
    for k, o in v["outcomes"].items():
        L.append(f"| {k} | " + " | ".join(o[s] for s in v["splits"]) + " |")
    L += ["", f"Expected (pre-registered): {json.dumps(v['pre_registered_expectation'])}", ""]
    for ds, D in out["datasets"].items():
        L += [f"## {ds}", ""]
        for s, S in D.items():
            L += [f"### {s}", "", "| model | MAE [95 % CI] |", "|---|---|"]
            for m, x in sorted(S["models"].items(), key=lambda kv: kv[1]["MAE"]):
                L.append(f"| {'**' + m + '**' if m.startswith('mlp') else m} | {x['MAE']:.3f} [{x['MAE_CI95'][0]:.3f}, {x['MAE_CI95'][1]:.3f}] |")
            L += ["", "| pair | Δ MAE [95 % CI] | outcome |", "|---|---|---|"]
            for k, p in S["pairs"].items():
                L.append(f"| {k} | {p['delta_MAE']:+.3f} [{p['CI95'][0]:+.3f}, {p['CI95'][1]:+.3f}] | {p['outcome']} |")
            L.append("")
    lc = out["learning_curve"]
    L += ["## Learning curve (primary, GKF×10; MAE eV, [2.5, 97.5 %] over the 10 repeats)", "",
          "| model | " + " | ".join(f"{int(f*100)} %" for f in lc["fractions"]) + " |", "|---|" + "---|" * len(lc["fractions"])]
    for m, v_ in lc["models"].items():
        L.append(f"| {m} | " + " | ".join(
            f"{v_[f'{f:.2f}']['MAE']:.3f} [{v_[f'{f:.2f}']['repeat_MAE_p2_5']:.3f}, {v_[f'{f:.2f}']['repeat_MAE_p97_5']:.3f}] (n≈{v_[f'{f:.2f}']['mean_n_train']:.0f})"
            for f in lc["fractions"]) + " |")
    L += ["", "## Hyper-parameters chosen by the inner search (count of outer folds)", ""]
    for k, c in out["chosen_params"].items():
        top = list(c.items())[:3]
        L.append(f"- {k}: " + "; ".join(f"{p} ×{n}" for p, n in top))
    write_atomic(paths.reports / "mlp_v2.md", "\n".join(L).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    if not torch_available():
        print("mlp: PyTorch not installed; skipping (pip install -r requirements-torch.lock)", file=sys.stderr)
        return 0
    out = run()
    v = out["verdict"]
    print(f"MLP beats XGB: {v['MLP beats XGB']} | MLP beats GPR: {v['MLP beats GPR']}")
    for k, o in v["outcomes"].items():
        print(f"  {k:<32} {o}")
    for ds, D in out["datasets"].items():
        for s, S in D.items():
            m = S["models"]
            print(f"[{ds:<13} {s:<4}] mlp_p9 {m['mlp_physics9']['MAE']:.3f} mlp_mg {m['mlp_magpie']['MAE']:.3f} "
                  f"xgb_p9 {m['xgb_physics9']['MAE']:.3f} xgb_mg {m['xgb_magpie']['MAE']:.3f} "
                  f"gpr {m['gpr_physics9']['MAE']:.3f} mean {m['mean']['MAE']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
