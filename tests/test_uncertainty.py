"""Week-6 uncertainty guards: coverage present per split/method, calibration fitted without
test rows, K2 recorded, holdout intervals, determinism."""
from __future__ import annotations

import csv
import json
import shutil

import numpy as np
import pytest

from a3bx3 import uncertainty as U
from a3bx3.config import ROOT, Paths, load_config
from a3bx3.evaluate import load_dataset
from a3bx3.features import featurize_rows
from a3bx3.splits import make_splits

CFG = load_config()
UC = CFG["uncertainty"]
OUT = ROOT / "reports" / "uncertainty_v2.json"


def _u():
    return json.loads(OUT.read_text(encoding="utf-8"))


def _csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_coverage_present_for_every_dataset_split_method_level():
    u = _u()
    methods = set(U.method_names(CFG))
    assert {"gpr_sigma", "gpr_sigma_scaled", "gpr_physics9_cvplus", "gpr_physics9_jackknife_plus",
            "ridge_physics9_cvplus", "mean_cvplus"} <= methods
    for ds in UC["datasets"]:
        assert set(u["coverage"][ds]) == set(UC["splits"])
        for s, S in u["coverage"][ds].items():
            assert set(S["methods"]) == methods, (ds, s)
            for m, e in S["methods"].items():
                for p in UC["report_levels"]:
                    v = e[f"{p:.2f}"]
                    assert 0.0 <= v["coverage"] <= 1.0 and v["mean_width"] > 0, (ds, s, m, p)
                    lo, hi = v["coverage_CI95"]
                    assert lo <= v["coverage"] <= hi, (ds, s, m, p)
                    assert v["n_unbounded"] == 0
                covs = [e[f"{p:.2f}"]["coverage"] for p in UC["report_levels"]]
                widths = [e[f"{p:.2f}"]["mean_width"] for p in UC["report_levels"]]
                assert widths == sorted(widths), (ds, s, m)     # nested intervals
                assert covs == sorted(covs), (ds, s, m)
            assert S["n_predictions"] == (len(load_dataset(Paths(), ds)) *
                                          (CFG["splits"]["gkf"]["n_repeats"] if s == "gkf" else 1))


def test_exchangeability_caveat_recorded_for_every_split():
    ex = _u()["exchangeability"]
    assert "violated" in ex["lobo"] and "violated" in ex["loao"] and "holds" in ex["gkf"]


def test_calibration_rows_never_include_the_fold_test_rows():
    """Fold bookkeeping: the rows used to fit the sigma scale c are a subset of the outer
    training fold, and disjoint from its test rows, for every dataset x split x fold."""
    book = _csv(ROOT / "reports" / "uncertainty_folds.csv")
    by = {}
    for r in book:
        by.setdefault((r["dataset"], r["split"], r["fold"]), {"test": set(), "calibration": set()})[
            r["role"]].add(r["formula"])
    for ds in UC["datasets"]:
        rows = load_dataset(Paths(), ds)
        formula = np.array([r["formula"] for r in rows])
        splits = make_splits(rows, CFG["splits"], CFG["seed"])
        for s in UC["splits"]:
            for fi, (tr, te) in enumerate(splits[s].folds):
                b = by[(ds, s, str(fi))]
                assert b["test"] == set(formula[te])
                assert not b["calibration"] & b["test"], (ds, s, fi)
                assert b["calibration"] == set(formula[tr]), (ds, s, fi)


def test_test_labels_are_never_read_when_building_intervals():
    """Structural check: replace the outer test labels by NaN. If any method touched them the
    intervals would be NaN or differ from the committed run."""
    rows = load_dataset(Paths(), "primary")
    X = featurize_rows(rows, "physics9")[0]
    y = np.array([r["Eg_eV"] for r in rows])
    g = np.array([r["formula"] for r in rows])
    tr, te = make_splits(rows, CFG["splits"], CFG["seed"])["lobo"].folds[0]
    y_blind = y.copy()
    y_blind[te] = np.nan
    r = U.fold_intervals(CFG, X, y_blind, g, tr, te, [0.9], CFG["seed"] + 0)
    committed = {(x["method"], x["formula"]): (float(x["lower_90"]), float(x["upper_90"]))
                 for x in _csv(ROOT / "reports" / "uncertainty_intervals_90.csv")
                 if x["dataset"] == "primary" and x["split"] == "lobo" and x["fold"] == "0"}
    assert committed
    for m, (lo, hi) in r["intervals"].items():
        assert np.isfinite(lo).all() and np.isfinite(hi).all(), m
        for j, i in enumerate(te):
            key = (m, rows[i]["formula"])
            if key in committed:
                assert (lo[j, 0], hi[j, 0]) == pytest.approx(committed[key], abs=1e-4), key


def test_k2_recorded_with_ci_and_verdict():
    k = _u()["k2"]
    for ds in UC["k2"]["datasets"]:
        d = k["datasets"][ds]
        assert d["verdict"] in {"PASS", "PASS vs mean only", "FAIL"}
        for lab in ("vs_mean", "vs_best_baseline"):
            v = d[lab]
            assert v["skill_CI95"][0] <= v["skill"] <= v["skill_CI95"][1]
            assert v["pass"] == (v["skill"] >= UC["k2"]["min_skill"] and v["MAE_diff_CI95"][1] < 0)
        m = json.loads((ROOT / "reports" / "metrics_v2.json").read_text(encoding="utf-8"))
        lobo = m["datasets"][ds]["results"]["lobo"]
        assert d["best_model"] == lobo["best_model"] and d["best_baseline"] == lobo["best_baseline"]
        assert d["best_model_MAE"] == pytest.approx(lobo["models"][d["best_model"]]["MAE"], abs=1e-4)


def test_holdout_intervals_for_the_three_targets():
    h = _u()["sr3bix3_holdout"]
    assert set(h["compounds"]) == set(CFG["data"]["holdout_formulas"])
    for f, e in h["compounds"].items():
        for m in U.method_names(CFG):
            iv = e["intervals"][m]["0.90"]
            assert iv["lower"] <= iv["upper"]
            assert iv["contains_target"] == (iv["lower"] <= e["target"] <= iv["upper"])


@pytest.mark.slow
def test_uncertainty_v2_is_byte_identical_on_rerun(tmp_path):
    """Rerun the uncertainty step from the committed processed data and OOF predictions into a
    temp dir; the JSON must equal the committed file byte for byte (same environment)."""
    paths = Paths(tmp_path).ensure()
    for sub in ("processed", "interim"):
        shutil.copytree(ROOT / "data" / sub, tmp_path / "data" / sub, dirs_exist_ok=True)
    shutil.copy(ROOT / "reports" / "oof_predictions.csv", paths.reports / "oof_predictions.csv")
    out = U.run(paths)
    fresh = paths.reports / "uncertainty_v2.json"
    if out["environment"] == _u()["environment"]:
        assert fresh.read_bytes() == OUT.read_bytes(), \
            "reports/uncertainty_v2.json is stale: rerun `python run_all.py uncertainty` and commit"
    else:
        pytest.skip("different package versions than the committed run")
