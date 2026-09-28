"""Week-8 MLP guards. Tests that only read the committed reports run everywhere (CI has no
torch); tests that train an MLP need the optional CPU torch and are skipped without it."""
from __future__ import annotations

import csv
import json
import re
import shutil

import numpy as np
import pytest

from a3bx3 import mlp as M
from a3bx3.config import ROOT, Paths, load_config
from a3bx3.evaluate import load_dataset
from a3bx3.features import featurize_rows
from a3bx3.manifest import sha256_text
from a3bx3.splits import make_splits

CFG = load_config()
MC = CFG["mlp"]
OUT = ROOT / "reports" / "mlp_v2.json"
needs_torch = pytest.mark.skipif(not M.torch_available(), reason="optional torch not installed")


def _o():
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_mlp_outputs_present_per_dataset_and_split():
    o = _o()
    assert set(o["datasets"]) == set(MC["datasets"])
    for ds, D in o["datasets"].items():
        assert set(D) == set(M.splits_for(CFG, ds))
        for s, S in D.items():
            for m in ("mlp_physics9", "mlp_magpie", "xgb_physics9", "xgb_magpie", "gpr_physics9", "mean"):
                e = S["models"][m]
                assert e["MAE_CI95"][0] <= e["MAE"] <= e["MAE_CI95"][1], (ds, s, m)
            for m, others in MC["pairs"].items():
                for other in others:
                    p = S["pairs"][f"{m} vs {other}"]
                    want = "WIN" if p["CI95"][1] < 0 else ("LOSS" if p["CI95"][0] > 0 else "TIE")
                    assert p["outcome"] == want
    lc = o["learning_curve"]
    assert set(lc["models"]) == set(MC["learning_curve"]["models"])
    for v in lc["models"].values():
        assert {f"{f:.2f}" for f in MC["learning_curve"]["fractions"]} == set(v)


def test_reference_models_match_metrics_v2():
    """The non-MLP rows are read from the committed OOF predictions, so they must equal
    metrics_v2 (same folds, same rows)."""
    o = _o()
    m2 = json.loads((ROOT / "reports" / "metrics_v2.json").read_text(encoding="utf-8"))
    for ds in MC["datasets"]:
        for s in M.splits_for(CFG, ds):
            for m in ("gpr_physics9", "xgb_magpie", "mean"):
                assert o["datasets"][ds][s]["models"][m]["MAE"] == pytest.approx(
                    m2["datasets"][ds]["results"][s]["models"][m]["MAE"], abs=1e-4), (ds, s, m)


def test_verdict_follows_the_preregistered_rule():
    o = _o()
    v = o["verdict"]
    D = o["datasets"][MC["verdict_dataset"]]
    sp = MC["verdict_splits"]
    xgb_of = {"mlp_physics9": "xgb_physics9", "mlp_magpie": "xgb_magpie"}
    oc = lambda m, other, s: D[s]["pairs"][f"{m} vs {other}"]["outcome"]
    beats_xgb = (any(all(oc(m, xgb_of[m], s) == "WIN" for s in sp) for m in xgb_of)
                 and not any(oc(m, xgb_of[m], s) == "LOSS" for m in xgb_of for s in sp))
    beats_gpr = any(all(oc(m, "gpr_physics9", s) == "WIN" for s in sp) for m in xgb_of)
    assert v["MLP beats XGB"] == ("YES" if beats_xgb else "NO")
    assert v["MLP beats GPR"] == ("YES" if beats_gpr else "NO")


def test_preregistration_recorded_and_unchanged():
    ex = _o()["expectations"]
    assert re.fullmatch(r"[0-9a-f]{40}", ex["commit"] or "")
    assert ex["sha256_lf"] == sha256_text(ROOT / MC["expectations_file"])


def test_inner_validation_rows_inside_training_fold_and_disjoint_from_test():
    book = {}
    with open(ROOT / "reports" / "mlp_folds.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = book.setdefault((r["dataset"], r["model"], r["split"], int(r["fold"])),
                                {"test": set(), "val": {}})
            if r["role"] == "test":
                b["test"].add(r["formula"])
            else:
                b["val"].setdefault(r["seed"], set()).add(r["formula"])
    n = 0
    for ds in MC["datasets"]:
        rows = load_dataset(Paths(), ds)
        formula = np.array([r["formula"] for r in rows])
        splits = make_splits(rows, CFG["splits"], CFG["seed"])
        for m in M.MLP_MODELS:
            for s in M.splits_for(CFG, ds):
                for fi, (tr, te) in enumerate(splits[s].folds):
                    b = book[(ds, m, s, fi)]
                    assert b["test"] == set(formula[te])
                    assert len(b["val"]) == MC["final_seeds"]
                    for v in b["val"].values():
                        assert v and v <= set(formula[tr]) and not v & b["test"], (ds, m, s, fi)
                    n += 1
    assert n == len(M.MLP_MODELS) * sum(
        len(make_splits(load_dataset(Paths(), d), CFG["splits"], CFG["seed"])[s].folds)
        for d in MC["datasets"] for s in M.splits_for(CFG, d))


def test_mlp_module_imports_and_fails_clearly_without_torch(monkeypatch):
    monkeypatch.setattr(M, "torch", None)
    assert not M.torch_available()
    with pytest.raises(ImportError, match="requirements-torch.lock"):
        M.TorchMLP().fit(np.zeros((4, 2)), np.zeros(4))
    assert M.main([]) == 0            # CLI skips cleanly


@needs_torch
def test_torch_mlp_is_deterministic_and_uses_only_given_rows():
    rows = load_dataset(Paths(), "primary")
    X = featurize_rows(rows, "physics9")[0]
    y = np.array([r["Eg_eV"] for r in rows])
    g = np.array([r["formula"] for r in rows])
    tr, te = make_splits(rows, CFG["splits"], CFG["seed"])["lobo"].folds[0]
    y_blind = y.copy()
    y_blind[te] = np.nan                        # test labels must never be read
    pipe = M.make_pipeline(CFG).set_params(mlp__n_seeds=2, mlp__max_epochs=200)
    p1 = pipe.fit(X[tr], y_blind[tr], mlp__groups=g[tr]).predict(X[te])
    p2 = M.make_pipeline(CFG).set_params(mlp__n_seeds=2, mlp__max_epochs=200) \
        .fit(X[tr], y[tr], mlp__groups=g[tr]).predict(X[te])
    assert np.isfinite(p1).all() and np.array_equal(p1, p2)
    for v in pipe.named_steps["mlp"].val_rows_:
        assert max(v) < len(tr)


@needs_torch
@pytest.mark.slow
def test_mlp_v2_same_machine_determinism(tmp_path):
    """Two fresh runs on this machine give identical mlp_v2.json (CRLF-normalised). Not
    compared with the committed file: torch CPU results differ across OS / thread settings."""
    hashes = []
    for k in ("a", "b"):
        paths = Paths(tmp_path / k).ensure()
        shutil.copytree(ROOT / "data" / "processed", tmp_path / k / "data" / "processed", dirs_exist_ok=True)
        shutil.copy(ROOT / "reports" / "oof_predictions.csv", paths.reports / "oof_predictions.csv")
        M.run(paths)
        hashes.append(sha256_text(paths.reports / "mlp_v2.json"))
    assert hashes[0] == hashes[1]


def test_readme_results_table_is_generated_from_json():
    """The README table must equal a fresh render from metrics_v2.json + mlp_v2.json
    (CRLF-normalised), i.e. nobody typed numbers into it by hand."""
    from a3bx3 import readme_table
    s = (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    block = s[s.index(readme_table.START) + len(readme_table.START):s.index(readme_table.END)].strip("\n")
    assert block == readme_table.render()
    assert "MLP (PyTorch), physics-9" in block
