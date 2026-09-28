"""Week-7 guards: explanation outputs present, SHAP backgrounds drawn from training rows only,
pre-registration recorded, verdicts follow the stated rule, notebook is narrative-only."""
from __future__ import annotations

import ast
import csv
import json
import re
import shutil

import numpy as np
import pytest

from a3bx3 import explain as X
from a3bx3.config import ROOT, Paths, load_config
from a3bx3.evaluate import load_dataset
from a3bx3.manifest import sha256_text
from a3bx3.splits import make_splits

CFG = load_config()
EC = CFG["explain"]
OUT = ROOT / "reports" / "explain_v2.json"
NB = ROOT / "notebooks" / "01_story.ipynb"
VERDICTS = {"CONFIRMED", "CONTRADICTED", "UNCLEAR", "REPORTED", "SUPPLEMENT (4 folds, no verdict)"}


def _e():
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_outputs_present_per_model_and_split():
    e = _e()
    assert set(e["models"]) == set(EC["models"])
    for m, mo in e["models"].items():
        assert set(mo["splits"]) == set(EC["splits"]), m
        for s, so in mo["splits"].items():
            assert so["n_folds"] == len(make_splits(load_dataset(Paths(), EC["dataset"]),
                                                    CFG["splits"], CFG["seed"])[s].folds)
            assert len(so["top10"]) == (9 if mo["feature_set"] == "physics9" else 10)
            assert so["mean_abs_shap"]
            assert {"kendall_tau_all_features", "kendall_tau_top10"} <= set(so["rank_stability"])
            assert so["permutation_importance_top10"]
            assert {f"{a}:{b}-{c}" for a, b, c, _, _ in X.PAIRS} == set(so["substitution"])
            for v in so["substitution"].values():
                assert v["verdict"] in VERDICTS
            if mo["feature_set"] == "physics9":
                assert {f for f, _, _ in X.SLOPES} == set(so["shap_slopes"])
                assert "site_groups" in so
    for fig in ("explain_substitution.png", "explain_rank_stability_gpr_physics9.png",
                "explain_dependence_gpr_physics9.png",
                *[f"explain_{k}_{m}.png" for k in ("bar", "beeswarm") for m in EC["models"]]):
        assert (ROOT / "reports" / "figures" / fig).exists(), fig


def test_shap_background_drawn_from_training_rows_only():
    """Fold bookkeeping: for every model x split x fold, the background rows are a subset of
    that fold's training rows and never include a held-out row."""
    book = {}
    with open(ROOT / "reports" / "explain_folds.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            book.setdefault((r["model"], r["split"], int(r["fold"])),
                            {"test": set(), "background": set()})[r["role"]].add(r["formula"])
    rows = load_dataset(Paths(), EC["dataset"])
    formula = np.array([r["formula"] for r in rows])
    splits = make_splits(rows, CFG["splits"], CFG["seed"])
    n = 0
    for m in EC["models"]:
        for s in EC["splits"]:
            for fi, (tr, te) in enumerate(splits[s].folds):
                b = book[(m, s, fi)]
                assert b["test"] == set(formula[te]), (m, s, fi)
                assert b["background"] <= set(formula[tr]), (m, s, fi)
                assert not b["background"] & b["test"], (m, s, fi)
                assert len(b["background"]) == min(EC["background_n"], len(tr))
                n += 1
    assert n == len(EC["models"]) * sum(len(splits[s].folds) for s in EC["splits"])


def test_preregistration_recorded_and_unchanged():
    ex = _e()["expectations"]
    assert re.fullmatch(r"[0-9a-f]{40}", ex["commit"] or ""), "expectations commit hash missing"
    assert ex["sha256_lf"] == sha256_text(ROOT / EC["expectations_file"]), \
        "expectations file changed after the results were computed"


def test_verdicts_follow_the_preregistered_rule():
    """Recompute each GKF verdict from its stored mean and percentile range."""
    for m, mo in _e()["models"].items():
        g = mo["splits"]["gkf"]
        items = list(g["substitution"].values()) + list(g.get("shap_slopes", {}).values())
        for v in items:
            sign = v["expected_sign"]
            if sign == 0:
                assert v["verdict"] == "REPORTED"
                continue
            if v.get("mean") is None:
                assert v["verdict"] == "UNCLEAR"
                continue
            excl = v["p2_5"] > 0 or v["p97_5"] < 0
            want = ("CONFIRMED" if np.sign(v["mean"]) == sign and excl else
                    "CONTRADICTED" if np.sign(v["mean"]) == -sign and excl else "UNCLEAR")
            assert v["verdict"] == want, (m, v)


def test_physics_check_has_every_expectation():
    pc = _e()["physics_check"]
    for m in EC["models"]:
        keys = " ".join(pc[m])
        for eid in ("E1", "E2", "E3"):
            assert eid in keys, (m, eid)
    assert {"E4 dchi_BX slope>0", "E6 A and X groups > B group (SHAP)"} <= set(pc["gpr_physics9"])


# ── notebook: narrative only ─────────────────────────────────────────────────
def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def test_notebook_executed_without_errors_and_small():
    nb = _nb()
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c.get("outputs") for c in code), "notebook was not executed"
    for c in code:
        for o in c["outputs"]:
            assert o["output_type"] != "error", o.get("ename")
            assert "image/png" not in o.get("data", {}), "embedded images: link figures instead"
    assert NB.stat().st_size < 200_000


def test_notebook_imports_nothing_that_computes():
    """Static check: code cells may import only json/pathlib and may read only committed
    report/data JSON files; no fitting, predicting or pipeline calls."""
    nb = _nb()
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= {"json", "pathlib"}
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in {"json", "pathlib"}
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"fit", "predict", "run", "run_offline", "train", "evaluate"}
    reads = re.findall(r'load\("([^"]+)"\)', src)
    assert reads and all(re.fullmatch(r"(reports/[\w]+\.json|data/[\w]+\.json)", r) for r in reads)
    for r in reads:
        assert (ROOT / r).exists(), r


@pytest.mark.slow
def test_explain_is_deterministic_on_the_same_machine(tmp_path):
    """Two fresh runs on this machine must give identical JSON (CRLF-normalised). Not compared
    with the committed file: SHAP of a GPR differs slightly across platforms."""
    outs = []
    for k in ("a", "b"):
        paths = Paths(tmp_path / k).ensure()
        shutil.copytree(ROOT / "data" / "processed", tmp_path / k / "data" / "processed",
                        dirs_exist_ok=True)
        X.run(paths)
        outs.append(sha256_text(paths.reports / "explain_v2.json"))
    assert outs[0] == outs[1]
