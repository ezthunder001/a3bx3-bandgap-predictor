"""v2 pipeline guards (design section 7). The v1 tests in test_metrics.py are untouched.

Run:  python -m pytest -q tests            (all, incl. the slow determinism rerun)
      python -m pytest -q tests -m "not slow"
"""
from __future__ import annotations

import ast
import csv
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from a3bx3 import curate as curate_mod
from a3bx3.config import ROOT, Paths, load_config
from a3bx3.features import CHI_A, CHI_B, CHI_X, RAD_A, RAD_B, RAD_X, read_csv
from a3bx3.fetch import literature, mp
from a3bx3.manifest import verify
from a3bx3.models import HalideRule, model_specs, unwrap
from a3bx3.splits import (RepeatedGroupKFold, assert_holdout_excluded, check_folds,
                          make_splits)

CFG = load_config()
PATHS = Paths()
HOLDOUT = set(CFG["data"]["holdout_formulas"])
BASELINES = {"mean", "halide_rule", "ridge_physics9", "ridge_magpie"}
METRICS = ROOT / "reports" / "metrics_v2.json"


def _family():
    return read_csv(PATHS.processed / "family.csv")


def _metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))


# ── data integrity ────────────────────────────────────────────────────────────
def test_raw_manifest_matches():
    assert verify() == []


def test_literature_raw_copy_is_the_v1_file():
    """v1 scripts read data/a3bx3_literature.csv; v2 reads the raw copy. They must not drift."""
    assert (ROOT / "data" / "a3bx3_literature.csv").read_bytes() == \
        (PATHS.literature / "a3bx3_literature_v1.csv").read_bytes()


def test_physics9_tables_are_the_v1_tables():
    tree = ast.parse((ROOT / "train_a3bx3_family.py").read_text(encoding="utf-8"))
    v1 = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
          for t in n.targets if isinstance(t, ast.Name) and t.id in
          {"RAD_A", "CHI_A", "RAD_B", "CHI_B", "RAD_X", "CHI_X"}}
    v2 = {"RAD_A": RAD_A, "CHI_A": CHI_A, "RAD_B": RAD_B, "CHI_B": CHI_B,
          "RAD_X": RAD_X, "CHI_X": CHI_X}
    assert v1 == v2


def test_curated_family_is_deduped_and_holdout_split_out():
    fam = _family()
    hold = read_csv(PATHS.processed / "holdout_sr3bix3.csv")
    formulas = [r["formula"] for r in fam]
    assert len(formulas) == len(set(formulas)) == 23
    assert {r["formula"] for r in hold} == HOLDOUT
    assert not HOLDOUT & set(formulas)


def test_every_data_row_has_a_doi_or_db_id_known_issues_listed():
    """Every literature row needs a well-formed DOI. Rows that do not have one must be the
    exact documented known-issue list in configs/v2.yaml: a new offender fails, and a fixed
    one fails until the list is updated. DOIs are never invented to make this pass."""
    rows = literature.load_all(PATHS.literature)
    offenders = {(r["formula"], r["doi"]) for r in rows
                 if curate_mod.doi_status(r["doi"]) != "ok"}
    documented = {(k["formula"], k["doi"]) for k in CFG["data"]["known_doi_issues"]}
    assert offenders == documented, (
        f"undocumented: {sorted(offenders - documented)}; fixed but still listed: "
        f"{sorted(documented - offenders)}")
    assert all((r["doi"] or "").strip() for r in rows), "a row has no source at all"
    for name, id_key in (("oqmd_a3bx3.json", "id"), ("jarvis_a3bx3.json", "jid"),
                         ("mp_a3bx3.json", "material_id"), ("mp_context.json", "material_id")):
        p = PATHS.raw / name
        if p.exists():
            recs = json.loads(p.read_text(encoding="utf-8"))["records"]
            assert all(r.get(id_key) for r in recs), f"{name}: record without a DB id"


def test_doi_status_flags():
    assert curate_mod.doi_status("10.1039/D4RA08680C") == "ok"
    assert curate_mod.doi_status("10.1016/j.mtcomm.2024") == "truncated DOI"
    assert curate_mod.doi_status("NJC 2026,50,522 Islam").startswith("not a DOI")
    assert curate_mod.doi_status("") == "missing"


# ── loader: both schemas, trainable flag ─────────────────────────────────────
def _harvest_csv(tmp_path: Path) -> Path:
    d = tmp_path / "lit"
    d.mkdir()
    rows = [
        ["Ba3PCl3", "Ba", "P", "Cl", "1.10", "PBE", "no", "indirect", "CASTEP", "6.4",
         "10.1234/abc.1", "yes", "", "2025", "Table 2", "ok", ""],
        ["Ba3PBr3", "Ba", "P", "Br", "1.00", "PBE+SOC", "yes", "", "", "", "10.1234/abc.1",
         "yes", "", "2025", "Table 2", "ok", ""],
        ["Ba3PI3", "Ba", "P", "I", "1.60", "HSE06", "no", "", "", "", "10.1234/abc.1", "yes",
         "", "2025", "Table 2", "ok", ""],
        ["Ba3AsI3", "Ba", "As", "I", "1.20", "PBE", "unknown", "", "", "", "",
         "", "2604.01942", "2026", "", "ok", "conflicts with v1 0.834"],
        ["Ba3SbI3", "Ba", "Sb", "I", "", "PBE", "", "", "", "", "10.1234/x", "", "", "2026",
         "", "pdf paywalled", ""],
    ]
    with open(d / "harvest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(literature.HARVEST_COLUMNS)
        w.writerows(rows)
    (d / "a3bx3_literature_v1.csv").write_bytes(
        (PATHS.literature / "a3bx3_literature_v1.csv").read_bytes())
    return d


def test_loader_accepts_both_schemas(tmp_path):
    d = _harvest_csv(tmp_path)
    rows = literature.load_all(d)
    schemas = {r["schema"] for r in rows}
    assert schemas == {"v1", "harvest"}
    h = {r["formula"]: r for r in rows if r["schema"] == "harvest"}
    assert h["Ba3PBr3"]["functional"] == "PBE+SOC" and h["Ba3PBr3"]["soc"] == "yes"
    assert h["Ba3SbI3"]["parse_error"]          # fetch_status != ok and no gap: logged, not dropped
    v1 = [r for r in rows if r["schema"] == "v1"]
    assert all(r["functional"] == "PBE" and r["soc"] == "unknown" for r in v1)


@pytest.mark.parametrize("include_soc,expect_soc_row", [(False, False), (True, True)])
def test_trainable_set_follows_functional_and_soc_flag(tmp_path, include_soc, expect_soc_row):
    d = _harvest_csv(tmp_path)
    paths = Paths(tmp_path / "out")
    s = curate_mod.curate(paths, literature_dir=d, include_soc=include_soc)
    fam = {r["formula"]: r for r in read_csv(paths.processed / "family.csv")}
    assert "Ba3PCl3" in fam
    assert "Ba3PI3" not in fam                  # HSE never enters the PBE target
    assert ("Ba3PBr3" in fam) is expect_soc_row
    # Ba3AsI3: v1 0.834 vs harvest 1.20 -> conflict logged, newer year picked, not averaged
    conf = list(csv.DictReader(open(paths.interim / "conflict_log.csv", encoding="utf-8")))
    assert {c["formula"] for c in conf} == {"Ba3AsI3"}
    assert fam["Ba3AsI3"]["Eg_eV"] == 1.20
    assert s["n_holdout"] == 3 and not HOLDOUT & set(fam)


# ── leakage guards ───────────────────────────────────────────────────────────
def test_no_formula_shared_between_train_and_test_in_any_fold():
    fam = _family()
    formula = np.array([r["formula"] for r in fam])
    splits = make_splits(fam, CFG["splits"], CFG["seed"])
    assert set(splits) == {"lobo", "loao", "gkf", "loo"}
    for s in splits.values():
        for tr, te in s.folds:
            assert not set(formula[tr]) & set(formula[te])
    # inner (nested) splitter inside an outer training fold, too
    tr, _ = splits["lobo"].folds[0]
    inner = RepeatedGroupKFold(CFG["splits"]["inner"]["n_splits"], 1, CFG["seed"])
    for itr, ite in inner.split(np.zeros((len(tr), 1)), groups=formula[tr]):
        assert not set(formula[tr][itr]) & set(formula[tr][ite])


def test_every_row_is_tested_exactly_once_per_repeat():
    fam = _family()
    splits = make_splits(fam, CFG["splits"], CFG["seed"])
    for s in splits.values():
        per_rep = len(s.folds) // s.n_repeats
        for rep in range(s.n_repeats):
            te = np.concatenate([t for _, t in s.folds[rep * per_rep:(rep + 1) * per_rep]])
            assert sorted(te.tolist()) == list(range(len(fam)))


def test_sr3bix3_absent_from_every_training_fold():
    fam = _family()
    formula = np.array([r["formula"] for r in fam])
    for s in make_splits(fam, CFG["splits"], CFG["seed"]).values():
        for tr, _ in s.folds:
            assert not HOLDOUT & set(formula[tr])
    oof = list(csv.DictReader(open(ROOT / "reports" / "oof_predictions.csv", encoding="utf-8")))
    assert not HOLDOUT & {r["formula"] for r in oof}, "holdout appears in CV predictions"


def test_guards_fire_on_a_leaky_fold():
    """Negative controls: the guards must actually catch the defects they exist for."""
    fam = _family() + [{"formula": "Sr3BiI3", "A": "Sr", "B": "Bi", "X": "I", "Eg_eV": 1.3}]
    n = len(fam)
    with pytest.raises(AssertionError, match="holdout"):
        check_folds(fam, [(np.arange(n - 1, n), np.arange(0, 1))], HOLDOUT)
    dup = fam[:3] + [dict(fam[0])]
    with pytest.raises(AssertionError, match="shared"):
        check_folds(dup, [(np.array([0, 1]), np.array([3]))], set())
    with pytest.raises(AssertionError):
        assert_holdout_excluded(["Ca3PI3", "Sr3BiCl3"], HOLDOUT)


def test_every_estimator_is_a_pipeline_and_searches_use_grouped_splitters():
    for spec in model_specs(CFG):
        est = spec.build(9)
        pipe = unwrap(est)
        assert isinstance(pipe, Pipeline), spec.name
        for _, step in pipe.steps[:-1]:
            assert isinstance(step, TransformerMixin), spec.name
        assert not isinstance(pipe.steps[-1][1], TransformerMixin), spec.name
        if isinstance(est, GridSearchCV):
            assert spec.searched
            assert isinstance(est.cv, RepeatedGroupKFold), f"{spec.name}: cv must be a grouped splitter"
    assert {s.name for s in model_specs(CFG) if s.role == "baseline"} == BASELINES


def test_no_integer_cv_or_unshuffled_cross_val_score_in_source():
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    assert not re.search(r"\bcv\s*=\s*\d", src)
    assert "cross_val_score" not in src


def test_halide_rule_is_per_halide_training_mean():
    X = np.zeros((4, 9)); X[:, 5] = [3.16, 3.16, 2.66, 2.66]
    m = HalideRule().fit(X, [1.0, 2.0, 3.0, 5.0])
    Xt = np.zeros((2, 9)); Xt[:, 5] = [3.16, 3.98]
    assert m.predict(Xt).tolist() == [1.5, 2.75]


# ── metrics file ─────────────────────────────────────────────────────────────
def test_baseline_present_for_every_metric_row():
    m = _metrics()
    for split in ("lobo", "loao", "gkf", "loo"):
        rows = m["results"][split]["models"]
        assert BASELINES <= set(rows), split
        for name, e in rows.items():
            assert e["MAE_CI95"][0] <= e["MAE"] <= e["MAE_CI95"][1], (split, name)
            others = {"mean", "ridge_physics9"} - {name}
            assert others <= set(e["vs_baseline"]), (split, name)
    assert BASELINES <= set(m["sr3bix3_holdout"]["models"])


def test_headline_mae_regression():
    m = _metrics()
    lobo = m["results"]["lobo"]["models"]
    assert lobo["gpr_physics9"]["MAE"] == pytest.approx(0.1458, abs=0.005)
    assert lobo["mean"]["MAE"] == pytest.approx(0.3290, abs=0.001)
    assert lobo["ridge_physics9"]["MAE"] == pytest.approx(0.1946, abs=0.001)


def test_v1_numbers_reproduce():
    v = _metrics()["v1_reproduction"]
    assert v["loo"]["abs_diff"] < 0.002 and v["lobo"]["abs_diff"] < 0.002
    assert v["sr3bix3"]["v2"] == pytest.approx(v["sr3bix3"]["v1"], abs=0.005)


# ── MP fetch failure path (no network, no key) ───────────────────────────────
def test_mp_fetch_without_key_exits_nonzero_with_message(monkeypatch, capsys):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.setattr(mp, "_load_dotenv", lambda: None)
    assert mp.main([]) == mp.EXIT_NO_KEY
    err = capsys.readouterr().err
    assert "MP_API_KEY is not set" in err
    with pytest.raises(mp.MissingKeyError):
        mp.get_api_key({})
    assert not (PATHS.raw / "mp_a3bx3.json").exists() or verify() == []


def test_mp_family_filter():
    pn, hal = set(CFG["data"]["mp"]["pnictogens"]), set(CFG["data"]["mp"]["halogens"])
    assert mp.is_family_member(["Ca", "P", "Cl"], pn, hal)
    assert not mp.is_family_member(["Ca", "P", "S"], pn, hal)
    assert not mp.is_family_member(["P", "As", "Cl"], pn, hal)


# ── determinism ──────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_metrics_v2_is_byte_identical_across_runs(tmp_path):
    """Rebuild the whole offline pipeline twice from data/raw into temp dirs: the two
    metrics_v2.json files must be byte-identical. In the environment that produced the
    committed file (same package versions), the rebuild must also equal the committed file."""
    from a3bx3.pipeline import run_offline
    a = run_offline(tmp_path / "a").reports / "metrics_v2.json"
    b = run_offline(tmp_path / "b").reports / "metrics_v2.json"
    assert a.read_bytes() == b.read_bytes()
    committed = _metrics()
    fresh = json.loads(a.read_text(encoding="utf-8"))
    if fresh["environment"] == committed["environment"]:
        assert a.read_bytes() == METRICS.read_bytes(), \
            "reports/metrics_v2.json is stale: rerun `python run_all.py` and commit"
    else:
        warnings.warn(f"committed metrics came from {committed['environment']}; this run "
                      f"used {fresh['environment']}. Two fresh runs matched each other, but "
                      f"the committed file was not compared.")
