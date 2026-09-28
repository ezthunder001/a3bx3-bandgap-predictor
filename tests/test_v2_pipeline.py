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
DATASETS = list(CFG["data"]["datasets"])
# Primary / LOBO headline recorded 2026-09-27 after the harvest; update deliberately.
HEADLINE = {"gpr": 0.1114, "mean": 0.5259, "ridge": 0.3340}


def _family(name: str = "primary"):
    return read_csv(PATHS.processed / f"family_{name}.csv")


def _metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))


def _corrected_rows(lit=None):
    lit = lit or PATHS.literature
    rows = literature.load_all(lit)
    curate_mod.apply_corrections(rows, literature.load_corrections(lit),
                                 CFG["data"]["corrections_apply_to"])
    return rows


def _label_rows():
    with open(PATHS.interim / "label_rows.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── data integrity ────────────────────────────────────────────────────────────
def test_raw_manifest_matches():
    assert verify() == []


def test_literature_raw_copy_is_the_v1_file():
    """v1 scripts read data/a3bx3_literature.csv; v2 reads the raw copy. They must not drift,
    and v2 fixes go in corrections.csv, never into the v1 file.

    Compared with line endings normalised: the v1 file is an ordinary text file that git
    checks out as CRLF on Windows and LF on Linux, while the raw copy is stored byte-exact
    (-text) so its SHA-256 holds everywhere. The content is what must not drift."""
    def content(p):
        return p.read_bytes().replace(b"\r\n", b"\n")
    assert content(ROOT / "data" / "a3bx3_literature.csv") == \
        content(PATHS.literature / "a3bx3_literature_v1.csv")


def test_physics9_tables_are_the_v1_tables():
    tree = ast.parse((ROOT / "train_a3bx3_family.py").read_text(encoding="utf-8"))
    v1 = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
          for t in n.targets if isinstance(t, ast.Name) and t.id in
          {"RAD_A", "CHI_A", "RAD_B", "CHI_B", "RAD_X", "CHI_X"}}
    v2 = {"RAD_A": RAD_A, "CHI_A": CHI_A, "RAD_B": RAD_B, "CHI_B": CHI_B,
          "RAD_X": RAD_X, "CHI_X": CHI_X}
    assert v1 == v2


def test_datasets_are_one_row_per_formula_and_exclude_the_holdout():
    hold = read_csv(PATHS.processed / "holdout_sr3bix3.csv")
    assert {r["formula"] for r in hold} == HOLDOUT
    assert all(r["doi"] == "10.1039/d5nj02800k" for r in hold)
    for name in DATASETS:
        formulas = [r["formula"] for r in _family(name)]
        assert len(formulas) == len(set(formulas)), name
        assert not HOLDOUT & set(formulas), name
    # no row for a holdout formula reached any label pool, from any source
    assert not HOLDOUT & {r["formula"] for r in _label_rows()}


def test_k1_checkpoint_recorded():
    k1 = _metrics()["k1"]
    assert k1["threshold"] == 35
    assert k1["primary_formulas"] == len(_family("primary"))
    assert k1["pass"] is (k1["primary_formulas"] >= 35)


def test_primary_excludes_the_mislabelled_mg3bi_hse_values():
    """v1 stored the HSE06 gaps of Mg3BiI3 (0.867) and Mg3BiBr3 (1.626) as PBE. v2 must not
    train on them: the v1 rows are relabelled HSE06 and the PBE values come from the harvest."""
    lab = [r for r in _label_rows() if r["formula"] in {"Mg3BiI3", "Mg3BiBr3"}]
    assert lab and all(r["functional"] in {"PBE", "PBE+SOC"} for r in lab)
    assert not {(r["formula"], float(r["Eg_eV"])) for r in lab} & \
        {("Mg3BiI3", 0.867), ("Mg3BiBr3", 1.626)}
    for name in DATASETS:
        fam = {r["formula"]: r["Eg_eV"] for r in _family(name)}
        assert fam.get("Mg3BiI3") != 0.867 and fam.get("Mg3BiBr3") != 1.626, name
    prim = {r["formula"]: r for r in _family("primary")}
    assert prim["Mg3BiI3"]["Eg_eV"] == 0.224
    assert "10.1039/d4ra09093d" in prim["Mg3BiBr3"]["sources"]
    exc = _csv(PATHS.interim / "excluded.csv")
    assert {(r["formula"], r["functional"]) for r in exc
            if r["source_file"] == "a3bx3_literature_v1.csv" and r["formula"].startswith("Mg3Bi")} \
        == {("Mg3BiI3", "HSE06"), ("Mg3BiBr3", "HSE06")}


def test_corrections_decisions_are_applied():
    v1 = {r["formula"]: r for r in _label_rows() if r["source_file"] == "a3bx3_literature_v1.csv"}
    for f in ("Ca3BiI3", "Ca3BiBr3", "Ca3BiCl3", "Ca3AsCl3"):
        assert v1[f]["use"] == "sensitivity_only" and v1[f]["status"] == "source_unverified", f
    for f in ("Ca3AsI3", "Ca3SbF3", "Ca3SbBr3", "Ca3SbI3"):
        assert f not in v1, f"{f}: v1 row must be excluded"
    assert float(v1["Ca3SbCl3"]["Eg_eV"]) == 1.814
    prim = {r["formula"]: r for r in _family("primary")}
    plus = {r["formula"] for r in _family("plus_unverified")}
    assert "Ca3AsCl3" not in prim and "Ca3AsCl3" in plus
    assert "Ca3AsI3" not in plus
    # Ca3BiI3 stays in primary only through a verified harvest source, not the v1 row
    assert "NJC" not in prim["Ca3BiI3"]["sources"]
    assert "functional_unconfirmed" in prim["Ca3SbCl3"]["flags"]


def test_soc_never_pooled_with_no_soc():
    lab = _label_rows()
    pbe = {(r["formula"], float(r["Eg_eV"])) for r in lab if r["functional"] == "PBE"}
    soc = {(r["formula"], float(r["Eg_eV"])) for r in lab if r["functional"] == "PBE+SOC"}
    for name in ("primary", "verified_only", "plus_unverified"):
        for r in _family(name):
            assert {(r["formula"], float(v)) for v in r["values"].split(";")} <= pbe, (name, r["formula"])
            assert r["soc"] in {"no", "unknown"}
    for r in _family("soc"):
        assert r["soc"] == "yes"
        assert {(r["formula"], float(v)) for v in r["values"].split(";")} <= soc


def test_labels_are_medians_and_conflicts_flagged():
    thr = CFG["data"]["conflict_threshold_eV"]
    for name in DATASETS:
        for r in _family(name):
            vals = sorted(float(v) for v in r["values"].split(";"))
            assert r["Eg_eV"] == pytest.approx(float(np.median(vals)), abs=1e-4)
            assert ("high_conflict" in r["flags"]) is (max(vals) - min(vals) > thr + 1e-9)


def test_verified_only_has_no_preprint_rows():
    for r in _family("verified_only"):
        assert "arXiv" not in r["sources"] and "NJC" not in r["sources"]
    prim = {r["formula"]: r for r in _family("primary")}
    ver = {r["formula"] for r in _family("verified_only")}
    only_pre = {f for f, r in prim.items() if "preprint_only" in r["flags"]}
    assert only_pre and not only_pre & ver
    assert set(prim) - only_pre == ver


def test_every_data_row_has_a_doi_or_db_id_known_issues_listed():
    """After corrections, every literature row needs a well-formed DOI or an arXiv id. Rows
    with neither must be the exact documented list in configs/v2.yaml: a new offender fails,
    and a fixed one fails until the list is updated. DOIs are never invented to pass this."""
    rows = _corrected_rows()
    offenders = {(r["formula"], r["doi"]) for r in rows
                 if curate_mod.doi_status(r["doi"]) != "ok" and not r.get("arxiv_id")}
    documented = {(k["formula"], k["doi"]) for k in CFG["data"]["known_doi_issues"]}
    assert offenders == documented, (
        f"undocumented: {sorted(offenders - documented)}; fixed but still listed: "
        f"{sorted(documented - offenders)}")
    prim_sources = ";".join(r["sources"] for r in _family("primary"))
    for _, doi in documented:       # every documented offender is kept out of primary
        assert doi not in prim_sources
    for name, id_key in (("oqmd_a3bx3.json", "id"), ("jarvis_a3bx3.json", "jid"),
                         ("mp_a3bx3.json", "material_id"), ("mp_context.json", "material_id")):
        p = PATHS.raw / name
        if p.exists():
            recs = json.loads(p.read_text(encoding="utf-8"))["records"]
            assert all(r.get(id_key) for r in recs), f"{name}: record without a DB id"


def test_doi_status_flags():
    assert curate_mod.doi_status("10.1039/d4ra09093d") == "ok"
    assert curate_mod.doi_status("10.1016/j.mtcomm.2024") == "truncated DOI"
    assert curate_mod.doi_status("NJC 2026,50,522 Islam").startswith("not a DOI")
    assert curate_mod.doi_status("") == "missing"


def test_a_stale_correction_stops_curate():
    rows = literature.load_all(PATHS.literature)
    bad = [{"formula": "Mg3BiI3", "field": "Eg_eV", "old": "9.99", "new": "1.0",
            "action": "correct_value", "evidence": "", "doi": ""}]
    with pytest.raises(ValueError, match="matched 0 rows"):
        curate_mod.apply_corrections(rows, bad, CFG["data"]["corrections_apply_to"])


# ── loader and curate on a synthetic harvest ─────────────────────────────────
def _harvest_csv(tmp_path: Path) -> Path:
    d = tmp_path / "lit"
    d.mkdir()
    rows = [
        ["Ba3PCl3", "Ba", "P", "Cl", "1.10", "PBE", "no", "indirect", "CASTEP", "6.4",
         "10.1234/abc.1", "true", "", "2025", "Table 2", "full_text", ""],
        ["Ba3PCl3", "Ba", "P", "Cl", "1.00", "PBE+SOC", "yes", "", "", "", "10.1234/abc.1",
         "true", "", "2025", "Table 2", "full_text", ""],
        ["Ba3PI3", "Ba", "P", "I", "1.60", "HSE06", "no", "", "", "", "10.1234/abc.1", "true",
         "", "2025", "Table 2", "full_text", ""],
        ["Ba3AsI3", "Ba", "As", "I", "1.20", "PBE", "unknown", "", "", "", "",
         "false", "2604.01942", "2026", "", "full_text", ""],
        ["Ba3SbI3", "Ba", "Sb", "I", "", "PBE", "", "", "", "", "10.1234/x", "", "", "2026",
         "", "pdf paywalled", ""],
        ["Sr3BiI3", "Sr", "Bi", "I", "1.164", "PBE", "no", "", "CASTEP", "", "10.1234/y",
         "true", "", "2025", "", "full_text", "must never be trained on"],
    ]
    with open(d / "harvest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(literature.HARVEST_COLUMNS)
        w.writerows(rows)
    for name in ("a3bx3_literature_v1.csv", "corrections.csv"):
        (d / name).write_bytes((PATHS.literature / name).read_bytes())
    return d


def test_loader_accepts_both_schemas(tmp_path):
    d = _harvest_csv(tmp_path)
    rows = literature.load_all(d)
    assert {r["schema"] for r in rows} == {"v1", "harvest"}
    assert not any(r["source_file"] == "corrections.csv" for r in rows)
    h = [r for r in rows if r["schema"] == "harvest"]
    assert any(r["functional"] == "PBE+SOC" and r["soc"] == "yes" for r in h)
    assert next(r for r in h if r["formula"] == "Ba3SbI3")["parse_error"]
    pre = next(r for r in h if r["formula"] == "Ba3AsI3")
    assert pre["preprint"] and pre["source_id"] == "arXiv:2604.01942"
    assert all(r["functional"] == "PBE" and r["soc"] == "unknown"
               for r in rows if r["schema"] == "v1")


def test_synthetic_harvest_builds_the_datasets(tmp_path):
    d = _harvest_csv(tmp_path)
    paths = Paths(tmp_path / "out")
    s = curate_mod.curate(paths, literature_dir=d)
    prim = {r["formula"]: r for r in read_csv(paths.processed / "family_primary.csv")}
    soc = {r["formula"]: r for r in read_csv(paths.processed / "family_soc.csv")}
    ver = {r["formula"]: r for r in read_csv(paths.processed / "family_verified_only.csv")}
    assert prim["Ba3PCl3"]["Eg_eV"] == 1.10 and soc["Ba3PCl3"]["Eg_eV"] == 1.00
    assert "Ba3PI3" not in prim                     # HSE never enters the PBE target
    assert prim["Ba3AsI3"]["Eg_eV"] == pytest.approx((0.834 + 1.20) / 2)   # median of two
    assert "high_conflict" in prim["Ba3AsI3"]["flags"]
    assert ver["Ba3AsI3"]["Eg_eV"] == 0.834         # preprint value dropped
    assert s["n_holdout"] == 3 and not HOLDOUT & set(prim)
    exc = _csv(paths.interim / "excluded.csv")
    assert any(r["formula"] == "Sr3BiI3" and "holdout" in r["reason"] for r in exc)


# ── leakage guards ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", DATASETS)
def test_no_formula_shared_between_train_and_test_in_any_fold(name):
    fam = _family(name)
    formula = np.array([r["formula"] for r in fam])
    splits = make_splits(fam, CFG["splits"], CFG["seed"])
    for s in splits.values():
        for tr, te in s.folds:
            assert not set(formula[tr]) & set(formula[te])
            assert not HOLDOUT & set(formula[tr])
    tr, _ = splits["lobo"].folds[0]
    inner = RepeatedGroupKFold(CFG["splits"]["inner"]["n_splits"], 1, CFG["seed"])
    for itr, ite in inner.split(np.zeros((len(tr), 1)), groups=formula[tr]):
        assert not set(formula[tr][itr]) & set(formula[tr][ite])


def test_every_row_is_tested_exactly_once_per_repeat():
    fam = _family()
    for s in make_splits(fam, CFG["splits"], CFG["seed"]).values():
        per_rep = len(s.folds) // s.n_repeats
        for rep in range(s.n_repeats):
            te = np.concatenate([t for _, t in s.folds[rep * per_rep:(rep + 1) * per_rep]])
            assert sorted(te.tolist()) == list(range(len(fam)))


def test_sr3bix3_absent_from_every_training_fold_and_oof():
    oof = _csv(ROOT / "reports" / "oof_predictions.csv")
    assert not HOLDOUT & {r["formula"] for r in oof}, "holdout appears in CV predictions"
    assert {r["dataset"] for r in oof} == set(DATASETS)


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
            assert isinstance(est.cv, RepeatedGroupKFold), f"{spec.name}: cv must be grouped"
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
    assert set(m["datasets"]) == set(DATASETS)
    for ds, D in m["datasets"].items():
        assert set(D["results"]) == set(CFG["evaluate"]["splits_by_dataset"][ds])
        for split, res in D["results"].items():
            rows = res["models"]
            assert BASELINES <= set(rows), (ds, split)
            for name, e in rows.items():
                assert e["MAE_CI95"][0] <= e["MAE"] <= e["MAE_CI95"][1], (ds, split, name)
                others = {"mean", "ridge_physics9"} - {name}
                assert others <= set(e["vs_baseline"]), (ds, split, name)
        if ds in CFG["evaluate"]["holdout_datasets"]:
            assert BASELINES <= set(D["sr3bix3_holdout"]["models"])
        else:
            assert D["sr3bix3_holdout"] is None


def test_headline_mae_regression():
    lobo = _metrics()["datasets"]["primary"]["results"]["lobo"]["models"]
    assert lobo["gpr_physics9"]["MAE"] == pytest.approx(HEADLINE["gpr"], abs=0.005)
    assert lobo["mean"]["MAE"] == pytest.approx(HEADLINE["mean"], abs=0.001)
    assert lobo["ridge_physics9"]["MAE"] == pytest.approx(HEADLINE["ridge"], abs=0.001)


def test_v1_numbers_reproduce():
    """The v2 code on the uncorrected v1 file still gives the published v1 numbers."""
    v = _metrics()["v1_reproduction"]
    assert v["loo"]["abs_diff"] < 0.002 and v["lobo"]["abs_diff"] < 0.002


# ── MP fetch failure path (no network, no key) ───────────────────────────────
def test_mp_fetch_without_key_exits_nonzero_with_message(monkeypatch, capsys):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.setattr(mp, "_load_dotenv", lambda: None)
    assert mp.main([]) == mp.EXIT_NO_KEY
    assert "MP_API_KEY is not set" in capsys.readouterr().err
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
    """Rebuild the whole offline pipeline from data/raw into a temp dir. In the environment
    that produced the committed file (same package versions) the rebuild must equal the
    committed reports/metrics_v2.json byte for byte, i.e. two runs are identical. Elsewhere a
    second fresh run is made and the two fresh runs must match."""
    from a3bx3.pipeline import run_offline
    a = run_offline(tmp_path / "a").reports / "metrics_v2.json"
    committed = _metrics()
    fresh = json.loads(a.read_text(encoding="utf-8"))
    if fresh["environment"] == committed["environment"]:
        assert a.read_bytes() == METRICS.read_bytes(), \
            "reports/metrics_v2.json is stale: rerun `python run_all.py` and commit"
    else:
        b = run_offline(tmp_path / "b").reports / "metrics_v2.json"
        assert a.read_bytes() == b.read_bytes()
        warnings.warn(f"committed metrics came from {committed['environment']}; this run "
                      f"used {fresh['environment']}. Two fresh runs matched each other.")
