"""Regression tests: the numbers quoted in README.md must match the committed metrics,
and the dataset must stay fully DOI-attributed.

Run:  python -m pytest tests/ -q     (or: python tests/test_metrics.py)
"""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _load(name: str) -> dict:
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


class TestFamilyModel(unittest.TestCase):
    """The headline A3BX3 family numbers."""

    def setUp(self) -> None:
        self.m = _load("a3bx3_family_metrics.json")

    def test_loo_matches_readme(self) -> None:
        loo = self.m["loo"]
        self.assertAlmostEqual(loo["MAE"], 0.1512, places=4)
        self.assertAlmostEqual(loo["R2"], 0.7396, places=4)
        self.assertEqual(loo["n"], 23)

    def test_leave_one_b_family_out(self) -> None:
        lobo = self.m["lobo"]
        self.assertAlmostEqual(lobo["MAE"], 0.1458, places=4)
        self.assertEqual(lobo["cv"], "leave-one-B-family-out")
        # Holding out an entire chemistry must not collapse the model.
        for b, mae in lobo["per_B"].items():
            self.assertLess(mae, 0.25, f"B-family {b} degraded past the reported range")

    def test_sr3bix3_holdout(self) -> None:
        val = self.m["sr3bix3_validation"]
        self.assertAlmostEqual(val["MAE"], 0.16, places=2)
        self.assertEqual(len(val["per_compound"]), 3)

    def test_named_baselines_are_stored(self) -> None:
        """Every headline score must travel with a named baseline on the same folds."""
        b = self.m["baselines"]
        for split in ("loo", "lobo", "sr3bix3_validation"):
            self.assertEqual(set(b[split]) - {"_note"}, {"mean", "ridge"}, split)
        self.assertAlmostEqual(b["loo"]["mean"]["MAE"], 0.3243, places=4)
        self.assertAlmostEqual(b["loo"]["ridge"]["MAE"], 0.2063, places=4)
        self.assertAlmostEqual(b["lobo"]["mean"]["MAE"], 0.3290, places=4)
        self.assertAlmostEqual(b["lobo"]["ridge"]["MAE"], 0.1946, places=4)

    def test_gpr_beats_both_baselines_in_cross_validation(self) -> None:
        """The README's skill claims (LOBO: 56 % vs mean, 25 % vs ridge)."""
        b = self.m["baselines"]
        for split in ("loo", "lobo"):
            for k in ("mean", "ridge"):
                self.assertLess(self.m[split]["MAE"], b[split][k]["MAE"], f"{split}/{k}")
        self.assertAlmostEqual(b["lobo"]["mean"]["skill_vs_mean"], 0.557, places=3)
        self.assertAlmostEqual(b["lobo"]["ridge"]["skill_vs_ridge"], 0.251, places=3)

    def test_sr3bix3_holdout_does_not_separate_models(self) -> None:
        """A documented negative: on the 3-compound hold-out the GPR is no better than the
        family mean. If this ever flips, the README paragraph must be rewritten."""
        b = self.m["baselines"]["sr3bix3_validation"]
        gpr = self.m["sr3bix3_validation"]["MAE"]
        self.assertGreaterEqual(gpr, b["mean"]["MAE"])
        self.assertGreaterEqual(gpr, b["ridge"]["MAE"])

    def test_ba3bii3_conflict_is_still_flagged(self) -> None:
        """The open conflict against the monotonic A-site trend is a documented finding.
        If this prediction ever moves near 1.6 eV, the README section must be rewritten."""
        ba = next(x for x in self.m["forward_ba3bix3"] if x["formula"] == "Ba3BiI3")
        self.assertAlmostEqual(ba["Eg_pred_eV"], 0.74, places=2)
        self.assertLess(ba["Eg_pred_eV"], 1.0)


class TestDeltaModel(unittest.TestCase):
    """The TBmBJ delta correction and its 53% error reduction claim."""

    def setUp(self) -> None:
        self.m = _load("mbj_delta_metrics.json")

    def test_best_model_is_rf_with_pbegap(self) -> None:
        self.assertEqual(self.m["best"], "RF+pbegap")
        best = self.m["models"]["RF+pbegap"]
        self.assertAlmostEqual(best["delta_MAE"], 0.275, places=3)
        self.assertAlmostEqual(best["corrected_gap_R2"], 0.9241, places=4)

    def test_correction_beats_raw_pbe(self) -> None:
        s = self.m["sr3bix3"]
        raw, corrected = s["raw_PBE_vs_HSE_MAE"], s["corrected_vs_HSE_MAE"]
        self.assertAlmostEqual(raw, 0.661, places=3)
        self.assertAlmostEqual(corrected, 0.3123, places=4)
        reduction = (raw - corrected) / raw
        self.assertGreater(reduction, 0.50, "README claims a 53% error reduction")

    def test_cross_stoichiometry_caveat_present(self) -> None:
        """The caveat travels with the metrics, not just the prose."""
        self.assertIn("cross-stoichiometry", self.m["caveat"])


class TestLeakageAudit(unittest.TestCase):
    """Grouped folds are the reported metric; the leaky variant is kept but marked."""

    def setUp(self) -> None:
        self.m = _load("jarvis_model_metrics.json")["bandgap"]

    def test_grouped_cv_is_the_reported_scheme(self) -> None:
        self.assertEqual(self.m["cv"], "GroupKFold(5) by formula")

    def test_leaky_variant_is_retained_and_labelled(self) -> None:
        self.assertIn("models_random_kfold_leaky", self.m)

    def test_random_folds_are_optimistic(self) -> None:
        """The whole point of the audit: random folds flatter the model."""
        honest = self.m["models"]["GPR"]
        leaky = self.m["models_random_kfold_leaky"]["GPR"]
        self.assertGreater(leaky["R2"], honest["R2"])
        self.assertLess(leaky["MAE"], honest["MAE"])
        self.assertAlmostEqual(honest["R2"], 0.6306, places=4)
        self.assertAlmostEqual(honest["MAE"], 0.5258, places=4)


class TestNegativeResults(unittest.TestCase):
    """Weak models are reported, not quietly dropped. These assertions exist so the
    negative numbers cannot disappear from the repo without a test failing."""

    def test_elastic_constants_are_reported_as_weak(self) -> None:
        m = _load("elastic_broad_metrics.json")["Cij_RandomForest"]
        self.assertLess(m["C44"]["R2"], 0.2)
        self.assertLess(m["C11"]["R2"], 0.5)

    def test_bulk_modulus_is_the_one_that_works(self) -> None:
        m = _load("elastic_broad_metrics.json")["Moduli_GradientBoosting"]
        self.assertGreater(m["BulkModulus_K"]["R2"], 0.7)

    def test_jarvis_c12_negative_r2_retained(self) -> None:
        m = _load("jarvis_model_metrics.json")["elastic"]["models"]["RandomForest"]
        self.assertLess(m["C12"]["R2"], 0.0)


class TestDataProvenance(unittest.TestCase):
    """No estimated values, no unverifiable DOIs — the dataset rule this project is built on."""

    def test_every_literature_row_has_a_doi(self) -> None:
        with open(DATA / "a3bx3_literature.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 23)
        missing = [r["formula"] for r in rows if not (r.get("doi") or "").strip()]
        self.assertEqual(missing, [], f"rows without a DOI: {missing}")

    def test_no_estimated_or_likely_entries(self) -> None:
        with open(DATA / "a3bx3_literature.csv", encoding="utf-8") as f:
            text = f.read().lower()
        for banned in ("dft_est", "likely", "estimated"):
            self.assertNotIn(banned, text, f"dataset reintroduced a '{banned}' entry")


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
