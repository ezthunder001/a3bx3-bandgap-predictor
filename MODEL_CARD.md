# Model card — A3BX3 PBE band-gap models (v2 draft)

Draft, 2026-09-27, after the literature harvest. Numbers come from `reports/metrics_v2.json`,
which `python run_all.py` generates. Every figure below is out-of-fold, never in-sample.

## Intended use

Screening **GGA-PBE band gaps without SOC** of A3BX3 inverse-perovskite pnictogen halides
(A ∈ Mg/Ca/Sr/Ba, B ∈ P/As/Sb/Bi, X ∈ F/Cl/Br/I), to decide which compositions are worth a
DFT calculation. It is a triage aid, not a substitute for DFT.

## Operating limits (binding on every caller)

- **PBE gaps, not real gaps.** PBE underestimates semiconductor gaps, typically by 30–50 %.
  A PBE+SOC gap is a different quantity, served only by the separate `soc` dataset.
- **Label noise floor: 0.1–0.2 eV.** Published PBE gaps for the same compound disagree by
  0.1–0.2 eV between codes: Ca3PI3 0.11, Ca3PBr3 0.13, Ca3PCl3 0.23 eV across QE, WIEN2k
  and QuantumATK. Treat this as **irreducible label noise**. An out-of-fold MAE near or
  below 0.1 eV means the model agrees with the median label about as well as two DFT codes
  agree with each other. It is not evidence of better-than-DFT accuracy, and further MAE
  gains below this floor are not meaningful.
- **n = 39 (primary).** 8 of these formulas come only from one preprint (arXiv 2604.01942).
  23 have SOC status unknown.
- **New A-site cations are not supported.** Under leave-one-A-out the GPR MAE is 0.505 eV
  against a mean baseline of 0.619 eV. Mg and Ca are predicted worst (0.77 and 0.73 eV). The
  best model there, rf_magpie at 0.431 eV, is still 4× the LOBO error. Do not use any model
  for a compound whose A-site is absent from training.
- **Family only.** Not valid outside the 4 × 4 × 4 space above, for mixed-halide or
  mixed-cation compositions, or for non-cubic polymorphs. The descriptors are compositional
  and cannot tell polymorphs apart.
- **The Sr3BiX3 holdout (n = 3) cannot rank models.** Its own literature spread is at least
  0.16 eV: Sr3BiI3 is 1.164 (CASTEP), 1.30 (WIEN2k) and 1.324 (target).
- **MAE is not comparable with v1.** The label range grew from 0.78–2.30 to 0.11–3.18 eV,
  which inflates the mean baseline. Compare skill against the baselines.

## Models and baselines (same folds for all)

Baselines:
- `mean`
- `halide_rule`: the training mean per halide
- `ridge_physics9`: as in v1
- `ridge_magpie`: VarianceThreshold + scaler + SelectKBest + Ridge, with k and alpha chosen
  by inner grouped CV

Models:
- `gpr_physics9`: ARD Matérn 5/2 + white noise, the v1 kernel
- `rf_*` and `xgb_*`: on physics-9 or Magpie features, grids chosen by nested grouped CV

Every transform sits inside a sklearn `Pipeline`.

## Held-out results (MAE eV, bootstrap 95 % CI)

**Primary** (39 formulas, no-SOC PBE, median labels):

| Split | gpr_physics9 | Best other | mean | ridge_physics9 | GPR − ridge [95 % CI] |
|---|---|---|---|---|---|
| **Leave-one-B-out (headline)** | **0.111 [0.080, 0.144]** | — (GPR best) | 0.526 | 0.334 | −0.223 [−0.322, −0.140] |
| Leave-one-A-out | 0.505 [0.377, 0.634] | rf_magpie 0.431 [0.330, 0.533] | 0.619 | 0.598 | −0.093 [−0.183, +0.009] |
| GroupKFold(5)×10 | 0.094 [0.075, 0.115] | — | 0.522 | 0.359 | −0.266 [−0.361, −0.187] |
| LOO | 0.090 [0.069, 0.115] | — | 0.521 | 0.350 | −0.260 [−0.348, −0.184] |

**Verified-only** (31 formulas; no preprint rows, no unverified sources):

| Split | gpr_physics9 | Best other | mean | ridge_physics9 | GPR − ridge [95 % CI] |
|---|---|---|---|---|---|
| Leave-one-B-out | 0.137 [0.098, 0.177] | — | 0.530 | 0.427 | −0.291 [−0.421, −0.182] |
| Leave-one-A-out | 0.371 [0.234, 0.535] | — | 0.584 | 0.741 | −0.371 [−0.518, −0.213] |
| GroupKFold(5)×10 | 0.127 [0.101, 0.153] | — | 0.520 | 0.370 | −0.243 [−0.346, −0.152] |

The sensitivity sets (`plus_unverified`, `soc`) are in `reports/metrics_v2.md`. On
leave-one-B-out, the SOC target is harder: GPR 0.364 eV, best xgb_physics9 0.310, mean 0.514,
n = 21.

**Sr3BiX3 locked holdout** (fit on the whole dataset, scored once):

| Dataset | GPR MAE | GPR bias | Mean baseline | Ridge baseline |
|---|---|---|---|---|
| primary | 0.060 | −0.047 | 0.187 | 0.320 |
| verified_only | 0.052 | +0.019 | 0.147 | 0.252 |

rf_physics9 and xgb_physics9 score 0.05–0.07 on the holdout, the same as the GPR. With n = 3,
these numbers are a consistency check, not a ranking.

Reading it honestly:

- With the larger and corrected dataset, the GPR now separates from **both** baselines on
  LOBO, GroupKFold and LOO: the CIs of the paired differences exclude zero. At n = 23 in v1
  it did not separate from ridge.
- The primary LOBO MAE (0.111) sits at the label noise floor. The verified-only LOBO MAE
  (0.137) is higher, but the CIs overlap. The primary set leans on the 12-compound Ba series
  of one internally consistent preprint, which may make that set easier to fit; quote both
  numbers.
- RF and XGB do not beat the GPR except on leave-one-A-out and on the SOC target, where
  every model is weak. This matches the pre-registered expectation.
- GPR σ calibration on primary LOBO (out-of-fold): 49 % of points fall within ±1σ against a
  nominal 68 %, and 72 % within ±2σ against 95 %. The intervals are **too narrow**; do not
  use them as confidence intervals. Conformal intervals are week-6 work.

## Reproduction

- **v1 check:** the v2 code on the uncorrected v1 file reproduces v1 exactly (LOO 0.1512,
  LOBO 0.1458 eV). That checks the code; the v1 data had two HSE06 labels, see DATA_CARD.
- **Rebuild:** `python run_all.py` rebuilds everything offline in about 6.5 minutes.
- **Determinism:** a rebuild gives a byte-identical `metrics_v2.json` (tested).
