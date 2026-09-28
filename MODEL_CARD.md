# Model card: A3BX3 PBE band-gap model (v2)

Version 2026-09-28 (week 7). Every number here is held out (out-of-fold or locked holdout),
never in-sample. The numbers come from committed files that `python run_all.py` regenerates
offline:

- `reports/metrics_v2.json` for accuracy
- `reports/uncertainty_v2.json` for intervals and K2
- `reports/explain_v2.json` for explanations

The narrative is in `notebooks/01_story.ipynb`.

## 1. Model

| | |
|---|---|
| Headline model | `gpr_physics9`: StandardScaler, then Gaussian process regression. The kernel is ARD Matérn(ν = 5/2) plus a white-noise term, `normalize_y`, 8 optimiser restarts, seed 42. It is the v1 kernel. |
| Input | Composition only. There are 9 crystal-chemistry descriptors: the radii rA, rB, rX; the electronegativities chiA, chiB, chiX; the differences dchi_BX and dchi_AX; and the ratio rB/rX. |
| Output | Predicted GGA-PBE band gap without SOC, in eV, with a predictive σ. Do not use raw σ as an interval (§6). |
| Compared against | Four named baselines on the same folds: mean, per-halide mean, ridge on physics-9, and ridge on Magpie. Also random forest and XGBoost on physics-9 and Magpie features. Every transform sits inside a `Pipeline`, and hyperparameters are chosen by nested grouped CV. |

## 2. Intended use

**Use it to screen GGA-PBE band gaps without SOC of cubic A3BX3 inverse-perovskite pnictogen
halides in the 4 × 4 × 4 space A ∈ {Mg, Ca, Sr, Ba}, B ∈ {P, As, Sb, Bi},
X ∈ {F, Cl, Br, I}.** The job is to rank compositions for a DFT follow-up. The model is a
triage tool; it does not replace DFT.

## 3. Out of scope

Do not use it for any of the following:

- Experimental or HSE gaps. PBE underestimates semiconductor gaps, typically by 30–50 %.
- PBE+SOC gaps. That is a separate target; see the `soc` dataset in `metrics_v2.md`.
- Compositions outside the 4 × 4 × 4 space, such as other A-site cations, chalcogenides or
  ABX3 perovskites. v1 showed that an ABX3-trained model misses Sr3BiX3 by about 0.85 eV.
- Mixed-halide or mixed-cation alloys.
- Non-cubic polymorphs. The descriptors cannot tell polymorphs apart.
- **Any compound whose A-site element is absent from the training data** (§6).
- Device-level claims such as efficiency.

## 4. Training data

Full provenance is in `DATA_CARD.md`.

| Set | Formulas | Label | Notes |
|---|---|---|---|
| **primary** | 39 | median no-SOC PBE gap per formula (57 values, 37 sources) | 8 formulas (21 %) come only from one preprint (arXiv 2604.01942, Ba3MX3); 4 more have a preprint value in their median. 23 are `soc_unknown`. 3 are `high_conflict` (spread > 0.2 eV). |
| verified_only | 31 | primary without any preprint row | the sensitivity set |
| plus_unverified / soc | 42 / 21 | see DATA_CARD | sensitivity sets only |
| Locked holdout | 3 | Sr3BiI3 / Br3 / Cl3 (Islam 2026, doi 10.1039/d5nj02800k) | never trained on, from any source |

**Known data issues, corrected in v2.**
- The v1 file stored the HSE06 gaps of Mg3BiI3 (0.867 eV) and Mg3BiBr3 (1.626 eV) as PBE.
  The PBE values are 0.224 and 1.071 eV.
- Several v1 rows cite a paper that does not contain their value.
- Fixes live in `data/raw/literature/corrections.csv`; the v1 file itself is unchanged.
- The published v1 metrics (LOO 0.151, LOBO 0.146 eV) were computed on the mislabelled data
  and are kept only as history.

## 5. Held-out accuracy

MAE in eV, bootstrap 95 % CI, same folds for every column.

**Primary (n = 39)**

| Split | gpr_physics9 | Best other | mean | ridge_physics9 | GPR − ridge [95 % CI] |
|---|---|---|---|---|---|
| **Leave-one-B-out (headline)** | **0.111 [0.080, 0.144]** | (GPR is best) | 0.526 | 0.334 | −0.223 [−0.322, −0.140] |
| Leave-one-A-out | 0.505 [0.377, 0.634] | rf_magpie 0.431 [0.330, 0.533] | 0.619 | 0.598 | −0.093 [−0.183, +0.009] |
| GroupKFold(5) × 10 | 0.094 [0.075, 0.115] | (GPR is best) | 0.522 | 0.359 | −0.266 [−0.361, −0.187] |
| LOO | 0.090 [0.069, 0.115] | (GPR is best) | 0.521 | 0.350 | −0.260 [−0.348, −0.184] |

The best baseline on primary LOBO is ridge_magpie, at 0.198 eV.

**Verified-only (n = 31)**

| Split | gpr_physics9 | mean | ridge_physics9 |
|---|---|---|---|
| LOBO | 0.137 [0.098, 0.177] | 0.530 | 0.427 |
| LOAO | 0.371 [0.234, 0.535] | 0.584 | 0.741 |
| GroupKFold(5) × 10 | 0.127 [0.101, 0.153] | 0.520 | 0.370 |

**Sr3BiX3 locked holdout** (fit on all of primary, scored once, n = 3)
- GPR MAE 0.060 eV, bias −0.047, against 0.187 for the mean baseline.
- With n = 3 this cannot rank models. The literature itself spreads by at least 0.16 eV for
  Sr3BiI3: 1.164 (CASTEP), 1.30 (WIEN2k) and 1.324 (target).

**Kill checkpoint K2 (design §9): PASS.**
- On primary LOBO, the GPR's skill is 0.79 [0.72, 0.84] against the mean baseline and
  **0.44 [0.14, 0.63]** against the best baseline (ridge_magpie).
- On verified-only, the same figures are 0.74 [0.63, 0.82] and 0.53 [0.26, 0.70].
- All paired-difference CIs exclude 0. **However, the lower bound of 0.14 against the best
  baseline is below the 30 % bar.** The pass is secure against the mean and not secure
  against ridge_magpie.

## 6. Uncertainty

Coverage is measured on held-out folds; every interval is fitted on the training fold only.
Primary set, nominal 90 %. Each cell gives coverage · mean width (eV):

| Method | LOBO | LOAO | GKF × 10 | LOO |
|---|---|---|---|---|
| GPR ±zσ (raw) | 0.67 · 0.34 | 0.77 · 1.91 | 0.78 · 0.35 | 0.85 · 0.33 |
| GPR ±z·c·σ (c fitted by inner CV) | **0.90 · 0.75** | 1.00 · 2.79 | 0.93 · 0.66 | 0.92 · 0.46 |
| GPR jackknife+ (MAPIE 1.5.0) | 0.87 · 0.52 | **0.21** · 0.42 | **0.95 · 0.48** | 0.92 · 0.46 |
| GPR CV+ | 0.92 · 0.79 | 0.36 · 0.54 | 0.97 · 0.68 | 0.95 · 0.55 |
| ridge CV+ (baseline) | 0.97 · 2.21 | 0.69 · 1.67 | 0.92 · 1.78 | 0.90 · 1.79 |
| mean CV+ (baseline) | 0.90 · 2.39 | 0.85 · 2.34 | 0.91 · 2.30 | 0.92 · 2.41 |

Which interval to use:
- **Inside the training chemistry, use GPR jackknife+.** Coverage there matches the nominal
  level, and the intervals are 4–5× narrower than conformal intervals around the baselines.
- **For a B-site family held out entirely, use the scaled σ.**
- **Never use the raw σ.**

The conformal guarantee (marginal coverage ≥ 1 − 2α) requires exchangeability. That holds for
GKF and approximately for LOO; it is violated for LOBO and LOAO. For Sr3BiX3, all three targets
fall inside every 90 % interval, and the Sr3BiI3 jackknife+ interval [1.07, 1.47] contains all
three literature values.

## 7. Operating limits (binding on every caller)

1. **Do not publish or use intervals for an unseen A-site.** Held-out conformal coverage there
   is 21–36 % at nominal 90 %. Scaled σ covers only by being 2.8 eV wide. LOAO point error is
   0.505 eV against a mean baseline of 0.62 eV.
2. **PBE, not reality.** Predictions inherit PBE's systematic underestimate against experiment
   and HSE. The v1 TBmBJ delta model is a separate, cross-stoichiometry correction.
3. **Label noise floor: 0.1–0.2 eV.** This is the code-to-code spread for one compound (for
   example, Ca3PCl3 1.98 / 2.04 / 2.21 eV). The headline MAE of 0.11 is at this floor.
   Improvements below it are not meaningful.
4. **Small n.** There are 39 formulas, 21 % of them preprint-only, and every CI is wide. Quote
   the verified-only number (LOBO 0.137 eV) beside the primary one.
5. **No SOC.** For Bi and Sb, the real (SOC) gap is lower; see the `soc` target.
6. **MAE is not comparable with v1.** The label range grew from 0.78–2.30 eV to 0.11–3.18 eV.

## 8. What the model learned (week 7, `reports/explain_v2.md`)

Physics expectations were **pre-registered before any result existed**
(`reports/explain_expectations.md`, commit 76f6425). They are tested on held-out rows of 50 GKF
fold-models, and every SHAP background comes from training rows only. The primary test changes
one element of a composition and records how the prediction moves:

| Expectation | GPR | RF (Magpie) | XGB (Magpie) |
|---|---|---|---|
| E1: gap rises I < Br < Cl < F | **CONFIRMED** (Br→I +0.42, Cl→Br +0.25, F→Cl +0.57 eV) | CONFIRMED | CONFIRMED |
| E2: gap falls P → Bi (decisive P − Bi) | **CONFIRMED** (+0.18 [0.06, 0.29]) | CONFIRMED | CONFIRMED |
| E3: gap falls Ca > Sr > Ba (decisive Ca − Ba) | **CONFIRMED** (+0.81 [0.67, 0.94]) | CONFIRMED | CONFIRMED |
| E4: dchi_BX SHAP slope > 0 | UNCLEAR | — | — |
| E6: A and X groups outweigh B | UNCLEAR: A > B confirmed, X > B unclear | — | — |

- **Top features.** For the GPR, mean |SHAP| ranks rA > dchi_AX > rX > rB > chiA.
- **Rank stability.** Kendall τ across folds is 0.48 on average for the GPR, and as low as
  −0.22. Credit moves between the collinear rX and dchi_AX from fold to fold. The tree models
  on Magpie are more stable, with τ of 0.87 (RF) and 0.80 (XGB). Their credit goes to atomic
  weight and atomic-number statistics.
- **Feature-level vs element-level results.** Feature-level SHAP signs are UNCLEAR for chiX
  and dchi_BX. The GPR's ARD kernel switches those features off and routes the halide effect
  through rX. chiB even leans against the expected sign, while rB carries the pnictogen effect.
  The element-level behaviour is still right: with correlated descriptors, attributions to
  single features are not interpretable one at a time.
- **Unregistered finding.** All three models predict Mg compounds **0.3–0.4 eV below** the
  matching Ca compounds. No prior was stated for this. It rests on 10 Mg formulas with mixed
  labels, so treat it as a hypothesis for DFT, not a result.

## 9. How to reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock   # use bin/ on Linux
python run_all.py            # data → features → train → evaluate → report → uncertainty → explain
python -m pytest -q tests -m "not slow"   # what CI runs
```

- **Rebuild time.** Offline, about 6.5 minutes for metrics, 8.5 minutes for uncertainty and
  3 minutes for explanations, on a 16-core Windows machine. Only `run_all.py fetch` uses the
  network.
- **Determinism.** Two runs on the same machine give byte-identical JSON; this is tested.
- **Across machines.** GPR outputs differ by about 0.005 eV between Linux and Windows. That is
  why cross-machine tests compare to documented tolerances, not to bytes.
