# Pre-registered decision rule — week 8, MLP vs XGB / GPR (design §8, November milestone)

Written on 2026-09-28. This file was committed before any MLP was trained and before any MLP
number existed. `reports/mlp_v2.json` records its commit hash and a content hash. Nothing here
may be edited after the results exist.

## Question

The November milestone is "MLP vs XGB on family-held-out splits". The question: does a small
PyTorch MLP, trained on the same features and the same folds, beat XGBoost and the GPR on
**leave-one-B-family-out (LOBO)** and **leave-one-A-out (LOAO)**?

## What is compared

- **Dataset:** `primary` (39 formulas) decides the verdict. `verified_only` (31) is a
  sensitivity check only.
- **Folds:** exactly the outer folds of `metrics_v2`:
  - LOBO and LOAO decide the verdict.
  - GroupKFold(5)×10 and LOO are reported only.
- **Competitors:** the existing out-of-fold predictions in `reports/oof_predictions.csv`. They
  are not recomputed.
- **Predefined pairs:** each MLP is compared on its own feature set.

  | MLP | against |
  |---|---|
  | `mlp_physics9` | `xgb_physics9`, `gpr_physics9` |
  | `mlp_magpie` | `xgb_magpie`, `gpr_physics9` (the headline model) |
  | both MLPs | `mean` (sanity check) |

- **MLP spec, fixed now:**
  - Architecture: 1 or 2 hidden layers, width 16 or 64, ReLU.
  - Training: Adam with weight decay 1e-3 or 1e-1, dropout 0 or 0.2.
  - Grid: the 16 combinations above, chosen by an inner GroupKFold(5) by formula on the outer
    training fold.
  - Early stopping: on an inner validation split (20 %, grouped by formula) drawn from the
    training data of that fit only.
  - Preprocessing: features scaled with StandardScaler inside a Pipeline; target
    standardised inside the fold.
  - Final model: an average of 5 seeds.

## Statistic

For each pair and split: Δ = MAE(MLP) − MAE(other), computed on the same held-out rows. The CI
is a paired bootstrap 95 % over formulas (2000 resamples, seeded).

| Outcome | Condition |
|---|---|
| **WIN** | CI entirely below 0 |
| **LOSS** | CI entirely above 0 |
| **TIE** | CI contains 0 |

## Milestone verdict (primary set)

| Verdict | Condition |
|---|---|
| **"MLP beats XGB"** | at least one MLP variant WINs against the XGB on its feature set on **both** LOBO and LOAO, and neither variant LOSEs to its XGB on either split |
| **"MLP beats GPR"** | at least one MLP variant WINs against `gpr_physics9` on **both** LOBO and LOAO |
| **NO** | anything else; each pair's WIN / TIE / LOSS is then reported per split |

Choosing the better of two MLP variants is optimistic in favour of the MLP. This is stated here
so that a "YES" can be discounted accordingly.

## Expected outcome (written down so it can be wrong)

At n ≈ 39, a neural network has more parameters than rows. The expected result:

- **vs GPR:** LOBO **LOSS**, because the GPR is at 0.111 eV, near the label noise floor.
  LOAO **TIE** or LOSS. Every model is weak on a new A-site.
- **vs XGB:** **TIE** on LOBO and on LOAO.
- **vs mean:** the MLP WINs on LOBO. That shows it learns something.
- **Milestone:** **NO** for both questions. The MLP ≤ XGB, and the MLP < GPR, in MAE.

## Learning curve (reported, no verdict)

- **Setup:** GKF×10 on primary. Each outer training fold is subsampled to 50 %, 75 % and
  100 % (seeded, grouped by formula, drawn only from that fold's training rows).
- **Models:** `mlp_physics9`, `gpr_physics9`, `xgb_physics9` and `xgb_magpie`.
- **Expectation:** every model improves with n. The MLP improves the most in relative terms
  but stays above the GPR at 100 %. There is no sign that more data would *not* help. That is,
  the family set is data-limited, not model-limited.
