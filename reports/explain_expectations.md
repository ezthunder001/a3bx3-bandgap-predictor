# Pre-registered physics expectations for the week-7 explanation step

Written 2026-09-28 and committed before any SHAP value, permutation importance or substitution
effect was computed. `reports/explain_v2.json` records the commit hash of this file, so the
order can be checked. Nothing below may be edited after the results exist. If an expectation
turns out wrong, the result is reported as CONTRADICTED.

**What was already known when this was written.** I had seen the curated labels
(`data/processed/family_primary.csv`) and the metrics. For example, Ca3PX3 gaps fall
Cl > Br > I, and the Ba3MX3 preprint series is nearly flat across M. So these expectations are
not blind to the data. What they test is whether the **model** learned these trends from 39
compounds, and whether its attributions agree with basic chemistry. They are not a test of the
chemistry itself.

## Model and data the checks apply to

- The headline model: `gpr_physics9` on the `primary` dataset.
- For comparison: `rf_magpie` and `xgb_magpie`, the tree models with the best leave-one-A-out
  scores.
- Each check is computed on the models fitted in the outer folds:
  - GroupKFold(5) × 10 repeats gives 50 fold-models. These give the interval.
  - Leave-one-B-out gives 4 fold-models. These are reported as a supplement, min–max only.
- The attributed rows are always that fold's held-out rows. Every SHAP background set is drawn
  from that fold's training rows only.

## Two kinds of evidence, one of them primary

1. **Primary: element-substitution effect.** This test holds up even when descriptors are
   correlated.
   - For every held-out row of a fold, predict the gap again with one site substituted. For
     example, keep A and B and set X to each of F, Cl, Br and I.
   - The features are rebuilt from the substituted composition.
   - The statistic is the mean predicted difference between two elements. An example is
     gap(X=Cl) − gap(X=Br), averaged over the held-out rows of the fold.
   - This needs no attribution method. It asks what the model does when only one site's
     element changes.
2. **Secondary: SHAP dependence slope.**
   - Fit an OLS slope of a feature's SHAP value against that feature's value, over the held-out
     rows of a fold.
   - The physics-9 descriptors are strongly correlated inside this family. rX, chiX, dchi_BX,
     dchi_AX and rB_over_rX all change together when X changes.
   - KernelExplainer and interventional SHAP assume the features are independent. So credit for
     one physical cause is split among correlated descriptors, and a single descriptor's sign
     can flip while the element-level effect stays right (Aas, Jullum & Løland, *Artif. Intell.*
     2021, doi 10.1016/j.artint.2021.103502, verified with CrossRef).
   - The feature-level signs are therefore recorded, but the element-level test decides the
     verdict.

**Verdict rule (applied to the 50 GKF fold-models).**

| Verdict | Condition |
|---|---|
| CONFIRMED | mean over folds has the expected sign, and the 2.5–97.5 percentile range across folds excludes 0 |
| CONTRADICTED | mean has the opposite sign, and the range excludes 0 |
| UNCLEAR | anything else |
| REPORTED (no verdict) | no strong prior was stated |

## Expectations

### E1 — Halide: the gap rises with halide electronegativity, I < Br < Cl < F (A and B held fixed)

**Strong prior.**

- **Why.** The halide supplies filled p states below the valence-band edge. A more
  electronegative halide binds its p electrons more tightly (Pauling electronegativity:
  F > Cl > Br > I; Pauling, *J. Am. Chem. Soc.* 1932, doi 10.1021/ja01348a011, verified with
  CrossRef). That makes the compound more ionic, which pushes the occupied and empty states
  apart. A smaller halide also shortens the lattice.
- **Element test.** Each adjacent difference should be > 0:
  - gap(F) − gap(Cl)
  - gap(Cl) − gap(Br)
  - gap(Br) − gap(I)
- **Caveat on F.** F appears in only 4 primary rows (all with Ca or Mg). The F − Cl step is
  therefore expected to be the least certain.
- **SHAP sign.** chiX slope > 0; rX slope < 0.

### E2 — Pnictogen: the gap falls down the group, P > As > Sb > Bi (A and X held fixed)

**Moderate prior overall; weak for adjacent pairs.**

- **Why.** Going down group 15, the valence np levels rise, electronegativity falls
  (P 2.19, As 2.18, Sb 2.05, Bi 2.02 in the v1 tables), and the orbitals grow and broaden the
  bands. In these antiperovskites the pnictogen p states sit at the valence-band top. So a
  heavier pnictogen should raise the band edge and narrow the gap.
- **Why the prior is only moderate.**
  - The labels are PBE **without** spin–orbit coupling, so the large SOC narrowing for Bi is
    absent from the target.
  - P and As have almost the same electronegativity.
- **Element test.**
  - gap(P) − gap(Bi) > 0: moderate prior, and the one that decides E2.
  - gap(P) − gap(As), gap(As) − gap(Sb), gap(Sb) − gap(Bi): each expected ≥ 0. P − As is
    expected to be UNCLEAR.
- **SHAP sign.** chiB slope > 0; rB slope < 0.

### E3 — A-site: the gap falls Ca > Sr > Ba; no prior for Mg

**Moderate prior for Ca → Sr → Ba; none for Mg.**

- **Why.**
  - The empty d states of the heavier alkaline earths (Ca 3d, Sr 4d, Ba 5d) lie lower down
    the group. They contribute to the conduction-band edge, which lowers it.
  - The lattice also expands from Ca to Ba.
  - Qualitatively, the same trend holds for simple alkaline-earth compounds, whose gaps shrink
    down the group.
  - Mg has no low-lying d states, so its conduction band has a different character. No
    ordering relative to Ca is predicted.
- **Element test.**
  - gap(Ca) − gap(Sr) > 0
  - gap(Sr) − gap(Ba) > 0
  - gap(Ca) − gap(Ba) > 0: this one decides E3.
  - gap(Mg) − gap(Ca): REPORTED, no verdict.
- **SHAP sign.**
  - rA slope < 0: moderate prior, confounded with Mg.
  - chiA: no prior, REPORTED. Mg has the highest chiA but the smallest radius, so the two
    A descriptors pull in different directions.

### E4 — B–X electronegativity difference: dchi_BX slope > 0

**Moderate prior.** A larger B–X electronegativity difference means a more ionic B–X
interaction, which should widen the gap.

- It is collinear with chiX, so E1 and E4 are not independent evidence.
- **SHAP sign only.**

### E5 — No prior

The SHAP slopes of dchi_AX and rB_over_rX, and the Magpie-feature rankings, are REPORTED
without a verdict.

### E6 — Which site matters most

**Moderate prior.** The A-site group (rA, chiA) and the X-site group (rX, chiX) should each
carry a larger total mean |SHAP| than the B-site group (rB, chiB).

- The three cross-site descriptors (dchi_BX, dchi_AX, rB_over_rX) are left out of the group
  sums, because they mix sites.
- **Why.** The pnictogen changes the gap less than the cation or the halide (E2 is the weakest
  prior). The data I had already seen agrees: the Ba3MX3 series is nearly flat across M.
- **Test.** Per fold, compute sum|SHAP|(A group) − sum|SHAP|(B group), and the same for X − B.
  Each difference is judged by the verdict rule. The permutation importance of the same groups
  on the held-out rows is a model-agnostic cross-check.

## Not expectations, reported descriptively

- Rank stability of mean |SHAP| across folds: Kendall τ between fold rankings.
- Agreement between the SHAP ranking and the held-out permutation importance.

## Tools named in advance

- GPR: `shap.KernelExplainer`, with a background of up to 20 training-fold rows chosen by a
  seeded sample.
- Tree models: `shap.TreeExplainer`, interventional, with the same background (Lundberg et al.,
  *Nat. Mach. Intell.* 2020, doi 10.1038/s42256-019-0138-9, verified with CrossRef).
- Permutation importance: `sklearn.inspection.permutation_importance` on the held-out rows,
  scoring by negative MAE, with seeded repeats.
