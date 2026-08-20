# A₃BX₃ Band-Gap Predictor

Machine-learning prediction of band gaps and elastic constants for **A₃BX₃ pnictogen-halide
inverse perovskites**, validated against published first-principles DFT.

The headline is not the accuracy number. It is that **picking the right structural family matters
more than picking the right model** — and that most of the work here went into proving the accuracy
number is real rather than leaked.

---

## The problem

Sr₃BiX₃ (X = I, Br, Cl) is a candidate lead-free photovoltaic absorber. Getting its band gap from
first principles is expensive, and hybrid-functional (HSE06) calculations more so.

The obvious shortcut — train on the large, well-populated set of ABX₃ halide perovskites and predict
Sr₃BiX₃ — **fails badly, by roughly 0.85 eV.** Sr₃BiX₃ is not an ABX₃ (1:1:3) perovskite. It belongs
to the **A₃BX₃ (3:1:3) inverse-perovskite** class: divalent alkaline-earth A-site, corner-sharing
BiX₆ octahedra in cubic Pm3̄m. Models trained on the wrong class are extrapolating, and they say so
only in their error.

Restricting the training set to the same A₃BX₃ pnictogen-halide family turns the same question into
an *interpolation* problem — with a 23-compound training set instead of hundreds.

---

## Results

### Band gap — family-restricted GPR

Gaussian process regression, ARD Matérn(ν = 5/2) kernel, crystal-structure descriptors derived from
ionic radii, electronegativities and atomic masses. Training set: 23 GGA-PBE literature band gaps
(A = Mg/Ca/Sr/Ba; B = P/As/Sb/Bi; X = F/Cl/Br/I), **every entry traceable to a primary DOI**.

| Test | Result |
|---|---|
| Leave-one-out CV (n = 23) | **MAE 0.151 eV, R² 0.740** |
| Leave-one-B-family-out CV | **MAE 0.146 eV** (As 0.165 · Bi 0.101 · P 0.198 · Sb 0.106) |
| Held-out Sr₃BiX₃ vs published DFT | **MAE 0.160 eV** |

The leave-one-B-family-out number is the one that matters. It holds out an entire chemistry — every
Bi compound at once — so the model cannot lean on a near-neighbour of the target. It scores *better*
than plain LOO, which is the behaviour you want from a family model rather than a memoriser.

### Beyond-PBE — TBmBJ delta correction

PBE systematically underestimates band gaps. Rather than predict the gap, predict the **correction**:
a RandomForest trained on 50 JARVIS halide perovskites carrying both OptB88vdW and TBmBJ gaps,
using the PBE gap itself as a feature.

| | MAE | R² |
|---|---|---|
| Δ model (GroupKFold by formula) | 0.275 eV | corrected-gap **0.924** |
| Sr₃BiX₃ raw PBE vs HSE06 reference | 0.661 eV | — |
| Sr₃BiX₃ **corrected** vs HSE06 reference | **0.312 eV** | — |

**A 53% error reduction against experiment-grade references at zero additional DFT cost.**

Caveat, carried in the metrics file itself and not just the paper: the Δ model is trained on ABX₃
halides, so applying it to A₃BX₃ is a cross-stoichiometry transfer. It is a correction, not a
guarantee.

### Elastic constants

RandomForest on 11 family compounds reproduces the reported Sr₃BiI₃ C₁₁ within ~7%
(64.5 vs 60.3 GPa). Family-level LOO MAE: C₁₁ 16.4, C₁₂ 10.6, C₄₄ 3.2 GPa. Derived moduli put the
Sr₃BiX₃ series at B/G = 1.76–1.80 — ductile, consistent with the reference DFT.

---

## The leakage audit

A companion model trained on the broad JARVIS-DFT perovskite set originally reported
**MAE 0.388 eV, R² 0.749**. That number was wrong — not arithmetically, but structurally.

JARVIS contains multiple polymorphs of the same formula. Under random K-fold, polymorphs of one
compound land in both train and test, and the model is scored on chemistry it has already seen.
Replacing random folds with **formula-grouped folds** gives the honest number:

| Fold scheme | MAE | R² |
|---|---|---|
| Random K-fold (leaky) | 0.388 eV | 0.749 |
| GroupKFold by formula | **0.526 eV** | **0.631** |

Both are kept in `data/jarvis_model_metrics.json`. The leaky one is stored under the key
`models_random_kfold_leaky` so it cannot be quoted by accident. The grouped number is the one
reported everywhere else.

Same audit applied to the elastic models, and it is less flattering: under grouped folds C₁₂ scores
**R² −0.217** and C₄₄ **R² 0.094**. Those are in the metrics file too. The bulk modulus model works
(R² 0.731); the individual stiffness constants largely do not, and the repo says so.

---

## An open conflict, not resolved

The family model predicts **Ba₃BiI₃ at 0.74 ± 0.15 eV**, anchored by published Ba₃AsI₃ and Ba₃SbI₃
values. This **contradicts** the monotonic Mg→Ca→Sr→Ba A-site trend asserted in earlier versions of
this work, which put it near 1.60 eV.

Both cannot be right. The prediction is flagged for DFT follow-up rather than quietly dropped or
smoothed into the trend line. If the trend is correct, this model has a specific, locatable failure;
if the model is correct, the trend claim needs revising.

---

## Install and reproduce

```bash
pip install -r requirements.txt
```

The datasets are committed, so training runs straight from a fresh clone:

```bash
python train_a3bx3_family.py
```

```bash
python train_mbj_delta.py
```

```bash
python train_jarvis.py
```

```bash
python sq_check.py
```

Each writes its metrics to `data/*_metrics.json`. Every number in this README comes from those files
— regenerate them and the tables above reproduce. Verified from a clean copy on 2026-08-20.

Rebuilding the dataset from source is optional and needs the extra `jarvis-tools` dependency:

```bash
python build_a3bx3.py
```

---

## Data provenance

`data/a3bx3_literature.csv` carries one row per literature compound with `year`, `doi` and a `note`
field naming the source table. No estimated values, no unverifiable DOIs, no "likely" entries — an
earlier revision of this dataset contained them and dropping them shrank the band-gap set from 34 to
20 entries before later additions brought it to 23. The smaller honest set predicts better.

JARVIS-DFT data is fetched via the public `jarvis-tools` API.

Reference DFT for validation: Islam et al., *New J. Chem.*, 2026, **50**, 522–536.

---

## Limitations

- 23 training compounds. This is a small-data regime; the GPR ±2σ bands are wide on purpose and
  should be reported with any prediction.
- Trained on **GGA-PBE** gaps. The Δ model lifts predictions toward HSE06 but does not replace a
  hybrid calculation.
- Cubic Pm3̄m A₃BX₃ only. Nothing here is validated on distorted or lower-symmetry variants.
- Elastic predictions are family-level and coarse — see the negative R² values above.
- Descriptors are compositional and lattice-derived. No electronic-structure features, so the model
  cannot distinguish polymorphs of the same formula.

---

## Citation

If this code or dataset is useful, cite the supporting manuscript (in preparation) and the reference
DFT study it validates against.

## License

MIT for the code. Literature-derived values in `data/` remain the property of their original
publications and are redistributed here only as extracted numerical values with DOI attribution.
