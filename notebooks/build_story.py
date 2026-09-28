"""Writes notebooks/01_story.ipynb (narrative only) and executes it headless.

The notebook computes nothing new: its code cells only read committed report files
(reports/*.json, data/*.json) and format them. Figures are linked, not embedded, to keep the
file small. Run:  python notebooks/build_story.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
NB = HERE / "01_story.ipynb"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# A3BX3 band gaps: small data, honest validation

This notebook tells the story. **It computes nothing.** Every number is read from a committed
report that `python run_all.py` regenerates:

- `reports/metrics_v2.json`
- `reports/uncertainty_v2.json`
- `reports/explain_v2.json`
- `data/a3bx3_family_metrics.json` (v1)"""),
    code("""import json
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))

M = load("reports/metrics_v2.json")
U = load("reports/uncertainty_v2.json")
E = load("reports/explain_v2.json")
V1 = load("data/a3bx3_family_metrics.json")

def ci(e):
    return f"{e['MAE']:.3f} [{e['MAE_CI95'][0]:.3f}, {e['MAE_CI95'][1]:.3f}]"
print("reports loaded:", M["version"], "| environment:", M["environment"]["scikit-learn"])"""),
    md("""## 1. The problem

**Goal.** Predict the GGA-PBE band gap of A3BX3 inverse-perovskite pnictogen halides:
A = Mg/Ca/Sr/Ba, B = P/As/Sb/Bi, X = F/Cl/Br/I.

**Why these compounds.** Sr3BiX3 is a candidate lead-free absorber.

**Why the obvious shortcut fails.** The obvious shortcut is to train on the large ABX3 perovskite
literature. But A3BX3 (3:1:3) is a different structural class from ABX3 (1:1:3)."""),
    md("""## 2. Why the structural family matters more than the model

In v1, a model trained on ABX3 halides missed Sr3BiX3 by about **0.85 eV**. That figure is quoted
in the v1 README and is not stored in a metrics file.

A model restricted to the A3BX3 family, trained on 23 literature rows, landed within about
0.16 eV. The same question stopped being an extrapolation and became an interpolation."""),
    code("""print("v1 family GPR on Sr3BiX3, MAE:", V1["sr3bix3_validation"]["MAE"], "eV")
print("v1 LOO / LOBO MAE:", V1["loo"]["MAE"], "/", V1["lobo"]["MAE"], "eV  (history; see section 3)")"""),
    md("""## 3. Data, and the corrections that v1 needed

Week 3 added a literature harvest of 37 sources. Checking every v1 row against its source found
real errors:

- **Mg3BiI3 and Mg3BiBr3 carried HSE06 gaps labelled as PBE** (0.867 and 1.626 eV instead of
  0.224 and 1.071 eV).
- Several rows cite a paper that does not contain their value.

The fixes live in `data/raw/literature/corrections.csv`. The v1 file itself is never edited.
The label is the **median** of the no-SOC PBE values for each formula."""),
    code("""for d in ("primary", "verified_only", "plus_unverified", "soc"):
    c = M["datasets"][d]["curate"]
    print(f"{d:<16} {c['n_formulas']:>3} formulas  {c['n_values']:>3} values  "
          f"high_conflict {c['n_high_conflict']}  preprint-only {c['n_preprint_only']}")
print("K1 checkpoint:", M["k1"])"""),
    md("""## 4. Honest validation

**How every score is produced:**

- Every score is out-of-fold, next to four named baselines on the **same folds**.
- Splits come from explicit grouped splitters.
- Every transform sits inside a `Pipeline`.
- Hyperparameters are chosen by nested CV.

**The headline split.** It is **leave-one-B-family-out**: a whole pnictogen chemistry is held
out at once.

![parity](../reports/figures/parity_lobo_gpr_physics9.png)"""),
    code("""print(f"{'set':<14}{'split':<6}{'gpr_physics9':<24}{'mean':<24}{'ridge_physics9':<24}best baseline")
for d in ("primary", "verified_only"):
    for s, r in M["datasets"][d]["results"].items():
        m = r["models"]
        print(f"{d:<14}{s:<6}{ci(m['gpr_physics9']):<24}{ci(m['mean']):<24}"
              f"{ci(m['ridge_physics9']):<24}{r['best_baseline']} {m[r['best_baseline']]['MAE']:.3f}")
k2 = U["k2"]["datasets"]["primary"]
print("\\nK2:", k2["verdict"], "| skill vs mean", k2["vs_mean"]["skill"], k2["vs_mean"]["skill_CI95"],
      "| vs best baseline", k2["best_baseline"], k2["vs_best_baseline"]["skill"], k2["vs_best_baseline"]["skill_CI95"])"""),
    md("""Two things temper the headline:

- The primary LOBO MAE (about 0.11 eV) is at the **label noise floor** of 0.1–0.2 eV, which is
  the code-to-code spread for a single compound.
- Against the best baseline, the lower CI bound of the skill is 0.14. That is below the 30 %
  K2 bar."""),
    md("""## 5. Uncertainty

**How the intervals are built.** Every interval is fitted on the training fold only:

- the GPR's σ, raw and rescaled by inner CV
- CV+ and jackknife+ (MAPIE), around the GPR and around the baselines

**Where the guarantee holds.** Exchangeability, and with it the conformal guarantee, holds for
GroupKFold. It does **not** hold when a whole chemistry is held out.

![reliability](../reports/figures/uncertainty_reliability_primary.png)"""),
    code("""methods = ["gpr_sigma", "gpr_sigma_scaled", "gpr_physics9_jackknife_plus", "ridge_physics9_cvplus", "mean_cvplus"]
print(f"{'method':<30}" + "".join(f"{s:>16}" for s in U["coverage"]["primary"]))
for mth in methods:
    row = ""
    for s, S in U["coverage"]["primary"].items():
        e = S["methods"][mth]["0.90"]
        row += f"{e['coverage']:>8.2f} ({e['mean_width']:.2f})"
    print(f"{mth:<30}{row}")
print("\\ncoverage at nominal 90 % (mean width, eV); exchangeability:", U["exchangeability"])"""),
    md("""## 6. What the model learned, tested against physics written down beforehand

The expectations were pre-registered and committed **before** any SHAP value existed, in
`reports/explain_expectations.md`.

**The primary test.** Swap one element and watch the prediction. This holds up even though
descriptors are correlated inside the family. SHAP dependence slopes are secondary evidence.

![substitution](../reports/figures/explain_substitution.png)"""),
    code("""print("expectations commit:", E["expectations"]["commit"])
for model, r in E["physics_check"].items():
    print(f"\\n{model}")
    for k, v in r.items():
        verdict = v.get("verdict") if isinstance(v, dict) and "verdict" in v else v
        print(f"  {k:<42} {verdict}")
g = E["models"]["gpr_physics9"]["splits"]["gkf"]
print("\\nGPR top features:", g["top10"][:5], "| rank stability (Kendall tau):", g["rank_stability"]["kendall_tau_all_features"])
print("Mg - Ca substitution (no prior):", g["substitution"]["A:Mg-Ca"]["mean"], "eV")"""),
    md("""**Findings:**

- **The element-level trends hold for all three models:**
  - halide I < Br < Cl < F
  - pnictogen P > Bi
  - A-site Ca > Sr > Ba
- **Single-descriptor SHAP signs are often UNCLEAR.** The GPR's ARD kernel switches off chiX and
  dchi_BX and routes the halide effect through rX. With correlated descriptors, attributions to
  one feature at a time are not interpretable.
- **An unregistered observation.** Mg compounds are predicted 0.3–0.4 eV below Ca compounds. It
  is a hypothesis for DFT, not a result.

![beeswarm](../reports/figures/explain_beeswarm_gpr_physics9.png)"""),
    md("""## 7. Limits (the binding list is MODEL_CARD.md §7)

- These are PBE gaps without SOC, not experimental ones.
- The label noise floor is 0.1–0.2 eV.
- n = 39, and 21 % of the formulas come only from a preprint.
- **No intervals and no trust for an unseen A-site.** LOAO conformal coverage is 21–36 % at
  nominal 90 %.
- The model covers the cubic 4 × 4 × 4 family only."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
ExecutePreprocessor(timeout=120, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
for c in nb.cells:                      # strip volatile metadata so reruns diff cleanly
    c.metadata = {}
    if c.cell_type == "code":
        c.execution_count = None
        for o in c.get("outputs", []):
            o.pop("execution_count", None)
nbf.write(nb, NB)
print("wrote", NB, NB.stat().st_size, "bytes")
