"""Generate the README results table from committed JSON (never typed by hand).

Reads reports/metrics_v2.json (GPR / XGB / RF rows, baselines, Sr3BiX3 holdout) and
reports/mlp_v2.json (MLP rows, same folds) and rewrites the block between
<!-- results-table:start --> and <!-- results-table:end --> in README.md.
`python -m a3bx3.readme_table` ; tests check the README block equals a fresh render.
"""
from __future__ import annotations

import json
import sys

from . import utf8_stdio
from .config import ROOT
from .manifest import write_atomic

START, END = "<!-- results-table:start -->", "<!-- results-table:end -->"
ROWS = [("gpr_physics9", "**GPR, physics-9 descriptors**", True),
        ("xgb_magpie", "XGBoost, Magpie", False),
        ("rf_magpie", "Random forest, Magpie", False),
        ("mlp_physics9", "MLP (PyTorch), physics-9", False),
        ("mlp_magpie", "MLP (PyTorch), Magpie", False)]


def _ci(e):
    return f"{e['MAE']:.3f} [{e['MAE_CI95'][0]:.3f}, {e['MAE_CI95'][1]:.3f}]"


def render() -> str:
    m = json.loads((ROOT / "reports" / "metrics_v2.json").read_text(encoding="utf-8"))["datasets"]["primary"]
    p = ROOT / "reports" / "mlp_v2.json"
    mlp = json.loads(p.read_text(encoding="utf-8"))["datasets"]["primary"] if p.exists() else None
    lobo, loao = m["results"]["lobo"], m["results"]["loao"]
    best = lobo["best_baseline"]
    mean = lobo["models"]["mean"]["MAE"]
    bm = lobo["models"][best]["MAE"]
    L = ["| Model | Leave-one-B-family-out | Leave-one-A-out | Mean baseline (LOBO) | Best baseline (LOBO) | Sr₃BiX₃ locked holdout |",
         "|---|---|---|---|---|---|"]
    for key, label, bold in ROWS:
        if key.startswith("mlp"):
            if mlp is None:
                continue
            a, b, hold = mlp["lobo"]["models"][key], mlp["loao"]["models"][key], "—"
        else:
            a, b = lobo["models"][key], loao["models"][key]
            hold = f"{m['sr3bix3_holdout']['models'][key]['MAE']:.3f}"
        f = (lambda x: f"**{x}**") if bold else (lambda x: x)
        base = f"{best.replace('_', '-').replace('ridge-magpie', 'ridge-Magpie')} {bm:.3f}" if bold else f"{bm:.3f}"
        L.append(f"| {label} | {f(_ci(a))} | {_ci(b)} | {mean:.3f} | {base} | "
                 f"{f(hold) if hold != '—' else hold}{' (n = 3)' if bold else ''} |")
    return "\n".join(L)


def update(path=None) -> bool:
    path = path or ROOT / "README.md"
    s = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    a, b = s.index(START) + len(START), s.index(END)
    new = s[:a] + "\n" + render() + "\n" + s[b:]
    changed = new != s
    if changed:
        write_atomic(path, new.encode("utf-8"))
    return changed


def main(argv=None) -> int:
    utf8_stdio()
    print("README table", "updated" if update() else "already current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
