import sys; sys.path.insert(0, ".")
from simulate_sr3bix3 import sq_efficiency

cases = [
    ("Sr3BiI3",  "PBE",  1.324),
    ("Sr3BiBr3", "PBE",  1.512),
    ("Sr3BiCl3", "PBE",  1.731),
    ("Sr3BiI3",  "HSE",  1.878),
    ("Sr3BiBr3", "HSE",  2.245),
    ("Sr3BiCl3", "HSE",  2.427),
    ("Sr3BiI3",  "GPR",  1.204),
    ("Sr3BiBr3", "GPR",  1.630),
    ("Sr3BiCl3", "GPR",  1.779),
    ("Ba3BiI3",  "GPR",  1.272),
    ("Ba3BiBr3", "GPR",  1.645),
    ("Ba3BiCl3", "GPR",  1.776),
]

print(f"{'Compound':<12} {'Level':<6} {'Eg':>6} {'PCE%':>7} {'Voc':>6} {'Jsc':>8} {'FF':>6}")
print("-" * 58)
for name, level, eg in cases:
    r = sq_efficiency(eg)
    print(f"{name:<12} {level:<6} {eg:>6.3f} {r['PCE_pct']:>7.2f} {r['Voc_V']:>6.3f} {r['Jsc_mAcm2']:>8.2f} {r['FF']:>6.3f}")
