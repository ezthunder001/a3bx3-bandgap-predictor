"""a3bx3 — v2 pipeline for PBE band gaps of A3BX3 inverse-perovskite pnictogen halides.

Layout (design doc section 7):
    fetch/      network pulls into data/raw/ (only `make fetch` touches the network)
    curate      dedupe, conflict log, SOC flag, locked holdout split
    features    physics-9 descriptors + matminer Magpie
    splits      LOBO, LOAO, repeated GroupKFold by formula, LOO, locked holdout
    models      named baselines + GPR/RF/XGB, every transform inside a Pipeline
    evaluate    out-of-fold metrics, bootstrap CI, per-family error, parity plot
"""
from __future__ import annotations

import sys

__version__ = "2.0.0.dev0"


def utf8_stdio() -> None:
    """Windows cp1252 consoles crash on non-ASCII glyphs; every CLI entry calls this."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
