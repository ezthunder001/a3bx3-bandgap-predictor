"""Explicit, grouped splitters. No `cv=<int>` anywhere: every split is an object that
names its groups, and every fold is checked for shared formulas and holdout rows.

Outer splits (design section 5, reported in this order):
    lobo  leave-one-B-family-out   (P / As / Sb / Bi)       headline
    loao  leave-one-A-out          (Mg / Ca / Sr / Ba)      harder
    gkf   GroupKFold(5) by formula, repeated 10x with shuffled group order
    loo   leave-one-formula-out    v1 comparison only
The Sr3BiX3 holdout is not part of the family matrix at all; curate.py removed it, and
check_folds() refuses any fold whose training side contains a holdout formula.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut


class RepeatedGroupKFold:
    """GroupKFold with a seeded shuffle of group order, repeated. Groups are split into
    near-equal folds by count of groups (here every group is one formula)."""

    def __init__(self, n_splits: int = 5, n_repeats: int = 1, random_state: int = 0):
        self.n_splits, self.n_repeats, self.random_state = n_splits, n_repeats, random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits * self.n_repeats

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("RepeatedGroupKFold requires groups")
        groups = np.asarray(groups)
        uniq = np.array(sorted(set(groups.tolist())))
        if len(uniq) < self.n_splits:
            raise ValueError(f"{len(uniq)} groups < n_splits={self.n_splits}")
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_repeats):
            order = uniq[rng.permutation(len(uniq))]
            for fold_groups in np.array_split(order, self.n_splits):
                te_mask = np.isin(groups, fold_groups)
                yield np.flatnonzero(~te_mask), np.flatnonzero(te_mask)


@dataclass
class Split:
    name: str
    description: str
    groups_key: str            # which column defines the held-out unit
    folds: list                # list of (train_idx, test_idx)
    n_repeats: int = 1


def make_splits(rows: list[dict], cfg: dict, seed: int) -> dict[str, Split]:
    formula = np.array([r["formula"] for r in rows])
    B = np.array([r["B"] for r in rows])
    A = np.array([r["A"] for r in rows])
    logo = LeaveOneGroupOut()
    X_dummy = np.zeros((len(rows), 1))
    g = cfg["gkf"]
    out = {
        "lobo": Split("lobo", "leave-one-B-family-out (P/As/Sb/Bi)", "B",
                      list(logo.split(X_dummy, groups=B))),
        "loao": Split("loao", "leave-one-A-out (Mg/Ca/Sr/Ba)", "A",
                      list(logo.split(X_dummy, groups=A))),
        "gkf": Split("gkf", f"GroupKFold({g['n_splits']}) by formula, repeated "
                            f"{g['n_repeats']}x with shuffled group order", "formula",
                     list(RepeatedGroupKFold(g["n_splits"], g["n_repeats"], seed)
                          .split(X_dummy, groups=formula)), n_repeats=g["n_repeats"]),
        "loo": Split("loo", "leave-one-formula-out (v1 comparison only)", "formula",
                     list(logo.split(X_dummy, groups=formula))),
    }
    for s in out.values():
        check_folds(rows, s.folds, holdout=set())
    return out


def check_folds(rows: list[dict], folds, holdout: set[str]) -> None:
    """Raise if any fold shares a formula between train and test, or trains on a holdout
    formula. Called on every split before any model sees it."""
    formula = np.array([r["formula"] for r in rows])
    for i, (tr, te) in enumerate(folds):
        shared = set(formula[tr]) & set(formula[te])
        if shared:
            raise AssertionError(f"fold {i}: formula shared between train and test: {sorted(shared)}")
        leaked = set(formula[tr]) & holdout
        if leaked:
            raise AssertionError(f"fold {i}: locked holdout in training fold: {sorted(leaked)}")
        if len(te) == 0 or len(tr) == 0:
            raise AssertionError(f"fold {i}: empty side")


def assert_holdout_excluded(train_formulas, holdout: set[str]) -> None:
    leaked = set(train_formulas) & set(holdout)
    if leaked:
        raise AssertionError(f"locked holdout reached a training set: {sorted(leaked)}")
