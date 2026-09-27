"""Named baselines and models. Every estimator returned here is a sklearn Pipeline (or a
GridSearchCV whose estimator is a Pipeline), so every fitted transform — scaling, variance
filter, feature selection — is fitted inside the training fold only. Hyperparameter
searches use an explicit inner grouped splitter (nested CV); nothing uses `cv=<int>`.

Baselines (design section 6, all printed beside every model):
    mean              training-fold mean
    halide_rule       training-fold mean per halide X (the chemist's rule of thumb)
    ridge_physics9    StandardScaler + RidgeCV on the nine v1 descriptors (same as v1)
    ridge_magpie      VarianceThreshold + StandardScaler + SelectKBest + Ridge on Magpie,
                      k and alpha chosen by inner grouped CV
Models:
    gpr_physics9      v1 champion: StandardScaler + GPR, ARD Matern(5/2) + white noise
    rf_physics9 / rf_magpie     RandomForest, small grid by inner grouped CV
    xgb_physics9 / xgb_magpie   XGBoost, shallow trees, small grid by inner grouped CV
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import CHIX_INDEX
from .splits import RepeatedGroupKFold


class HalideRule(RegressorMixin, BaseEstimator):
    """Predict the training-fold mean gap of rows with the same halide. The halide is read
    from one feature column (physics-9 chiX, which is unique per halide). A halide absent
    from the training fold falls back to the training-fold mean."""

    def __init__(self, column: int = CHIX_INDEX):
        self.column = column

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.global_mean_ = float(y.mean())
        keys = np.round(X[:, self.column], 6)
        self.means_ = {float(k): float(y[keys == k].mean()) for k in np.unique(keys)}
        return self

    def predict(self, X):
        keys = np.round(np.asarray(X, float)[:, self.column], 6)
        return np.array([self.means_.get(float(k), self.global_mean_) for k in keys])


def make_gpr_kernel(nf: int):
    """Identical to v1 train_a3bx3_family.make_gpr."""
    return (ConstantKernel(1.0, (1e-2, 1e4)) *
            Matern(length_scale=np.ones(nf), length_scale_bounds=(1e-2, 1e3), nu=2.5) +
            WhiteKernel(0.02, (1e-4, 0.5)))


@dataclass
class ModelSpec:
    name: str
    feature_set: str
    role: str                       # "baseline" | "model"
    description: str
    build: Callable[[int], object]  # nf -> estimator
    searched: bool = False          # GridSearchCV: fit needs groups


def _inner(cfg, seed):
    return RepeatedGroupKFold(cfg["splits"]["inner"]["n_splits"], 1, seed)


def model_specs(cfg: dict) -> list[ModelSpec]:
    seed = cfg["seed"]
    m = cfg["models"]
    ls = m["ridge_alphas_logspace"]
    alphas = np.logspace(ls["start"], ls["stop"], ls["num"])
    inner = _inner(cfg, seed)

    def grid(pipe, param_grid):
        return GridSearchCV(pipe, param_grid, cv=inner, scoring="neg_mean_absolute_error",
                            refit=True, n_jobs=1)

    rf_grid, xgb_grid = m["rf"]["grid"], m["xgb"]["grid"]

    def rf(nf):
        return grid(Pipeline([("rf", RandomForestRegressor(
            n_estimators=m["rf"]["n_estimators"], random_state=seed, n_jobs=1))]), rf_grid)

    def xgb(nf):
        from xgboost import XGBRegressor
        return grid(Pipeline([("xgb", XGBRegressor(random_state=seed, **m["xgb"]["fixed"]))]),
                    xgb_grid)

    g = m["gpr"]
    return [
        ModelSpec("mean", "physics9", "baseline", "training-fold mean",
                  lambda nf: Pipeline([("mean", DummyRegressor(strategy="mean"))])),
        ModelSpec("halide_rule", "physics9", "baseline", "training-fold mean per halide X",
                  lambda nf: Pipeline([("halide", HalideRule())])),
        ModelSpec("ridge_physics9", "physics9", "baseline",
                  "StandardScaler + RidgeCV(alphas 1e-3..1e3) on physics-9 (as v1)",
                  lambda nf: Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=alphas))])),
        ModelSpec("ridge_magpie", "magpie", "baseline",
                  "VarianceThreshold + StandardScaler + SelectKBest(f_regression) + Ridge on Magpie; "
                  "k, alpha by inner grouped CV",
                  lambda nf: grid(Pipeline([("var", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                                            ("select", SelectKBest(f_regression)), ("ridge", Ridge())]),
                                  m["ridge_magpie_grid"]), searched=True),
        ModelSpec("gpr_physics9", "physics9", "model",
                  "StandardScaler + GPR, ARD Matern(5/2) + white noise (v1 kernel)",
                  lambda nf: Pipeline([("scale", StandardScaler()), ("gpr", GaussianProcessRegressor(
                      kernel=make_gpr_kernel(nf), n_restarts_optimizer=g["n_restarts_optimizer"],
                      normalize_y=True, random_state=g["random_state"]))])),
        ModelSpec("rf_physics9", "physics9", "model", "RandomForest, grid by inner grouped CV",
                  rf, searched=True),
        ModelSpec("rf_magpie", "magpie", "model", "RandomForest, grid by inner grouped CV",
                  rf, searched=True),
        ModelSpec("xgb_physics9", "physics9", "model", "XGBoost (shallow), grid by inner grouped CV",
                  xgb, searched=True),
        ModelSpec("xgb_magpie", "magpie", "model", "XGBoost (shallow), grid by inner grouped CV",
                  xgb, searched=True),
    ]


def unwrap(est):
    """The Pipeline inside an estimator (GridSearchCV -> its estimator)."""
    return est.estimator if isinstance(est, GridSearchCV) else est


def fit(spec: ModelSpec, est, X, y, groups):
    if spec.searched:
        return est.fit(X, y, groups=groups)
    return est.fit(X, y)


def predict(est, X, want_std: bool = False):
    final = unwrap(est)
    if isinstance(est, GridSearchCV):
        final = est.best_estimator_
    last = final.steps[-1][1]
    if want_std and isinstance(last, GaussianProcessRegressor):
        mu, sd = final.predict(X, return_std=True)
        return np.asarray(mu), np.asarray(sd)
    return np.asarray(est.predict(X)), None
