"""Main model: LightGBM, with its complexity controlled on purpose (decision D14).

How memorisation is held back:

- early stopping on the log loss of the validation window, not on its average precision: the
  log loss gets worse as soon as the model grows over-confident on new data, while the average
  precision can keep creeping up by ten-thousandths as the model memorises the training rows;
- a search over regularisation only: leaves per tree, minimum transactions per leaf and L2;
- rolling-origin cross-validation in time (data.time_folds): four windows, each trained on the
  months before it, so a configuration has to work across the year, not on one period;
- the one-standard-error rule (choose): among the configurations whose mean cross-validated
  average precision is within one standard error of the best, the simplest wins.

No resampling (SMOTE or undersampling) and no class weight: the first search (D14) showed that
weighting the frauds never improved the selection period, and it distorts the probabilities.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Final

import lightgbm as lgb
import numpy as np
from scipy.special import expit

from pipeline.modeling import metrics
from pipeline.modeling.data import FeatureSpec, Split

SEED: Final = 20_261_115
NUM_THREADS: Final = 8  # fixed, with deterministic=True: same data, threads and seed, same model
MAX_ROUNDS: Final = 5_000
EARLY_STOPPING_ROUNDS: Final = 100
BASE_PARAMS: Final = {
    "objective": "binary",
    "metric": ["binary_logloss", "average_precision"],  # the first one stops the training
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_bin": 255,
    "verbosity": -1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": NUM_THREADS,
}
GRID: Final = tuple(
    {"num_leaves": leaves, "min_child_samples": minimum, "lambda_l2": l2}
    for leaves in (7, 15)
    for minimum in (300, 1_000)
    for l2 in (1.0, 10.0)
)
CURVE_POINTS: Final = 12


@dataclass
class Trained:
    booster: lgb.Booster  # holds exactly best_iteration trees
    params: dict[str, Any]
    best_iteration: int
    seconds: float


def fit(train: Split, validation: Split, spec: FeatureSpec, config: dict[str, Any]) -> Trained:
    """Train one configuration; stop when the validation log loss stops improving."""
    params = {**BASE_PARAMS, **config}
    names, categorical = list(spec.columns), [spec.category_index]
    train_set = lgb.Dataset(train.x, label=train.y, feature_name=names,
                            categorical_feature=categorical, free_raw_data=False)
    validation_set = lgb.Dataset(validation.x, label=validation.y, reference=train_set,
                                 feature_name=names, categorical_feature=categorical,
                                 free_raw_data=False)
    start = time.perf_counter()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True,
                                      verbose=False)],
    )
    seconds = time.perf_counter() - start
    best = booster.best_iteration or booster.current_iteration()
    final = lgb.Booster(model_str=booster.model_to_string(num_iteration=best))
    return Trained(final, params, best, seconds)


def raw_score(booster: lgb.Booster, x: np.ndarray, trees: int | None = None) -> np.ndarray:
    """Log-odds of fraud before calibration (with the first trees only, when given)."""
    return booster.predict(x, raw_score=True, num_iteration=trees, num_threads=NUM_THREADS)


def fold_result(trained: Trained, fitted_on: Split, validated_on: Split) -> dict[str, float]:
    """Validation quality and how much better the model does on the rows it learned from."""
    validation = raw_score(trained.booster, validated_on.x)
    seen = raw_score(trained.booster, fitted_on.x)
    log_loss = metrics.log_loss(validated_on.y, expit(validation))
    train_log_loss = metrics.log_loss(fitted_on.y, expit(seen))
    return {
        "trees": trained.best_iteration,
        "seconds": trained.seconds,
        "average_precision": metrics.average_precision(validated_on.y, validation),
        "log_loss": log_loss,
        "train_log_loss": train_log_loss,
        "gap": log_loss / train_log_loss,
    }


def summarize(config: dict[str, Any], folds: list[dict[str, float]]) -> dict[str, Any]:
    """Mean and standard error over the windows."""
    precision = np.array([fold["average_precision"] for fold in folds])
    return {
        **config,
        "cv_average_precision": float(precision.mean()),
        "cv_average_precision_se": float(precision.std(ddof=1) / math.sqrt(len(precision))),
        "cv_log_loss": float(np.mean([fold["log_loss"] for fold in folds])),
        "cv_gap": float(np.mean([fold["gap"] for fold in folds])),
        "cv_trees": float(np.mean([fold["trees"] for fold in folds])),
        "seconds": float(sum(fold["seconds"] for fold in folds)),
        "folds": folds,
    }


def complexity(row: dict[str, Any]) -> tuple[float, ...]:
    """Smaller is simpler: fewer leaves, then more rows per leaf, then more L2, then fewer trees."""
    return (row["num_leaves"], -row["min_child_samples"], -row["lambda_l2"], row["cv_trees"])


def choose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One-standard-error rule: the simplest row within one standard error of the best."""
    best = max(rows, key=lambda row: row["cv_average_precision"])
    floor = best["cv_average_precision"] - best["cv_average_precision_se"]
    return min((row for row in rows if row["cv_average_precision"] >= floor), key=complexity)


def learning_curve(booster: lgb.Booster, sets: dict[str, Split]) -> list[dict[str, float]]:
    """Log loss and average precision on each set with the first k trees, for a few k."""
    total = booster.num_trees()
    checkpoints = np.unique(np.geomspace(min(25, total), total, CURVE_POINTS).round().astype(int))
    curve = []
    for trees in checkpoints:
        row: dict[str, float] = {"trees": int(trees)}
        for name, split in sets.items():
            raw = raw_score(booster, split.x, int(trees))
            row[f"{name}_log_loss"] = metrics.log_loss(split.y, expit(raw))
            row[f"{name}_average_precision"] = metrics.average_precision(split.y, raw)
        curve.append(row)
    return curve
