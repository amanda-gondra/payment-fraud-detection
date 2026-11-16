"""Tests for the control of memorisation: time windows, early stopping, the choice rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from pipeline import features
from pipeline.modeling import data, lgbm, metrics
from pipeline.modeling.data import FeatureSpec, Split


def row(leaves: int, minimum: int, l2: float, mean: float, *, se: float = 0.002,
        trees: float = 1_000) -> dict:
    return {"num_leaves": leaves, "min_child_samples": minimum, "lambda_l2": l2,
            "cv_trees": trees, "cv_average_precision": mean, "cv_average_precision_se": se}


def test_the_simplest_configuration_within_one_standard_error_wins() -> None:
    rows = [row(15, 300, 1.0, 0.9900), row(7, 1_000, 10.0, 0.9885), row(7, 300, 1.0, 0.9850)]
    assert lgbm.choose(rows) is rows[1]  # 0.9885 >= 0.9900 - 0.002, and it is the simplest


def test_the_best_stays_when_no_simpler_one_is_close() -> None:
    rows = [row(15, 300, 1.0, 0.9900, se=0.0005), row(7, 1_000, 10.0, 0.9850)]
    assert lgbm.choose(rows) is rows[0]


def test_ties_in_size_prefer_more_regularisation_then_fewer_trees() -> None:
    rows = [row(7, 300, 1.0, 0.99), row(7, 1_000, 1.0, 0.99), row(7, 1_000, 10.0, 0.99, trees=900),
            row(7, 1_000, 10.0, 0.99, trees=800)]
    assert lgbm.choose(rows) is rows[3]


def test_summary_mean_and_standard_error() -> None:
    folds = [{"average_precision": value, "log_loss": 0.01, "gap": 2.0, "trees": 100,
              "seconds": 1.0} for value in (0.90, 0.92, 0.94, 0.96)]
    summary = lgbm.summarize({"num_leaves": 7}, folds)
    assert summary["cv_average_precision"] == pytest.approx(0.93)
    assert summary["cv_average_precision_se"] == pytest.approx(np.std([.90, .92, .94, .96],
                                                                      ddof=1) / 2)
    assert summary["seconds"] == 4.0


def timeline(start: str, end: str, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    moments = pd.date_range(start, end, freq="6h")
    frame = pd.DataFrame({name: rng.random(len(moments)) for name in features.FEATURE_COLUMNS})
    frame["category"] = "misc_net"
    frame["timestamp"] = moments
    frame["is_fraud"] = (rng.random(len(moments)) < 0.1).astype("int8")
    spec = FeatureSpec(columns=tuple(features.FEATURE_COLUMNS), categories=("misc_net",))
    return Split.of(frame, spec)


def test_every_window_learns_only_from_the_past() -> None:
    train = timeline("2019-01-01", "2019-12-31 23:00", 1)
    selection = timeline("2020-01-01", "2020-03-31 23:00", 2)
    folds = data.time_folds(train, selection)
    assert [label for label, _, _ in folds] == [
        *(label for label, _, _ in data.CV_WINDOWS), data.FINAL_WINDOW]
    for (_, start, end), (_, fitted, checked) in zip(data.CV_WINDOWS, folds, strict=False):
        assert fitted.frame["timestamp"].max() < pd.Timestamp(start)
        assert checked.frame["timestamp"].min() >= pd.Timestamp(start)
        assert checked.frame["timestamp"].max() < pd.Timestamp(end)
        assert len(fitted.y) == len(fitted.x) == len(fitted.amount)
    assert folds[-1][1] is train and folds[-1][2] is selection


def small_problem(seed: int, n: int = 3_000) -> tuple[Split, Split, FeatureSpec]:
    rng = np.random.default_rng(seed)
    spec = FeatureSpec(columns=tuple(features.FEATURE_COLUMNS), categories=("misc_net",))
    splits = []
    for _ in range(2):
        x = rng.normal(size=(n, len(spec.columns)))
        x[:, spec.category_index] = 0
        y = (x[:, 0] + rng.normal(0, 1.2, n) > 2.2).astype("int8")
        splits.append(Split(pd.DataFrame(index=range(n)), x, y, np.ones(n)))
    return splits[0], splits[1], spec


def test_training_stops_at_the_lowest_validation_log_loss(monkeypatch) -> None:
    monkeypatch.setitem(lgbm.BASE_PARAMS, "num_threads", 1)
    fitted, checked, spec = small_problem(4)
    trained = lgbm.fit(fitted, checked, spec, {"num_leaves": 15, "min_child_samples": 20,
                                                "lambda_l2": 0.0})
    losses = [metrics.log_loss(checked.y, expit(lgbm.raw_score(trained.booster, checked.x, k)))
              for k in range(1, trained.best_iteration + 1)]
    assert int(np.argmin(losses)) + 1 == trained.best_iteration
    result = lgbm.fold_result(trained, fitted, checked)
    assert result["gap"] == pytest.approx(result["log_loss"] / result["train_log_loss"])


def test_learning_curve_ends_at_the_final_model() -> None:
    fitted, checked, spec = small_problem(5)
    trained = lgbm.fit(fitted, checked, spec, {"num_leaves": 7, "min_child_samples": 50,
                                                "lambda_l2": 1.0})
    curve = lgbm.learning_curve(trained.booster, {"train": fitted, "selection": checked})
    assert [point["trees"] for point in curve] == sorted({point["trees"] for point in curve})
    assert curve[-1]["trees"] == trained.booster.num_trees()
    assert {"train_log_loss", "selection_log_loss", "train_average_precision",
            "selection_average_precision"} <= set(curve[-1])
