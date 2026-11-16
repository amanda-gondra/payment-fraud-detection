"""Tests for the model matrix (data.py), the metrics, the baselines and the group results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from pipeline import build, download, features
from pipeline.modeling import baselines, data, evaluation, metrics, report_selection
from pipeline.modeling.data import FeatureSpec, Split
from pipeline.modeling.decision import APPROVE, BLOCK, REVIEW, Costs
from pipeline.modeling.evaluation import Scored


def signal_frame(n: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.random(n) * (i + 1) for i, name in
                          enumerate(features.FEATURE_COLUMNS)})
    for name in baselines.FLAG_FEATURES:
        frame[name] = rng.random(n) > 0.5
    frame["category"] = pd.Categorical(rng.choice(["misc_net", "grocery_pos"], n))
    frame["hour"] = rng.integers(0, 24, n).astype("int8")
    frame["day_of_week"] = rng.integers(0, 7, n).astype("int8")
    return frame


# --- model matrix ---------------------------------------------------------------------------

def test_the_matrix_follows_the_saved_order_and_encoding() -> None:
    frame = signal_frame()
    spec = FeatureSpec.from_train(frame)
    matrix = spec.matrix(frame)
    assert spec.categories == ("grocery_pos", "misc_net")
    for position, name in enumerate(spec.columns):
        if name == "category":
            expected = frame[name].astype(str).map({"grocery_pos": 0, "misc_net": 1})
        else:
            expected = frame[name].astype(float)
        assert np.array_equal(matrix[:, position], expected.to_numpy(dtype=float))


def test_the_frame_column_order_and_extra_columns_do_not_matter() -> None:
    frame = signal_frame()
    spec = FeatureSpec.from_train(frame)
    extra = frame.assign(gender="F", age_years=40, card_key="abc", is_fraud=1,
                         transaction_id="t")
    shuffled = extra[list(reversed(extra.columns))]
    assert np.array_equal(spec.matrix(shuffled), spec.matrix(frame))
    assert not {"gender", "age_years", "card_key", "is_fraud"} & set(spec.columns)


def test_missing_values_stay_missing_and_problems_are_refused() -> None:
    frame = signal_frame()
    spec = FeatureSpec.from_train(frame)
    frame.loc[2, "amount_mean_1h"] = np.nan
    assert np.isnan(spec.matrix(frame)[2, spec.columns.index("amount_mean_1h")])
    with pytest.raises(ValueError, match="missing features"):
        spec.matrix(frame.drop(columns="amount"))
    unknown = frame.assign(category=["travel"] * len(frame))
    with pytest.raises(ValueError, match="unknown category"):
        spec.matrix(unknown)


def test_the_spec_survives_json() -> None:
    spec = FeatureSpec.from_train(signal_frame())
    assert FeatureSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_processed_files_are_checked_before_reading(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "data" / "processed" / "train.parquet"
    path.parent.mkdir(parents=True)
    signal_frame().to_parquet(path)
    manifest = tmp_path / "data" / "processed" / "manifest.json"
    record = {"path": "data/processed/train.parquet", "sha256": download.sha256(path)}
    manifest.write_text(json.dumps({"outputs": {"train": record}}), encoding="utf-8")
    monkeypatch.setattr(download, "ROOT", tmp_path)
    monkeypatch.setattr(build, "MANIFEST", manifest)
    assert len(data.read_verified("train")) == 6
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="differs"):
        data.read_verified("train")


# --- metrics ---------------------------------------------------------------------------------

def test_classification_counts_by_hand() -> None:
    result = metrics.classification(np.array([1, 1, 0, 0, 1]), np.array([1, 0, 1, 0, 1]))
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 1)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert np.isnan(metrics.classification(np.array([1, 0]), np.array([0, 0]))["precision"])


def test_recall_at_a_fixed_precision_by_hand() -> None:
    y = np.array([1, 1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    result = metrics.recall_at_precision(y, score, target=0.9)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["precision"] == 1.0
    assert metrics.recall_at_precision(y, score, target=0.7)["recall"] == pytest.approx(1.0)


def test_wilson_interval() -> None:
    low, high = metrics.wilson(0, 10)
    assert low == 0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = metrics.wilson(5, 10)
    assert low + high == pytest.approx(1.0)


def test_log_loss_matches_scikit_learn_and_reliability_bands() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    assert metrics.log_loss(y, p) == pytest.approx(sk_log_loss(y, p))
    table = metrics.reliability(np.array([0, 1, 1]), np.array([0.0005, 0.95, 1.0]))
    assert table["band"].tolist() == ["0-0.001", "0.9-1"]
    assert table["transactions"].tolist() == [1, 2]


# --- baselines --------------------------------------------------------------------------------

def test_the_amount_rule_takes_the_cut_with_the_best_f1() -> None:
    rng = np.random.default_rng(3)
    amount = np.round(rng.uniform(1, 1000, 400), 0)
    y = (rng.random(400) < amount / 2500).astype(int)
    rule = baselines.AmountRule.fit(amount, y)
    best = max(np.unique(amount), key=lambda cut: metrics.classification(y, amount >= cut)["f1"])
    assert metrics.classification(y, rule.flag(amount))["f1"] == pytest.approx(
        metrics.classification(y, amount >= best)["f1"])


def test_every_signal_gets_exactly_one_logistic_treatment() -> None:
    groups = (baselines.LOG_FEATURES, baselines.SIGNED_LOG_FEATURES, baselines.LINEAR_FEATURES,
              baselines.FLAG_FEATURES, baselines.ONE_HOT_FEATURES)
    names = [name for group in groups for name in group]
    assert sorted(names) == sorted(features.FEATURE_COLUMNS)


def test_the_logistic_model_copes_with_missing_values_and_new_categories() -> None:
    frame = signal_frame(80, seed=4)
    frame.loc[::3, "amount_mean_1h"] = np.nan
    y = (frame["amount"] > frame["amount"].median()).astype(int)
    model = baselines.logistic_model(1.0, None).fit(baselines.logistic_inputs(frame), y)
    later = frame.assign(category="travel")
    assert np.isfinite(baselines.logistic_raw_score(model, later)).all()


def test_signed_log_keeps_the_sign() -> None:
    values = baselines.signed_log1p(np.array([-(np.e - 1), 0.0, np.e - 1]))
    assert np.allclose(values, [-1.0, 0.0, 1.0])


def unlisted_function(values: np.ndarray) -> np.ndarray:
    return values


def test_skops_refuses_types_off_the_trusted_list() -> None:
    model = make_pipeline(FunctionTransformer(unlisted_function)).fit(np.ones((3, 1)))
    with pytest.raises(ValueError, match="untrusted types"):
        baselines.load_logistic(baselines.dump_logistic(model))


# --- report wording -------------------------------------------------------------------------------

def test_the_report_never_calls_the_model_probability_calibrated_when_it_is_not() -> None:
    own = report_selection.chosen_text("none")
    assert "sem calibrador" in own and "ajustada" not in own
    assert "ajustada" in report_selection.chosen_text("platt")
    assert "sem calibrador" in report_selection.METHOD_PHRASES["none"]


# --- results by group ---------------------------------------------------------------------------

def test_results_by_group_flag_small_samples() -> None:
    groups = pd.Series(["a"] * 4 + ["b"] * 3)
    y = np.array([1, 0, 0, 1, 0, 0, 0])
    score = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.2, 0.1])
    decided = np.array([BLOCK, APPROVE, REVIEW, REVIEW, APPROVE, APPROVE, APPROVE])
    split = Split(pd.DataFrame(index=range(7)), np.zeros((7, 1)), y, np.full(7, 50.0))
    table = evaluation.by_group(groups, split, Scored(score, None, decided), Costs())
    a, b = table.set_index("group").loc["a"], table.set_index("group").loc["b"]
    assert (a["frauds"], a["frauds_flagged"], a["false_positives"]) == (2, 2, 1)
    assert a["average_precision"] == pytest.approx(1.0)
    assert np.isnan(b["average_precision"]) and b["frauds"] == 0
    assert not a["reliable"]
