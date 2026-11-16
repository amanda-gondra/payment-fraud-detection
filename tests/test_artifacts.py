"""Tests for pipeline/modeling/artifacts.py: signed bundles, safe loading, inference."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from pipeline import features
from pipeline.modeling import artifacts, baselines, calibration
from pipeline.modeling.artifacts import FraudModel, IntegrityError
from pipeline.modeling.data import FeatureSpec
from pipeline.modeling.decision import Thresholds

CATEGORIES = ("grocery_pos", "misc_net", "shopping_net")


def signals(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.random(n) * 100 for name in features.FEATURE_COLUMNS})
    for name in ("is_late_night", "is_weekend", "has_history", "is_new_category"):
        frame[name] = rng.random(n) > 0.5
    frame["category"] = rng.choice(CATEGORIES, n)
    frame["hour"] = rng.integers(0, 24, n)
    frame["day_of_week"] = rng.integers(0, 7, n)
    frame.loc[::5, "amount_mean_1h"] = np.nan
    return frame


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, lgb.Booster, FeatureSpec, pd.DataFrame]:
    frame = signals(600)
    spec = FeatureSpec(columns=tuple(features.FEATURE_COLUMNS), categories=CATEGORIES)
    x = spec.matrix(frame)
    y = ((frame["amount"] > 70) & frame["is_late_night"]).astype(int).to_numpy()
    data = lgb.Dataset(x, label=y, feature_name=list(spec.columns),
                       categorical_feature=[spec.category_index])
    booster = lgb.train({"objective": "binary", "verbosity": -1, "num_leaves": 7, "seed": 1},
                        data, num_boost_round=15)
    metadata = {"model_version": "1.0.0", "created_at": "2026-11-15T00:00:00+00:00"}
    parts = artifacts.ModelParts(booster, spec, calibration.Platt(a=0.8, b=-0.3),
                                 Thresholds(0.2, 0.7), {"review": 5.0}, metadata)
    artifacts.save_model(parts, directory=tmp_path / "bundle")
    return tmp_path / "bundle", booster, spec, frame


def test_a_saved_bundle_loads_and_predicts_like_the_original(bundle) -> None:
    directory, booster, spec, frame = bundle
    model = FraudModel.load(directory)
    expected = calibration.Platt(a=0.8, b=-0.3).apply(
        booster.predict(spec.matrix(frame), raw_score=True))
    probability, decided = model.decide(frame)
    assert np.allclose(probability, expected)
    assert np.array_equal(decided, Thresholds(0.2, 0.7).decide(expected))
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [*artifacts.FILES, artifacts.MANIFEST_FILE])


def test_answers_are_readable_and_rounded(bundle) -> None:
    directory, _, _, frame = bundle
    answers = FraudModel.load(directory).assess(frame.head(5))
    for answer in answers:
        assert answer.label in {"Aprovar", "Revisão humana", "Bloquear"}
        assert answer.probability == round(answer.probability, 2)
        assert answer.reasons and all(isinstance(text, str) for text in answer.reasons)
        assert answer.model_version == "1.0.0"


def test_column_order_of_the_frame_does_not_change_the_answer(bundle) -> None:
    directory, _, _, frame = bundle
    model = FraudModel.load(directory)
    shuffled = frame[list(reversed(frame.columns))]
    assert np.array_equal(model.probability(frame), model.probability(shuffled))


@pytest.mark.parametrize("name", artifacts.FILES)
def test_a_changed_file_is_refused(bundle, name: str) -> None:
    directory = bundle[0]
    path = directory / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(IntegrityError, match=f"{name} changed"):
        FraudModel.load(directory)


def test_missing_or_unexpected_files_are_refused(bundle) -> None:
    directory = bundle[0]
    (directory / "extra.pkl").write_bytes(b"not allowed")
    with pytest.raises(IntegrityError, match="differ from the manifest"):
        FraudModel.load(directory)
    (directory / "extra.pkl").unlink()
    (directory / "calibrator.json").unlink()
    with pytest.raises(IntegrityError, match="differ from the manifest"):
        FraudModel.load(directory)


def test_an_edited_manifest_is_refused(bundle) -> None:
    directory = bundle[0]
    path = directory / artifacts.MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"]["decision.json"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IntegrityError, match="bundle_sha256"):
        FraudModel.load(directory)


def test_columns_that_differ_from_the_model_are_refused(bundle, tmp_path: Path) -> None:
    _, booster, spec, _ = bundle
    swapped = FeatureSpec(columns=(spec.columns[1], spec.columns[0], *spec.columns[2:]),
                          categories=spec.categories)
    parts = artifacts.ModelParts(booster, swapped, calibration.Identity(), Thresholds(0.2, 0.7),
                                 {}, {"model_version": "1.0.0", "created_at": "x"})
    artifacts.save_model(parts, directory=tmp_path / "swapped")
    with pytest.raises(IntegrityError, match="columns differ"):
        FraudModel.load(tmp_path / "swapped")


def test_saving_over_a_folder_with_stray_files_is_refused(tmp_path: Path) -> None:
    (tmp_path / "old.joblib").write_bytes(b"x")
    with pytest.raises(IntegrityError, match="unexpected files"):
        artifacts.write_signed(tmp_path, {"a.json": b"{}"}, {})


def test_baselines_round_trip(tmp_path: Path) -> None:
    frame = signals(400, seed=2)
    y = (frame["amount"] > 60).astype(int).to_numpy()
    model = baselines.logistic_model(1.0, None).fit(baselines.logistic_inputs(frame), y)
    saved = artifacts.Baselines(baselines.AmountRule(60.0), model, calibration.Platt(1.0, 0.0),
                                Thresholds(0.3, 0.8), {"C": 1.0, "class_weight": "none"})
    saved.save(tmp_path)
    loaded = artifacts.Baselines.load(tmp_path)
    assert loaded.rule == saved.rule and loaded.thresholds == saved.thresholds
    assert np.array_equal(baselines.logistic_raw_score(loaded.logistic, frame),
                          baselines.logistic_raw_score(model, frame))
