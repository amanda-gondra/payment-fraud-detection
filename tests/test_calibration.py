"""Tests for pipeline/modeling/calibration.py."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pipeline.modeling import calibration


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    score = rng.normal(-3, 2, 4_000)
    y = (rng.random(4_000) < expit(0.7 * score + 0.4)).astype(int)
    return score, y


def test_platt_matches_an_unregularised_logistic_regression(scored) -> None:
    score, y = scored
    mine = calibration.Platt.fit(score, y)
    reference = LogisticRegression(C=1e12, max_iter=10_000).fit(score.reshape(-1, 1), y)
    assert mine.a == pytest.approx(reference.coef_[0, 0], abs=1e-3)
    assert mine.b == pytest.approx(reference.intercept_[0], abs=1e-3)


def test_isotonic_from_json_matches_scikit_learn(scored) -> None:
    score, y = scored
    reference = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(score, y)
    stored = calibration.from_dict(calibration.Isotonic.fit(score, y).to_dict())
    probe = np.r_[score, score.min() - 10, score.max() + 10,
                  np.linspace(score.min(), score.max(), 2_001)]
    assert np.allclose(stored.apply(probe), reference.predict(probe))


@pytest.mark.parametrize("method", calibration.METHODS)
def test_every_method_survives_json_and_keeps_order(scored, method: str) -> None:
    score, y = scored
    fitted = calibration.fit(method, score, y)
    again = calibration.from_dict(fitted.to_dict())
    probe = np.sort(np.r_[score, -50.0, 50.0])
    assert np.array_equal(again.apply(probe), fitted.apply(probe))
    assert np.all(np.diff(again.apply(probe)) >= 0)
    assert again.apply(probe).min() >= 0 and again.apply(probe).max() <= 1


def test_none_is_the_model_sigmoid() -> None:
    score = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(calibration.Identity().apply(score), expit(score))


@pytest.mark.parametrize("data", [
    {"method": "isotonic", "x": [1.0, 0.5], "y": [0.1, 0.2]},
    {"method": "isotonic", "x": [0.0, 1.0], "y": [0.5, 0.1]},
    {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.5]},
    {"method": "pickle"},
])
def test_invalid_calibrators_are_refused(data: dict) -> None:
    with pytest.raises(ValueError):
        calibration.from_dict(data)


def test_time_blocks_are_consecutive_and_even() -> None:
    blocks = calibration.time_blocks(103, 5)
    assert np.all(np.diff(blocks) >= 0)
    assert set(np.bincount(blocks)) <= {20, 21}


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_a_block_never_sees_its_own_labels(scored, method: str) -> None:
    score, y = scored
    blocks = calibration.time_blocks(len(score))
    first = calibration.cross_fit(method, score, y)
    flipped = y.copy()
    flipped[blocks == 2] = 1 - flipped[blocks == 2]
    second = calibration.cross_fit(method, score, flipped)
    assert np.array_equal(first[blocks == 2], second[blocks == 2])
    assert not np.array_equal(first[blocks == 1], second[blocks == 1])
