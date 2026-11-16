"""Tests for pipeline/modeling/decision.py: costs, the three bands and the threshold search."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.modeling import decision
from pipeline.modeling.decision import APPROVE, BLOCK, REVIEW, Costs, Thresholds


@pytest.mark.parametrize("review, block", [(0.6, 0.5), (-0.1, 0.5), (0.2, 1.1)])
def test_thresholds_must_be_ordered_and_inside_zero_one(review: float, block: float) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        Thresholds(review, block)


def test_a_probability_on_a_threshold_goes_up_a_band() -> None:
    decided = Thresholds(0.2, 0.7).decide(np.array([0.0, 0.19999, 0.2, 0.5, 0.7, 1.0]))
    assert decided.tolist() == [APPROVE, APPROVE, REVIEW, REVIEW, BLOCK, BLOCK]


def test_equal_thresholds_leave_no_review_band() -> None:
    assert Thresholds(0.5, 0.5).decide(np.array([0.49, 0.5])).tolist() == [APPROVE, BLOCK]


def test_a_missing_probability_is_refused() -> None:
    with pytest.raises(ValueError, match="NaN"):
        Thresholds(0.2, 0.7).decide(np.array([0.1, np.nan]))


def test_cost_of_each_decision_by_hand() -> None:
    costs = Costs(review=5, review_friction=2, block_legit=25, review_detection=0.9)
    table = costs.per_decision(np.array([100.0, 100.0]), np.array([1, 0]))
    np.testing.assert_allclose(table, [[100, 15, 0], [0, 7, 25]])


def random_case(seed: int, n: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    probability = np.clip(np.round(0.6 * y + rng.normal(0.2, 0.2, n), 2), 0, 0.99)
    amount = rng.uniform(1, 400, n)
    return probability, amount, y


def test_the_summary_adds_up_to_the_same_cost_as_the_table() -> None:
    probability, amount, y = random_case(1)
    decided = Thresholds(0.3, 0.6).decide(probability)
    summary = decision.summarize(decided, y, amount)
    expected = decision.BASE_COSTS.per_decision(amount, y)[np.arange(len(y)), decided].sum()
    assert summary["total_cost"] == pytest.approx(expected)
    assert summary["approve_transactions"] + summary["review_transactions"] + summary[
        "block_transactions"] == len(y)


def brute_force(probability: np.ndarray, amount: np.ndarray, y: np.ndarray,
                costs: Costs) -> float:
    """Cheapest pair over every possible cut, without the prefix-sum shortcut."""
    table = costs.per_decision(amount, y)
    cuts = np.r_[np.unique(probability), np.inf]
    best = np.inf
    for review in cuts:
        for block in cuts[cuts >= review]:
            decided = np.where(probability >= block, BLOCK,
                               np.where(probability >= review, REVIEW, APPROVE))
            if costs.max_review_share is not None and np.mean(
                    decided == REVIEW) > costs.max_review_share:
                continue
            best = min(best, table[np.arange(len(y)), decided].sum())
    return float(best)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("share", [None, 0.05])
def test_the_search_finds_the_cheapest_pair(seed: int, share: float | None) -> None:
    probability, amount, y = random_case(seed)
    costs = Costs(max_review_share=share)
    result = decision.search(probability, amount, y, costs)
    decided = result.thresholds.decide(probability)
    realised = decision.summarize(decided, y, amount, costs)["total_cost"]
    assert result.total_cost == pytest.approx(realised)
    assert result.total_cost == pytest.approx(brute_force(probability, amount, y, costs))
    if share is not None:
        assert np.mean(decided == REVIEW) <= share


def test_thinned_cuts_stay_exact_and_dense_among_the_highest(monkeypatch) -> None:
    monkeypatch.setattr(decision, "MAX_CANDIDATES", 60)
    ranked = np.sort(np.random.default_rng(4).random(5_000))[::-1]
    counts, values = decision._cuts(ranked)
    assert len(counts) <= 70
    assert counts[:4].tolist() == [0, 1, 2, 3]
    assert counts[-1] == len(ranked)
    for count, value in zip(counts, values, strict=True):
        assert int((ranked >= value).sum()) == count
