"""Leakage tests for pipeline/features.py (decision D8).

Every behaviour signal must come only from the card's transactions strictly before the current
instant. The tests check that in four independent ways:

1. against a row-by-row reference (tests/reference_history.py) on hand-made edge cases and on
   random data, with a fixed seed;
2. by changing the future, the current transaction or the labels and checking what must not move;
3. by introducing leaks on purpose and checking that the reference catches every one of them;
4. by hand-computed values at the window limits and for transactions in the same second.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from reference_history import make_base, mismatches, random_base

from pipeline import features

SEED = 20_261_115
HISTORY_SIGNALS = [
    name for name in features.FEATURE_COLUMNS
    if name not in ("amount", "category", "hour", "day_of_week", "is_late_night", "is_weekend",
                    "distance_km", "amount_to_history_mean", "amount_minus_history_mean",
                    "amount_log_zscore")
]


@pytest.fixture
def edge_base() -> pd.DataFrame:
    """One card with exact window limits and a tie in the same second; one card used once."""
    return make_base(
        [
            ("A", "2020-01-01 10:00:00", "grocery_pos", 10.0),  # 0: first purchase of A
            ("A", "2020-01-01 11:00:00", "grocery_pos", 20.0),  # 1: exactly 1 h after row 0
            ("A", "2020-01-01 11:00:01", "shopping_net", 30.0),  # 2: 1 h and 1 s after row 0
            ("A", "2020-01-01 12:00:00", "shopping_net", 40.0),  # 3: same second as row 4
            ("A", "2020-01-01 12:00:00", "travel", 50.0),  # 4: same second as row 3
            ("A", "2020-01-01 12:00:01", "travel", 60.0),  # 5: sees rows 3 and 4
            ("B", "2020-01-01 11:30:00", "grocery_pos", 99.0),  # 6: only purchase of B
        ]
    )


def test_matches_the_reference_on_edge_cases(edge_base: pd.DataFrame) -> None:
    built = features.build_features(edge_base)
    assert mismatches(edge_base, built, range(len(edge_base))) == []


def test_matches_the_reference_on_random_data() -> None:
    base = random_base()
    built = features.build_features(base)
    sample = np.random.default_rng(SEED).choice(len(base), size=300, replace=False)
    assert mismatches(base, built, sample) == []


def test_window_includes_its_lower_limit_and_excludes_the_present(edge_base: pd.DataFrame) -> None:
    built = features.build_features(edge_base)
    assert built.loc[1, "tx_count_1h"] == 1  # row 0 is exactly 1 h before: included
    assert built.loc[2, "tx_count_1h"] == 1  # row 0 is 1 h 1 s before: out; row 1 is in
    assert built.loc[2, "amount_sum_1h"] == 20.0
    assert built.loc[0, "tx_count_1h"] == 0  # the transaction itself never counts


def test_transactions_in_the_same_second_do_not_see_each_other(edge_base: pd.DataFrame) -> None:
    built = features.build_features(edge_base)
    for row in (3, 4):
        assert built.loc[row, "tx_count_24h"] == 3  # rows 0, 1 and 2, not the other tie
        assert built.loc[row, "seconds_since_previous"] == 3_599  # since row 2, not 0
    assert built.loc[4, "is_new_category"]  # row 3 is shopping_net, row 4 is travel
    assert built.loc[5, "tx_count_1h"] == 3  # rows 2, 3 and 4
    assert not built.loc[5, "is_new_category"]  # travel was used one second earlier

    swapped = edge_base.copy()
    swapped.loc[[3, 4], "source_row"] = [4, 3]
    swapped = swapped.iloc[[0, 1, 2, 4, 3, 5, 6]].reset_index(drop=True)
    rebuilt = features.build_features(swapped)
    assert rebuilt.loc[3, HISTORY_SIGNALS].equals(built.loc[4, HISTORY_SIGNALS])


def test_cards_without_history_get_explicit_values(edge_base: pd.DataFrame) -> None:
    built = features.build_features(edge_base)
    for row in (0, 6):
        assert not built.loc[row, "has_history"]
        assert built.loc[row, "is_new_category"]
        assert built.loc[row, "tx_count_30d"] == 0
        assert built.loc[row, "amount_sum_30d"] == 0.0
        for name in ("amount_mean_30d", "history_mean_amount", "amount_to_history_mean",
                     "seconds_since_previous", "history_share_same_hour"):
            assert math.isnan(built.loc[row, name])


def test_future_transactions_do_not_change_past_signals() -> None:
    base = random_base()
    cut = base["timestamp"].quantile(0.5)
    past = base["timestamp"] < cut
    future = base.copy()
    later = ~past
    future.loc[later, "amount"] = future.loc[later, "amount"] * 37 + 1
    future.loc[later, "category"] = "cat0"
    extra = future[later].copy()
    extra["transaction_id"] = extra["transaction_id"] + "x"
    future = pd.concat([future, extra], ignore_index=True)

    before = features.build_features(base)[past]
    after = features.build_features(future).iloc[: len(base)][past]
    pd.testing.assert_frame_equal(before, after)


def test_the_current_transaction_is_not_part_of_its_own_history() -> None:
    base = random_base()
    target = int(base["timestamp"].sort_values().index[len(base) // 2])
    changed = base.copy()
    changed.loc[target, "amount"] = 99_999.0
    original = features.build_features(base)
    altered = features.build_features(changed)
    # The history of the changed transaction does not move...
    assert original.loc[target, HISTORY_SIGNALS].equals(altered.loc[target, HISTORY_SIGNALS])
    # ...and neither does anything about the transactions before it.
    earlier = base["timestamp"] < base.loc[target, "timestamp"]
    pd.testing.assert_frame_equal(original[earlier], altered[earlier])


def test_labels_are_never_read() -> None:
    base = random_base()
    shuffled = base.assign(is_fraud=np.random.default_rng(SEED).permutation(base["is_fraud"] + 1))
    pd.testing.assert_frame_equal(features.build_features(base),
                                  features.build_features(shuffled))
    pd.testing.assert_frame_equal(features.build_features(base),
                                  features.build_features(base.drop(columns="is_fraud")))


LEAKS = {
    "the transaction sees itself and its ties": lambda key, window: (
        _start(key, window), np.searchsorted(key, key, side="right")
    ),
    "same-second ties are visible in file order": lambda key, window: (
        _start(key, window), np.arange(len(key))
    ),
    "the whole future of the card is visible": lambda key, window: (
        _start(key, window),
        np.searchsorted(key, key - key % features.GROUP_STRIDE + features.GROUP_STRIDE, "left"),
    ),
    "the window drops its lower limit": lambda key, window: (
        _start(key, window, side="right"), np.searchsorted(key, key, side="left")
    ),
}


def _start(key: np.ndarray, window: int | None, side: str = "left") -> np.ndarray:
    if window is None:
        return np.searchsorted(key, key - key % features.GROUP_STRIDE, side="left")
    return np.searchsorted(key, key - window, side=side)


@pytest.mark.parametrize("leak", list(LEAKS))
def test_leaks_introduced_on_purpose_are_caught(
    leak: str, edge_base: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    bounds = LEAKS[leak]
    monkeypatch.setattr(features._SortedHistory, "earlier",
                        lambda self, window=None: bounds(self.key, window))
    base = pd.concat([edge_base, random_base()], ignore_index=True)
    built = features.build_features(base)
    sample = np.random.default_rng(SEED).choice(len(base), size=200, replace=False)
    assert mismatches(base, built, [*range(len(edge_base)), *sample]), f"missed: {leak}"


def test_every_feature_is_documented() -> None:
    document = (Path(__file__).parents[1] / "docs" / "dados-processados.md").read_text(
        encoding="utf-8")
    missing = [name for name in features.FEATURE_COLUMNS if f"`{name}`" not in document]
    assert missing == []


def test_distance_is_missing_for_invalid_coordinates() -> None:
    distance = features.haversine_km([0, 95, np.nan, 10], [0, 0, 0, 200], [0, 0, 0, 0],
                                     [1, 0, 0, 0])
    assert math.isclose(distance[0], 111.19, abs_tol=0.01)  # 1 degree of longitude at the equator
    assert np.isnan(distance[1:]).all()
