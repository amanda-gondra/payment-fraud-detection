"""Behaviour signals of each transaction, built only from what was known before it.

The single rule of this module: the history of a transaction made at instant t is made of the
earlier transactions of the same card whose timestamp is *strictly* before t (decision D8).

- The transaction itself is never part of its own history.
- Transactions of the same card in the same second do not see each other: with one-second
  timestamps there is no way to know which came first, and the file order is not time order.
- A window of length w covers [t - w, t): the lower bound is included, the instant t is not.
- The fraud label is never an input: build_features receives only the columns listed in
  INPUT_COLUMNS.

Every history signal comes from one place, _SortedHistory.earlier(), so the rule above is
written once. tests/test_features.py checks it against an independent row-by-row computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

INPUT_COLUMNS: Final = ("timestamp", "card_key", "category", "amount", "distance_km")
WINDOWS: Final = {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400, "30d": 30 * 86_400}
# 84.8% of the frauds of the training file happen between 22h and 3h59 (exploratory analysis).
LATE_NIGHT_HOURS: Final = (22, 23, 0, 1, 2, 3)
WEEKEND_DAYS: Final = (5, 6)  # Saturday and Sunday (Monday is 0)
# Below this many earlier transactions, a standard deviation of the card's amounts is too noisy.
MIN_HISTORY_FOR_DEVIATION: Final = 10
# A variance this small means identical amounts; dividing by it would only amplify rounding.
MIN_VARIANCE: Final = 1e-12
# Composite sort key: group * GROUP_STRIDE + seconds. The stride exceeds any Unix timestamp plus
# the longest window, so a window never reaches into the previous group.
GROUP_STRIDE: Final = 10**10
EARTH_RADIUS_KM: Final = 6371.0088

WINDOW_FEATURES: Final = tuple(
    f"{kind}_{name}" for name in WINDOWS for kind in ("tx_count", "amount_sum", "amount_mean")
)
FEATURE_COLUMNS: Final = (
    "amount",
    "category",
    "hour",
    "day_of_week",
    "is_late_night",
    "is_weekend",
    "distance_km",
    *WINDOW_FEATURES,
    "has_history",
    "history_mean_amount",
    "amount_to_history_mean",
    "amount_minus_history_mean",
    "amount_log_zscore",
    "seconds_since_previous",
    "is_new_category",
    "history_share_same_hour",
    "history_share_late_night",
)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km; NaN when a coordinate is missing or out of range."""
    lat1, lon1, lat2, lon2 = (
        np.asarray(values, dtype=float) for values in (lat1, lon1, lat2, lon2)
    )
    valid = (
        (np.abs(lat1) <= 90) & (np.abs(lat2) <= 90) & (np.abs(lon1) <= 180) & (np.abs(lon2) <= 180)
    )  # NaN fails every comparison, so missing coordinates are invalid too
    lat1, lon1, lat2, lon2 = (np.radians(values) for values in (lat1, lon1, lat2, lon2))
    half_chord = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(half_chord, 0, 1)))
    return np.where(valid, distance, np.nan)


def epoch_seconds(timestamps: pd.Series) -> np.ndarray:
    """Whole seconds since 1970-01-01, as int64."""
    return ((timestamps - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy(np.int64)


@dataclass(frozen=True)
class _SortedHistory:
    """Rows sorted by (group, second), to find each row's earlier rows with binary search."""

    order: np.ndarray  # original positions, in (group, second) order
    key: np.ndarray  # sorted composite keys

    @classmethod
    def of(cls, groups: np.ndarray, seconds: np.ndarray) -> _SortedHistory:
        if groups.min() < 0:
            raise ValueError("missing card or category: every row needs a group")
        if seconds.min() < 0 or seconds.max() + max(WINDOWS.values()) >= GROUP_STRIDE:
            raise ValueError("timestamps outside the range the composite key supports")
        key = groups.astype(np.int64) * GROUP_STRIDE + seconds
        order = np.argsort(key, kind="stable")
        return cls(order=order, key=key[order])

    def earlier(self, window: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Sorted positions [start, end) of the same-group rows strictly before each row.

        With a window, only rows at or after t - window. Rows in the same second as the current
        one are excluded, and so is the current row itself.
        """
        end = np.searchsorted(self.key, self.key, side="left")
        if window is None:
            group_start = self.key - self.key % GROUP_STRIDE
            start = np.searchsorted(self.key, group_start, side="left")
        else:
            start = np.searchsorted(self.key, self.key - window, side="left")
        return start, end

    def to_rows(self, sorted_values: np.ndarray) -> np.ndarray:
        """Values computed in sorted order, back in the original row order."""
        values = np.empty_like(sorted_values)
        values[self.order] = sorted_values
        return values


def _running(values: np.ndarray) -> np.ndarray:
    """Prefix sums with a leading zero: total of positions [a, b) is result[b] - result[a]."""
    return np.concatenate([[0.0], np.cumsum(values, dtype=float)])


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division; NaN where the denominator is zero (no history)."""
    result = np.full(len(numerator), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _count_earlier(groups: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    """For each row, in the original order: rows of the same group strictly before it."""
    history = _SortedHistory.of(groups, seconds)
    start, end = history.earlier()
    return history.to_rows(end - start)


def calendar_features(timestamps: pd.Series) -> pd.DataFrame:
    """Hour of day, day of week and the late-night and weekend flags."""
    hour = timestamps.dt.hour.astype("int8")
    day_of_week = timestamps.dt.day_of_week.astype("int8")
    return pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "is_late_night": hour.isin(LATE_NIGHT_HOURS),
            "is_weekend": day_of_week.isin(WEEKEND_DAYS),
        },
        index=timestamps.index,
    )


def card_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Signals from the earlier transactions of the same card (see the module docstring)."""
    cards = pd.factorize(frame["card_key"])[0]
    seconds = epoch_seconds(frame["timestamp"])
    hours = frame["timestamp"].dt.hour.to_numpy()
    history = _SortedHistory.of(cards, seconds)
    # Everything below, up to the conversion back to row order, is in (card, second) order.
    amount = frame["amount"].to_numpy(float)[history.order]
    when = seconds[history.order]
    log_amount = np.log1p(amount)
    late_night = np.isin(hours[history.order], LATE_NIGHT_HOURS).astype(float)
    signals: dict[str, np.ndarray] = {}

    amount_total = _running(amount)

    for name, window in WINDOWS.items():
        start, end = history.earlier(window)
        count = end - start
        total = amount_total[end] - amount_total[start]
        signals[f"tx_count_{name}"] = count.astype(np.int32)
        signals[f"amount_sum_{name}"] = total
        signals[f"amount_mean_{name}"] = _ratio(total, count)

    start, end = history.earlier()
    count = end - start
    mean = _ratio(amount_total[end] - amount_total[start], count)
    sum_log = _running(log_amount)[end] - _running(log_amount)[start]
    sum_log_sq = _running(log_amount**2)[end] - _running(log_amount**2)[start]
    log_mean = _ratio(sum_log, count)
    log_variance = _ratio(sum_log_sq - sum_log * log_mean, count - 1)  # sample variance
    deviation_ok = (count >= MIN_HISTORY_FOR_DEVIATION) & (log_variance > MIN_VARIANCE)
    log_std = np.sqrt(np.where(deviation_ok, log_variance, np.nan))
    previous = np.maximum(end - 1, 0)  # last earlier row; only read where count > 0

    signals["has_history"] = count > 0
    signals["history_mean_amount"] = mean
    signals["amount_to_history_mean"] = _ratio(amount, mean)
    signals["amount_minus_history_mean"] = amount - mean
    signals["amount_log_zscore"] = (log_amount - log_mean) / log_std
    signals["seconds_since_previous"] = np.where(count > 0, when - when[previous], np.nan)
    signals["history_share_late_night"] = _ratio(
        _running(late_night)[end] - _running(late_night)[start], count
    )

    in_rows = {name: history.to_rows(values) for name, values in signals.items()}
    same_hour_before = _count_earlier(cards * 24 + hours, seconds)
    in_rows["history_share_same_hour"] = _ratio(same_hour_before, history.to_rows(count))
    return pd.DataFrame(in_rows, index=frame.index)


def category_history(frame: pd.DataFrame) -> pd.Series:
    """True when the card made no earlier purchase in the transaction's category."""
    cards = pd.factorize(frame["card_key"])[0]
    categories, names = pd.factorize(frame["category"])
    before = _count_earlier(cards * len(names) + categories, epoch_seconds(frame["timestamp"]))
    return pd.Series(before == 0, index=frame.index, name="is_new_category")


def build_features(base: pd.DataFrame) -> pd.DataFrame:
    """All model features, in FEATURE_COLUMNS order. The fraud label is never read."""
    inputs = base[list(INPUT_COLUMNS)]
    features = pd.concat(
        [
            inputs[["amount", "category"]],
            calendar_features(inputs["timestamp"]),
            inputs[["distance_km"]],
            card_history(inputs),
            category_history(inputs),
        ],
        axis=1,
    )
    return features[list(FEATURE_COLUMNS)]
