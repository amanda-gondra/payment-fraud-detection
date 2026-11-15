"""Independent reference for the behaviour signals, one transaction at a time.

pipeline.features computes every signal at once with sorted keys and binary search. This module
does it the slow, obvious way: for one transaction, filter the earlier rows of its card with a
plain comparison and compute each signal from them. It shares only the definitions (window
lengths, late-night hours, minimum history), never the logic, so a mistake in the fast version
shows up as a difference here.

Also holds the helpers that build small synthetic bases for the tests.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipeline.features import (
    FEATURE_COLUMNS,
    LATE_NIGHT_HOURS,
    MIN_HISTORY_FOR_DEVIATION,
    MIN_VARIANCE,
    WEEKEND_DAYS,
    WINDOWS,
)

FLOAT_TOLERANCE = {"rtol": 1e-6, "atol": 1e-6}


def make_base(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """A prepared base from (card, 'YYYY-MM-DD HH:MM:SS', category, amount) tuples."""
    cards, moments, categories, amounts = zip(*rows, strict=True)
    count = len(rows)
    return pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(count)],
            "source_row": np.arange(count, dtype=np.int64),
            "timestamp": pd.to_datetime(list(moments)).astype("datetime64[ms]"),
            "card_key": pd.Categorical(cards),
            "category": pd.Categorical(categories),
            "amount": np.asarray(amounts, dtype=float),
            "distance_km": np.full(count, 10.0),
            "is_fraud": np.zeros(count, dtype="int8"),
        }
    )


def random_base(seed: int = 7, cards: int = 6, per_card: int = 250) -> pd.DataFrame:
    """Random cards over 60 days, with repeated seconds and exact window boundaries on purpose."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2019-03-01")
    rows = []
    for card in range(cards):
        offsets = np.sort(rng.integers(0, 60 * 86_400, per_card))
        offsets[10:14] = offsets[10]  # four transactions in the same second
        offsets[40] = offsets[39] + 3_600  # exactly one hour apart
        offsets[41] = offsets[40] + 86_400  # exactly one day apart
        offsets = np.sort(offsets)
        for offset in offsets:
            rows.append(
                (
                    f"card{card}",
                    str(start + pd.Timedelta(seconds=int(offset))),
                    f"cat{rng.integers(0, 5)}",
                    float(np.round(rng.lognormal(3.5, 1.2), 2)),
                )
            )
    order = rng.permutation(len(rows))  # the file is not in time order
    return make_base([rows[i] for i in order])


def reference_signals(base: pd.DataFrame, position: int) -> dict[str, object]:
    """Every signal of the transaction at this position, from its card's strictly earlier rows."""
    row = base.iloc[position]
    moment = row["timestamp"]
    card_rows = base[base["card_key"] == row["card_key"]]
    history = card_rows[card_rows["timestamp"] < moment]
    count = len(history)
    signals: dict[str, object] = {
        "amount": row["amount"],
        "category": row["category"],
        "hour": moment.hour,
        "day_of_week": moment.day_of_week,
        "is_late_night": moment.hour in LATE_NIGHT_HOURS,
        "is_weekend": moment.day_of_week in WEEKEND_DAYS,
        "distance_km": row["distance_km"],
    }
    for name, seconds in WINDOWS.items():
        window = history[history["timestamp"] >= moment - pd.Timedelta(seconds=seconds)]
        signals[f"tx_count_{name}"] = len(window)
        signals[f"amount_sum_{name}"] = float(window["amount"].sum())
        signals[f"amount_mean_{name}"] = float(window["amount"].mean()) if len(window) else math.nan

    mean = float(history["amount"].mean()) if count else math.nan
    logs = np.log1p(history["amount"].to_numpy())
    variance = float(np.var(logs, ddof=1)) if count >= 2 else math.nan
    deviation_ok = count >= MIN_HISTORY_FOR_DEVIATION and variance > MIN_VARIANCE
    hours = history["timestamp"].dt.hour
    signals.update(
        {
            "has_history": count > 0,
            "history_mean_amount": mean,
            "amount_to_history_mean": row["amount"] / mean if count else math.nan,
            "amount_minus_history_mean": row["amount"] - mean if count else math.nan,
            "amount_log_zscore": (
                (math.log1p(row["amount"]) - logs.mean()) / math.sqrt(variance)
                if deviation_ok
                else math.nan
            ),
            "seconds_since_previous": (
                (moment - history["timestamp"].max()).total_seconds() if count else math.nan
            ),
            "is_new_category": not history["category"].eq(row["category"]).any(),
            "history_share_same_hour": float(hours.eq(moment.hour).mean()) if count else math.nan,
            "history_share_late_night": (
                float(hours.isin(LATE_NIGHT_HOURS).mean()) if count else math.nan
            ),
        }
    )
    return signals


def mismatches(base: pd.DataFrame, stored: pd.DataFrame, positions) -> list[str]:
    """Signals where the stored features differ from the reference, as readable messages."""
    found = []
    for position in positions:
        expected = reference_signals(base, position)
        actual = stored.iloc[position]
        for column in FEATURE_COLUMNS:
            want, got = expected[column], actual[column]
            if isinstance(want, float) or isinstance(got, float):
                same = math.isclose(float(want), float(got), rel_tol=FLOAT_TOLERANCE["rtol"],
                                    abs_tol=FLOAT_TOLERANCE["atol"]) or (
                    math.isnan(float(want)) and math.isnan(float(got))
                )
            else:
                same = want == got
            if not same:
                found.append(f"row {position} {column}: expected {want!r}, got {got!r}")
    return found
