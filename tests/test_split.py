"""Tests for pipeline/split.py: the cut instant, overlap checks and the summary."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline import split


def moments(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values))).astype("datetime64[ms]")


def test_the_cut_instant_opens_the_validation_set() -> None:
    timestamps = moments("2019-12-31 23:59:59", "2020-01-01 00:00:00", "2020-01-01 00:00:01")
    assert list(split.assign(timestamps)) == ["train", "validation", "validation"]


def test_check_accepts_a_clean_split() -> None:
    timestamps = moments("2019-06-01", "2020-02-01")
    split.check(timestamps, split.assign(timestamps))


def test_check_stops_on_overlap_or_an_empty_set() -> None:
    timestamps = moments("2019-06-01", "2020-02-01")
    overlapping = pd.Series(["validation", "train"]).astype(pd.CategoricalDtype(split.SPLITS))
    with pytest.raises(SystemExit, match="overlap"):
        split.check(timestamps, overlapping)
    only_train = moments("2019-06-01", "2019-07-01")
    with pytest.raises(SystemExit, match="both splits"):
        split.check(only_train, split.assign(only_train))


def test_summarize_counts_each_set() -> None:
    timestamps = moments("2019-06-01", "2019-07-01", "2020-02-01")
    frame = pd.DataFrame(
        {"timestamp": timestamps, "is_fraud": [0, 1, 1], "card_key": ["a", "b", "a"]}
    ).assign(split=lambda data: split.assign(data["timestamp"]))
    summary = split.summarize(frame)
    assert summary.loc["train", "transactions"] == 2
    assert summary.loc["train", "frauds"] == 1
    assert summary.loc["validation", "fraud_rate"] == 1.0
    assert summary.loc["train", "cards"] == 2
