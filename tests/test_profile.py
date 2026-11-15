"""Tests for pipeline/profile.py, on small synthetic CSV files shaped like the Sparkov ones.

The rows are invented for the tests (names, streets and card numbers included); nothing here
reads data/raw/.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd
import pytest

from pipeline import download, profile

BASE_ROW = {
    "trans_date_trans_time": "2019-01-01 10:00:00",
    "cc_num": "4000000000000001",
    "merchant": "fraud_Example Shop",
    "category": "grocery_pos",
    "amt": "10.50",
    "first": "Marta",
    "last": "Exemplar",
    "gender": "F",
    "street": "12 Sample Street",
    "city": "Sampleton",
    "state": "TX",
    "zip": "75001",
    "lat": "32.9",
    "long": "-96.8",
    "city_pop": "1000",
    "job": "Lighthouse keeper",
    "dob": "1990-05-17",
    "trans_num": "t0",
    "unix_time": "1325412000",
    "merch_lat": "33.0",
    "merch_long": "-96.9",
    "is_fraud": "0",
}
PERSONAL_VALUES = ("Marta", "Exemplar", "12 Sample Street", "4000000000000001", "1990-05-17")


def rows(*changes: dict[str, str]) -> list[dict[str, str]]:
    """One row per change, each a copy of BASE_ROW with its own trans_num."""
    return [{**BASE_ROW, "trans_num": f"t{i}", **change} for i, change in enumerate(changes)]


def write_csv(
    path: Path,
    data: list[dict[str, str]],
    header: tuple[str, ...] = profile.EXPECTED_COLUMNS,
    first_index: int = 0,
) -> Path:
    """Write rows the way the Sparkov files are written: an unnamed row index comes first."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *header])
        for position, row in enumerate(data, start=first_index):
            writer.writerow([position, *(row[column] for column in profile.EXPECTED_COLUMNS)])
    return path


def frame(tmp_path: Path, *changes: dict[str, str]) -> pd.DataFrame:
    return profile.load(write_csv(tmp_path / "file.csv", rows(*changes)))


def test_load_drops_the_leftover_index_and_parses_the_clock(tmp_path: Path) -> None:
    loaded = frame(tmp_path, {}, {"trans_date_trans_time": "2019-01-01 10:00:05"})
    assert tuple(loaded.columns) == profile.EXPECTED_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(loaded[profile.TIMESTAMP])
    assert len(loaded) == 2


def test_load_stops_on_a_renamed_column(tmp_path: Path) -> None:
    renamed = tuple("amount" if c == "amt" else c for c in profile.EXPECTED_COLUMNS)
    path = write_csv(tmp_path / "file.csv", rows({}), header=renamed)
    with pytest.raises(SystemExit, match="unexpected columns"):
        profile.load(path)


def test_load_stops_when_the_first_column_is_not_a_row_index(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "file.csv", rows({}, {}), first_index=5)
    with pytest.raises(SystemExit, match="row index"):
        profile.load(path)


def test_load_stops_on_a_target_other_than_0_and_1(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "file.csv", rows({}, {"is_fraud": "2"}))
    with pytest.raises(SystemExit, match="other than 0 and 1"):
        profile.load(path)


def test_profile_split_counts_frauds_ties_repeats_and_order(tmp_path: Path) -> None:
    card_b = {"cc_num": "4000000000000002", "trans_date_trans_time": "2019-01-01 09:00:00"}
    loaded = frame(
        tmp_path,
        {"amt": "10.00"},
        {"amt": "20.00", "is_fraud": "1"},  # same card and second as the first row
        card_b,  # earlier than the rows above: the file is out of time order
        card_b,  # the same transaction again, with another trans_num
    )
    split = profile.profile_split(loaded)

    assert (split.rows, split.frauds, split.fraud_rate) == (4, 1, 0.25)
    assert (split.tie_groups, split.tie_rows) == (2, 4)
    assert split.duplicated_rows == 1  # trans_num differs, everything else repeats
    assert split.duplicated_ids == 0
    assert split.in_time_order is False


def test_profile_values_counts_impossible_values(tmp_path: Path) -> None:
    loaded = frame(
        tmp_path,
        {"amt": "0"},
        {"lat": "95.0"},
        {"city_pop": "0", "gender": "M"},
        {"dob": "2019-02-01"},  # born after the transaction
    )
    values = profile.profile_values(loaded)

    assert values.non_positive_amounts == 1
    assert values.coordinates_out_of_range == 1
    assert values.non_positive_city_population == 1
    assert values.born_after_transaction == 1
    assert values.genders == ("F", "M")


def test_compare_reports_overlapping_periods_and_shared_ids(tmp_path: Path) -> None:
    train = profile.load(write_csv(tmp_path / "train.csv", rows({})))
    test = profile.load(
        write_csv(tmp_path / "test.csv", rows({"trans_date_trans_time": "2019-01-01 09:59:00"}))
    )
    overlap = profile.compare(train, test)

    assert overlap.gap < pd.Timedelta(0)
    assert overlap.shared_ids == 1  # both files have trans_num "t0"
    assert (overlap.test_cards, overlap.test_cards_seen_in_train) == (1, 1)


def test_monthly_fraud_keeps_a_month_shared_by_both_files_apart(tmp_path: Path) -> None:
    train = profile.load(write_csv(tmp_path / "train.csv", rows({}, {"is_fraud": "1"})))
    test = profile.load(write_csv(tmp_path / "test.csv", rows({})))
    months = profile.monthly_fraud({"train": train, "test": test})

    assert months[["split", "transactions", "frauds"]].to_dict("records") == [
        {"split": "train", "transactions": 2, "frauds": 1},
        {"split": "test", "transactions": 1, "frauds": 0},
    ]


def test_report_never_shows_personal_values(tmp_path: Path) -> None:
    frames = {
        "train": profile.load(write_csv(tmp_path / "train.csv", rows({}, {"is_fraud": "1"}))),
        "test": profile.load(
            write_csv(tmp_path / "test.csv", rows({"trans_date_trans_time": "2019-01-02 08:00:00"}))
        ),
    }
    files = {name: {"bytes": 1, "sha256": "0" * 64} for name in download.FILES}
    report = profile.render(profile.build_profile(files, frames))

    assert "## Valores" in report
    for value in PERSONAL_VALUES:
        assert value not in report


def test_main_stops_with_a_clear_message_when_a_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    with pytest.raises(SystemExit, match=re.escape("run python -m pipeline.download first")):
        profile.main()
