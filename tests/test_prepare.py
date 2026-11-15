"""Tests for pipeline/prepare.py: the secret key, the card pseudonym, minimisation and order.

All data here is synthetic; the tests never read data/raw/ nor the project's .env.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from sparkov_csv import PERSONAL_VALUES, rows, write_csv

from pipeline import download, prepare, profile

KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))


@pytest.fixture
def no_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the project root at an empty folder, so the real .env is never loaded."""
    monkeypatch.setattr(download, "ROOT", tmp_path)
    monkeypatch.delenv(prepare.KEY_VARIABLE, raising=False)


@pytest.mark.parametrize("value", ["", "not-hex", "ab" * 31])
def test_load_key_refuses_a_missing_or_short_key(
    value: str, no_env_file: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(prepare.KEY_VARIABLE, value)
    with pytest.raises(SystemExit, match=re.escape(".env.example")):
        prepare.load_key()


def test_load_key_reads_the_environment(no_env_file: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(prepare.KEY_VARIABLE, KEY.hex())
    assert prepare.load_key() == KEY


def test_card_keys_are_stable_keyed_and_unique() -> None:
    numbers = pd.Series([4000000000000001, 4000000000000002, 4000000000000001])
    first = prepare.card_keys(numbers, KEY)
    assert list(first) == list(prepare.card_keys(numbers, KEY))  # stable for one key
    assert first[0] == first[2] != first[1]  # one pseudonym per card
    assert set(first).isdisjoint(prepare.card_keys(numbers, OTHER_KEY))  # depends on the key
    assert all(re.fullmatch(r"[0-9a-f]{16}", value) for value in first)
    assert not any(str(number) in value for number in numbers for value in first)


def test_key_fingerprint_does_not_reveal_the_key() -> None:
    fingerprint = prepare.key_fingerprint(KEY)
    assert fingerprint != prepare.key_fingerprint(OTHER_KEY)
    assert KEY.hex()[:12] not in fingerprint


def raw_frame(tmp_path: Path) -> pd.DataFrame:
    data = rows(
        {"trans_date_trans_time": "2020-03-01 10:00:00", "dob": "1990-03-01"},  # birthday
        {"trans_date_trans_time": "2020-02-29 23:59:59", "dob": "1990-03-01"},  # day before
        {"trans_date_trans_time": "2020-03-01 10:00:00", "cc_num": "4000000000000002",
         "gender": "M", "lat": "95.0"},  # same second as row 0; invalid latitude
    )
    return profile.load(write_csv(tmp_path / "fraudTrain.csv", data))


def test_prepare_keeps_only_the_minimised_columns(tmp_path: Path) -> None:
    base, audit = prepare.prepare(raw_frame(tmp_path), KEY)
    assert tuple(base.columns) == prepare.BASE_COLUMNS
    assert tuple(audit.columns) == prepare.AUDIT_COLUMNS
    text = base.astype(str).to_csv() + audit.astype(str).to_csv()
    assert not [value for value in PERSONAL_VALUES if value in text]


def test_prepare_orders_by_time_and_keeps_file_order_in_ties(tmp_path: Path) -> None:
    base, audit = prepare.prepare(raw_frame(tmp_path), KEY)
    assert list(base["source_row"]) == [1, 0, 2]  # 23:59:59 first; rows 0 and 2 tie
    assert list(audit["transaction_id"]) == list(base["transaction_id"])
    assert base["timestamp"].is_monotonic_increasing
    assert str(base["timestamp"].dtype) == prepare.TIMESTAMP_DTYPE


def test_prepare_computes_age_and_flags_invalid_coordinates(tmp_path: Path) -> None:
    base, audit = prepare.prepare(raw_frame(tmp_path), KEY)
    ages = dict(zip(base["source_row"], audit["age_years"], strict=True))
    assert ages[0] == 30  # birthday on the day of the transaction
    assert ages[1] == 29  # one day before the birthday
    distance = dict(zip(base["source_row"], base["distance_km"], strict=True))
    assert pd.isna(distance[2]) and distance[0] > 0


def test_verify_train_file_stops_on_a_changed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    train = write_csv(raw_dir / prepare.TRAIN_FILE, rows({}))
    manifest = tmp_path / "manifest.json"
    files = {prepare.TRAIN_FILE: {"bytes": train.stat().st_size, "sha256": download.sha256(train)}}
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    monkeypatch.setattr(download, "RAW_DIR", raw_dir)
    monkeypatch.setattr(download, "MANIFEST", manifest)

    assert prepare.verify_train_file() == files[prepare.TRAIN_FILE]["sha256"]
    train.write_bytes(train.read_bytes().replace(b"10.50", b"10.51"))
    with pytest.raises(SystemExit, match="differs"):
        prepare.verify_train_file()
