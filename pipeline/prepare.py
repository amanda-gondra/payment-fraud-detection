"""Prepare the training file: check it, minimise it, pseudonymise the card and order the rows.

Only fraudTrain.csv is read. fraudTest.csv is not opened in this phase: it stays untouched for
the final evaluation (decisions D7 and D11).

What happens to each raw column (docs/dados-processados.md explains every choice):

- kept: trans_num (as transaction_id), trans_date_trans_time (as timestamp), category, amt (as
  amount) and is_fraud;
- transformed: cc_num becomes card_key, an HMAC-SHA256 pseudonym that cannot be reversed without
  the secret key; the customer and merchant coordinates become distance_km and are dropped;
- audit only: gender and the age at the transaction (from dob) go to a separate frame that never
  reaches the model (decision D5);
- dropped: first, last, street, city, state, zip, city_pop, job, merchant, unix_time (D6) and
  the leftover row index, which survives as source_row.

The rows are ordered by timestamp; transactions in the same second keep their order in the
file (source_row), so the order is always the same (decision D6).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Final

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pipeline import download, features

TRAIN_FILE: Final = "fraudTrain.csv"
KEY_VARIABLE: Final = "CARD_PSEUDONYM_KEY"
KEY_BYTES: Final = 32
CARD_KEY_LENGTH: Final = 16  # hex characters: 64 bits, enough for a few thousand cards
TIMESTAMP_DTYPE: Final = "datetime64[ms]"
BASE_COLUMNS: Final = (
    "transaction_id", "source_row", "timestamp", "card_key", "category", "amount", "distance_km",
    "is_fraud",
)
AUDIT_COLUMNS: Final = ("transaction_id", "gender", "age_years")


def load_key() -> bytes:
    """The secret for the card pseudonym, from the environment or the local .env file."""
    load_dotenv(download.ROOT / ".env", override=False)
    value = os.environ.get(KEY_VARIABLE, "").strip()
    try:
        key = bytes.fromhex(value)
    except ValueError:
        key = b""
    if len(key) < KEY_BYTES:
        raise SystemExit(
            f"{KEY_VARIABLE} must hold at least {KEY_BYTES} random bytes in hex. Create it once, "
            "in the .env file (never committed): see .env.example"
        )
    return key


def key_fingerprint(key: bytes) -> str:
    """A short public tag that changes when the key changes, without revealing the key."""
    return hashlib.sha256(b"card-pseudonym-key:" + key).hexdigest()[:12]


def card_keys(card_numbers: pd.Series, key: bytes) -> pd.Series:
    """HMAC-SHA256 pseudonym of each card number: stable for one key, not reversible without it."""
    unique = pd.unique(card_numbers)
    pseudonyms = {
        number: hmac.new(key, str(number).encode(), hashlib.sha256).hexdigest()[:CARD_KEY_LENGTH]
        for number in unique
    }
    if len(set(pseudonyms.values())) != len(unique):
        raise ValueError("two cards got the same pseudonym: increase CARD_KEY_LENGTH")
    return card_numbers.map(pseudonyms).astype("category").rename("card_key")


def verify_train_file() -> str:
    """Check the training file against data/manifest.json; return its SHA-256."""
    path = download.RAW_DIR / TRAIN_FILE
    if not path.is_file():
        raise SystemExit(f"missing data/raw/{TRAIN_FILE}: run python -m pipeline.download first")
    expected = json.loads(download.MANIFEST.read_text(encoding="utf-8"))["files"][TRAIN_FILE]
    actual = {"bytes": path.stat().st_size, "sha256": download.sha256(path)}
    if actual != expected:
        raise SystemExit(f"{TRAIN_FILE} differs from data/manifest.json")
    return str(actual["sha256"])


def age_in_years(birth_dates: pd.Series, timestamps: pd.Series) -> pd.Series:
    """Completed years of age on the day of each transaction."""
    birth = pd.to_datetime(birth_dates)
    years = timestamps.dt.year - birth.dt.year
    before_birthday = (timestamps.dt.month < birth.dt.month) | (
        (timestamps.dt.month == birth.dt.month) & (timestamps.dt.day < birth.dt.day)
    )
    return (years - before_birthday.astype(int)).astype("int16")


def prepare(raw: pd.DataFrame, key: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimised, pseudonymised base in time order, and the separate audit attributes.

    raw is the output of pipeline.profile.load: columns checked, leftover index dropped, rows in
    file order (RangeIndex), timestamp already parsed.
    """
    # Parquet has no one-second unit; milliseconds store the whole seconds without any change.
    timestamps = raw["trans_date_trans_time"].astype(TIMESTAMP_DTYPE)
    base = pd.DataFrame(
        {
            "transaction_id": raw["trans_num"].astype(str),
            "source_row": np.arange(len(raw), dtype=np.int64),
            "timestamp": timestamps,
            "card_key": card_keys(raw["cc_num"], key),
            "category": raw["category"].astype("category"),
            "amount": raw["amt"].astype(float),
            "distance_km": features.haversine_km(
                raw["lat"], raw["long"], raw["merch_lat"], raw["merch_long"]
            ),
            "is_fraud": raw["is_fraud"].astype("int8"),
        }
    )
    audit = pd.DataFrame(
        {
            "transaction_id": base["transaction_id"],
            "gender": raw["gender"].astype("category"),
            "age_years": age_in_years(raw["dob"], timestamps),
        }
    )
    # Stable sort: rows in the same second keep the file order (source_row).
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    base = base.iloc[order].reset_index(drop=True)
    audit = audit.iloc[order].reset_index(drop=True)
    return base[list(BASE_COLUMNS)], audit[list(AUDIT_COLUMNS)]
