"""Small synthetic CSV files shaped like the Sparkov ones, shared by the tests.

The rows are invented for the tests, names, streets and card numbers included.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pipeline import profile

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
PERSONAL_VALUES = ("Marta", "Exemplar", "12 Sample Street", "4000000000000001", "1990-05-17",
                   "Sampleton", "75001", "Lighthouse keeper")


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
