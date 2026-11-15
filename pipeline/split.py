"""Time split of the training file into train and validation (decision D9).

- train: from the first transaction (2019-01-01) up to, not including, VALIDATION_START;
- validation: from VALIDATION_START to the last transaction of the file (2020-06-21);
- test: fraudTest.csv, kept apart and untouched until the final evaluation (D11).

The cut is a fixed instant, so the two sets cannot overlap: every transaction before it is in
train and every transaction at or after it is in validation. History signals of validation rows
may look back into the train period, which is what happens in production; train rows never look
forward, because history only uses earlier transactions (D8).
"""

from __future__ import annotations

from typing import Final

import pandas as pd

VALIDATION_START: Final = pd.Timestamp("2020-01-01 00:00:00")
SPLITS: Final = ("train", "validation")


def assign(timestamps: pd.Series) -> pd.Series:
    """'train' before VALIDATION_START, 'validation' from it on."""
    labels = timestamps.ge(VALIDATION_START).map({False: "train", True: "validation"})
    return labels.astype(pd.CategoricalDtype(SPLITS)).rename("split")


def check(timestamps: pd.Series, splits: pd.Series) -> None:
    """Stop if a split is empty or the periods overlap."""
    if set(splits.unique()) != set(SPLITS):
        raise SystemExit(f"expected both splits {SPLITS}, got {sorted(splits.unique())}")
    last_train = timestamps[splits.eq("train")].max()
    first_validation = timestamps[splits.eq("validation")].min()
    if not last_train < VALIDATION_START <= first_validation:
        raise SystemExit("train and validation overlap in time")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Period, transactions, frauds, fraud rate and cards of each split."""
    grouped = frame.groupby("split", observed=True)
    table = grouped.agg(
        first=("timestamp", "min"),
        last=("timestamp", "max"),
        transactions=("is_fraud", "size"),
        frauds=("is_fraud", "sum"),
        cards=("card_key", "nunique"),
    )
    table["fraud_rate"] = table["frauds"] / table["transactions"]
    return table.reindex(list(SPLITS))
