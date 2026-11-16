"""Model inputs: the verified processed sets, the internal cut of the validation period and the
model matrix shared by training and inference (decision D12).

Development sets, all from fraudTrain.csv:

- train (2019): fits every model;
- selection (validation, 2020-01-01 up to CALIBRATION_START): early stopping and the choice of
  hyperparameters;
- calibration (validation, from CALIBRATION_START to 2020-06-21): calibrators, their comparison
  by cross-fitting and the decision thresholds.

The test file stays closed until python -m pipeline.evaluate (D11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from pipeline import build, download, features

CALIBRATION_START: Final = pd.Timestamp("2020-04-01 00:00:00")
DEVELOPMENT_SETS: Final = ("train", "selection", "calibration")
TARGET: Final = build.TARGET
CATEGORY: Final = "category"


def processed_manifest() -> dict[str, Any]:
    return json.loads(build.MANIFEST.read_text(encoding="utf-8"))


def read_verified(name: str) -> pd.DataFrame:
    """A processed file, after checking its SHA-256 against data/processed/manifest.json."""
    record = processed_manifest()["outputs"][name]
    path = download.ROOT / record["path"]
    if not path.is_file() or download.sha256(path) != record["sha256"]:
        raise SystemExit(
            f"{record['path']} differs from data/processed/manifest.json: "
            "run python -m pipeline.build again"
        )
    return pd.read_parquet(path)


def development_sets() -> dict[str, pd.DataFrame]:
    """train, selection and calibration, each in time order and with both classes."""
    train = read_verified("train")
    validation = read_verified("validation")
    later = validation["timestamp"].ge(CALIBRATION_START)
    sets = {"train": train, "selection": validation[~later], "calibration": validation[later]}
    for name, frame in sets.items():
        if frame[TARGET].nunique() != 2:
            raise SystemExit(f"the {name} set needs frauds and legitimate transactions")
    return {name: frame.reset_index(drop=True) for name, frame in sets.items()}


def data_version() -> dict[str, str]:
    """Signatures of the source file and of the processed sets, recorded with every experiment."""
    manifest = processed_manifest()
    version = {"source_sha256": manifest["source"]["sha256"]}
    for name in ("train", "validation"):
        version[f"{name}_sha256"] = manifest["outputs"][name]["sha256"]
        version[f"{name}_content_sha256"] = manifest["outputs"][name]["content_sha256"]
    return version


@dataclass(frozen=True)
class FeatureSpec:
    """Order of the model columns and the code of each category, saved with the model.

    The same object builds the matrix for training, evaluation and the API, so a column can never
    change place between them. Only the 28 signals are read: identifiers, the label and the audit
    attributes (gender, age) never reach the matrix even when the frame carries them.
    """

    columns: tuple[str, ...]
    categories: tuple[str, ...]

    @classmethod
    def from_train(cls, train: pd.DataFrame) -> FeatureSpec:
        categories = tuple(sorted(train[CATEGORY].astype(str).unique()))
        return cls(columns=tuple(features.FEATURE_COLUMNS), categories=categories)

    @property
    def category_index(self) -> int:
        return self.columns.index(CATEGORY)

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """float64 matrix in column order: category as its code, booleans as 0/1, NaN kept."""
        missing = [name for name in self.columns if name not in frame.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")
        labels = frame[CATEGORY].astype(str)
        known = labels.isin(self.categories)
        if not known.all():
            raise ValueError(f"unknown category: {sorted(set(labels[~known]))}")
        codes = labels.map({name: code for code, name in enumerate(self.categories)})
        matrix = np.empty((len(frame), len(self.columns)), dtype=np.float64)
        for position, name in enumerate(self.columns):
            if name == CATEGORY:
                matrix[:, position] = codes.to_numpy(dtype=np.float64)
            else:
                matrix[:, position] = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
        return matrix

    def category_label(self, code: float) -> str:
        return self.categories[int(code)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "categories": list(self.categories),
            "categorical_columns": [CATEGORY],
            "encoding": "float64; category as its position in categories; booleans as 0/1; "
            "missing values as NaN",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSpec:
        return cls(columns=tuple(data["columns"]), categories=tuple(data["categories"]))


@dataclass
class Split:
    """A set ready for the models: its frame, model matrix, labels and amounts."""

    frame: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    amount: np.ndarray

    @classmethod
    def of(cls, frame: pd.DataFrame, spec: FeatureSpec) -> Split:
        return cls(frame, spec.matrix(frame), frame[TARGET].to_numpy(dtype=np.int8),
                   frame["amount"].to_numpy(dtype=float))

    def subset(self, mask: np.ndarray) -> Split:
        return Split(self.frame.loc[mask].reset_index(drop=True), self.x[mask], self.y[mask],
                     self.amount[mask])

    def between(self, start: str | None, end: str) -> Split:
        """Rows from start (inclusive; None for the beginning) up to end (exclusive)."""
        timestamps = self.frame["timestamp"]
        mask = timestamps.lt(pd.Timestamp(end))
        if start is not None:
            mask &= timestamps.ge(pd.Timestamp(start))
        return self.subset(mask.to_numpy())


# Rolling-origin windows inside 2019 (decision D14): each trains on everything before its window
# and validates on the window. A fourth window trains on all of 2019 and validates on the
# selection period; its model is the final one.
CV_WINDOWS: Final = (
    ("jul-ago 2019", "2019-07-01", "2019-09-01"),
    ("set-out 2019", "2019-09-01", "2019-11-01"),
    ("nov-dez 2019", "2019-11-01", "2020-01-01"),
)
FINAL_WINDOW: Final = "jan-mar 2020"


def time_folds(train: Split, selection: Split) -> list[tuple[str, Split, Split]]:
    """(label, rows to fit, rows to validate) for each window, in time order."""
    folds = [(label, train.between(None, start), train.between(start, end))
             for label, start, end in CV_WINDOWS]
    return [*folds, (FINAL_WINDOW, train, selection)]
