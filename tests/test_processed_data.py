"""Regression tests on the real processed data (skipped until python -m pipeline.build runs).

They sample transactions with a fixed seed, recompute every signal with the independent
row-by-row reference and compare with the stored values, so a future change to the pipeline
that introduces a leak fails here even if the synthetic tests miss it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from reference_history import mismatches

from pipeline import build, split

SEED = 20_261_115
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in (build.BASE_FILE, *build.SPLIT_FILES.values())),
    reason="processed data missing: run python -m pipeline.build",
)


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return pd.read_parquet(build.BASE_FILE)


@pytest.fixture(scope="module")
def stored(base: pd.DataFrame) -> pd.DataFrame:
    """Train and validation features, aligned row by row with the base."""
    sets = pd.concat([pd.read_parquet(path) for path in build.SPLIT_FILES.values()])
    return base[["transaction_id"]].merge(sets, on="transaction_id", how="left", validate="1:1")


def test_stored_signals_match_the_reference(base: pd.DataFrame, stored: pd.DataFrame) -> None:
    sample = np.random.default_rng(SEED).choice(len(base), size=300, replace=False)
    assert mismatches(base, stored, sample) == []


def test_no_transaction_is_lost_or_duplicated(base: pd.DataFrame, stored: pd.DataFrame) -> None:
    assert stored["timestamp"].notna().all()  # every base row is in train or validation
    assert stored["transaction_id"].is_unique


def test_the_sets_follow_the_cut_and_stay_inside_the_training_file(base: pd.DataFrame) -> None:
    train, validation = (pd.read_parquet(build.SPLIT_FILES[name]) for name in split.SPLITS)
    assert train["timestamp"].max() < split.VALIDATION_START <= validation["timestamp"].min()
    assert validation["timestamp"].max() == base["timestamp"].max()
    manifest = json.loads(build.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source"]["file"] == "fraudTrain.csv"
