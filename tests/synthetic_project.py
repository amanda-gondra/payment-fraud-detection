"""A small synthetic project for end-to-end tests: raw files, built sets and every output path
pointed at a temporary folder.

Twelve invented cards buy every two hours from 2019-01-01 to 2020-12-31. About 3% of the
purchases are frauds, most of them late at night, large and online, with noise on both sides so
no signal separates the classes perfectly. Names, streets and card numbers are invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sparkov_csv import rows, write_csv

from pipeline import build, download, evaluate, train
from pipeline.modeling import artifacts, baselines, lgbm, tracking

KEY = bytes(range(32))
CARDS = tuple(f"40000000000000{number:02d}" for number in range(12))
CATEGORIES = ("grocery_pos", "shopping_net", "gas_transport", "misc_net")
TRAIN_END = "2020-06-21 12:00:00"


def transactions(start: str, end: str, prefix: str, seed: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    data = []
    for step, moment in enumerate(pd.date_range(start, end, freq="2h")):
        number = step % len(CARDS)
        fraud = rng.random() < 0.03
        typical = rng.random() < 0.8  # 20% of the frauds look ordinary, and 3% of the legit not
        unusual = fraud == typical or (not fraud and rng.random() < 0.03)
        hour = int(rng.choice([22, 23, 0, 1])) if unusual else int(rng.integers(7, 21))
        when = moment.normalize() + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
        data.append({
            "trans_date_trans_time": f"{when:%Y-%m-%d %H:%M:%S}",
            "cc_num": CARDS[number],
            "category": "shopping_net" if unusual else CATEGORIES[int(rng.integers(0, 4))],
            "amt": f"{rng.uniform(300, 900) if unusual else rng.uniform(5, 120):.2f}",
            "is_fraud": "1" if fraud else "0",
            "gender": "F" if number % 2 else "M",
            "dob": f"{1950 + 4 * number}-03-15",
            "merch_lat": f"{32.9 + rng.uniform(-1, 1):.4f}",
            "merch_long": f"{-96.8 + rng.uniform(-1, 1):.4f}",
        })
    result = rows(*data)
    for position, row in enumerate(result):
        row["trans_num"] = f"{prefix}{position}"
    return result


def make_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the raw files and their manifest, build the sets and redirect every output."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    files = {
        "fraudTrain.csv": transactions("2019-01-01", TRAIN_END, "a", seed=1),
        "fraudTest.csv": transactions("2020-06-22", "2020-12-31", "b", seed=2),
    }
    entries = {}
    for name, data in files.items():
        path = write_csv(raw / name, data)
        entries[name] = {"bytes": path.stat().st_size, "sha256": download.sha256(path)}
    manifest = tmp_path / "data" / "manifest.json"
    manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")
    processed = tmp_path / "data" / "processed"
    models = tmp_path / "models"
    patches = [
        (download, {"ROOT": tmp_path, "RAW_DIR": raw, "MANIFEST": manifest}),
        (build, {
            "BASE_FILE": processed / "transactions.parquet",
            "SPLIT_FILES": {name: processed / f"{name}.parquet" for name in build.split.SPLITS},
            "AUDIT_FILE": processed / "audit" / "attributes.parquet",
            "MANIFEST": processed / "manifest.json",
            "REPORT": tmp_path / "reports" / "temporal-split.md",
        }),
        (tracking, {"TRACKING_DIR": tmp_path / "mlruns"}),
        (artifacts, {"MODELS_DIR": models, "BUNDLE_DIR": models / "fraud-detector",
                     "BASELINE_DIR": models / "baselines"}),
        (train, {"REPORT": tmp_path / "reports" / "model-selection.md",
                 "FIGURES": tmp_path / "reports" / "figures"}),
        (evaluate, {"REPORT": tmp_path / "reports" / "test-evaluation.md",
                    "FIGURES": tmp_path / "reports" / "figures",
                    "RECORD": models / "test-evaluation.json",
                    "TEST_OUTPUT": processed / "test.parquet"}),
        # a smaller search keeps the tests fast; the real grids are tested elsewhere
        (lgbm, {"GRID": lgbm.GRID[:2]}),
        (baselines, {"LOGISTIC_GRID": baselines.LOGISTIC_GRID[:2]}),
    ]
    for module, values in patches:
        for name, value in values.items():
            monkeypatch.setattr(module, name, value)
    monkeypatch.setenv("CARD_PSEUDONYM_KEY", KEY.hex())
    build.main()
    return tmp_path
