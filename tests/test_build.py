"""End-to-end tests for pipeline/build.py on a synthetic project folder.

The folder has a fraudTrain.csv and a manifest that knows only that file: there is no
fraudTest.csv at all, which proves the build never needs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sparkov_csv import PERSONAL_VALUES, rows, write_csv

from pipeline import build, download, features, prepare

KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))
CARDS = ("4000000000000001", "4000000000000002", "4000000000000003")


def synthetic_rows() -> list[dict[str, str]]:
    """Three cards, one purchase every 3 days from Oct 2019 to Feb 2020, a few frauds."""
    data = []
    moments = pd.date_range("2019-10-01 09:00:00", "2020-02-28", freq="3D")
    for number, card in enumerate(CARDS):
        for step, moment in enumerate(moments):
            when = moment + pd.Timedelta(hours=number)
            data.append(
                {
                    "trans_date_trans_time": f"{when:%Y-%m-%d %H:%M:%S}",
                    "cc_num": card,
                    "category": ("grocery_pos", "shopping_net", "travel")[step % 3],
                    "amt": f"{10 + step * 7 + number:.2f}",
                    "is_fraud": "1" if step % 17 == 5 else "0",
                }
            )
    return rows(*data)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    train = write_csv(raw / prepare.TRAIN_FILE, synthetic_rows())
    manifest = tmp_path / "data" / "manifest.json"
    entry = {"bytes": train.stat().st_size, "sha256": download.sha256(train)}
    manifest.write_text(json.dumps({"files": {prepare.TRAIN_FILE: entry}}), encoding="utf-8")
    processed = tmp_path / "data" / "processed"
    for name, value in {"ROOT": tmp_path, "RAW_DIR": raw, "MANIFEST": manifest}.items():
        monkeypatch.setattr(download, name, value)
    for name, value in {
        "BASE_FILE": processed / "transactions.parquet",
        "SPLIT_FILES": {split: processed / f"{split}.parquet" for split in build.split.SPLITS},
        "AUDIT_FILE": processed / "audit" / "attributes.parquet",
        "MANIFEST": processed / "manifest.json",
        "REPORT": tmp_path / "reports" / "temporal-split.md",
    }.items():
        monkeypatch.setattr(build, name, value)
    monkeypatch.setenv(prepare.KEY_VARIABLE, KEY.hex())
    return tmp_path


def outputs() -> dict[str, Path]:
    return {"base": build.BASE_FILE, **build.SPLIT_FILES, "audit": build.AUDIT_FILE}


def test_build_writes_minimised_files_without_the_test_file(project: Path) -> None:
    assert not (download.RAW_DIR / "fraudTest.csv").exists()
    assert build.main() == 0

    expected = [*build.ID_COLUMNS, *features.FEATURE_COLUMNS, build.TARGET]
    for name in build.split.SPLITS:
        frame = pd.read_parquet(build.SPLIT_FILES[name])
        assert list(frame.columns) == expected
        assert set(frame["timestamp"].lt(build.split.VALIDATION_START)) == {name == "train"}
    assert tuple(pd.read_parquet(build.AUDIT_FILE).columns) == prepare.AUDIT_COLUMNS

    model_files = (build.BASE_FILE, *build.SPLIT_FILES.values())
    text = "".join(pd.read_parquet(path).astype(str).to_csv() for path in model_files)
    assert not [value for value in PERSONAL_VALUES if value in text]
    assert "gender" not in text and "age_years" not in text


def test_build_records_the_signature_of_every_file(project: Path) -> None:
    build.main()
    manifest = json.loads(build.MANIFEST.read_text(encoding="utf-8"))
    for name, path in outputs().items():
        assert manifest["outputs"][name]["sha256"] == download.sha256(path)
    assert prepare.KEY_VARIABLE in manifest["card_key"]["key_variable"]
    assert KEY.hex() not in build.MANIFEST.read_text(encoding="utf-8")
    assert KEY.hex() not in build.REPORT.read_text(encoding="utf-8")


def test_build_is_reproducible(project: Path) -> None:
    build.main()
    first = {name: download.sha256(path) for name, path in outputs().items()}
    report = build.REPORT.read_text(encoding="utf-8")
    build.main()
    assert {name: download.sha256(path) for name, path in outputs().items()} == first
    assert build.REPORT.read_text(encoding="utf-8") == report


def test_another_key_changes_only_the_card_pseudonym(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build.main()
    first = json.loads(build.MANIFEST.read_text(encoding="utf-8"))["outputs"]
    report = build.REPORT.read_text(encoding="utf-8")
    monkeypatch.setenv(prepare.KEY_VARIABLE, OTHER_KEY.hex())
    build.main()
    second = json.loads(build.MANIFEST.read_text(encoding="utf-8"))["outputs"]
    for name in ("base", *build.split.SPLITS):
        assert first[name]["sha256"] != second[name]["sha256"]  # card_key changed
        assert first[name]["content_sha256"] == second[name]["content_sha256"]  # nothing else
    assert build.REPORT.read_text(encoding="utf-8") == report


def test_build_stops_without_the_key(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(prepare.KEY_VARIABLE)
    with pytest.raises(SystemExit, match=prepare.KEY_VARIABLE):
        build.main()
    assert not build.BASE_FILE.exists()
