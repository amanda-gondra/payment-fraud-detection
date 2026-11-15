"""Tests for pipeline/download.py: the manifest must catch any file that is not the recorded one.

Every test works in a temporary folder with small synthetic files. Nothing here touches
data/raw/, data/manifest.json or the network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline import download

CC0_METADATA = {
    "licenseName": download.EXPECTED_LICENSE,
    "currentVersionNumber": 1,
    "ownerName": "Example Owner",
    "lastUpdated": "2020-08-05T15:20:55.217Z",
}
NON_COMMERCIAL_METADATA = {**CC0_METADATA, "licenseName": "CC BY-NC-SA 4.0"}


@pytest.fixture
def raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary project root with data/raw/ holding one small synthetic file per name."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for position, name in enumerate(download.FILES):
        (raw / name).write_text(f"col\n{position}\n", encoding="utf-8")
    monkeypatch.setattr(download, "ROOT", tmp_path)
    monkeypatch.setattr(download, "RAW_DIR", raw)
    monkeypatch.setattr(download, "MANIFEST", tmp_path / "data" / "manifest.json")
    return raw


def write_manifest(files: dict[str, download.FileInfo]) -> None:
    download.MANIFEST.write_text(json.dumps({"files": files}), encoding="utf-8")


def test_sha256_matches_hashlib_across_several_blocks(tmp_path: Path) -> None:
    path = tmp_path / "big.bin"
    content = bytes(range(256)) * (download.CHUNK_BYTES // 256 * 2 + 7)  # just over 2 blocks
    path.write_bytes(content)
    assert download.sha256(path) == hashlib.sha256(content).hexdigest()


def test_verify_accepts_the_recorded_files(raw_dir: Path) -> None:
    files = download.describe_files()
    write_manifest(files)
    download.verify(files)  # must not raise


def test_verify_stops_when_one_byte_changes_without_changing_the_size(raw_dir: Path) -> None:
    write_manifest(download.describe_files())
    target = raw_dir / download.FILES[0]
    original = target.read_bytes()
    target.write_bytes(original.replace(b"0", b"9"))
    tampered = download.describe_files()

    assert tampered[target.name]["bytes"] == len(original)
    with pytest.raises(SystemExit, match=target.name):
        download.verify(tampered)


def test_verify_stops_without_a_manifest(raw_dir: Path) -> None:
    with pytest.raises(SystemExit, match="--record"):
        download.verify(download.describe_files())


def test_record_refuses_a_licence_other_than_cc0(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "fetch_metadata", lambda: NON_COMMERCIAL_METADATA)
    with pytest.raises(SystemExit, match="review decision D1"):
        download.record(download.describe_files())
    assert not download.MANIFEST.exists()


def test_record_writes_source_licence_and_hashes(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "fetch_metadata", lambda: CC0_METADATA)
    files = download.describe_files()
    download.record(files)

    manifest = json.loads(download.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["dataset"] == download.DATASET
    assert manifest["version"] == download.VERSION
    assert manifest["licence"] == download.EXPECTED_LICENSE
    assert manifest["files"] == files


def test_download_skips_kaggle_when_every_file_is_there(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("kagglehub must not be called when the files already exist")

    monkeypatch.setattr(download.kagglehub, "dataset_download", fail)
    download.download()


def test_download_stops_when_kaggle_leaves_a_file_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    monkeypatch.setattr(download, "RAW_DIR", raw)

    def only_first_file(*_args: object, **_kwargs: object) -> None:
        (raw / download.FILES[0]).write_text("col\n0\n", encoding="utf-8")

    monkeypatch.setattr(download.kagglehub, "dataset_download", only_first_file)
    with pytest.raises(SystemExit, match=download.FILES[1]):
        download.download()
