"""Download the Sparkov card transactions from Kaggle and prove which files were used.

Source: kartik2112/fraud-detection on Kaggle, version 1, licence CC0 (decision D1).
The CSV files go to data/raw/ and stay out of Git. data/manifest.json is versioned: it records
the source, the licence and the SHA-256 of each file, so anyone can prove they used exactly
the same data (decision D4).

    python -m pipeline.download            # download if needed, then check against the manifest
    python -m pipeline.download --record   # first run only: check the licence, write the manifest

No credential is needed for this public dataset. If Kaggle ever asks for one, kagglehub reads it
from ~/.kaggle/access_token by itself; this script never reads or prints it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import kagglehub

DATASET: Final = "kartik2112/fraud-detection"
VERSION: Final = 1
EXPECTED_LICENSE: Final = "CC0: Public Domain"
FILES: Final = ("fraudTrain.csv", "fraudTest.csv")

ROOT: Final = Path(__file__).resolve().parents[1]
RAW_DIR: Final = ROOT / "data" / "raw"
MANIFEST: Final = ROOT / "data" / "manifest.json"
METADATA_URL: Final = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}"
CHUNK_BYTES: Final = 1024 * 1024

FileInfo = dict[str, int | str]


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MB blocks so a 335 MB file never sits whole in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download() -> None:
    """Fetch the pinned version into data/raw/, unless every file is already there."""
    if all((RAW_DIR / name).is_file() for name in FILES):
        print("data/raw/ already has the files; checking them instead of downloading")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    kagglehub.dataset_download(f"{DATASET}/versions/{VERSION}", output_dir=str(RAW_DIR))
    missing = [name for name in FILES if not (RAW_DIR / name).is_file()]
    if missing:
        raise SystemExit(f"download incomplete, missing: {', '.join(missing)}")


def describe_files() -> dict[str, FileInfo]:
    """Size and SHA-256 of each expected file."""
    return {
        name: {"bytes": (RAW_DIR / name).stat().st_size, "sha256": sha256(RAW_DIR / name)}
        for name in FILES
    }


def fetch_metadata() -> dict[str, object]:
    """Public metadata of the dataset on Kaggle (licence, owner, versions); no credential."""
    request = urllib.request.Request(METADATA_URL, headers={"Accept": "application/json"})
    # S310 / B310 warn that urlopen also accepts file:// and custom schemes. METADATA_URL is a
    # fixed https constant, never built from input, so neither can reach this call.
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310  # nosec B310
        return json.load(response)


def record(files: dict[str, FileInfo]) -> None:
    """Check the licence and write data/manifest.json. Refuses a licence other than CC0."""
    metadata = fetch_metadata()
    licence = metadata.get("licenseName")
    if licence != EXPECTED_LICENSE:
        raise SystemExit(
            f"licence is {licence!r}, expected {EXPECTED_LICENSE!r}: review decision D1 first"
        )
    manifest = {
        "dataset": DATASET,
        "url": f"https://www.kaggle.com/datasets/{DATASET}",
        "version": VERSION,
        "latest_version_on_kaggle": metadata.get("currentVersionNumber"),
        "licence": licence,
        "owner": metadata.get("ownerName"),
        "published_at": metadata.get("lastUpdated"),
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT).as_posix()}")


def verify(files: dict[str, FileInfo]) -> None:
    """Stop if any file differs, in size or SHA-256, from what the manifest recorded."""
    if not MANIFEST.is_file():
        raise SystemExit("data/manifest.json does not exist yet: run once with --record")
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    different = [name for name in FILES if expected.get(name) != files[name]]
    if different:
        raise SystemExit(f"files differ from data/manifest.json: {', '.join(different)}")
    print("files match data/manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--record", action="store_true", help="check the licence and write data/manifest.json"
    )
    args = parser.parse_args(argv)

    download()
    files = describe_files()
    if args.record:
        record(files)
    else:
        verify(files)
    for name, info in files.items():
        # MB as Kaggle shows it (1 MB = 1024 * 1024 bytes); ASCII only, for the Windows console.
        print(f"{name}  {int(info['bytes']) / 2**20:.1f} MB  sha256 {str(info['sha256'])[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
