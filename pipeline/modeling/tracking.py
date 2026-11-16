"""Experiment tracking with MLflow, local and private (decision D17).

Runs are kept in mlruns/ (an SQLite database and the artifacts), outside Git. Only parameters,
aggregated metrics and files about the model are logged: no transaction, no card pseudonym and
no personal attribute. MLflow's telemetry is switched off before MLflow is imported.

To browse the runs:

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil

# subprocess runs only git, with fixed arguments and no shell (see git_state)
import subprocess  # nosec B404
from importlib.metadata import version
from types import ModuleType
from typing import Final

from pipeline import download

TRACKING_DIR: Final = download.ROOT / "mlruns"
EXPERIMENT: Final = "payment-fraud-detection"
LIBRARIES: Final = ("lightgbm", "scikit-learn", "scipy", "numpy", "pandas", "pyarrow", "mlflow",
                    "skops")


def mlflow_client() -> ModuleType:
    """MLflow pointed at the local store, with telemetry off and the experiment selected."""
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["DO_NOT_TRACK"] = "true"
    mlflow = importlib.import_module("mlflow")  # only now, after telemetry is off
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(TRACKING_DIR / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT,
                                 artifact_location=(TRACKING_DIR / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def libraries() -> dict[str, str]:
    return {name: version(name) for name in LIBRARIES}


def code_digest() -> str:
    """SHA-256 of every Python file under pipeline/, line endings normalised."""
    digest = hashlib.sha256()
    folder = download.ROOT / "pipeline"
    for path in sorted(folder.rglob("*.py"), key=lambda item: item.relative_to(folder).as_posix()):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return digest.hexdigest()


def git_state() -> dict[str, str]:
    """Commit of the code and whether there were uncommitted changes when the run started."""
    git = shutil.which("git")
    if git is None:
        return {"git_commit": "unknown", "git_dirty": "unknown"}

    def run(*args: str) -> str:
        # fixed arguments, no shell, the executable resolved by shutil.which
        result = subprocess.run(  # noqa: S603  # nosec B603
            [git, *args], cwd=download.ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain"))
    return {"git_commit": commit or "unknown", "git_dirty": str(dirty).lower()}
