"""End-to-end tests of python -m pipeline.train and python -m pipeline.evaluate on a synthetic
project: test isolation, signed artifacts, privacy of the outputs and the single evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from reference_history import mismatches
from synthetic_project import KEY, make_project

from pipeline import build, download, evaluate, prepare, profile, train
from pipeline.modeling import artifacts


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return make_project(tmp_path, monkeypatch)


def texts_under(folder: Path) -> str:
    """Every readable file under folder, decoded loosely (binary files included)."""
    return "".join(path.read_bytes().decode("utf-8", errors="ignore")
                   for path in folder.rglob("*") if path.is_file())


def test_train_works_without_the_test_file_and_writes_a_signed_bundle(project: Path) -> None:
    (download.RAW_DIR / "fraudTest.csv").unlink()  # training must not need it
    assert train.main() == 0
    model = artifacts.FraudModel.load()
    assert model.spec.columns == tuple(build.features.FEATURE_COLUMNS)
    assert model.thresholds.review <= model.thresholds.block
    assert artifacts.Baselines.load().rule.threshold > 0
    report = train.REPORT.read_text(encoding="utf-8")
    assert "## O que fica congelado para o teste" in report
    assert model.manifest["bundle_sha256"] in report
    names = {path.suffix for path in artifacts.MODELS_DIR.rglob("*") if path.is_file()}
    assert not names & {".pkl", ".pickle", ".joblib"}


def test_training_twice_gives_the_same_model(project: Path) -> None:
    train.main()
    names = ("model.txt", "features.json", "calibrator.json", "decision.json")
    first = {name: (artifacts.BUNDLE_DIR / name).read_bytes() for name in names}
    train.main()
    assert {name: (artifacts.BUNDLE_DIR / name).read_bytes() for name in names} == first


def test_outputs_keep_no_secret_and_no_pseudonym(project: Path) -> None:
    train.main()
    cards = set(pd.read_parquet(build.BASE_FILE)["card_key"].astype(str))
    for folder in (artifacts.MODELS_DIR, project / "mlruns" / "artifacts", project / "reports"):
        text = texts_under(folder)
        assert KEY.hex() not in text
        assert not [card for card in cards if card in text]


def test_reloaded_model_answers_like_the_trained_one(project: Path) -> None:
    train.main()
    model = artifacts.FraudModel.load()
    frame = pd.read_parquet(build.SPLIT_FILES["validation"]).head(50)
    answers = model.assess(frame)
    probability, decided = model.decide(frame)
    assert [answer.probability for answer in answers] == [round(p, 2) for p in probability]
    labels = {"approve": "Aprovar", "review": "Revisão humana", "block": "Bloquear"}
    assert [answer.label for answer in answers] == [labels[a.decision] for a in answers]
    assert all(answer.reasons and answer.model_version == "1.0.0" for answer in answers)
    assert np.array_equal([("approve", "review", "block").index(a.decision) for a in answers],
                          decided)


def test_evaluate_runs_once_with_signals_from_the_past(project: Path) -> None:
    train.main()
    assert evaluate.main() == 0
    record = json.loads(evaluate.RECORD.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert "## Resultados principais" in evaluate.REPORT.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="used once"):
        evaluate.main()

    # the test signals equal an independent recomputation over train and test together
    past = pd.read_parquet(build.BASE_FILE)
    current, _ = prepare.prepare(profile.load(download.RAW_DIR / evaluate.TEST_FILE), KEY)
    combined = pd.concat([past, current], ignore_index=True)
    combined = combined.iloc[np.argsort(combined["timestamp"].to_numpy(), kind="stable")]
    combined = combined.reset_index(drop=True)
    stored = combined[["transaction_id"]].merge(pd.read_parquet(evaluate.TEST_OUTPUT),
                                                on="transaction_id", how="left", validate="1:1")
    test_rows = np.flatnonzero(stored["timestamp"].notna().to_numpy())
    sample = np.random.default_rng(3).choice(test_rows, size=60, replace=False)
    assert mismatches(combined, stored, sample) == []


def test_evaluate_refuses_a_model_changed_after_the_test_was_opened(project: Path) -> None:
    train.main()
    evaluate.write_record({"status": "opened", "bundle_sha256": "0" * 64, "attempts": []})
    with pytest.raises(SystemExit, match="another model bundle"):
        evaluate.main()
    assert not evaluate.TEST_OUTPUT.exists()


def test_evaluate_refuses_a_tampered_bundle_before_opening_the_test(project: Path) -> None:
    train.main()
    path = artifacts.BUNDLE_DIR / "decision.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["thresholds"]["block"] = 0.99
    path.write_text(artifacts.to_json(config), encoding="utf-8")
    with pytest.raises(artifacts.IntegrityError, match=r"decision\.json changed"):
        evaluate.main()
    assert not evaluate.RECORD.exists()
    assert not evaluate.TEST_OUTPUT.exists()
