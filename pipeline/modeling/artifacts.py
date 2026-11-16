"""Signed model bundles, saved without pickle (decision D18).

models/fraud-detector/ holds everything inference needs:

- model.txt: LightGBM's own text format (trees and numbers, no code);
- features.json: order of the 28 columns and the code of each category;
- calibrator.json: calibration method and its numbers;
- decision.json: thresholds, decision names and the cost hypotheses behind them;
- metadata.json: version, data signatures, parameters, libraries and code version;
- manifest.json: SHA-256 of each file above and of the whole bundle.

models/baselines/ holds the amount rule and the logistic regression (skops, trusted types only)
with their calibrator and thresholds, signed the same way, so the test can score all three.

Loading reads each file once, checks its SHA-256 before parsing those same bytes, and refuses
missing or unexpected files; afterwards it checks the files again and the model's columns
against features.json. No secret is stored: the model never uses card_key and the pseudonym key
stays in .env.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from pipeline import download
from pipeline.modeling import baselines, calibration, explain
from pipeline.modeling.data import FeatureSpec
from pipeline.modeling.decision import DECISION_LABELS, DECISIONS, Thresholds
from pipeline.modeling.lgbm import NUM_THREADS

MODELS_DIR: Final = download.ROOT / "models"
BUNDLE_DIR: Final = MODELS_DIR / "fraud-detector"
BASELINE_DIR: Final = MODELS_DIR / "baselines"
MODEL_VERSION: Final = "1.0.0"
MANIFEST_FILE: Final = "manifest.json"
FILES: Final = ("model.txt", "features.json", "calibrator.json", "decision.json", "metadata.json")
BASELINE_FILES: Final = ("amount-rule.json", "logistic.skops", "logistic.json")
PROBABILITY_DIGITS: Final = 2  # rounded in responses, so the thresholds cannot be probed finely


class IntegrityError(RuntimeError):
    """A bundle file is missing, unexpected or changed since it was signed."""


def to_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_digest(files: dict[str, str]) -> str:
    return sha256("".join(f"{name}:{files[name]}\n" for name in sorted(files)).encode())


def write_signed(directory: Path, contents: dict[str, bytes], header: dict[str, Any]) -> dict:
    """Write the files, then manifest.json with their SHA-256, then check them back."""
    directory.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in directory.iterdir()
                  if path.is_file() and path.name not in (*contents, MANIFEST_FILE)]
    if unexpected:
        raise IntegrityError(f"unexpected files in {directory}: {sorted(unexpected)}")
    for name, data in contents.items():
        (directory / name).write_bytes(data)
    files = {name: sha256((directory / name).read_bytes()) for name in contents}
    manifest = {**header, "files": files, "bundle_sha256": bundle_digest(files)}
    (directory / MANIFEST_FILE).write_bytes(to_json(manifest).encode())
    read_signed(directory, tuple(contents))
    return manifest


def read_signed(directory: Path, names: tuple[str, ...]) -> tuple[dict, dict[str, bytes]]:
    """The manifest and the verified bytes of each file; IntegrityError on any mismatch."""
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise IntegrityError(f"{path} is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if set(files) != set(names):
        raise IntegrityError(f"{path.name} does not list exactly {sorted(names)}")
    present = {item.name for item in directory.iterdir() if item.is_file()} - {MANIFEST_FILE}
    if present != set(names):
        raise IntegrityError(f"files differ from the manifest: {sorted(present ^ set(names))}")
    if bundle_digest(files) != manifest.get("bundle_sha256"):
        raise IntegrityError("bundle_sha256 does not match the listed files")
    contents = {}
    for name in names:
        data = (directory / name).read_bytes()
        if sha256(data) != files[name]:
            raise IntegrityError(f"{name} changed: its SHA-256 differs from {path.name}")
        contents[name] = data
    return manifest, contents


def decision_config(thresholds: Thresholds, costs: dict[str, Any]) -> dict[str, Any]:
    return {
        "probability": "probability of fraud: calibrator.json applied to the raw score",
        "thresholds": thresholds.to_dict(),
        "rule": "p < review: approve; review <= p < block: review; p >= block: block",
        "decisions": [{"code": code, "name": name, "label": DECISION_LABELS[name]}
                      for code, name in enumerate(DECISIONS)],
        "costs": costs,
    }


@dataclass(frozen=True)
class ModelParts:
    """What python -m pipeline.train hands over to be saved."""

    booster: lgb.Booster
    spec: FeatureSpec
    calibrator: calibration.Calibrator
    thresholds: Thresholds
    costs: dict[str, Any]
    metadata: dict[str, Any]


def save_model(parts: ModelParts, directory: Path | None = None) -> dict:
    contents = {
        "model.txt": parts.booster.model_to_string().encode(),
        "features.json": to_json(parts.spec.to_dict()).encode(),
        "calibrator.json": to_json(parts.calibrator.to_dict()).encode(),
        "decision.json": to_json(decision_config(parts.thresholds, parts.costs)).encode(),
        "metadata.json": to_json(parts.metadata).encode(),
    }
    header = {"bundle": "fraud-detector", "model_version": parts.metadata["model_version"],
              "created_at": parts.metadata["created_at"]}
    return write_signed(directory or BUNDLE_DIR, contents, header)


@dataclass(frozen=True)
class Assessment:
    """What the system answers about one transaction."""

    decision: str
    label: str
    probability: float
    reasons: list[str]
    model_version: str


@dataclass(frozen=True)
class FraudModel:
    booster: lgb.Booster
    spec: FeatureSpec
    calibrator: calibration.Calibrator
    thresholds: Thresholds
    metadata: dict[str, Any]
    manifest: dict[str, Any]

    @classmethod
    def load(cls, directory: Path | None = None) -> FraudModel:
        directory = directory or BUNDLE_DIR
        manifest, contents = read_signed(directory, FILES)
        text = {name: data.decode("utf-8") for name, data in contents.items()}
        booster = lgb.Booster(model_str=text["model.txt"])
        spec = FeatureSpec.from_dict(json.loads(text["features.json"]))
        model = cls(
            booster=booster,
            spec=spec,
            calibrator=calibration.from_dict(json.loads(text["calibrator.json"])),
            thresholds=Thresholds(**json.loads(text["decision.json"])["thresholds"]),
            metadata=json.loads(text["metadata.json"]),
            manifest=manifest,
        )
        if booster.feature_name() != list(spec.columns):
            raise IntegrityError("the model's columns differ from features.json")
        if booster.model_to_string() != text["model.txt"]:
            raise IntegrityError("the loaded model differs from model.txt")
        if read_signed(directory, FILES)[0] != manifest:
            raise IntegrityError("the bundle changed while it was loaded")
        return model

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    def raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self.spec.matrix(frame), raw_score=True,
                                    num_threads=NUM_THREADS)

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.apply(self.raw_score(frame))

    def decide(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        probability = self.probability(frame)
        return probability, self.thresholds.decide(probability)

    def assess(self, frame: pd.DataFrame, top: int = explain.TOP_REASONS) -> list[Assessment]:
        """Decision, rounded probability and reasons for each transaction."""
        x = self.spec.matrix(frame)
        probability = self.calibrator.apply(
            self.booster.predict(x, raw_score=True, num_threads=NUM_THREADS))
        decided = self.thresholds.decide(probability)
        contribution = explain.contributions(self.booster, x)
        answers = []
        for row in range(len(x)):
            name = DECISIONS[int(decided[row])]
            found = explain.reasons(x[row], contribution[row], self.spec, top)
            answers.append(Assessment(
                decision=name,
                label=DECISION_LABELS[name],
                probability=round(float(probability[row]), PROBABILITY_DIGITS),
                reasons=[reason.text for reason in found],
                model_version=self.version,
            ))
        return answers


@dataclass(frozen=True)
class Baselines:
    rule: baselines.AmountRule
    logistic: Pipeline
    calibrator: calibration.Calibrator
    thresholds: Thresholds
    config: dict[str, Any]

    def save(self, directory: Path | None = None) -> dict:
        directory = directory or BASELINE_DIR
        logistic = {"config": self.config, "calibrator": self.calibrator.to_dict(),
                    "thresholds": self.thresholds.to_dict()}
        contents = {
            "amount-rule.json": to_json(self.rule.to_dict()).encode(),
            "logistic.skops": baselines.dump_logistic(self.logistic),
            "logistic.json": to_json(logistic).encode(),
        }
        return write_signed(directory, contents, {"bundle": "baselines"})

    @classmethod
    def load(cls, directory: Path | None = None) -> Baselines:
        _, contents = read_signed(directory or BASELINE_DIR, BASELINE_FILES)
        rule = json.loads(contents["amount-rule.json"])
        logistic = json.loads(contents["logistic.json"])
        return cls(
            rule=baselines.AmountRule(threshold=float(rule["threshold"])),
            logistic=baselines.load_logistic(contents["logistic.skops"]),
            calibrator=calibration.from_dict(logistic["calibrator"]),
            thresholds=Thresholds(**logistic["thresholds"]),
            config=logistic["config"],
        )
