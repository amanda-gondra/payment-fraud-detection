"""The single evaluation on fraudTest.csv (decision D17).

    python -m pipeline.evaluate

Runs after python -m pipeline.train has frozen the model, the calibrator and the thresholds
(reports/model-selection.md). Nothing it measures goes back into the model in this phase.

1. verifies the signed model and baselines (SHA-256) and the guard file: the test is evaluated
   for one frozen model only, and never again once the evaluation is complete;
2. checks fraudTest.csv against data/manifest.json and prepares it with the same code and the
   same key as the build, so a card keeps its pseudonym across the two files;
3. computes the test signals with the training file as the past (D8, D11) and writes
   data/processed/test.parquet;
4. scores the three models and applies their frozen thresholds;
5. writes the pre-registered metrics, by month and by category, the SHAP explanations and the
   fairness audit to reports/test-evaluation.md, and the aggregates to MLflow.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from types import ModuleType
from typing import Any, Final

import numpy as np
import pandas as pd

from pipeline import build, download, features, prepare, profile
from pipeline.modeling import (
    artifacts,
    baselines,
    data,
    decision,
    evaluation,
    lgbm,
    metrics,
    plots,
    tracking,
)
from pipeline.modeling.data import Split
from pipeline.modeling.evaluation import Scored
from pipeline.modeling.report_test import render
from pipeline.train import FIGURES, MODELS, fairness_stage, finite, shap_stage

log = logging.getLogger("pipeline.evaluate")
TEST_FILE: Final = "fraudTest.csv"
TEST_OUTPUT: Final = build.PROCESSED / "test.parquet"
RECORD: Final = artifacts.MODELS_DIR / "test-evaluation.json"
REPORT: Final = download.ROOT / "reports" / "test-evaluation.md"
TEST_FIGURES: Final = ("calibration-test.png", "precision-recall-test.png",
                       "shap-summary-test.png")


def check_guard(bundle_sha256: str) -> list[str]:
    """Earlier attempts; stop if the evaluation is complete or the model changed since."""
    if not RECORD.is_file():
        return []
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    if record.get("status") == "completed":
        raise SystemExit(f"the test was already evaluated on {record['completed_at']} "
                         f"({RECORD.relative_to(download.ROOT).as_posix()}): it is used once")
    if record.get("bundle_sha256") != bundle_sha256:
        raise SystemExit("the test was opened for another model bundle: a model changed after "
                         "seeing the test cannot be evaluated on it")
    return list(record.get("attempts", []))


def write_record(record: dict[str, Any]) -> None:
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(artifacts.to_json(record), encoding="utf-8", newline="\n")


def test_signals(key: bytes) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Signals of every test transaction, with the training file as its past."""
    built = data.processed_manifest()
    if built["card_key"]["key_fingerprint"] != prepare.key_fingerprint(key):
        raise SystemExit(f"{prepare.KEY_VARIABLE} differs from the key used by "
                         "python -m pipeline.build: the card histories would not connect")
    source_sha256 = prepare.verify_raw_file(TEST_FILE)
    past = data.read_verified("base")
    raw = profile.load(download.RAW_DIR / TEST_FILE)
    current, audit = prepare.prepare(raw, key)
    del raw  # personal columns leave memory as soon as the base is built
    if set(current["transaction_id"]) & set(past["transaction_id"]):
        raise SystemExit("some test transactions are also in the training file")
    combined = pd.concat([past.assign(is_test=False), current.assign(is_test=True)],
                         ignore_index=True)
    combined = combined.iloc[np.argsort(combined["timestamp"].to_numpy(), kind="stable")]
    combined = combined.reset_index(drop=True)
    for column in ("card_key", "category"):
        combined[column] = combined[column].astype(str).astype("category")
    signals = features.build_features(combined)
    dataset = combined[list(build.ID_COLUMNS)].join(signals).join(combined[[build.TARGET]])
    test = dataset[combined["is_test"].to_numpy()].reset_index(drop=True)
    return test, audit, source_sha256


def score(model: artifacts.FraudModel, reference: artifacts.Baselines,
          split: Split) -> dict[str, Scored]:
    """The three models with their frozen thresholds."""
    raw = lgbm.raw_score(model.booster, split.x)
    probability = model.calibrator.apply(raw)
    logistic_raw = baselines.logistic_raw_score(reference.logistic, split.frame)
    logistic_probability = reference.calibrator.apply(logistic_raw)
    rule = np.where(reference.rule.flag(split.amount), decision.REVIEW,
                    decision.APPROVE).astype(np.int8)
    return {
        "rule": Scored(split.amount, None, rule),
        "logistic": Scored(logistic_raw, logistic_probability,
                           reference.thresholds.decide(logistic_probability)),
        "lightgbm": Scored(raw, probability, model.thresholds.decide(probability)),
    }


def measure(model: artifacts.FraudModel, split: Split, scored: dict[str, Scored],
            audit: pd.DataFrame) -> dict[str, Any]:
    """Every pre-registered result, the figures and the explanations."""
    costs = decision.Costs(**model.metadata["costs"])
    main = scored["lightgbm"]
    test = split.frame
    reliability = {"Regressão logística": metrics.reliability(split.y,
                                                             scored["logistic"].probability),
                   "LightGBM": metrics.reliability(split.y, main.probability)}
    plots.reliability(reliability, FIGURES / TEST_FIGURES[0], "Calibração no teste",
                      colours=(plots.SERIES[2], plots.SERIES[1]))
    plots.precision_recall(
        {"Regra do valor": (split.y, scored["rule"].score), "LightGBM": (split.y, main.score),
         "Regressão logística": (split.y, scored["logistic"].score)},
        FIGURES / TEST_FIGURES[1], "Precisão e recall no teste")
    return {
        "overall": {name: evaluation.overall(split, scored[name], costs) for name in MODELS},
        "months": evaluation.by_group(test["timestamp"].dt.strftime("%Y-%m"), split, main,
                                      costs),
        "categories": evaluation.by_group(test["category"].astype(str), split, main, costs),
        "sensitivity": [(label, decision.summarize(main.decisions, split.y, split.amount,
                                                   variant))
                        for label, variant in decision.SENSITIVITY],
        "reliability": reliability,
        "shap": shap_stage(model.booster, model.spec, split, main, TEST_FIGURES[2]),
        "fairness": fairness_stage(split, main.decisions, audit),
        "costs": costs,
    }


def log_run(mlflow: ModuleType, model: artifacts.FraudModel, values: dict[str, Any]) -> str:
    """Aggregates, tables and figures of the evaluation (no transactions)."""
    record = values["record"]
    with mlflow.start_run(run_name="test-evaluation") as run:
        mlflow.set_tags({"phase": "4", "training_run_id": model.metadata["mlflow_run_id"],
                         "bundle_sha256": record["bundle_sha256"],
                         "model_version": model.version,
                         "test_source_sha256": record["test_source_sha256"],
                         **tracking.git_state(), "code_sha256": tracking.code_digest()})
        for name, result in values["overall"].items():
            mlflow.log_metrics(finite(result, f"test_{name}_"))
        mlflow.log_metric("evaluation_seconds", values["seconds"])
        mlflow.log_table(values["months"], "tables/test_months.json")
        mlflow.log_table(values["categories"], "tables/test_categories.json")
        mlflow.log_table(values["shap"]["importance"], "tables/test_shap_importance.json")
        for attribute, result in values["fairness"].items():
            mlflow.log_table(result["table"], f"tables/test_fairness_{attribute}.json")
        for figure in TEST_FIGURES:
            mlflow.log_artifact(str(FIGURES / figure), "figures")
        return str(run.info.run_id)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    started = time.perf_counter()
    model = artifacts.FraudModel.load()
    reference = artifacts.Baselines.load()
    baseline_sha256 = artifacts.read_signed(artifacts.BASELINE_DIR,
                                            artifacts.BASELINE_FILES)[0]["bundle_sha256"]
    bundle_sha256 = model.manifest["bundle_sha256"]
    attempts = check_guard(bundle_sha256)
    opened = datetime.now(UTC).isoformat(timespec="seconds")
    record = {"status": "opened", "model_version": model.version, "bundle_sha256": bundle_sha256,
              "baselines_sha256": baseline_sha256, "attempts": [*attempts, opened]}
    write_record(record)  # from here on the test counts as seen by this model

    test, audit, source_sha256 = test_signals(prepare.load_key())
    output = build.write_parquet(test, TEST_OUTPUT)
    split = Split.of(test, model.spec)
    log.info("test: %d transactions, %d frauds", len(split.y), int(split.y.sum()))
    values = measure(model, split, score(model, reference, split), audit)
    del audit

    completed = datetime.now(UTC).isoformat(timespec="seconds")
    headline = {key: values["overall"]["lightgbm"][key] for key in (
        "average_precision", "recall_at_target_precision", "alert_precision", "alert_recall",
        "alert_f1", "brier", "log_loss", "total_cost")}
    record |= {"status": "completed", "completed_at": completed,
               "test_source_sha256": source_sha256, "test_features": output,
               "precision_target": metrics.PRECISION_TARGET, "headline": headline}
    values |= {
        "completed": completed, "record": record, "thresholds": model.thresholds,
        "model": model, "rule": reference.rule, "seconds": time.perf_counter() - started,
        "period": {"first": test["timestamp"].min(), "last": test["timestamp"].max(),
                   "transactions": len(split.y), "frauds": int(split.y.sum()),
                   "cards": test["card_key"].nunique()},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(values), encoding="utf-8", newline="\n")
    record["mlflow_run_id"] = log_run(tracking.mlflow_client(), model, values)
    write_record(record)
    log.info("test evaluation complete: PR-AUC %.4f, run %s",
             values["overall"]["lightgbm"]["average_precision"], record["mlflow_run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
