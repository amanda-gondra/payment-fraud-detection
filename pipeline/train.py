"""Model development on train and validation only: fraudTest.csv stays closed.

    python -m pipeline.train

1. reads train and validation after checking their SHA-256 and cuts the validation period in
   two (D12): selection (January to March 2020) and calibration (April to June 21, 2020);
2. baselines (D13): the amount rule, fitted on train, and a small logistic regression grid;
3. LightGBM grid (D14), chosen by average precision on the selection period;
4. calibration (D15): none, Platt and isotonic, compared by cross-fitting on the calibration
   period; the best is fitted on the whole period;
5. decision thresholds (D16) that minimise the hypothetical costs, with a sensitivity analysis;
6. SHAP explanations (D19) and the fairness audit (D20) on the calibration period;
7. saves models/fraud-detector/ and models/baselines/, signed (D18), records everything in
   MLflow (mlruns/) and writes reports/model-selection.md: the choices frozen before the single
   evaluation on the test (D17).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import ModuleType
from typing import Any, Final

import numpy as np
import pandas as pd

from pipeline import download, features
from pipeline.modeling import (
    artifacts,
    baselines,
    calibration,
    data,
    decision,
    explain,
    fairness,
    lgbm,
    metrics,
    plots,
    tracking,
)
from pipeline.modeling.data import FeatureSpec, Split
from pipeline.modeling.evaluation import Scored
from pipeline.modeling.report_selection import render

log = logging.getLogger("pipeline.train")
REPORT: Final = download.ROOT / "reports" / "model-selection.md"
FIGURES: Final = download.ROOT / "reports" / "figures"
SHAP_SAMPLE: Final = 20_000
SEED: Final = 20_261_115
MODELS: Final = ("rule", "logistic", "lightgbm")


@dataclass
class Calibrated:
    method: str
    calibrator: calibration.Calibrator
    oof: dict[str, np.ndarray]  # cross-fitted probabilities of every method
    quality: dict[str, dict[str, float]]
    reliability: dict[str, pd.DataFrame]

    @property
    def probability(self) -> np.ndarray:
        return self.oof[self.method]


@dataclass
class Results:
    """Everything the report needs."""

    sets: dict[str, dict[str, Any]]
    values: dict[str, Any]


def finite(values: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """The numeric, finite entries of a dict, ready for mlflow.log_metrics."""
    return {f"{prefix}{key}": float(value) for key, value in values.items()
            if isinstance(value, int | float | np.integer | np.floating) and np.isfinite(value)}


def load_sets() -> tuple[FeatureSpec, dict[str, Split]]:
    frames = data.development_sets()
    spec = FeatureSpec.from_train(frames["train"])
    return spec, {name: Split.of(frame, spec) for name, frame in frames.items()}


def describe_sets(sets: dict[str, Split]) -> dict[str, dict[str, Any]]:
    return {
        name: {"first": split.frame["timestamp"].min(), "last": split.frame["timestamp"].max(),
               "transactions": len(split.y), "frauds": int(split.y.sum()),
               "fraud_rate": float(split.y.mean()), "cards": split.frame["card_key"].nunique()}
        for name, split in sets.items()
    }


def rule_stage(sets: dict[str, Split], mlflow: ModuleType) -> tuple[baselines.AmountRule, dict]:
    rule = baselines.AmountRule.fit(sets["train"].amount, sets["train"].y)
    results = {
        name: metrics.classification(split.y, rule.flag(split.amount))
        | {"average_precision": metrics.average_precision(split.y, split.amount)}
        for name, split in sets.items()
    }
    with mlflow.start_run(run_name="amount-rule", nested=True):
        mlflow.log_params({"model": "amount_rule", "threshold": rule.threshold,
                           "criterion": "maximum F1 on train", "balancing": "none"})
        for name, values in results.items():
            mlflow.log_metrics(finite(values, f"{name}_"))
    log.info("amount rule: X = %.2f", rule.threshold)
    return rule, results


def logistic_stage(
    sets: dict[str, Split], mlflow: ModuleType
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    inputs = {name: baselines.logistic_inputs(split.frame) for name, split in sets.items()}
    rows, models = [], []
    for config in baselines.LOGISTIC_GRID:
        start = time.perf_counter()
        model = baselines.logistic_model(config["C"], config["class_weight"])
        model.fit(inputs["train"], sets["train"].y)
        iterations = int(np.max(model[-1].n_iter_))
        row = {
            "C": config["C"],
            "class_weight": config["class_weight"] or "none",
            "iterations": iterations,
            "converged": iterations < baselines.MAX_ITER,
            "seconds": time.perf_counter() - start,
            "train_average_precision": metrics.average_precision(
                sets["train"].y, model.decision_function(inputs["train"])),
            "selection_average_precision": metrics.average_precision(
                sets["selection"].y, model.decision_function(inputs["selection"])),
        }
        with mlflow.start_run(run_name=f"logistic C={row['C']} weight={row['class_weight']}",
                              nested=True):
            mlflow.log_params({"model": "logistic_regression", "C": row["C"], "penalty": "l2",
                               "class_weight": row["class_weight"],
                               "balancing": f"class_weight={row['class_weight']}",
                               "max_iter": baselines.MAX_ITER})
            mlflow.log_metrics(finite(row))
        log.info("logistic C=%s weight=%s: selection AP %.4f (%.0f s)", row["C"],
                 row["class_weight"], row["selection_average_precision"], row["seconds"])
        rows.append(row)
        models.append(model)
    best = int(np.argmax([row["selection_average_precision"] for row in rows]))
    scores = {name: models[best].decision_function(inputs[name]) for name in sets}
    return models[best], rows[best], rows, scores


def config_key(config: dict[str, Any]) -> tuple[float, ...]:
    return (config["num_leaves"], config["min_child_samples"], config["lambda_l2"])


def config_key_columns(config: dict[str, Any]) -> dict[str, float]:
    return {name: config[name] for name in ("num_leaves", "min_child_samples", "lambda_l2")}


def lightgbm_stage(sets: dict[str, Split], spec: FeatureSpec,
                   mlflow: ModuleType) -> dict[str, Any]:
    """Every configuration on the four time windows; the one-standard-error rule picks one."""
    folds = data.time_folds(sets["train"], sets["selection"])
    windows = [{"window": label, "fit_first": fitted.frame["timestamp"].min(),
                "fit_last": fitted.frame["timestamp"].max(),
                "validation_first": checked.frame["timestamp"].min(),
                "validation_last": checked.frame["timestamp"].max(),
                "fit_transactions": len(fitted.y), "validation_transactions": len(checked.y),
                "validation_frauds": int(checked.y.sum())}
               for label, fitted, checked in folds]
    rows, finals = [], {}
    for config in lgbm.GRID:
        results = []
        for label, fitted, checked in folds:
            trained = lgbm.fit(fitted, checked, spec, config)
            results.append({"window": label, **lgbm.fold_result(trained, fitted, checked)})
        finals[config_key(config)] = trained  # last window: all of 2019, stopped on selection
        row = lgbm.summarize(config, results)
        name = (f"lightgbm leaves={config['num_leaves']} min_child={config['min_child_samples']}"
                f" l2={config['lambda_l2']:g}")
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params({**trained.params, "model": "lightgbm", "balancing": "none",
                               "cross_validation": f"{len(folds)} rolling-origin windows",
                               "final_trees": trained.best_iteration})
            mlflow.log_metrics(finite(row))
            for number, result in enumerate(results, start=1):
                mlflow.log_metrics(finite(result, f"window{number}_"))
        log.info("%s: CV AP %.4f ± %.4f, log loss gap %.1fx, %d trees in the last window "
                 "(%.0f s)", name, row["cv_average_precision"], row["cv_average_precision_se"],
                 row["cv_gap"], trained.best_iteration, row["seconds"])
        rows.append(row)
    chosen = lgbm.choose(rows)
    best = finals[config_key(chosen)]
    return {
        "trained": best,
        "best": chosen,
        "rows": rows,
        "windows": windows,
        "scores": {name: lgbm.raw_score(best.booster, split.x) for name, split in sets.items()},
        "curve": lgbm.learning_curve(best.booster, {"train": sets["train"],
                                                    "selection": sets["selection"]}),
    }


def calibration_stage(raw: np.ndarray, y: np.ndarray) -> Calibrated:
    oof, quality, tables = {}, {}, {}
    for method in calibration.METHODS:
        oof[method] = calibration.cross_fit(method, raw, y)
        quality[method] = metrics.probability_quality(y, oof[method])
        tables[method] = metrics.reliability(y, oof[method])
    chosen = min(calibration.METHODS, key=lambda method: quality[method]["log_loss"])
    return Calibrated(chosen, calibration.fit(chosen, raw, y), oof, quality, tables)


def decision_stage(
    cal: Split, main: Calibrated, logistic: Calibrated, rule: baselines.AmountRule
) -> dict[str, Any]:
    base = decision.BASE_COSTS
    chosen = decision.search(main.probability, cal.amount, cal.y, base)
    unlimited = decision.search(main.probability, cal.amount, cal.y,
                                replace(base, max_review_share=None))
    own = decision.search(logistic.probability, cal.amount, cal.y, base)
    half = decision.Thresholds(0.5, 0.5)
    rule_decisions = np.where(rule.flag(cal.amount), decision.REVIEW, decision.APPROVE)
    strategies = [
        ("LightGBM: limites escolhidos, revisão até 1%", chosen.thresholds,
         chosen.thresholds.decide(main.probability)),
        ("LightGBM: limites sem limite de capacidade", unlimited.thresholds,
         unlimited.thresholds.decide(main.probability)),
        ("LightGBM: só bloquear a partir de 0,5", half, half.decide(main.probability)),
        ("Regressão logística: limites próprios, revisão até 1%", own.thresholds,
         own.thresholds.decide(logistic.probability)),
        ("Regra do valor: revisar a partir de X", None, rule_decisions),
        ("Aprovar tudo", None, np.zeros(len(cal.y), dtype=np.int8)),
    ]
    rows = [(label, thresholds, decision.summarize(decided, cal.y, cal.amount, base))
            for label, thresholds, decided in strategies]
    sensitivity = []
    for label, costs in decision.SENSITIVITY:
        result = decision.search(main.probability, cal.amount, cal.y, costs)
        decided = result.thresholds.decide(main.probability)
        sensitivity.append((label, costs, result,
                            decision.summarize(decided, cal.y, cal.amount, costs)))
    return {"chosen": chosen, "logistic": own, "strategies": rows, "sensitivity": sensitivity,
            "decided": chosen.thresholds.decide(main.probability),
            "rule_decisions": rule_decisions}


def shap_stage(booster: Any, spec: FeatureSpec, split: Split, scored: Scored,
               figure: str) -> dict[str, Any]:
    """Global SHAP importance on a random sample, the summary figure and typical cases."""
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(split.y), size=min(SHAP_SAMPLE, len(split.y)),
                                replace=False))
    contribution = explain.contributions(booster, split.x[sample])
    explain.summary_plot(contribution, split.x[sample], spec, FIGURES / figure)
    return {
        "importance": explain.global_importance(contribution, spec, booster),
        "examples": explain.explain_cases(booster, spec, split, scored),
        "sample": len(sample),
    }


def fairness_stage(split: Split, decided: np.ndarray, audit: pd.DataFrame) -> dict[str, Any]:
    """Error rates by gender and age band; the attributes are joined here and nowhere else."""
    attributes = fairness.attributes(audit, split.frame["transaction_id"])
    results = {}
    for attribute in ("gender", "age_band"):
        frame = pd.DataFrame({"group": attributes[attribute].astype(str),
                              "card_key": split.frame["card_key"], "is_fraud": split.y,
                              "decision": decided})
        table, draws = fairness.audit(frame)
        reference = str(table.loc[table["transactions"].idxmax(), "group"])
        results[attribute] = {"table": table, "reference": reference,
                              "differences": fairness.differences(table, draws, reference)}
    return results


def comparison_stage(sets: dict[str, Split], selection_scores: dict[str, np.ndarray],
                     scored: dict[str, Scored]) -> dict[str, dict[str, float]]:
    """The three models side by side: selection and calibration periods, own thresholds."""
    cal = sets["calibration"]
    table = {}
    for model in MODELS:
        current = scored[model]
        at_target = metrics.recall_at_precision(cal.y, current.score)
        row = {
            "selection_average_precision": metrics.average_precision(
                sets["selection"].y, selection_scores[model]),
            "average_precision": metrics.average_precision(cal.y, current.score),
            "recall_at_target_precision": at_target["recall"],
            "target_precision_reached": at_target["precision"],
        }
        if current.probability is not None:
            row |= metrics.probability_quality(cal.y, current.probability)
        summary = decision.summarize(current.decisions, cal.y, cal.amount, decision.BASE_COSTS)
        row |= {"total_cost": summary["total_cost"], "frauds_flagged": summary["frauds_flagged"],
                "false_positives": summary["review_legit"] + summary["block_legit"]}
        table[model] = row
    return table


def develop(spec: FeatureSpec, sets: dict[str, Split], mlflow: ModuleType) -> dict[str, Any]:
    """Every development stage, in order, timed."""
    cal = sets["calibration"]
    timings: dict[str, float] = {}
    stage = time.perf_counter()
    rule, rule_results = rule_stage(sets, mlflow)
    logistic_model, logistic_best, logistic_rows, logistic_scores = logistic_stage(sets, mlflow)
    timings["baselines_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    lightgbm = lightgbm_stage(sets, spec, mlflow)
    trained, lightgbm_scores = lightgbm["trained"], lightgbm["scores"]
    timings["lightgbm_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    main_cal = calibration_stage(lightgbm_scores["calibration"], cal.y)
    logistic_cal = calibration_stage(logistic_scores["calibration"], cal.y)
    decided = decision_stage(cal, main_cal, logistic_cal, rule)
    timings["calibration_decision_seconds"] = time.perf_counter() - stage

    scored = {
        "rule": Scored(cal.amount, None, decided["rule_decisions"]),
        "logistic": Scored(logistic_scores["calibration"], logistic_cal.probability,
                           decided["logistic"].thresholds.decide(logistic_cal.probability)),
        "lightgbm": Scored(lightgbm_scores["calibration"], main_cal.probability,
                           decided["decided"]),
    }
    stage = time.perf_counter()
    shap = shap_stage(trained.booster, spec, cal, scored["lightgbm"], "shap-summary.png")
    audit = data.read_verified("audit")
    equity = fairness_stage(cal, decided["decided"], audit)
    del audit  # the attributes leave memory as soon as the audit is done
    timings["explain_fairness_seconds"] = time.perf_counter() - stage

    selection_scores = {"rule": sets["selection"].amount,
                        "logistic": logistic_scores["selection"],
                        "lightgbm": lightgbm_scores["selection"]}
    labels = {"Sem calibração": "none", "Isotônica": "isotonic", "Platt": "platt"}
    plots.reliability({label: main_cal.reliability[method] for label, method in labels.items()},
                      FIGURES / "calibration.png",
                      "LightGBM: calibração por validação cruzada no período de calibração")
    plots.precision_recall(
        {"Regra do valor": (cal.y, cal.amount), "LightGBM": (cal.y, lightgbm_scores["calibration"]),
         "Regressão logística": (cal.y, logistic_scores["calibration"])},
        FIGURES / "precision-recall.png", "Precisão e recall no período de calibração")
    plots.learning_curve(lightgbm["curve"], FIGURES / "learning-curve.png",
                         "LightGBM escolhido: log loss conforme as árvores são somadas")
    return {
        "spec": spec, "rule": rule, "rule_results": rule_results,
        "logistic_model": logistic_model, "logistic_rows": logistic_rows,
        "logistic_best": logistic_best, "logistic_scores": logistic_scores,
        "lightgbm_rows": lightgbm["rows"], "lightgbm_best": lightgbm["best"],
        "windows": lightgbm["windows"], "learning_curve": lightgbm["curve"], "trained": trained,
        "lightgbm_scores": lightgbm_scores, "main_cal": main_cal, "logistic_cal": logistic_cal,
        "decision": decided, "thresholds": decided["chosen"].thresholds,
        "comparison": comparison_stage(sets, selection_scores, scored), "shap": shap,
        "fairness": equity, "timings": timings,
    }


def save_bundles(values: dict[str, Any], cal: Split, run_id: str) -> tuple[dict, dict]:
    """Save, reload and check the main model and the baselines."""
    trained, main_cal = values["trained"], values["main_cal"]
    metadata = {
        "model_version": artifacts.MODEL_VERSION,
        "created_at": values["created"],
        "created_by": "python -m pipeline.train",
        "model": "LightGBM binary classifier, calibrated",
        "intended_use": "demonstration on simulated data (Sparkov); not for real payments",
        "trees": trained.best_iteration,
        "params": trained.params,
        "features": list(features.FEATURE_COLUMNS),
        "selection": {
            "method": "rolling-origin cross-validation in time, one-standard-error rule",
            "windows": [label for label, _, _ in data.CV_WINDOWS] + [data.FINAL_WINDOW],
            "cv_average_precision": values["lightgbm_best"]["cv_average_precision"],
            "cv_average_precision_se": values["lightgbm_best"]["cv_average_precision_se"],
            "selection_average_precision":
                values["lightgbm_best"]["folds"][-1]["average_precision"],
        },
        "calibration": {"method": main_cal.method,
                        "fitted_on": f"{data.CALIBRATION_START:%Y-%m-%d} to 2020-06-21",
                        "cross_fit_blocks": calibration.BLOCKS},
        "thresholds": values["thresholds"].to_dict(),
        "costs": decision.BASE_COSTS.to_dict(),
        "data": data.data_version(),
        "libraries": tracking.libraries(),
        "code": {**tracking.git_state(), "code_sha256": tracking.code_digest()},
        "mlflow_run_id": run_id,
    }
    manifest = artifacts.save_model(artifacts.ModelParts(
        trained.booster, values["spec"], main_cal.calibrator, values["thresholds"],
        decision.BASE_COSTS.to_dict(), metadata))
    expected = main_cal.calibrator.apply(values["lightgbm_scores"]["calibration"])
    if not np.array_equal(artifacts.FraudModel.load().probability(cal.frame), expected):
        raise SystemExit("the reloaded model predicts differently from the trained one")
    baseline_manifest = artifacts.Baselines(
        rule=values["rule"], logistic=values["logistic_model"],
        calibrator=values["logistic_cal"].calibrator,
        thresholds=values["decision"]["logistic"].thresholds,
        config={"C": values["logistic_best"]["C"],
                "class_weight": values["logistic_best"]["class_weight"]},
    ).save()
    reloaded = artifacts.Baselines.load()
    if not np.array_equal(baselines.logistic_raw_score(reloaded.logistic, cal.frame),
                          values["logistic_scores"]["calibration"]):
        raise SystemExit("the reloaded logistic baseline predicts differently")
    return manifest, baseline_manifest


def log_run(mlflow: ModuleType, values: dict[str, Any]) -> None:
    """Parameters, aggregated metrics and model files of the parent run (no transactions)."""
    thresholds, main_cal = values["thresholds"], values["main_cal"]
    mlflow.log_params({
        "model": "lightgbm", "calibration_method": main_cal.method,
        "threshold_review": thresholds.review, "threshold_block": thresholds.block,
        "decision_strategy": "three bands chosen by minimum hypothetical cost",
        **{f"cost_{key}": value for key, value in decision.BASE_COSTS.to_dict().items()},
        "bundle_sha256": values["manifest"]["bundle_sha256"],
    })
    for model, row in values["comparison"].items():
        mlflow.log_metrics(finite(row, f"calibration_{model}_"))
    mlflow.log_metrics(finite(main_cal.quality[main_cal.method], "calibrated_"))
    mlflow.log_metrics(finite(values["timings"]))
    mlflow.log_artifacts(str(artifacts.BUNDLE_DIR), "fraud-detector")
    mlflow.log_artifacts(str(artifacts.BASELINE_DIR), "baselines")
    mlflow.log_artifacts(str(FIGURES), "figures")
    mlflow.log_table(values["shap"]["importance"], "tables/shap_importance.json")
    grid = [{key: value for key, value in row.items() if key != "folds"}
            for row in values["lightgbm_rows"]]
    windows = [{**config_key_columns(row), **fold} for row in values["lightgbm_rows"]
               for fold in row["folds"]]
    mlflow.log_table(pd.DataFrame(grid), "tables/lightgbm_grid.json")
    mlflow.log_table(pd.DataFrame(windows), "tables/lightgbm_windows.json")
    mlflow.log_table(pd.DataFrame(values["learning_curve"]), "tables/learning_curve.json")
    mlflow.log_table(pd.DataFrame(values["logistic_rows"]), "tables/logistic_grid.json")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    started = time.perf_counter()
    spec, sets = load_sets()
    mlflow = tracking.mlflow_client()
    with mlflow.start_run(run_name="model-selection") as run:
        mlflow.set_tags({**data.data_version(), **tracking.git_state(),
                         "code_sha256": tracking.code_digest(), "phase": "4",
                         "test_set": "closed (fraudTest.csv not read)"})
        mlflow.log_dict(spec.to_dict(), "features.json")
        mlflow.log_params({"features": len(spec.columns),
                           "calibration_start": f"{data.CALIBRATION_START:%Y-%m-%d}",
                           "validation_start": "2020-01-01"})
        values = develop(spec, sets, mlflow)
        values["created"] = datetime.now(UTC).isoformat(timespec="seconds")
        values["run_id"] = run.info.run_id
        values["manifest"], values["baseline_manifest"] = save_bundles(
            values, sets["calibration"], run.info.run_id)
        values["timings"]["total_seconds"] = time.perf_counter() - started
        log_run(mlflow, values)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(render(Results(describe_sets(sets), values)), encoding="utf-8",
                          newline="\n")
        mlflow.set_tag("report", REPORT.relative_to(download.ROOT).as_posix())

    best, thresholds = values["lightgbm_best"], values["thresholds"]
    log.info("chosen: LightGBM %s, %s calibration, review >= %.4f, block >= %.4f",
             config_key_columns(best),
             values["main_cal"].method, thresholds.review, thresholds.block)
    log.info("bundle %s, run %s, %.0f s", values["manifest"]["bundle_sha256"][:12],
             run.info.run_id, values["timings"]["total_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
