"""Metrics for a rare event: ranking, decisions at a threshold and quality of the probabilities.

Accuracy is never used: approving everything is right 99.4% of the time and catches nothing.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn import metrics as sk

PRECISION_TARGET: Final = 0.90  # pre-registered before the test is opened (decision D17)
RELIABILITY_EDGES: Final = (
    0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)
EPSILON: Final = 1e-15
Z_95: Final = 1.959964


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision): ranking quality for frauds."""
    return float(sk.average_precision_score(y, score))


def brier(y: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((probability - y) ** 2))


def log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, EPSILON, 1 - EPSILON)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def classification(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    """Confusion counts, precision, recall and F1 of a yes/no decision."""
    y = y.astype(bool)
    flagged = flagged.astype(bool)
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    tn = int(np.sum(~flagged & ~y))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": f1}


def recall_at_precision(
    y: np.ndarray, score: np.ndarray, target: float = PRECISION_TARGET
) -> dict[str, float]:
    """Highest recall among the score cuts whose precision reaches the target."""
    precision, recall, thresholds = sk.precision_recall_curve(y, score)
    # the last point (precision 1, recall 0) has no threshold
    ok = precision[:-1] >= target
    if not ok.any():
        return {"recall": 0.0, "precision": float("nan"), "threshold": float("nan")}
    best = np.flatnonzero(ok)[np.argmax(recall[:-1][ok])]
    return {"recall": float(recall[best]), "precision": float(precision[best]),
            "threshold": float(thresholds[best])}


def wilson(successes: float, n: float, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson interval of a proportion (good with few events)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def reliability(
    y: np.ndarray, probability: np.ndarray, edges: tuple[float, ...] = RELIABILITY_EDGES
) -> pd.DataFrame:
    """Observed fraud rate per band of predicted probability, with a 95% interval."""
    bands = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for band in range(len(edges) - 1):
        inside = bands == band
        n = int(inside.sum())
        if not n:
            continue
        frauds = float(y[inside].sum())
        low, high = wilson(frauds, n)
        rows.append({
            "band": f"{edges[band]:g}-{edges[band + 1]:g}",
            "transactions": n,
            "mean_predicted": float(probability[inside].mean()),
            "observed_rate": frauds / n,
            "low": low,
            "high": high,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Gap between predicted and observed rates, weighted by the transactions of each band."""
    weights = table["transactions"] / table["transactions"].sum()
    return float(np.sum(weights * (table["mean_predicted"] - table["observed_rate"]).abs()))


def probability_quality(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier(y, probability),
        "log_loss": log_loss(y, probability),
        "ece": expected_calibration_error(reliability(y, probability)),
    }
