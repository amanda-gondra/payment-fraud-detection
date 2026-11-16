"""Results of a scored set: overall, by month and by category (used by train and evaluate).

Alerted means review or block: it is the yes/no reading of the three decisions used for
precision, recall and F1. Blocked is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from pipeline.modeling import decision, metrics
from pipeline.modeling.data import Split

MIN_FRAUDS: Final = 30  # below this, a group's precision and recall are flagged as unreliable


@dataclass(frozen=True)
class Scored:
    """One model's output on a set, row by row."""

    score: np.ndarray  # ranking score (raw log-odds, or the amount for the rule)
    probability: np.ndarray | None  # calibrated probability; None for the rule
    decisions: np.ndarray  # 0 approve, 1 review, 2 block


def overall(split: Split, scored: Scored, costs: decision.Costs) -> dict[str, float]:
    """Every pre-registered metric of one model on one set."""
    y = split.y
    result: dict[str, float] = {"transactions": len(y), "frauds": int(y.sum()),
                                "average_precision": metrics.average_precision(y, scored.score)}
    at_target = metrics.recall_at_precision(y, scored.score)
    result["recall_at_target_precision"] = at_target["recall"]
    result["target_precision_reached"] = at_target["precision"]
    if scored.probability is not None:
        result |= metrics.probability_quality(y, scored.probability)
    alerted = metrics.classification(y, scored.decisions >= decision.REVIEW)
    blocked = metrics.classification(y, scored.decisions == decision.BLOCK)
    result |= {f"alert_{name}": value for name, value in alerted.items()}
    result |= {f"block_{name}": value for name, value in blocked.items()}
    result |= decision.summarize(scored.decisions, y, split.amount, costs)
    return result


def by_group(groups: pd.Series, split: Split, scored: Scored,
             costs: decision.Costs) -> pd.DataFrame:
    """Volume, frauds, ranking, decisions and cost per group (month or category)."""
    rows = []
    labels = pd.Series(groups).reset_index(drop=True)
    for group in labels.drop_duplicates().sort_values():
        inside = (labels == group).to_numpy()
        y, decided = split.y[inside], scored.decisions[inside]
        frauds = int(y.sum())
        alerted = metrics.classification(y, decided >= decision.REVIEW)
        summary = decision.summarize(decided, y, split.amount[inside], costs)
        rows.append({
            "group": group,
            "transactions": len(y),
            "frauds": frauds,
            "fraud_rate": frauds / len(y),
            "average_precision": metrics.average_precision(y, scored.score[inside])
            if 0 < frauds < len(y) else float("nan"),
            "precision": alerted["precision"],
            "recall": alerted["recall"],
            "frauds_flagged": alerted["tp"],
            "false_positives": alerted["fp"],
            "approve_share": summary["approve_transactions"] / len(y),
            "review_share": summary["review_transactions"] / len(y),
            "block_share": summary["block_transactions"] / len(y),
            "total_cost": summary["total_cost"],
            "reliable": frauds >= MIN_FRAUDS,
        })
    return pd.DataFrame(rows)
