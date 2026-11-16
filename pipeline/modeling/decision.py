"""Cost-based decision: approve, send to human review or block (decision D16).

The costs are hypotheses, not measured values: a simulated base has no real cost data. The
threshold search is repeated under other hypotheses in a sensitivity analysis.

Cost of each decision for one transaction of amount a, in the units of the base:

| decision | legitimate | fraud |
| --- | --- | --- |
| approve | 0 | a (financial loss) |
| review | review + review_friction | review + (1 - review_detection) * a |
| block | block_legit | 0 (the loss is avoided) |

review is operational (an analyst's time), review_friction and block_legit are costs to the
customer's experience and to the business, a is financial.

Two probability thresholds split the transactions: p < review: approve; review <= p < block:
human review; p >= block: block. They are chosen on the calibration period by minimising the
total cost, with at most max_review_share of the transactions in review (team capacity).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Final

import numpy as np

DECISIONS: Final = ("approve", "review", "block")
DECISION_LABELS: Final = {"approve": "Aprovar", "review": "Revisão humana", "block": "Bloquear"}
APPROVE: Final = 0
REVIEW: Final = 1
BLOCK: Final = 2
MAX_CANDIDATES: Final = 1_500


@dataclass(frozen=True)
class Costs:
    review: float = 5.0  # operational: one manual review
    review_friction: float = 2.0  # experience: a legitimate customer waits for the review
    block_legit: float = 25.0  # experience and business: a legitimate purchase is refused
    review_detection: float = 0.9  # share of the reviewed frauds the analysts stop
    max_review_share: float | None = 0.01  # capacity: at most 1% of the transactions in review

    def per_decision(self, amount: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Cost of approving, reviewing and blocking each transaction (one column each)."""
        fraud = np.asarray(y, dtype=float)
        legit = 1.0 - fraud
        return np.column_stack([
            fraud * amount,
            self.review + legit * self.review_friction
            + fraud * (1.0 - self.review_detection) * amount,
            legit * self.block_legit,
        ])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BASE_COSTS: Final = Costs()
SENSITIVITY: Final = (
    ("Hipótese base", BASE_COSTS),
    ("Revisão custa 2", replace(BASE_COSTS, review=2.0)),
    ("Revisão custa 10", replace(BASE_COSTS, review=10.0)),
    ("Revisão custa 20", replace(BASE_COSTS, review=20.0)),
    ("Bloqueio indevido custa 10", replace(BASE_COSTS, block_legit=10.0)),
    ("Bloqueio indevido custa 50", replace(BASE_COSTS, block_legit=50.0)),
    ("Bloqueio indevido custa 100", replace(BASE_COSTS, block_legit=100.0)),
    ("Revisão detecta 80% das fraudes", replace(BASE_COSTS, review_detection=0.8)),
    ("Revisão detecta todas as fraudes", replace(BASE_COSTS, review_detection=1.0)),
    ("Capacidade de revisão 0,5%", replace(BASE_COSTS, max_review_share=0.005)),
    ("Capacidade de revisão 2%", replace(BASE_COSTS, max_review_share=0.02)),
    ("Sem limite de capacidade", replace(BASE_COSTS, max_review_share=None)),
)


@dataclass(frozen=True)
class Thresholds:
    review: float
    block: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.review <= self.block <= 1.0:
            raise ValueError(
                f"thresholds must satisfy 0 <= review <= block <= 1, got {self.review}, "
                f"{self.block}"
            )

    def decide(self, probability: np.ndarray) -> np.ndarray:
        """0 approve, 1 review, 2 block. A probability equal to a threshold goes up a band."""
        p = np.asarray(probability, dtype=float)
        if np.isnan(p).any():
            raise ValueError("a probability is NaN")
        return np.where(p >= self.block, BLOCK, np.where(p >= self.review, REVIEW, APPROVE)).astype(
            np.int8
        )

    def to_dict(self) -> dict[str, float]:
        return {"review": self.review, "block": self.block}


@dataclass(frozen=True)
class SearchResult:
    thresholds: Thresholds
    total_cost: float
    review_share: float


def _cuts(ranked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Possible cuts of probabilities sorted from high to low.

    Returns, for each cut, how many transactions are at or above it and the threshold value. A
    cut never splits equal probabilities. When there are too many, the cuts are thinned on a
    geometric scale of that count, so they stay dense among the highest probabilities, where the
    decisions are made.
    """
    n = len(ranked)
    counts = np.r_[np.flatnonzero(ranked[1:] < ranked[:-1]) + 1, n]
    if ranked[0] < 1.0:
        counts = np.r_[0, counts]  # nobody at or above the cut
    if len(counts) > MAX_CANDIDATES:
        wanted = np.unique(np.geomspace(1, n, MAX_CANDIDATES).round().astype(int))
        keep = np.unique(np.clip(np.searchsorted(counts, wanted), 0, len(counts) - 1))
        counts = np.unique(np.r_[counts[0], counts[keep], n])
    values = np.empty(len(counts), dtype=float)
    for position, count in enumerate(counts):
        if count == 0:
            values[position] = (ranked[0] + 1.0) / 2
        elif count == n:
            values[position] = ranked[-1]
        else:
            values[position] = (ranked[count - 1] + ranked[count]) / 2
    return counts, values


def search(
    probability: np.ndarray, amount: np.ndarray, y: np.ndarray, costs: Costs = BASE_COSTS
) -> SearchResult:
    """The pair of thresholds with the lowest total cost on these transactions."""
    order = np.argsort(-probability, kind="stable")
    ranked = probability[order]
    cost = costs.per_decision(amount[order], y[order])
    prefix = np.vstack([np.zeros(3), np.cumsum(cost, axis=0)])
    n = len(ranked)
    counts, values = _cuts(ranked)
    at_review = counts[:, None]  # transactions at or above the review threshold (rows)
    at_block = counts[None, :]  # transactions at or above the block threshold (columns)
    total = (
        prefix[at_block, BLOCK]
        + prefix[at_review, REVIEW] - prefix[at_block, REVIEW]
        + prefix[n, APPROVE] - prefix[at_review, APPROVE]
    )
    valid = at_review >= at_block
    if costs.max_review_share is not None:
        valid &= (at_review - at_block) <= costs.max_review_share * n
    total = np.where(valid, total, np.inf)
    row, column = np.unravel_index(np.argmin(total), total.shape)
    return SearchResult(
        thresholds=Thresholds(review=float(values[row]), block=float(values[column])),
        total_cost=float(total[row, column]),
        review_share=float(counts[row] - counts[column]) / n,
    )


def summarize(
    decision: np.ndarray, y: np.ndarray, amount: np.ndarray, costs: Costs = BASE_COSTS
) -> dict[str, float]:
    """Transactions, frauds and costs of a set of decisions, split by kind of cost."""
    fraud = np.asarray(y).astype(bool)
    summary: dict[str, float] = {}
    for code, name in enumerate(DECISIONS):
        inside = decision == code
        summary[f"{name}_transactions"] = int(inside.sum())
        summary[f"{name}_frauds"] = int((inside & fraud).sum())
        summary[f"{name}_legit"] = int((inside & ~fraud).sum())
    approved = float(amount[(decision == APPROVE) & fraud].sum())
    reviewed = float(amount[(decision == REVIEW) & fraud].sum())
    blocked = float(amount[(decision == BLOCK) & fraud].sum())
    summary["fraud_amount"] = approved + reviewed + blocked
    summary["loss_avoided"] = blocked + costs.review_detection * reviewed
    summary["financial_cost"] = approved + (1.0 - costs.review_detection) * reviewed
    summary["operational_cost"] = costs.review * summary["review_transactions"]
    summary["experience_cost"] = (
        costs.review_friction * summary["review_legit"] + costs.block_legit * summary["block_legit"]
    )
    summary["total_cost"] = (
        summary["financial_cost"] + summary["operational_cost"] + summary["experience_cost"]
    )
    summary["frauds_flagged"] = summary["review_frauds"] + summary["block_frauds"]
    summary["review_share"] = summary["review_transactions"] / len(decision)
    return summary
