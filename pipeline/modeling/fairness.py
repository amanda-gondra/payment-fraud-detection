"""Fairness audit by gender and age band (decision D20).

Gender and age never enter the model. They live in a separate file (data/processed/audit/ for
train and validation; kept in memory only for the test) and are joined here, after the decisions
are made, only to compare errors between groups.

Each card belongs to one customer, and the transactions of a card are not independent (a fraud
episode hits several transactions of the same card). The 95% intervals therefore come from a
cluster bootstrap: whole cards are drawn with replacement inside each group. A difference whose
interval excludes zero is an indication to investigate, not proof of discrimination; one whose
interval includes zero is not proof of equality. The data are simulated, so nothing here
generalises to a real population.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from pipeline.modeling import decision
from pipeline.modeling.evaluation import MIN_FRAUDS

AGE_BANDS: Final = (
    (0, 29, "Até 29 anos"),
    (30, 44, "30 a 44 anos"),
    (45, 59, "45 a 59 anos"),
    (60, 200, "60 anos ou mais"),
)
GENDER_LABELS: Final = {"F": "Feminino", "M": "Masculino"}
BOOTSTRAP: Final = 1_000
SEED: Final = 20_261_115
COUNTS: Final = ("transactions", "legit", "frauds", "alerted_legit", "alerted_fraud",
                 "blocked_legit", "reviewed")
RATES: Final = ("fraud_rate", "fpr", "fnr", "precision", "recall", "legit_block_rate",
                "review_rate")


def age_band(ages: pd.Series) -> pd.Series:
    labels = [label for _, _, label in AGE_BANDS]
    edges = [AGE_BANDS[0][0] - 1] + [high for _, high, _ in AGE_BANDS]
    return pd.cut(ages, bins=edges, labels=labels).astype(pd.CategoricalDtype(labels, ordered=True))


def attributes(audit: pd.DataFrame, transaction_ids: pd.Series) -> pd.DataFrame:
    """Gender and age band of the given transactions, in their order."""
    wanted = pd.DataFrame({"transaction_id": transaction_ids.to_numpy()})
    joined = wanted.merge(audit, on="transaction_id", how="left", validate="1:1")
    if joined["gender"].isna().any():
        raise ValueError("audit attributes missing for some transactions")
    return pd.DataFrame({
        "gender": joined["gender"].astype(str).map(GENDER_LABELS).fillna("Outro"),
        "age_band": age_band(joined["age_years"]),
    })


def _card_counts(frame: pd.DataFrame) -> pd.DataFrame:
    fraud = frame["is_fraud"].astype(bool)
    alerted = frame["decision"].ge(decision.REVIEW)
    counts = pd.DataFrame({
        "group": frame["group"],
        "card": frame["card_key"].astype(str),
        "transactions": 1,
        "legit": (~fraud).astype(int),
        "frauds": fraud.astype(int),
        "alerted_legit": (alerted & ~fraud).astype(int),
        "alerted_fraud": (alerted & fraud).astype(int),
        "blocked_legit": (frame["decision"].eq(decision.BLOCK) & ~fraud).astype(int),
        "reviewed": frame["decision"].eq(decision.REVIEW).astype(int),
    })
    return counts.groupby(["group", "card"], observed=True)[list(COUNTS)].sum().reset_index()


def _rates(totals: np.ndarray) -> dict[str, np.ndarray]:
    """Rates from count totals (last axis in COUNTS order); NaN when the denominator is zero."""
    c = {name: totals[..., i].astype(float) for i, name in enumerate(COUNTS)}
    with np.errstate(divide="ignore", invalid="ignore"):
        alerted = c["alerted_legit"] + c["alerted_fraud"]
        return {
            "fraud_rate": c["frauds"] / c["transactions"],
            "fpr": c["alerted_legit"] / c["legit"],
            "fnr": 1 - c["alerted_fraud"] / c["frauds"],
            "precision": c["alerted_fraud"] / alerted,
            "recall": c["alerted_fraud"] / c["frauds"],
            "legit_block_rate": c["blocked_legit"] / c["legit"],
            "review_rate": c["reviewed"] / c["transactions"],
        }


def audit(
    frame: pd.DataFrame, boot: int = BOOTSTRAP, seed: int = SEED
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Counts and error rates per group, with 95% cluster-bootstrap intervals.

    frame has one row per transaction with group, card_key, is_fraud and decision. Alerted means
    review or block. Returns the table and the bootstrap draws of each group (for differences).
    """
    rng = np.random.default_rng(seed)
    cards = _card_counts(frame)
    rows, draws = [], {}
    for group, part in cards.groupby("group", observed=True, sort=True):
        counts = part[list(COUNTS)].to_numpy(dtype=float)
        weights = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=boot)
        sample = _rates(weights @ counts)
        point = _rates(counts.sum(axis=0))
        draws[str(group)] = sample
        row = {"group": str(group), "cards": len(counts)}
        row |= {name: int(total) for name, total in zip(COUNTS, counts.sum(axis=0), strict=True)}
        for name in RATES:
            low, high = np.nanpercentile(sample[name], [2.5, 97.5])
            row |= {name: float(point[name]), f"{name}_low": float(low),
                    f"{name}_high": float(high)}
        row["reliable"] = row["frauds"] >= MIN_FRAUDS
        rows.append(row)
    return pd.DataFrame(rows), draws


def differences(
    table: pd.DataFrame, draws: dict[str, dict[str, np.ndarray]], reference: str
) -> pd.DataFrame:
    """Each group minus the reference group, with a 95% interval from the bootstrap draws.

    reliable is true only when both groups have enough frauds (MIN_FRAUDS): with fewer, the
    interval says little even when it excludes zero.
    """
    point = table.set_index("group")
    rows = []
    for group in point.index:
        if group == reference:
            continue
        for name in ("fpr", "fnr", "precision", "legit_block_rate", "review_rate"):
            gap = draws[group][name] - draws[reference][name]
            low, high = np.nanpercentile(gap, [2.5, 97.5])
            rows.append({
                "group": group,
                "reference": reference,
                "metric": name,
                "difference": float(point.loc[group, name] - point.loc[reference, name]),
                "low": float(low),
                "high": float(high),
                "excludes_zero": bool(low > 0 or high < 0),
                "reliable": bool(point.loc[group, "reliable"] and point.loc[reference,
                                                                           "reliable"]),
            })
    return pd.DataFrame(rows)
