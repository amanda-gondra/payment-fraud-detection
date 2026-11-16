"""Reference models the main model has to beat (decision D13).

- AmountRule: flags a transaction when its amount reaches X. X is the cut that maximises F1 on
  the training set (2019): F1 weighs precision and recall equally and needs no cost hypothesis,
  so the rule is fixed without looking at the validation period.
- Logistic regression on the 28 signals: log of the heavy-tailed amounts, counts and durations,
  median imputation with a missing-value indicator, standardisation, one-hot encoding of the
  category, the hour and the weekday, and L2 regularisation. C and the class weight are chosen
  on the selection period.

The fitted logistic regression is saved with skops (no pickle) and loaded only when every type
in the file is on an explicit trusted list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from pipeline import features

LOG_FEATURES: Final = (
    "amount", "tx_count_1h", "amount_sum_1h", "amount_mean_1h", "tx_count_24h",
    "amount_sum_24h", "amount_mean_24h", "tx_count_7d", "amount_sum_7d", "amount_mean_7d",
    "tx_count_30d", "amount_sum_30d", "amount_mean_30d", "history_mean_amount",
    "amount_to_history_mean", "seconds_since_previous",
)
SIGNED_LOG_FEATURES: Final = ("amount_minus_history_mean",)
LINEAR_FEATURES: Final = (
    "distance_km", "amount_log_zscore", "history_share_same_hour", "history_share_late_night",
)
FLAG_FEATURES: Final = ("is_late_night", "is_weekend", "has_history", "is_new_category")
ONE_HOT_FEATURES: Final = ("category", "hour", "day_of_week")
LOGISTIC_GRID: Final = tuple(
    {"C": c, "class_weight": weight} for c in (0.01, 0.1, 1.0) for weight in (None, "balanced")
)
MAX_ITER: Final = 3_000
# Types skops does not trust by default that the saved logistic baseline contains: the dtype of
# the one-hot output (a description of a number format) and our own signed_log1p function.
LOGISTIC_TRUSTED: Final = ("numpy.dtype", "pipeline.modeling.baselines.signed_log1p")


@dataclass(frozen=True)
class AmountRule:
    """Flag when the amount is at least threshold."""

    threshold: float

    @classmethod
    def fit(cls, amount: np.ndarray, y: np.ndarray) -> AmountRule:
        """The cut with the highest F1 on the given (training) transactions."""
        order = np.argsort(-amount, kind="stable")
        ranked = amount[order]
        caught = np.cumsum(y[order])
        flagged = np.arange(1, len(amount) + 1)
        # a cut can only fall where the amount changes: keep the last row of each value
        last_of_value = np.r_[ranked[1:] != ranked[:-1], True]
        f1 = 2 * caught / (flagged + y.sum())
        best = np.flatnonzero(last_of_value)[np.argmax(f1[last_of_value])]
        return cls(threshold=float(ranked[best]))

    def flag(self, amount: np.ndarray) -> np.ndarray:
        return amount >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "rule": "flag when amount >= threshold",
                "fitted_on": "train (2019), maximum F1"}


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """log(1 + |x|) with the sign of x, for differences that can be negative."""
    return np.sign(values) * np.log1p(np.abs(values))


def _numeric(transform: Any = None) -> Pipeline:
    steps = [] if transform is None else [
        FunctionTransformer(transform, feature_names_out="one-to-one")
    ]
    return make_pipeline(
        *steps, SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
    )


def logistic_model(c: float, class_weight: str | None) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("log", _numeric(np.log1p), list(LOG_FEATURES)),
            ("signed_log", _numeric(signed_log1p), list(SIGNED_LOG_FEATURES)),
            ("linear", _numeric(), list(LINEAR_FEATURES)),
            ("flags", "passthrough", list(FLAG_FEATURES)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(ONE_HOT_FEATURES)),
        ],
        sparse_threshold=0.0,
    )
    model = LogisticRegression(C=c, class_weight=class_weight, max_iter=MAX_ITER)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def logistic_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """The 28 signals with plain types: floats, booleans as 0/1, category as text."""
    data = frame[list(features.FEATURE_COLUMNS)].copy()
    for name in (*LOG_FEATURES, *SIGNED_LOG_FEATURES, *LINEAR_FEATURES, *FLAG_FEATURES):
        data[name] = data[name].astype(float)
    data["category"] = data["category"].astype(str)
    return data


def logistic_raw_score(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Log-odds of fraud (the score the calibrators work on)."""
    return model.decision_function(logistic_inputs(frame))


def dump_logistic(model: Pipeline) -> bytes:
    return sio.dumps(model)


def load_logistic(data: bytes) -> Pipeline:
    """Load skops bytes only if every type they need is trusted by default or on our list."""
    untrusted = set(sio.get_untrusted_types(data=data))
    unexpected = untrusted - set(LOGISTIC_TRUSTED)
    if unexpected:
        raise ValueError(f"refusing to load the logistic baseline: untrusted types "
                         f"{sorted(unexpected)}")
    return sio.loads(data, trusted=sorted(untrusted))
