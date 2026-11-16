"""Exact Shapley values of a small LightGBM model by brute force, independent of TreeSHAP.

The value of a coalition S of features is the expected output of each tree when the features in
S are fixed to the transaction's values and the others follow the training data, as recorded in
the number of training rows that reached each node (the path-dependent definition TreeSHAP
computes). Every coalition is enumerated, so this only works for a handful of features. Numeric
splits only, no missing values.
"""

from __future__ import annotations

import math
from itertools import combinations

import lightgbm as lgb
import numpy as np


def _rows(node: dict) -> float:
    return float(node.get("internal_count", node.get("leaf_count")))


def _expected(node: dict, x: np.ndarray, known: frozenset[int]) -> float:
    if "leaf_value" in node:
        return float(node["leaf_value"])
    left, right = node["left_child"], node["right_child"]
    feature = node["split_feature"]
    if feature in known:
        return _expected(left if x[feature] <= node["threshold"] else right, x, known)
    weight_left, weight_right = _rows(left), _rows(right)
    return (weight_left * _expected(left, x, known) + weight_right * _expected(right, x, known)) / (
        weight_left + weight_right)


def shapley(booster: lgb.Booster, x: np.ndarray) -> np.ndarray:
    """SHAP value of each feature and, last, the expected value: the layout of pred_contrib."""
    trees = [tree["tree_structure"] for tree in booster.dump_model()["tree_info"]]
    features = len(x)

    def value(known: frozenset[int]) -> float:
        return sum(_expected(tree, x, known) for tree in trees)

    phi = np.zeros(features)
    for feature in range(features):
        others = [j for j in range(features) if j != feature]
        for size in range(features):
            weight = (math.factorial(size) * math.factorial(features - size - 1)
                      / math.factorial(features))
            for coalition in combinations(others, size):
                known = frozenset(coalition)
                phi[feature] += weight * (value(known | {feature}) - value(known))
    return np.r_[phi, value(frozenset())]
