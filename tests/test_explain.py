"""Tests for pipeline/modeling/explain.py: SHAP values and the reasons written from them."""

from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from reference_shap import shapley

from pipeline import features
from pipeline.modeling import explain
from pipeline.modeling.data import FeatureSpec


def small_model(seed: int = 3, categorical: bool = False) -> tuple[lgb.Booster, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2_000, 4))
    if categorical:
        x[:, 3] = rng.integers(0, 4, len(x))
    x[:, 2] = 1.0  # a constant column: never split on
    y = (x[:, 0] + 0.6 * x[:, 1] + 0.3 * (x[:, 3] > 1) + rng.normal(0, 0.8, len(x)) > 1.2)
    data = lgb.Dataset(x, label=y.astype(int), categorical_feature=[3] if categorical else "auto")
    params = {"objective": "binary", "num_leaves": 7, "learning_rate": 0.2, "verbosity": -1,
              "seed": seed, "num_threads": 1, "deterministic": True}
    return lgb.train(params, data, num_boost_round=12), x


def test_contributions_equal_brute_force_shapley_values() -> None:
    booster, x = small_model()
    contribution = explain.contributions(booster, x[:6])
    for row in range(6):
        assert np.allclose(contribution[row], shapley(booster, x[row]), atol=1e-9)


@pytest.mark.parametrize("categorical", [False, True])
def test_contributions_add_up_to_the_raw_score(categorical: bool) -> None:
    booster, x = small_model(categorical=categorical)
    contribution = explain.contributions(booster, x[:200])
    assert np.allclose(contribution.sum(axis=1), booster.predict(x[:200], raw_score=True))
    assert np.all(contribution[:, 2] == 0)  # the column the model never uses gets no credit


SPEC = FeatureSpec(columns=tuple(features.FEATURE_COLUMNS),
                   categories=("grocery_pos", "misc_net", "shopping_net"))


@pytest.mark.parametrize("feature", [name for name in features.FEATURE_COLUMNS
                                     if name != "category"])
@pytest.mark.parametrize("value", [0.0, 1.0, 3.0, math.nan])
def test_every_signal_has_a_sentence_for_any_value(feature: str, value: float) -> None:
    if math.isnan(value) and feature in {"amount", "hour", "day_of_week", "is_late_night",
                                          "is_weekend", "has_history", "is_new_category"} | {
            name for name in features.FEATURE_COLUMNS if name.startswith(("tx_count_",
                                                                          "amount_sum_"))}:
        pytest.skip("this signal is never missing")
    text = explain.describe(feature, value, SPEC)
    assert text and "nan" not in text.lower()


@pytest.mark.parametrize("feature, value, expected", [
    ("amount_to_history_mean", 6.12, "Valor 6,1 vezes a média das compras anteriores do cartão"),
    ("amount_to_history_mean", math.nan, "Sem compras anteriores para comparar o valor"),
    ("amount_minus_history_mean", -1234.5, "Valor 1.234,50 abaixo da média das compras "
                                           "anteriores do cartão"),
    ("is_new_category", 1.0, "Primeira compra do cartão nesta categoria"),
    ("tx_count_1h", 3.0, "3 compras do cartão na última hora"),
    ("tx_count_24h", 1.0, "1 compra do cartão nas últimas 24 horas"),
    ("hour", 3.0, "Compra às 3h"),
    ("day_of_week", 5.0, "Compra num sábado"),
    ("day_of_week", 0.0, "Compra numa segunda-feira"),
    ("distance_km", 120.4, "120 km entre o endereço do cliente e o lojista"),
    ("seconds_since_previous", 90.0, "90 segundos desde a compra anterior do cartão"),
    ("seconds_since_previous", 3 * 86_400.0, "3,0 dias desde a compra anterior do cartão"),
    ("history_share_same_hour", 0.0, "Nenhuma compra anterior do cartão nesta hora do dia"),
    ("history_share_late_night", 0.25, "25% das compras anteriores do cartão foram na madrugada"),
    ("category", 2.0, "Categoria do lojista: shopping_net"),
])
def test_sentences_state_the_value(feature: str, value: float, expected: str) -> None:
    assert explain.describe(feature, value, SPEC) == expected


def test_reasons_come_strongest_first_with_their_direction() -> None:
    row = np.zeros(len(SPEC.columns))
    contribution = np.zeros(len(SPEC.columns) + 1)
    position = {name: i for i, name in enumerate(SPEC.columns)}
    row[position["amount"]] = 950.0
    row[position["hour"]] = 23.0
    contribution[position["amount"]] = 1.5
    contribution[position["hour"]] = -2.0
    contribution[position["is_weekend"]] = 0.2
    contribution[-1] = -6.0  # the expected value is never a reason
    found = explain.reasons(row, contribution, SPEC, top=5)
    assert [reason.feature for reason in found] == ["hour", "amount", "is_weekend"]
    assert found[0].text == "Compra às 23h (reduziu o risco)"
    assert found[1].text == "Valor da compra: 950,00 (aumentou o risco)"
    assert [reason.effect for reason in found] == ["decrease", "increase", "increase"]


def test_typical_cases_pick_one_case_per_outcome() -> None:
    y = np.array([1, 1, 1, 0, 0, 1, 0])
    probability = np.array([0.9, 0.95, 0.5, 0.8, 0.4, 0.05, 0.01])
    decided = np.array([2, 2, 1, 2, 1, 0, 0])
    cases = dict(explain.typical_cases(y, probability, decided))
    assert cases == {"Fraude bloqueada": 1, "Fraude enviada para revisão": 2,
                     "Cliente legítimo bloqueado": 3, "Cliente legítimo em revisão": 4,
                     "Fraude aprovada": 5}


def test_summary_plot_is_written(tmp_path) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, len(SPEC.columns)))
    x[:, SPEC.category_index] = rng.integers(0, 3, 300)
    contribution = rng.normal(size=(300, len(SPEC.columns) + 1))
    explain.summary_plot(contribution, x, SPEC, tmp_path / "shap.png", top=8)
    assert (tmp_path / "shap.png").stat().st_size > 10_000


def test_global_importance_ranks_by_mean_absolute_contribution() -> None:
    booster, _ = small_model()
    spec = FeatureSpec(columns=("a", "b", "c", "d"), categories=())
    contribution = np.array([[0.1, -2.0, 0.0, 0.5, -3.0], [0.3, 1.0, 0.0, -0.5, -3.0]])
    table = explain.global_importance(contribution, spec, booster)
    assert table["feature"].tolist()[:3] == ["b", "d", "a"]
    assert table["shap_share"].sum() == pytest.approx(1.0)
    assert isinstance(table, pd.DataFrame)
