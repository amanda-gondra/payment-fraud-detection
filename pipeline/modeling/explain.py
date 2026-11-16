"""Explanations: SHAP contributions of the LightGBM model and the reasons written from them
(decision D19).

LightGBM computes the SHAP values itself (TreeSHAP, exact for trees: predict with
pred_contrib=True). For one transaction, the contributions of the 28 signals plus the expected
value add up to the model's raw score (log-odds) before calibration. Calibration is increasing,
so a signal that raises the raw score also raises the calibrated probability.

A reason is written from the value of the signal in that transaction, never from a guess: the
text states the fact ("Valor 6,1 vezes a média das compras anteriores do cartão") and whether the
signal pushed the risk up or down in the model. SHAP explains the model, not the world: a signal
that weighs towards fraud is not a proven cause, and correlated signals share the credit.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import lightgbm as lgb
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pipeline import build
from pipeline.modeling import decision
from pipeline.modeling.data import CATEGORY, FeatureSpec, Split
from pipeline.modeling.evaluation import Scored
from pipeline.modeling.lgbm import NUM_THREADS

WEEKDAYS: Final = ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira",
                   "sábado", "domingo")
WINDOW_TEXT: Final = {"1h": "na última hora", "24h": "nas últimas 24 horas",
                      "7d": "nos últimos 7 dias", "30d": "nos últimos 30 dias"}
EFFECT_TEXT: Final = {"increase": "aumentou o risco", "decrease": "reduziu o risco"}
TOP_REASONS: Final = 4  # in each answer of the model
REPORT_REASONS: Final = 5  # in the examples of the reports
PLOT_SEED: Final = 7  # vertical jitter of the summary plot
INK: Final = "#2b2b33"
MUTED: Final = "#8c8c96"
GRID: Final = "#e4e4ea"


def contributions(booster: lgb.Booster, x: np.ndarray) -> np.ndarray:
    """SHAP value of each signal (columns in model order) plus the expected value (last column)."""
    return booster.predict(x, pred_contrib=True, num_threads=NUM_THREADS)


def decimal(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}".translate(build.PT_BR)


def money(value: float) -> str:
    return decimal(value, 2)


def share(value: float) -> str:
    return f"{value:.0%}".translate(build.PT_BR)


def duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} segundos"
    if seconds < 2 * 3_600:
        return f"{seconds / 60:.0f} minutos"
    if seconds < 2 * 86_400:
        return f"{decimal(seconds / 3_600)} horas"
    return f"{decimal(seconds / 86_400)} dias"


def _window_templates() -> dict[str, Callable[[float], str]]:
    templates: dict[str, Callable[[float], str]] = {}
    for window, when in WINDOW_TEXT.items():
        def count(v: float, when: str = when) -> str:
            if v == 0:
                return f"Nenhuma compra do cartão {when}"
            return f"{v:.0f} compra{'s' if v != 1 else ''} do cartão {when}"

        def total(v: float, when: str = when) -> str:
            if v == 0:
                return f"Nenhum valor gasto pelo cartão {when}"
            return f"Soma das compras do cartão {when}: {money(v)}"

        def mean(v: float, when: str = when) -> str:
            if math.isnan(v):
                return f"Nenhuma compra do cartão {when}, sem média para comparar"
            return f"Média das compras do cartão {when}: {money(v)}"

        templates[f"tx_count_{window}"] = count
        templates[f"amount_sum_{window}"] = total
        templates[f"amount_mean_{window}"] = mean
    return templates


NO_HISTORY: Final = "Sem compras anteriores do cartão"
TEMPLATES: Final[dict[str, Callable[[float], str]]] = {
    "amount": lambda v: f"Valor da compra: {money(v)}",
    "hour": lambda v: f"Compra às {v:.0f}h",
    "day_of_week": lambda v: f"Compra {'numa' if v < 5 else 'num'} {WEEKDAYS[int(v)]}",
    "is_late_night": lambda v: "Compra na madrugada (22h às 3h59)" if v else
    "Compra fora da madrugada",
    "is_weekend": lambda v: "Compra no fim de semana" if v else "Compra em dia útil",
    "distance_km": lambda v: "Distância indisponível (coordenada inválida)" if math.isnan(v)
    else f"{decimal(v, 0)} km entre o endereço do cliente e o lojista",
    **_window_templates(),
    "has_history": lambda v: "O cartão tem compras anteriores" if v else
    "Primeira compra registrada do cartão",
    "history_mean_amount": lambda v: NO_HISTORY if math.isnan(v) else
    f"Média das compras anteriores do cartão: {money(v)}",
    "amount_to_history_mean": lambda v: "Sem compras anteriores para comparar o valor"
    if math.isnan(v) else f"Valor {decimal(v)} vezes a média das compras anteriores do cartão",
    "amount_minus_history_mean": lambda v: "Sem compras anteriores para comparar o valor"
    if math.isnan(v) else f"Valor {money(abs(v))} {'acima' if v >= 0 else 'abaixo'} da média "
    "das compras anteriores do cartão",
    "amount_log_zscore": lambda v: "Menos de 10 compras anteriores: desvio em relação ao padrão "
    "do cartão não calculado" if math.isnan(v) else f"Valor {decimal(abs(v))} desvios padrão "
    f"{'acima' if v >= 0 else 'abaixo'} do padrão de valores do cartão",
    "seconds_since_previous": lambda v: "Sem compra anterior do cartão" if math.isnan(v) else
    f"{duration(v)} desde a compra anterior do cartão",
    "is_new_category": lambda v: "Primeira compra do cartão nesta categoria" if v else
    "O cartão já comprou nesta categoria",
    "history_share_same_hour": lambda v: NO_HISTORY if math.isnan(v) else
    "Nenhuma compra anterior do cartão nesta hora do dia" if v == 0 else
    f"{share(v)} das compras anteriores do cartão foram nesta hora do dia",
    "history_share_late_night": lambda v: NO_HISTORY if math.isnan(v) else
    "Nenhuma compra anterior do cartão na madrugada" if v == 0 else
    f"{share(v)} das compras anteriores do cartão foram na madrugada",
}


def describe(feature: str, value: float, spec: FeatureSpec) -> str:
    """The fact a signal states about one transaction, in Portuguese."""
    if feature == CATEGORY:
        return f"Categoria do lojista: {spec.category_label(value)}"
    return TEMPLATES[feature](float(value))


@dataclass(frozen=True)
class Reason:
    feature: str
    value: float | None
    contribution: float  # SHAP value, in log-odds of the model score
    effect: str  # "increase" or "decrease"
    text: str


def reasons(
    row: np.ndarray, contribution: np.ndarray, spec: FeatureSpec, top: int = TOP_REASONS
) -> list[Reason]:
    """The signals with the largest contributions to one prediction, strongest first."""
    weights = contribution[: len(spec.columns)]
    order = [i for i in np.argsort(-np.abs(weights), kind="stable") if weights[i] != 0][:top]
    found = []
    for position in order:
        feature = spec.columns[position]
        value = float(row[position])
        effect = "increase" if weights[position] > 0 else "decrease"
        fact = describe(feature, value, spec)
        found.append(Reason(
            feature=feature,
            value=None if math.isnan(value) else value,
            contribution=float(weights[position]),
            effect=effect,
            text=f"{fact} ({EFFECT_TEXT[effect]})",
        ))
    return found


def typical_cases(
    y: np.ndarray, probability: np.ndarray, decided: np.ndarray
) -> list[tuple[str, int]]:
    """One case of each outcome: the median-probability blocked and reviewed frauds, the most
    confident wrong block and review, and the fraud that came closest to being caught."""
    cases = [
        ("Fraude bloqueada", (y == 1) & (decided == decision.BLOCK), "median"),
        ("Fraude enviada para revisão", (y == 1) & (decided == decision.REVIEW), "median"),
        ("Cliente legítimo bloqueado", (y == 0) & (decided == decision.BLOCK), "max"),
        ("Cliente legítimo em revisão", (y == 0) & (decided == decision.REVIEW), "max"),
        ("Fraude aprovada", (y == 1) & (decided == decision.APPROVE), "max"),
    ]
    picked = []
    for label, mask, how in cases:
        rows = np.flatnonzero(mask)
        if not len(rows):
            continue
        ordered = rows[np.argsort(probability[rows], kind="stable")]
        picked.append((label, int(ordered[len(ordered) // 2] if how == "median" else ordered[-1])))
    return picked


def explain_cases(
    booster: lgb.Booster, spec: FeatureSpec, split: Split, scored: Scored
) -> list[dict]:
    """The reasons behind each typical case, for the reports (never logged to MLflow)."""
    if scored.probability is None:
        raise ValueError("typical cases need calibrated probabilities")
    examples = []
    for label, row in typical_cases(split.y, scored.probability, scored.decisions):
        single = contributions(booster, split.x[[row]])[0]
        examples.append({
            "case": label,
            "probability": float(scored.probability[row]),
            "decision": decision.DECISIONS[int(scored.decisions[row])],
            "reasons": reasons(split.x[row], single, spec, top=REPORT_REASONS),
        })
    return examples


def global_importance(
    contribution: np.ndarray, spec: FeatureSpec, booster: lgb.Booster
) -> pd.DataFrame:
    """Mean |SHAP| of each signal, its share of the total, and LightGBM's gain for comparison."""
    weights = contribution[:, : len(spec.columns)]
    mean_abs = np.abs(weights).mean(axis=0)
    gain = booster.feature_importance(importance_type="gain")
    table = pd.DataFrame({
        "feature": spec.columns,
        "mean_abs_shap": mean_abs,
        "shap_share": mean_abs / mean_abs.sum(),
        "mean_shap": weights.mean(axis=0),
        "gain_share": gain / gain.sum(),
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    table["shap_rank"] = np.arange(1, len(table) + 1)
    table["gain_rank"] = table["gain_share"].rank(ascending=False, method="min").astype(int)
    return table


def summary_plot(
    contribution: np.ndarray,
    x: np.ndarray,
    spec: FeatureSpec,
    path: Path,
    top: int = 15,
) -> None:
    """Beeswarm summary: one row per signal, one dot per transaction, colour = value of the signal.

    Colour runs from light to dark purple as the value goes from low to high within the signal
    (percentile); grey for the category (no order) and for missing values.
    """
    rng = np.random.default_rng(PLOT_SEED)
    weights = contribution[:, : len(spec.columns)]
    order = np.argsort(-np.abs(weights).mean(axis=0))[:top][::-1]
    colormap = mpl.colormaps["Purples"]
    fig = Figure(figsize=(8.6, 0.42 * top + 1.6), dpi=150, layout="tight")
    ax = fig.subplots()
    for row, position in enumerate(order):
        values = weights[:, position]
        counts, edges = np.histogram(values, bins=80)
        density = counts / counts.max()
        bin_of = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(counts) - 1)
        jitter = rng.uniform(-1, 1, len(values)) * 0.38 * density[bin_of]
        signal = x[:, position]
        colours = np.tile(np.array(mpl.colors.to_rgba(MUTED)), (len(values), 1))
        if spec.columns[position] != CATEGORY:
            known = ~np.isnan(signal)
            if known.sum() > 1:
                ranks = pd.Series(signal[known]).rank(pct=True).to_numpy()
                colours[known] = colormap(0.3 + 0.7 * ranks)
        ax.scatter(values, row + jitter, c=colours, s=4, linewidths=0, rasterized=True)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(order)), [spec.columns[i] for i in order], fontsize=8.5, color=INK)
    ax.set_xlabel("Contribuição SHAP (log-odds do modelo)", color=INK)
    ax.tick_params(axis="x", colors=INK, labelsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    scale = mpl.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(
        colormap(np.linspace(0.3, 1.0, 256))))
    bar = fig.colorbar(scale, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
    bar.ax.set_yticklabels(["baixo", "alto"], fontsize=8, color=INK)
    bar.set_label("Valor do sinal", fontsize=8.5, color=INK)
    bar.outline.set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
