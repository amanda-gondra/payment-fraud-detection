"""Figures of the reports: reliability (calibration) and precision-recall curves.

Drawn with matplotlib's object API (no global state). Series are told apart by colour and by a
legend; the axes and grid stay in grey.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

INK: Final = "#2b2b33"
MUTED: Final = "#8c8c96"
GRID: Final = "#e4e4ea"
# Fixed order, validated for colour-blind separation: reference (orange), main (purple), other
# (green). Callers pass the series in that order.
SERIES: Final = ("#c0661c", "#6a3d9a", "#2e8b57")


def _style(ax: Axes) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def reliability(tables: dict[str, pd.DataFrame], path: Path, title: str,
                colours: tuple[str, ...] = SERIES) -> None:
    """Observed fraud rate against mean predicted probability, per probability band."""
    fig = Figure(figsize=(6.4, 5.4), dpi=150, layout="tight")
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Calibração perfeita")
    for colour, (name, table) in zip(colours, tables.items(), strict=False):
        error = np.vstack([table["observed_rate"] - table["low"],
                           table["high"] - table["observed_rate"]])
        ax.errorbar(table["mean_predicted"], table["observed_rate"], yerr=error, color=colour,
                    marker="o", markersize=4, linewidth=1.6, capsize=2, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidade prevista (média da faixa)", color=INK)
    ax.set_ylabel("Taxa de fraude observada (IC 95%)", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def learning_curve(curve: list[dict[str, float]], path: Path, title: str) -> None:
    """Log loss on the training rows and on new data as trees are added (log scale)."""
    trees = [point["trees"] for point in curve]
    fig = Figure(figsize=(6.4, 4.6), dpi=150, layout="tight")
    ax = fig.subplots()
    ax.plot(trees, [point["train_log_loss"] for point in curve], color=SERIES[0], marker="o",
            markersize=4, linewidth=1.8, label="Treino (dados que o modelo viu)")
    ax.plot(trees, [point["selection_log_loss"] for point in curve], color=SERIES[1],
            marker="o", markersize=4, linewidth=1.8, label="Seleção (dados novos)")
    ax.set_yscale("log")
    ax.set_xlabel("Árvores", color=INK)
    ax.set_ylabel("Log loss (escala logarítmica)", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def precision_recall(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path,
                     title: str) -> None:
    """Precision against recall for each model (y, score)."""
    fig = Figure(figsize=(6.4, 5.0), dpi=150, layout="tight")
    ax = fig.subplots()
    for colour, (name, (y, score)) in zip(SERIES, curves.items(), strict=False):
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, color=colour, linewidth=1.8, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (fraudes encontradas)", color=INK)
    ax.set_ylabel("Precisão (alertas que eram fraude)", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
