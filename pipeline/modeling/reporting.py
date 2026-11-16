"""Markdown pieces for the Portuguese reports: Brazilian number format and tables."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from pipeline import build

DASH = "—"
TIMES = "\u00d7"  # the multiplication sign, as in "1,4x"


def times(value: float | None, digits: int = 1) -> str:
    return DASH if _missing(value) else f"{decimal(value, digits)}{TIMES}"


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def integer(value: float | None) -> str:
    return DASH if _missing(value) else build.number(value)


def decimal(value: float | None, digits: int = 2) -> str:
    return DASH if _missing(value) else f"{value:,.{digits}f}".translate(build.PT_BR)


def pct(value: float | None, digits: int = 1) -> str:
    return DASH if _missing(value) else f"{value:.{digits}%}".translate(build.PT_BR)


def interval(low: float, high: float, digits: int = 1) -> str:
    return f"{pct(low, digits)} a {pct(high, digits)}"


def moment(iso: str) -> str:
    """An ISO timestamp in UTC as 15/11/2026 às 23:54 (UTC)."""
    return f"{datetime.fromisoformat(iso):%d/%m/%Y às %H:%M} (UTC)"


def band_label(band: str) -> str:
    """A reliability band '0.005-0.01' as '0,5% a 1,0%'."""
    low, high = (float(edge) for edge in band.split("-"))
    return interval(low, high, 1)


def target_recall(recall: float, precision_reached: float) -> str:
    """Recall at the target precision, or a note when no cut reaches the target."""
    return "não atinge a meta" if _missing(precision_reached) else pct(recall)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: str = "") -> list[str]:
    """A Markdown table; align has one letter per column: l (left) or r (right), default l."""
    marks = ["---:" if (align[i:i + 1] == "r") else "---" for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(marks) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines
