"""Calibration of a model score into a probability of fraud (decision D15).

The model gives a raw score s (log-odds). Three readings of it are compared:

- none: sigmoid(s), the model's own probability;
- platt: sigmoid(a * s + b), with a and b fitted by maximum likelihood;
- isotonic: an increasing function of s fitted by isotonic regression, applied by linear
  interpolation between its points and constant beyond the first and last.

They are compared by cross-fitting on the calibration period: the period is cut into
consecutive time blocks and each block is predicted by a calibrator fitted on the other blocks,
so no method is scored on the rows it was fitted on. The chosen method is then fitted on the
whole period. A calibrator is stored as plain numbers (JSON), never as a pickled object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Final, Protocol

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

METHODS: Final = ("none", "platt", "isotonic")
BLOCKS: Final = 5


class Calibrator(Protocol):
    method: ClassVar[str]

    def apply(self, score: np.ndarray) -> np.ndarray: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Identity:
    method: ClassVar[str] = "none"

    @classmethod
    def fit(cls, score: np.ndarray, y: np.ndarray) -> Identity:
        return cls()

    def apply(self, score: np.ndarray) -> np.ndarray:
        return expit(np.asarray(score, dtype=float))

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method}


@dataclass(frozen=True)
class Platt:
    a: float
    b: float
    method: ClassVar[str] = "platt"

    @classmethod
    def fit(cls, score: np.ndarray, y: np.ndarray) -> Platt:
        score = np.asarray(score, dtype=float)
        y = np.asarray(y, dtype=float)

        def loss(params: np.ndarray) -> tuple[float, np.ndarray]:
            z = params[0] * score + params[1]
            gap = expit(z) - y
            value = float(np.mean(np.logaddexp(0.0, z) - y * z))
            return value, np.array([np.mean(gap * score), np.mean(gap)])

        result = minimize(loss, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"Platt scaling did not converge: {result.message}")
        return cls(a=float(result.x[0]), b=float(result.x[1]))

    def apply(self, score: np.ndarray) -> np.ndarray:
        return expit(self.a * np.asarray(score, dtype=float) + self.b)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "a": self.a, "b": self.b}


@dataclass(frozen=True)
class Isotonic:
    x: tuple[float, ...]
    y: tuple[float, ...]
    method: ClassVar[str] = "isotonic"

    def __post_init__(self) -> None:
        x, y = np.asarray(self.x), np.asarray(self.y)
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("isotonic calibrator needs at least two matching points")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0) or y.min() < 0 or y.max() > 1:
            raise ValueError("isotonic calibrator points must increase and stay in [0, 1]")

    @classmethod
    def fit(cls, score: np.ndarray, y: np.ndarray) -> Isotonic:
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(score, y)
        return cls(x=tuple(map(float, model.X_thresholds_)),
                   y=tuple(map(float, model.y_thresholds_)))

    def apply(self, score: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(score, dtype=float), self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "x": list(self.x), "y": list(self.y)}


FITTERS: Final = {"none": Identity, "platt": Platt, "isotonic": Isotonic}


def fit(method: str, score: np.ndarray, y: np.ndarray) -> Calibrator:
    return FITTERS[method].fit(score, y)


def from_dict(data: dict[str, Any]) -> Calibrator:
    method = data.get("method")
    if method == "none":
        return Identity()
    if method == "platt":
        return Platt(a=float(data["a"]), b=float(data["b"]))
    if method == "isotonic":
        return Isotonic(x=tuple(map(float, data["x"])), y=tuple(map(float, data["y"])))
    raise ValueError(f"unknown calibration method: {method!r}")


def time_blocks(n: int, blocks: int = BLOCKS) -> np.ndarray:
    """Block number of each row, rows being in time order: consecutive blocks of equal size."""
    return np.arange(n) * blocks // n


def cross_fit(method: str, score: np.ndarray, y: np.ndarray, blocks: int = BLOCKS) -> np.ndarray:
    """Probabilities where each time block comes from a calibrator fitted on the other blocks."""
    block = time_blocks(len(score), blocks)
    probability = np.empty(len(score), dtype=float)
    for number in range(blocks):
        inside = block == number
        calibrator = fit(method, score[~inside], y[~inside])
        probability[inside] = calibrator.apply(score[inside])
    return probability
