"""Auditable split-conformal Adaptive Prediction Sets (Plan §10)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def _probability_matrix(probabilities: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] == 0:
        raise ValueError("probabilities must have shape [N, C] with N>0 and C>1")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    sums = values.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("probability rows must sum to one")
    return values


def aps_scores(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> np.ndarray:
    values = _probability_matrix(probabilities)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (values.shape[0],):
        raise ValueError("labels must have shape [N]")
    if (y < 0).any() or (y >= values.shape[1]).any():
        raise ValueError("labels contain an invalid class index")
    scores = np.empty(values.shape[0], dtype=np.float64)
    for row, (probability, label) in enumerate(zip(values, y)):
        order = np.argsort(-probability, kind="stable")
        cumulative = np.cumsum(probability[order])
        true_rank = int(np.flatnonzero(order == label)[0])
        scores[row] = cumulative[true_rank]
    return scores


def finite_sample_quantile(scores: Sequence[float], alpha: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = min(values.size, math.ceil((values.size + 1) * (1.0 - alpha)))
    return float(np.sort(values, kind="stable")[rank - 1])


def prediction_sets(
    probabilities: Sequence[Sequence[float]],
    threshold: float,
) -> List[List[int]]:
    values = _probability_matrix(probabilities)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0 + 1.0e-8:
        raise ValueError("APS threshold must be finite and in [0, 1]")
    output: List[List[int]] = []
    for probability in values:
        order = np.argsort(-probability, kind="stable")
        cumulative = np.cumsum(probability[order])
        output.append([int(index) for index, total in zip(order, cumulative) if total <= threshold])
    return output


@dataclass(frozen=True)
class APSThreshold:
    alpha: float
    target_coverage: float
    threshold: float
    calibration_count: int


def fit_aps(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    alpha: float,
) -> APSThreshold:
    scores = aps_scores(probabilities, labels)
    return APSThreshold(
        alpha=float(alpha),
        target_coverage=1.0 - float(alpha),
        threshold=finite_sample_quantile(scores, alpha),
        calibration_count=int(scores.size),
    )


def evaluate_sets(prediction_set: Sequence[Sequence[int]], labels: Sequence[int]) -> Dict[str, float]:
    y = list(labels)
    if len(prediction_set) != len(y) or not y:
        raise ValueError("prediction sets and labels must have the same positive length")
    sizes = np.asarray([len(value) for value in prediction_set], dtype=np.float64)
    covered = np.asarray(
        [int(int(label) in set(values)) for values, label in zip(prediction_set, y)],
        dtype=np.float64,
    )
    return {
        "coverage": float(covered.mean()),
        "average_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "empty_set_rate": float((sizes == 0).mean()),
    }

