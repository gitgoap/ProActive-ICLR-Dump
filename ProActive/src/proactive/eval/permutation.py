"""Permutation-stability metrics from Plan §20.

All functions operate on predictions for the *same* evidence set under
different arrival orders. They fail closed on malformed probabilities and do
not average across unrelated states.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def distinct_permutations(
    values: Sequence[int], *, maximum: int = 10, seed_key: str = ""
) -> List[List[int]]:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if len(values) != len(set(values)):
        raise ValueError("Permutation input must contain distinct values")
    candidates = list(itertools.permutations(values))
    candidates.sort(
        key=lambda value: hashlib.sha256(
            f"{seed_key}|{'|'.join(map(str, value))}".encode("utf-8")
        ).hexdigest()
    )
    return [list(value) for value in candidates[:maximum]]


def _probabilities(values: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("Expected a non-empty [permutations, classes] matrix")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("Probability rows must sum to one")
    return array


def mean_js_drift(probabilities: Sequence[Sequence[float]]) -> float:
    values = _probabilities(probabilities)
    mean = values.mean(axis=0)
    epsilon = np.finfo(np.float64).tiny
    divergences = []
    for row in values:
        middle = 0.5 * (row + mean)
        left = np.sum(np.where(row > 0, row * np.log((row + epsilon) / middle), 0.0))
        right = np.sum(np.where(mean > 0, mean * np.log((mean + epsilon) / middle), 0.0))
        divergences.append(0.5 * (left + right))
    return float(np.mean(divergences))


def mean_bit_drift(bit_probabilities: Sequence[Sequence[float]]) -> float:
    values = np.asarray(bit_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 3:
        raise ValueError("Expected source-bit probabilities with shape [P, 3]")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Source-bit probabilities must lie in [0, 1]")
    return float(np.abs(values - values.mean(axis=0)).sum(axis=1).mean())


def set_disagreement(prediction_sets: Sequence[Sequence[int]]) -> float:
    sets = [set(int(value) for value in prediction_set) for prediction_set in prediction_sets]
    if not sets:
        raise ValueError("At least one prediction set is required")
    if len(sets) == 1:
        return 0.0
    similarities = []
    for left, right in itertools.combinations(sets, 2):
        union = left | right
        similarities.append(1.0 if not union else len(left & right) / len(union))
    return float(1.0 - np.mean(similarities))


def action_agreement(actions: Sequence[str]) -> float:
    if not actions or any(not isinstance(value, str) or not value for value in actions):
        raise ValueError("Actions must be a non-empty list of strings")
    count = Counter(actions)
    # The value, not an arbitrary tie winner, defines agreement.
    return float(max(count.values()) / len(actions))


def hidden_state_drift(hidden_states: Sequence[Sequence[float]], epsilon: float = 1e-12) -> float:
    values = np.asarray(hidden_states, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("Hidden states must be a finite [P, D] matrix")
    mean = values.mean(axis=0)
    denominator = float(np.linalg.norm(mean) + epsilon)
    return float(np.linalg.norm(values - mean, axis=1).mean() / denominator)


def state_permutation_metrics(
    *,
    six_way_probabilities: Sequence[Sequence[float]],
    bit_probabilities: Sequence[Sequence[float]],
    hidden_states: Sequence[Sequence[float]],
    prediction_sets: Sequence[Sequence[int]] | None = None,
    actions: Sequence[str] | None = None,
) -> Dict[str, float]:
    count = len(six_way_probabilities)
    if len(bit_probabilities) != count or len(hidden_states) != count:
        raise ValueError("Permutation prediction blocks have different lengths")
    result = {
        "permutation_count": float(count),
        "js_drift": mean_js_drift(six_way_probabilities),
        "bit_l1_drift": mean_bit_drift(bit_probabilities),
        "hidden_relative_l2_drift": hidden_state_drift(hidden_states),
    }
    if prediction_sets is not None:
        if len(prediction_sets) != count:
            raise ValueError("Prediction-set permutation count mismatch")
        result["set_disagreement"] = set_disagreement(prediction_sets)
    if actions is not None:
        if len(actions) != count:
            raise ValueError("Action permutation count mismatch")
        result["action_agreement"] = action_agreement(actions)
    return result


def calibration_ranges(
    coverage_by_permutation: Sequence[float],
    average_set_size_by_permutation: Sequence[float],
) -> Dict[str, float]:
    coverage = np.asarray(coverage_by_permutation, dtype=np.float64)
    sizes = np.asarray(average_set_size_by_permutation, dtype=np.float64)
    if coverage.ndim != 1 or coverage.size < 2 or sizes.shape != coverage.shape:
        raise ValueError("Calibration vectors must have matching length >= 2")
    if not np.isfinite(coverage).all() or not np.isfinite(sizes).all():
        raise ValueError("Calibration vectors must be finite")
    if (coverage < 0).any() or (coverage > 1).any() or (sizes < 0).any():
        raise ValueError("Invalid coverage or set-size value")
    return {
        "coverage_range": float(coverage.max() - coverage.min()),
        "average_set_size_range": float(sizes.max() - sizes.min()),
    }


def summarize_state_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("Cannot summarize zero permutation states")
    numeric = [
        "js_drift",
        "bit_l1_drift",
        "hidden_relative_l2_drift",
        "set_disagreement",
        "action_agreement",
    ]
    summary: Dict[str, Any] = {"state_count": len(values)}
    for field in numeric:
        observed = np.asarray([row[field] for row in values if field in row], dtype=np.float64)
        if observed.size:
            summary[field] = {
                "mean": float(observed.mean()),
                "median": float(np.median(observed)),
                "max": float(observed.max()),
            }
    return summary
