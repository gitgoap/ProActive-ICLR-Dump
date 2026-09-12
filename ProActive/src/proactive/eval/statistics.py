"""Grouped statistical utilities for frozen Week 8 evaluation."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score


MetricFunction = Callable[[Sequence[Mapping[str, Any]]], float]


def base_group(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Trajectory row lacks metadata")
    group_id = metadata.get("group_id")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("Trajectory row lacks a non-empty group_id")
    return group_id


def source_macro_f1(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ValueError("Cannot score an empty trajectory collection")
    truth = np.asarray([row["targets"]["source_bits"] for row in rows], dtype=np.int64)
    pred = (
        np.asarray([row["bit_probabilities"] for row in rows], dtype=np.float64) >= 0.5
    ).astype(np.int64)
    return float(f1_score(truth, pred, average="macro", zero_division=0))


def six_way_macro_f1(rows: Sequence[Mapping[str, Any]]) -> float:
    truth = np.asarray([row["targets"]["six_way"] for row in rows], dtype=np.int64)
    pred = np.asarray(
        [int(np.argmax(row["six_way_probabilities"])) for row in rows], dtype=np.int64
    )
    return float(f1_score(truth, pred, average="macro", zero_division=0))


def empirical_coverage(rows: Sequence[Mapping[str, Any]]) -> float:
    values = []
    for row in rows:
        prediction_set = row.get("prediction_set")
        if not isinstance(prediction_set, list):
            raise ValueError("Prediction set missing from calibrated trajectory row")
        values.append(int(row["targets"]["six_way"]) in prediction_set)
    return float(np.mean(values))


def average_set_size(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(np.mean([len(row["prediction_set"]) for row in rows]))


def mean_cost(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(np.mean([float(row["acquisition_cost"]) for row in rows]))


METRICS: Dict[str, MetricFunction] = {
    "source_bit_macro_f1": source_macro_f1,
    "six_way_macro_f1": six_way_macro_f1,
    "empirical_coverage": empirical_coverage,
    "average_set_size": average_set_size,
    "mean_acquisition_cost": mean_cost,
}


def grouped_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: MetricFunction,
    resamples: int,
    confidence: float,
    seed: int,
) -> Dict[str, float]:
    """Bootstrap complete visual groups, never individual partial states."""

    if resamples < 1000:
        raise ValueError("Week 8 requires at least 1,000 bootstrap resamples")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[base_group(row)].append(row)
    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError("Grouped bootstrap requires at least two base groups")
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample: List[Mapping[str, Any]] = []
        for _ in groups:
            sample.extend(grouped[rng.choice(groups)])
        estimates.append(metric(sample))
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": metric(rows),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
        "resamples": resamples,
        "group_count": len(groups),
        "row_count": len(rows),
    }


def paired_grouped_bootstrap_difference(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    metric: MetricFunction,
    resamples: int,
    confidence: float,
    seed: int,
) -> Dict[str, float]:
    """Paired grouped bootstrap of ``metric(left) - metric(right)``."""

    left_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    right_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in left:
        left_groups[base_group(row)].append(row)
    for row in right:
        right_groups[base_group(row)].append(row)
    if set(left_groups) != set(right_groups):
        raise ValueError("Paired methods do not contain identical base groups")
    groups = sorted(left_groups)
    if resamples < 1000 or len(groups) < 2:
        raise ValueError("Paired bootstrap requires >=1000 resamples and >=2 groups")
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        left_sample: List[Mapping[str, Any]] = []
        right_sample: List[Mapping[str, Any]] = []
        for _ in groups:
            group = rng.choice(groups)
            left_sample.extend(left_groups[group])
            right_sample.extend(right_groups[group])
        estimates.append(metric(left_sample) - metric(right_sample))
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    # Add-one smoothing prevents a zero Monte Carlo p-value. The test is
    # two-sided because a control or ablation can outperform the main method.
    lower_tail = (1 + sum(value <= 0.0 for value in estimates)) / (resamples + 1)
    upper_tail = (1 + sum(value >= 0.0 for value in estimates)) / (resamples + 1)
    return {
        "estimate": metric(left) - metric(right),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
        "resamples": resamples,
        "group_count": len(groups),
        "row_count": len(left),
        "p_value_two_sided": min(1.0, 2.0 * min(lower_tail, upper_tail)),
    }


def holm_bonferroni(
    p_values: Sequence[float], alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """Return Holm-adjusted p-values and familywise rejection decisions."""

    if not p_values:
        raise ValueError("Holm correction requires at least one p-value")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in p_values
    ):
        raise ValueError("p-values must be finite values in [0, 1]")
    ordered = sorted(
        range(len(p_values)), key=lambda index: (float(p_values[index]), index)
    )
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(ordered):
        candidate = (len(p_values) - rank) * float(p_values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return [
        {
            "p_value_holm": adjusted[index],
            "reject_familywise": adjusted[index] <= alpha,
        }
        for index in range(len(p_values))
    ]


def stable_example_rank(seed: int, label: str, row: Mapping[str, Any]) -> str:
    metadata = row["metadata"]
    identity = f"{metadata.get('model_id')}|{metadata.get('instance_id')}"
    return hashlib.sha256(f"{seed}|{label}|{identity}".encode("utf-8")).hexdigest()
