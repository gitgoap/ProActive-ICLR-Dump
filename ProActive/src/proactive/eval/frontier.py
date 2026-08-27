"""Matched-cost frontier aggregation and dominance checks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from proactive.conformal.aps import evaluate_sets
from proactive.eval.diagnostic_metrics import diagnostic_metrics


def summarize_trajectories(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty trajectory collection")
    identities = [
        (row["metadata"]["model_id"], row["metadata"]["instance_id"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Trajectory condition contains duplicate model-instance rows")
    bit_probability = [row["bit_probabilities"] for row in rows]
    six_probability = [row["six_way_probabilities"] for row in rows]
    bit_target = [row["targets"]["source_bits"] for row in rows]
    six_target = [row["targets"]["six_way"] for row in rows]
    signature_prediction = [row["signature_prediction"] for row in rows]
    signature_target = [row["targets"]["signature"] for row in rows]
    sets = [row.get("prediction_set") for row in rows]
    if any(value is None for value in sets) and not all(value is None for value in sets):
        raise ValueError("Trajectory condition mixes calibrated and uncalibrated rows")
    actions = Counter(action for row in rows for action in row["actions"])
    return {
        "row_count": len(rows),
        "mean_acquisition_cost": float(np.mean([row["acquisition_cost"] for row in rows])),
        "diagnostic": diagnostic_metrics(
            bit_probability,
            six_probability,
            bit_target,
            six_target,
            signature_prediction,
            signature_target,
        ),
        "prediction_sets": None if all(value is None for value in sets) else evaluate_sets(sets, six_target),
        "action_counts": dict(actions),
    }


def pareto_flags(
    points: Sequence[Mapping[str, float]],
    *,
    cost_field: str = "mean_acquisition_cost",
    quality_field: str = "source_bit_macro_f1",
) -> list[bool]:
    """Return nondominance flags for lower cost and higher quality."""

    output = []
    for index, point in enumerate(points):
        cost = float(point[cost_field])
        quality = float(point[quality_field])
        dominated = False
        for other_index, other in enumerate(points):
            if other_index == index:
                continue
            other_cost = float(other[cost_field])
            other_quality = float(other[quality_field])
            if other_cost <= cost and other_quality >= quality and (
                other_cost < cost or other_quality > quality
            ):
                dominated = True
                break
        output.append(not dominated)
    return output


def matched_budget(rows: Iterable[Mapping[str, Any]], maximum_budget: int) -> None:
    if maximum_budget < 0:
        raise ValueError("maximum_budget must be nonnegative")
    for row in rows:
        cost = row.get("acquisition_cost")
        if isinstance(cost, bool) or not isinstance(cost, int) or not 0 <= cost <= maximum_budget:
            raise ValueError("Trajectory violates matched acquisition budget")
