from __future__ import annotations

import math

from proactive.eval.permutation import (
    action_agreement,
    calibration_ranges,
    distinct_permutations,
    set_disagreement,
    state_permutation_metrics,
)


def test_distinct_permutation_cap_and_determinism() -> None:
    first = distinct_permutations([1, 2, 3, 4], maximum=10, seed_key="x")
    second = distinct_permutations([1, 2, 3, 4], maximum=10, seed_key="x")
    assert first == second
    assert len(first) == 10
    assert len({tuple(value) for value in first}) == 10


def test_zero_drift_for_identical_outputs() -> None:
    metrics = state_permutation_metrics(
        six_way_probabilities=[[0.6, 0.4], [0.6, 0.4]],
        bit_probabilities=[[0.2, 0.3, 0.4], [0.2, 0.3, 0.4]],
        hidden_states=[[1.0, 2.0], [1.0, 2.0]],
        prediction_sets=[[0], [0]],
        actions=["blur", "blur"],
    )
    assert metrics["js_drift"] == 0.0
    assert metrics["bit_l1_drift"] == 0.0
    assert metrics["hidden_relative_l2_drift"] == 0.0
    assert metrics["set_disagreement"] == 0.0
    assert metrics["action_agreement"] == 1.0


def test_set_action_and_calibration_drift() -> None:
    assert math.isclose(set_disagreement([[0], [0, 1]]), 0.5)
    assert action_agreement(["blur", "blur", "crop"]) == 2 / 3
    drift = calibration_ranges([0.9, 0.8], [1.2, 1.5])
    assert math.isclose(drift["coverage_range"], 0.1)
    assert math.isclose(drift["average_set_size_range"], 0.3)
