from __future__ import annotations

import copy

import pytest

from proactive.train.voi import (
    add_cached_observation,
    build_multi_cost_voi_record,
    realized_voi,
)
from scripts.build_voi_targets import _jobs_for_records
from tests._week5_fixtures import make_state, make_teacher


def test_realized_voi_formula_and_sign() -> None:
    value = realized_voi(
        current_set_size=4,
        next_set_size=2,
        current_diagnostic_loss=1.0,
        next_diagnostic_loss=0.6,
        eta_loss=0.25,
        cost_multiplier=0.2,
    )
    assert value == pytest.approx(1.9)


def test_counterfactual_reveals_only_requested_observation() -> None:
    state = make_state(acquired=("blank",))
    teacher = make_teacher(state)
    original = copy.deepcopy(state)
    result = add_cached_observation(state, teacher, "blur")
    assert state == original
    assert result["learner_input"]["acquired_probe_names"] == ["blank", "blur"]
    assert len(result["learner_input"]["acquired_observations"]) == 2
    assert "crop" not in result["learner_input"]["acquired_probe_names"]


def test_multi_cost_target_uses_stop_for_nonpositive_actions() -> None:
    state = make_state(acquired=("blank", "blur", "crop"))
    legal = ["brightness", "noise", "grounding"]
    action_results = {
        action: {"set_size": 2, "diagnostic_loss": 1.0, "entropy": 0.8}
        for action in legal
    }
    record = build_multi_cost_voi_record(
        state_record=state,
        max_budget=4,
        current_loss=1.0,
        current_set_size=2,
        current_entropy=0.8,
        action_results=action_results,
        eta_loss=0.25,
        cost_multipliers=[0.0, 0.2],
        provenance={"checkpoint_sha256": "a" * 64},
    )
    assert record["targets_by_cost_multiplier"]["0"]["best_action"] == "stop"
    assert record["targets_by_cost_multiplier"]["0.2"]["best_action"] == "stop"
    assert set(record["counterfactual_components"]) == set(legal)


def test_voi_action_coverage_fails_closed() -> None:
    state = make_state(acquired=("blank", "blur", "crop"))
    with pytest.raises(ValueError, match="coverage mismatch"):
        build_multi_cost_voi_record(
            state_record=state,
            max_budget=4,
            current_loss=1.0,
            current_set_size=2,
            current_entropy=0.8,
            action_results={},
            eta_loss=0.25,
            cost_multipliers=[0.0],
            provenance={},
        )


def test_voi_pilot_job_limit_stops_counterfactual_expansion() -> None:
    states = [
        make_state(acquired=(), instance_id=f"pilot-{index}")
        for index in range(10)
    ]
    teacher = {
        (state["metadata"]["model_id"], state["metadata"]["instance_id"]): make_teacher(state)
        for state in states
    }
    vectors, jobs = _jobs_for_records(states, teacher=teacher, budgets=[1, 2, 4], max_jobs=2)
    assert len(jobs) == 2
    assert len(vectors) <= 2 * 8
