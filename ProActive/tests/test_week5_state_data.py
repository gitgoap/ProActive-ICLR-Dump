from __future__ import annotations

import copy

import pytest
import torch

from proactive.train.state_data import ACTION_ORDER, vectorize_state
from proactive.train.vectorized import (
    BudgetProjectedDataset,
    TensorStateDataset,
    empty_state_view,
    project_model_input,
    stack_vectorized_rows,
)
from tests._week5_fixtures import make_state


def test_vectorization_recomputes_budget_and_never_emits_identity() -> None:
    state = make_state(acquired=("blank", "blur"))
    vector = vectorize_state(state, max_budget=4)
    assert int(vector.model_input["remaining_budget"]) == 2
    assert vector.model_input["action_mask"].shape == (len(ACTION_ORDER),)
    assert not ({"dataset", "model_id", "instance_id"} & set(vector.model_input))
    assert vector.metadata["dataset"] == "pope"


def test_vectorization_rejects_leakage_and_budget_overrun() -> None:
    state = make_state(acquired=("blank", "blur"))
    leaked = copy.deepcopy(state)
    leaked["learner_input"]["dataset_id"] = "pope"
    with pytest.raises(ValueError, match="schema mismatch|Forbidden"):
        vectorize_state(leaked, max_budget=4)
    with pytest.raises(ValueError, match="exceeds"):
        vectorize_state(state, max_budget=1)


def test_empty_view_has_one_row_per_model_instance() -> None:
    rows = [
        vectorize_state(make_state(acquired=(), instance_id="a"), max_budget=7),
        vectorize_state(make_state(acquired=("blank",), instance_id="a"), max_budget=7),
        vectorize_state(make_state(acquired=(), instance_id="b"), max_budget=7),
    ]
    base = TensorStateDataset([stack_vectorized_rows(rows, "train")], "train")
    view = empty_state_view(base)
    assert len(view) == 2
    projected = BudgetProjectedDataset(view, [1, 2, 4])
    assert len(projected) == 6
    assert all(not projected[index]["model_input"]["acquired_mask"].any() for index in range(6))


def test_budget_projection_masks_all_acquisitions_at_zero_remaining() -> None:
    vector = vectorize_state(make_state(acquired=("blank", "blur")), max_budget=7)
    projected = project_model_input(vector.model_input, 2)
    assert int(projected["remaining_budget"]) == 0
    assert not projected["action_mask"][:-1].any()
    assert bool(projected["action_mask"][-1])
