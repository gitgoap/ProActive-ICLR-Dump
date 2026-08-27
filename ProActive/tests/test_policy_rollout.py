from __future__ import annotations

import pytest
import torch

from proactive.networks.diagnostic import build_diagnostic_model
from proactive.policy.rollout import acquire_cached_tensor, oracle_best_subset_rollout, teacher_observation
from proactive.train.state_data import FeatureNormalizer, PROBE_ORDER, vectorize_state
from tests._week5_fixtures import make_state, make_teacher


def test_acquisition_reveals_one_probe_and_updates_budget() -> None:
    state = make_state(acquired=())
    teacher = make_teacher(state)
    vector = vectorize_state(state, max_budget=2)
    updated = acquire_cached_tensor(
        vector.model_input,
        action="blur",
        observation=teacher_observation(teacher, "blur"),
    )
    slot = PROBE_ORDER.index("blur")
    assert bool(updated["acquired_mask"][slot])
    assert not bool(updated["action_mask"][slot])
    assert int(updated["remaining_budget"]) == 1
    assert int(updated["acquired_mask"].sum()) == 1
    assert int(vector.model_input["acquired_mask"].sum()) == 0


def test_acquisition_refuses_repeat_and_inapplicable_relation() -> None:
    state = make_state(acquired=("blur",))
    vector = vectorize_state(state, max_budget=2)
    teacher = make_teacher(state)
    with pytest.raises(ValueError, match="already"):
        acquire_cached_tensor(
            vector.model_input,
            action="blur",
            observation=teacher_observation(teacher, "blur"),
        )
    with pytest.raises(ValueError, match="valid applicable"):
        teacher_observation(teacher, "relation")


def test_oracle_subset_batches_candidates_and_respects_budget() -> None:
    empty_state = make_state(acquired=())
    acquired_state = make_state(acquired=("blank",), instance_id="normalizer")
    empty = vectorize_state(empty_state, max_budget=2)
    acquired = vectorize_state(acquired_state, max_budget=2)
    batched = {
        key: torch.stack([acquired.model_input[key]]) for key in acquired.model_input
    }
    normalizer = FeatureNormalizer.fit([batched])
    architecture = {
        "state_dim": 16,
        "dropout": 0.0,
        "probe_token_dim": 8,
        "action_embedding_dim": 4,
        "budget_embedding_dim": 4,
        "phi_hidden_dim": 16,
        "pooled_dim": 16,
        "max_budget": 7,
    }
    model = build_diagnostic_model("deep_sets", architecture).eval()
    result = oracle_best_subset_rollout(
        diagnostic_model=model,
        initial_model_input=empty.model_input,
        targets=empty.targets,
        teacher_record=make_teacher(empty_state),
        normalizer=normalizer,
        device=torch.device("cpu"),
        aps_threshold=1.0,
        bit_pos_weight=torch.ones(3),
        class_weight=torch.ones(6),
        six_way_weight=0.5,
        signature_weight=0.1,
    )
    assert result["acquisition_cost"] <= 2
    assert result["actions"][-1] == "stop"
