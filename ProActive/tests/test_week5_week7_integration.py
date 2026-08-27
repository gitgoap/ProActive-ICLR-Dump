from __future__ import annotations

import copy

import torch

from proactive.conformal.aps import fit_aps, prediction_sets
from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.losses import diagnostic_loss, policy_loss
from proactive.train.state_data import ACTION_ORDER, FeatureNormalizer, SIX_WAY_LABELS, vectorize_state
from proactive.train.voi import add_cached_observation, build_multi_cost_voi_record
from tests._week5_fixtures import make_state, make_teacher


def _batch(vectors: list) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = {
        key: torch.stack([vector.model_input[key] for vector in vectors])
        for key in vectors[0].model_input
    }
    targets = {
        key: torch.stack([vector.targets[key] for vector in vectors])
        for key in vectors[0].targets
    }
    return inputs, targets


def test_synthetic_diagnostic_aps_voi_policy_chain() -> None:
    bits = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
    )
    states = []
    vectors = []
    for index, (label, target_bits) in enumerate(zip(SIX_WAY_LABELS, bits)):
        state = make_state(
            acquired=("blank", "blur", "crop", "brightness", "noise"),
            instance_id=f"integration-{index}",
        )
        state["targets"]["teacher_label6"] = label
        state["targets"]["teacher_bits"] = dict(zip(("visual", "language", "alignment"), target_bits))
        state["learner_input"]["clean_features"]["answer_prob"] -= index * 0.05
        states.append(state)
        vectors.append(vectorize_state(state, max_budget=6))
    inputs, targets = _batch(vectors)
    normalizer = FeatureNormalizer.fit([inputs])
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
    model = build_diagnostic_model("deep_sets", architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    output = model(normalizer.transform(inputs))
    loss = diagnostic_loss(
        output,
        targets,
        bit_pos_weight=torch.ones(3),
        class_weight=torch.ones(6),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()
    probabilities = torch.softmax(output.six_way_logits.detach(), dim=-1)
    fitted = fit_aps(probabilities.tolist(), list(range(6)), alpha=0.1)
    current = vectors[0]
    current_output = model(
        normalizer.transform({key: value.unsqueeze(0) for key, value in current.model_input.items()})
    )
    current_loss = diagnostic_loss(
        current_output,
        {key: value.unsqueeze(0) for key, value in current.targets.items()},
        bit_pos_weight=torch.ones(3),
        class_weight=torch.ones(6),
        reduction="none",
    ).total.item()
    current_probability = torch.softmax(current_output.six_way_logits, dim=-1)[0].tolist()
    current_size = len(prediction_sets([current_probability], fitted.threshold)[0])
    teacher = make_teacher(states[0])
    next_state = add_cached_observation(copy.deepcopy(states[0]), teacher, "grounding")
    next_vector = vectorize_state(next_state, max_budget=6)
    next_output = model(
        normalizer.transform({key: value.unsqueeze(0) for key, value in next_vector.model_input.items()})
    )
    next_loss = diagnostic_loss(
        next_output,
        {key: value.unsqueeze(0) for key, value in next_vector.targets.items()},
        bit_pos_weight=torch.ones(3),
        class_weight=torch.ones(6),
        reduction="none",
    ).total.item()
    next_probability = torch.softmax(next_output.six_way_logits, dim=-1)[0]
    next_size = len(prediction_sets([next_probability.tolist()], fitted.threshold)[0])
    entropy = lambda value: float(-(value * value.clamp_min(1.0e-12).log()).sum())
    record = build_multi_cost_voi_record(
        state_record=states[0],
        max_budget=6,
        current_loss=current_loss,
        current_set_size=current_size,
        current_entropy=entropy(torch.tensor(current_probability)),
        action_results={
            "grounding": {
                "set_size": next_size,
                "diagnostic_loss": next_loss,
                "entropy": entropy(next_probability),
            }
        },
        eta_loss=0.25,
        cost_multipliers=[0.0, 0.2],
        provenance={"checkpoint_sha256": "a" * 64},
    )
    target_values = torch.full((1, len(ACTION_ORDER)), -1.0)
    legal = torch.zeros_like(target_values, dtype=torch.bool)
    for action in record["legal_actions"]:
        legal[0, ACTION_ORDER.index(action)] = True
        target_values[0, ACTION_ORDER.index(action)] = record["targets_by_cost_multiplier"]["0.2"][
            "realized_voi"
        ][action]
    objective = policy_loss(torch.zeros_like(target_values), target_values, legal)
    assert torch.isfinite(loss.total)
    assert torch.isfinite(objective.total)
    assert record["targets_by_cost_multiplier"]["0.2"]["best_action"] in ACTION_ORDER
