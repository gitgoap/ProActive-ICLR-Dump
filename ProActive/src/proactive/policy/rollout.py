"""Leakage-safe sequential rollout over cached independent observations."""

from __future__ import annotations

import math
import itertools
from typing import Any, Dict, Mapping, Sequence

import torch

from proactive.conformal.aps import prediction_sets
from proactive.networks.losses import diagnostic_loss
from proactive.policy.controller import select_action, validate_trace
from proactive.train.state_data import (
    ACTION_ORDER,
    DEFAULT_SEVERITIES,
    OBSERVATION_FEATURES,
    PROBE_NUMERIC_FEATURES,
    PROBE_ORDER,
)


def acquire_cached_tensor(
    model_input: Mapping[str, torch.Tensor],
    *,
    action: str,
    observation: Mapping[str, Any],
    severities: Mapping[str, float] = DEFAULT_SEVERITIES,
) -> Dict[str, torch.Tensor]:
    """Reveal exactly one cached action and update the learner state."""

    if action not in PROBE_ORDER:
        raise ValueError(f"Unknown acquisition action: {action}")
    output = {key: value.clone() for key, value in model_input.items()}
    if output["clean_features"].ndim != 1:
        raise ValueError("acquire_cached_tensor expects one unbatched state")
    slot = PROBE_ORDER.index(action)
    if bool(output["acquired_mask"][slot]):
        raise ValueError(f"Probe {action} was already acquired")
    if not bool(output["action_mask"][slot]):
        raise ValueError(f"Probe {action} is not legal")
    remaining = int(output["remaining_budget"])
    if remaining <= 0:
        raise ValueError("No remaining acquisition budget")
    required = {"probe_id", *OBSERVATION_FEATURES}
    if not required.issubset(observation):
        raise ValueError(f"Cached observation {action} is missing required fields")
    if observation.get("probe_id") != action or observation.get("applicable") not in (1, True):
        raise ValueError(f"Cached observation {action} identity/applicability mismatch")

    def number(name: str) -> float:
        value = observation.get(name)
        if isinstance(value, bool) and name not in {"flip", "applicable"}:
            raise ValueError(f"Observation {action}.{name} must be numeric")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Observation {action}.{name} must be numeric") from exc
        if not math.isfinite(result):
            raise ValueError(f"Observation {action}.{name} must be finite")
        return result

    values = torch.tensor(
        [
            1.0,
            number("applicable"),
            number("flip"),
            number("conf_shift"),
            number("entropy_shift"),
            number("margin_shift"),
            number("exact_match"),
            number("semantic_match"),
            1.0,
            float(severities[action]),
        ],
        dtype=output["probe_numeric"].dtype,
        device=output["probe_numeric"].device,
    )
    if values.numel() != len(PROBE_NUMERIC_FEATURES):
        raise AssertionError("Probe numeric schema drift")
    acquired_count = int(output["acquired_mask"].sum())
    output["probe_numeric"][slot] = values
    output["acquired_mask"][slot] = True
    output["sequence_indices"][acquired_count] = slot
    output["action_mask"][slot] = False
    output["remaining_budget"] = torch.tensor(
        remaining - 1, dtype=torch.long, device=output["remaining_budget"].device
    )
    if remaining - 1 == 0:
        output["action_mask"][:-1] = False
    output["action_mask"][-1] = True
    return output


def teacher_observation(teacher_record: Mapping[str, Any], action: str) -> Dict[str, Any]:
    probes = teacher_record.get("probes")
    payload = probes.get(action) if isinstance(probes, Mapping) else None
    if not isinstance(payload, Mapping) or payload.get("valid") is not True or payload.get("applicable") is not True:
        raise ValueError(f"Teacher has no valid applicable observation for {action}")
    return {
        "probe_id": action,
        "flip": int(bool(payload["flip"])),
        "conf_shift": float(payload["conf_shift"]),
        "entropy_shift": float(payload["entropy_shift"]),
        "margin_shift": float(payload["margin_shift"]),
        "exact_match": float(payload["exact_match"]),
        "semantic_match": float(payload["semantic_match"]),
        "applicable": 1,
    }


@torch.no_grad()
def diagnostic_snapshot(
    *,
    diagnostic_model: Any,
    model_input: Mapping[str, torch.Tensor],
    normalizer: Any,
    device: torch.device,
    targets: Mapping[str, torch.Tensor] | None = None,
    bit_pos_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
    six_way_weight: float = 0.5,
    signature_weight: float = 0.1,
) -> Dict[str, Any]:
    batch = {key: value.unsqueeze(0).to(device) for key, value in model_input.items()}
    output = diagnostic_model(normalizer.transform(batch))
    result: Dict[str, Any] = {
        "bit_probabilities": torch.sigmoid(output.bit_logits)[0].cpu().tolist(),
        "six_way_probabilities": torch.softmax(output.six_way_logits, dim=-1)[0].cpu().tolist(),
        "signature_prediction": output.signature[0].cpu().tolist(),
        "hidden_state": output.hidden[0].cpu().tolist(),
    }
    if targets is not None:
        if bit_pos_weight is None or class_weight is None:
            raise ValueError("Per-row diagnostic loss requires frozen training weights")
        batched_targets = {key: value.unsqueeze(0).to(device) for key, value in targets.items()}
        loss = diagnostic_loss(
            output,
            batched_targets,
            bit_pos_weight=bit_pos_weight,
            class_weight=class_weight,
            six_way_weight=six_way_weight,
            signature_weight=signature_weight,
            reduction="none",
        )
        result["diagnostic_loss"] = float(loss.total[0].cpu())
    return result


def _finish(
    *,
    actions: list[str],
    current: Mapping[str, torch.Tensor],
    snapshot: Mapping[str, Any],
    max_budget: int,
    relation_legal: bool,
) -> Dict[str, Any]:
    if not actions or actions[-1] != "stop":
        actions.append("stop")
    validate_trace(actions, max_budget=max_budget, relation_legal=relation_legal)
    return {
        "actions": actions,
        "acquisition_cost": int(current["acquired_mask"].sum()),
        "final_model_input": current,
        **dict(snapshot),
    }


def baseline_policy_rollout(
    *,
    acquisition_policy: Any,
    diagnostic_model: Any,
    initial_model_input: Mapping[str, torch.Tensor],
    teacher_record: Mapping[str, Any],
    normalizer: Any,
    device: torch.device,
    instance_id: str,
) -> Dict[str, Any]:
    current = {key: value.clone() for key, value in initial_model_input.items()}
    max_budget = int(current["max_budget"])
    actions: list[str] = []
    relation_legal = bool(current["action_mask"][PROBE_ORDER.index("relation")])
    while int(current["remaining_budget"]) > 0:
        action = acquisition_policy.select(
            instance_id=instance_id,
            step=len(actions),
            acquired=[PROBE_ORDER[index] for index in torch.nonzero(current["acquired_mask"], as_tuple=False).flatten().tolist()],
            legal_mask=current["action_mask"].tolist(),
            predicted_values=None,
        )
        actions.append(action)
        if action == "stop":
            break
        current = acquire_cached_tensor(
            current,
            action=action,
            observation=teacher_observation(teacher_record, action),
        )
    snapshot = diagnostic_snapshot(
        diagnostic_model=diagnostic_model,
        model_input=current,
        normalizer=normalizer,
        device=device,
    )
    return _finish(
        actions=actions,
        current=current,
        snapshot=snapshot,
        max_budget=max_budget,
        relation_legal=relation_legal,
    )


def oracle_next_rollout(
    *,
    diagnostic_model: Any,
    initial_model_input: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    teacher_record: Mapping[str, Any],
    normalizer: Any,
    device: torch.device,
    aps_threshold: float,
    eta_loss: float,
    cost_multiplier: float,
    bit_pos_weight: torch.Tensor,
    class_weight: torch.Tensor,
    six_way_weight: float,
    signature_weight: float,
) -> Dict[str, Any]:
    current = {key: value.clone() for key, value in initial_model_input.items()}
    max_budget = int(current["max_budget"])
    relation_legal = bool(current["action_mask"][PROBE_ORDER.index("relation")])
    actions: list[str] = []
    while int(current["remaining_budget"]) > 0:
        before = diagnostic_snapshot(
            diagnostic_model=diagnostic_model,
            model_input=current,
            normalizer=normalizer,
            device=device,
            targets=targets,
            bit_pos_weight=bit_pos_weight,
            class_weight=class_weight,
            six_way_weight=six_way_weight,
            signature_weight=signature_weight,
        )
        before_size = len(prediction_sets([before["six_way_probabilities"]], aps_threshold)[0])
        best_action = "stop"
        best_value = 0.0
        best_state = None
        legal_indices = torch.nonzero(current["action_mask"][:-1], as_tuple=False).flatten().tolist()
        for index in legal_indices:
            action = PROBE_ORDER[index]
            candidate = acquire_cached_tensor(
                current,
                action=action,
                observation=teacher_observation(teacher_record, action),
            )
            after = diagnostic_snapshot(
                diagnostic_model=diagnostic_model,
                model_input=candidate,
                normalizer=normalizer,
                device=device,
                targets=targets,
                bit_pos_weight=bit_pos_weight,
                class_weight=class_weight,
                six_way_weight=six_way_weight,
                signature_weight=signature_weight,
            )
            after_size = len(prediction_sets([after["six_way_probabilities"]], aps_threshold)[0])
            value = (
                before_size
                - after_size
                + eta_loss * (before["diagnostic_loss"] - after["diagnostic_loss"])
                - cost_multiplier
            )
            if value > best_value:
                best_value = float(value)
                best_action = action
                best_state = candidate
        actions.append(best_action)
        if best_action == "stop":
            break
        current = best_state
    snapshot = diagnostic_snapshot(
        diagnostic_model=diagnostic_model,
        model_input=current,
        normalizer=normalizer,
        device=device,
    )
    return _finish(
        actions=actions,
        current=current,
        snapshot=snapshot,
        max_budget=max_budget,
        relation_legal=relation_legal,
    )


def oracle_best_subset_rollout(
    *,
    diagnostic_model: Any,
    initial_model_input: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    teacher_record: Mapping[str, Any],
    normalizer: Any,
    device: torch.device,
    aps_threshold: float,
    bit_pos_weight: torch.Tensor,
    class_weight: torch.Tensor,
    six_way_weight: float,
    signature_weight: float,
) -> Dict[str, Any]:
    max_budget = int(initial_model_input["max_budget"])
    relation_legal = bool(initial_model_input["action_mask"][PROBE_ORDER.index("relation")])
    legal = [
        PROBE_ORDER[index]
        for index in torch.nonzero(initial_model_input["action_mask"][:-1], as_tuple=False).flatten().tolist()
    ]
    candidate_states: list[Dict[str, torch.Tensor]] = []
    candidate_actions: list[list[str]] = []
    for size in range(0, min(max_budget, len(legal)) + 1):
        for subset in itertools.combinations(legal, size):
            current = {key: value.clone() for key, value in initial_model_input.items()}
            for action in subset:
                current = acquire_cached_tensor(
                    current,
                    action=action,
                    observation=teacher_observation(teacher_record, action),
                )
            candidate_states.append(current)
            candidate_actions.append(list(subset))
    if not candidate_states:
        raise AssertionError("Oracle subset search produced no candidate")
    batched_input = {
        key: torch.stack([state[key] for state in candidate_states]).to(device)
        for key in candidate_states[0]
    }
    with torch.no_grad():
        output = diagnostic_model(normalizer.transform(batched_input))
        repeated_targets = {
            key: value.unsqueeze(0).expand(len(candidate_states), *value.shape).to(device)
            for key, value in targets.items()
        }
        losses = diagnostic_loss(
            output,
            repeated_targets,
            bit_pos_weight=bit_pos_weight,
            class_weight=class_weight,
            six_way_weight=six_way_weight,
            signature_weight=signature_weight,
            reduction="none",
        ).total.detach().cpu()
        bit_probability = torch.sigmoid(output.bit_logits).detach().cpu()
        six_probability = torch.softmax(output.six_way_logits, dim=-1).detach().cpu()
        signatures = output.signature.detach().cpu()
        hidden = output.hidden.detach().cpu()
    sets = prediction_sets(six_probability.tolist(), aps_threshold)
    best_index = min(
        range(len(candidate_states)),
        key=lambda index: (
            len(sets[index]),
            float(losses[index]),
            len(candidate_actions[index]),
            tuple(candidate_actions[index]),
        ),
    )
    best_state = candidate_states[best_index]
    best_actions = candidate_actions[best_index]
    best_snapshot = {
        "bit_probabilities": bit_probability[best_index].tolist(),
        "six_way_probabilities": six_probability[best_index].tolist(),
        "signature_prediction": signatures[best_index].tolist(),
        "hidden_state": hidden[best_index].tolist(),
    }
    return _finish(
        actions=best_actions,
        current=best_state,
        snapshot=best_snapshot,
        max_budget=max_budget,
        relation_legal=relation_legal,
    )


@torch.no_grad()
def learned_policy_rollout(
    *,
    policy_model: Any,
    initial_model_input: Mapping[str, torch.Tensor],
    teacher_record: Mapping[str, Any],
    normalizer: Any,
    device: torch.device,
    force_full_budget: bool = False,
) -> Dict[str, Any]:
    current = {key: value.clone() for key, value in initial_model_input.items()}
    max_budget = int(current["max_budget"])
    actions: list[str] = []
    values_by_step: list[Dict[str, float]] = []
    relation_legal = bool(current["action_mask"][PROBE_ORDER.index("relation")])
    while int(current["remaining_budget"]) > 0:
        batched = {key: value.unsqueeze(0).to(device) for key, value in current.items()}
        predicted = policy_model(normalizer.transform(batched))[0].detach().cpu().tolist()
        if force_full_budget:
            legal = torch.nonzero(current["action_mask"][:-1], as_tuple=False).flatten().tolist()
            if not legal:
                action = "stop"
            else:
                best = max(legal, key=lambda index: (float(predicted[index]), -index))
                action = PROBE_ORDER[best]
        else:
            action = select_action(predicted, current["action_mask"].tolist())
        values_by_step.append(dict(zip(ACTION_ORDER, map(float, predicted))))
        actions.append(action)
        if action == "stop":
            break
        current = acquire_cached_tensor(
            current,
            action=action,
            observation=teacher_observation(teacher_record, action),
        )
    if not actions or actions[-1] != "stop":
        actions.append("stop")
    validate_trace(actions, max_budget=max_budget, relation_legal=relation_legal)
    final_batch = {key: value.unsqueeze(0).to(device) for key, value in current.items()}
    diagnostic = policy_model.diagnostic(normalizer.transform(final_batch))
    return {
        "actions": actions,
        "values_by_step": values_by_step,
        "acquisition_cost": int(current["acquired_mask"].sum()),
        "final_model_input": current,
        "bit_probabilities": torch.sigmoid(diagnostic.bit_logits)[0].cpu().tolist(),
        "six_way_probabilities": torch.softmax(diagnostic.six_way_logits, dim=-1)[0].cpu().tolist(),
        "signature_prediction": diagnostic.signature[0].cpu().tolist(),
        "hidden_state": diagnostic.hidden[0].cpu().tolist(),
    }
