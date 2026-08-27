"""Realized counterfactual VOI targets from the frozen teacher cache."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from proactive.conformal.aps import prediction_sets
from proactive.train.state_data import ACTION_ORDER, PROBE_ORDER, vectorize_state
from proactive.utils.hashing import hash_dict


def add_cached_observation(
    state_record: Mapping[str, Any],
    teacher_record: Mapping[str, Any],
    action: str,
) -> Dict[str, Any]:
    """Create S∪{a} while exposing no other unacquired teacher observation."""

    if action not in PROBE_ORDER:
        raise ValueError(f"Unknown non-stop action: {action}")
    state = copy.deepcopy(dict(state_record))
    learner = state.get("learner_input")
    probes = teacher_record.get("probes")
    if not isinstance(learner, dict) or not isinstance(probes, Mapping):
        raise ValueError("Malformed state/teacher blocks")
    names = learner.get("acquired_probe_names")
    observations = learner.get("acquired_observations")
    if not isinstance(names, list) or not isinstance(observations, list):
        raise ValueError("Malformed acquired state")
    if action in names:
        raise ValueError(f"Cannot reacquire probe {action}")
    payload = probes.get(action)
    if not isinstance(payload, Mapping) or payload.get("valid") is not True or payload.get("applicable") is not True:
        raise ValueError(f"Cached action {action} is absent, invalid, or inapplicable")
    observations.append(
        {
            "probe_id": action,
            "flip": int(bool(payload["flip"])),
            "conf_shift": float(payload["conf_shift"]),
            "entropy_shift": float(payload["entropy_shift"]),
            "margin_shift": float(payload["margin_shift"]),
            "exact_match": float(payload["exact_match"]),
            "semantic_match": float(payload["semantic_match"]),
            "applicable": int(bool(payload["applicable"])),
        }
    )
    names.append(action)
    learner["remaining_budget"] = 7 - len(names)
    learner["action_mask"][action] = 0
    state["state_id"] = f"counterfactual:{hash_dict({'source': state_record['state_id'], 'action': action})[:24]}"
    return state


def legal_non_stop_actions(vectorized: Any) -> list[str]:
    mask = vectorized.model_input["action_mask"].tolist()
    return [name for name, legal in zip(ACTION_ORDER[:-1], mask[:-1]) if legal]


def realized_voi(
    *,
    current_set_size: int,
    next_set_size: int,
    current_diagnostic_loss: float,
    next_diagnostic_loss: float,
    eta_loss: float,
    cost_multiplier: float,
    action_cost: float = 1.0,
) -> float:
    values = (
        current_set_size,
        next_set_size,
        current_diagnostic_loss,
        next_diagnostic_loss,
        eta_loss,
        cost_multiplier,
        action_cost,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("VOI components must be finite")
    if current_set_size < 0 or next_set_size < 0 or action_cost < 0:
        raise ValueError("Set sizes and action cost must be nonnegative")
    return float(
        (current_set_size - next_set_size)
        + eta_loss * (current_diagnostic_loss - next_diagnostic_loss)
        - cost_multiplier * action_cost
    )


def aps_set_size(probabilities: Sequence[float], threshold: float) -> int:
    return len(prediction_sets([probabilities], threshold)[0])


def build_voi_record(
    *,
    state_record: Mapping[str, Any],
    max_budget: int,
    current_loss: float,
    current_set_size: int,
    action_results: Mapping[str, Mapping[str, float]],
    eta_loss: float,
    cost_multiplier: float,
    provenance: Mapping[str, str],
) -> Dict[str, Any]:
    current = vectorize_state(state_record, max_budget=max_budget)
    legal = legal_non_stop_actions(current)
    if set(action_results) != set(legal):
        raise ValueError(
            f"VOI action coverage mismatch: expected {sorted(legal)}, got {sorted(action_results)}"
        )
    values: Dict[str, float] = {"stop": 0.0}
    components: Dict[str, Any] = {}
    for action in legal:
        result = action_results[action]
        value = realized_voi(
            current_set_size=current_set_size,
            next_set_size=int(result["set_size"]),
            current_diagnostic_loss=current_loss,
            next_diagnostic_loss=float(result["diagnostic_loss"]),
            eta_loss=eta_loss,
            cost_multiplier=cost_multiplier,
        )
        values[action] = value
        components[action] = {
            "next_set_size": int(result["set_size"]),
            "next_diagnostic_loss": float(result["diagnostic_loss"]),
        }
    positive_actions = [action for action in legal if values[action] > 0.0]
    best_action = (
        max(positive_actions, key=lambda action: (values[action], -ACTION_ORDER.index(action)))
        if positive_actions
        else "stop"
    )
    metadata = state_record["metadata"]
    identity = {
        "state_id": state_record["state_id"],
        "max_budget": max_budget,
        "cost_multiplier": cost_multiplier,
    }
    record: Dict[str, Any] = {
        "record_type": "voi_target_v1",
        "voi_id": f"voi:{hash_dict(identity)[:24]}",
        "state_id": state_record["state_id"],
        "max_budget": max_budget,
        "remaining_budget": int(current.model_input["remaining_budget"]),
        "metadata": {
            key: metadata[key]
            for key in ("instance_id", "group_id", "dataset", "split", "model_id")
        },
        "acquired_probe_names": list(state_record["learner_input"]["acquired_probe_names"]),
        "legal_actions": legal + ["stop"],
        "realized_voi": values,
        "best_action": best_action,
        "current_set_size": int(current_set_size),
        "current_diagnostic_loss": float(current_loss),
        "counterfactual_components": components,
        "eta_loss": float(eta_loss),
        "cost_multiplier": float(cost_multiplier),
        "provenance": dict(provenance),
    }
    record["record_sha256"] = hash_dict(record)
    return record


def build_multi_cost_voi_record(
    *,
    state_record: Mapping[str, Any],
    max_budget: int,
    current_loss: float,
    current_set_size: int,
    current_entropy: float,
    action_results: Mapping[str, Mapping[str, float]],
    eta_loss: float,
    cost_multipliers: Sequence[float],
    provenance: Mapping[str, str],
) -> Dict[str, Any]:
    """Store one counterfactual evaluation and all validation lambda targets.

    The expensive diagnostic forward passes do not depend on the cost
    multiplier. Storing the loss/set-size components once avoids a fivefold
    duplication of the Week 6 target corpus while preserving the exact VOI
    value for every declared multiplier.
    """

    current = vectorize_state(state_record, max_budget=max_budget)
    if not all(
        np.isfinite(float(value))
        for value in (current_loss, current_set_size, current_entropy, eta_loss)
    ):
        raise ValueError("Current VOI components must be finite")
    if current_set_size < 0:
        raise ValueError("Current APS set size must be nonnegative")
    legal = legal_non_stop_actions(current)
    if set(action_results) != set(legal):
        raise ValueError(
            f"VOI action coverage mismatch: expected {sorted(legal)}, got {sorted(action_results)}"
        )
    lambdas = [float(value) for value in cost_multipliers]
    if not lambdas or len(lambdas) != len(set(lambdas)):
        raise ValueError("Cost multipliers must be non-empty and unique")
    if any(not np.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError("Cost multipliers must be finite and nonnegative")
    components: Dict[str, Any] = {}
    for action in legal:
        result = action_results[action]
        required = (result.get("set_size"), result.get("diagnostic_loss"), result.get("entropy"))
        if not all(value is not None and np.isfinite(float(value)) for value in required):
            raise ValueError(f"Counterfactual components must be finite for action {action}")
        if int(result["set_size"]) < 0:
            raise ValueError(f"Counterfactual APS set size must be nonnegative for action {action}")
        components[action] = {
            "set_size_reduction": int(current_set_size) - int(result["set_size"]),
            "diagnostic_loss_reduction": float(current_loss) - float(result["diagnostic_loss"]),
            "next_set_size": int(result["set_size"]),
            "next_diagnostic_loss": float(result["diagnostic_loss"]),
            "next_entropy": float(result["entropy"]),
            "entropy_reduction": float(current_entropy) - float(result["entropy"]),
            "cost": 1.0,
        }
    targets_by_multiplier: Dict[str, Any] = {}
    for multiplier in lambdas:
        values = {"stop": 0.0}
        for action in legal:
            item = components[action]
            values[action] = realized_voi(
                current_set_size=current_set_size,
                next_set_size=item["next_set_size"],
                current_diagnostic_loss=current_loss,
                next_diagnostic_loss=item["next_diagnostic_loss"],
                eta_loss=eta_loss,
                cost_multiplier=multiplier,
                action_cost=item["cost"],
            )
        positive_actions = [action for action in legal if values[action] > 0.0]
        best = (
            max(positive_actions, key=lambda action: (values[action], -ACTION_ORDER.index(action)))
            if positive_actions
            else "stop"
        )
        targets_by_multiplier[format(multiplier, ".12g")] = {
            "realized_voi": values,
            "best_action": best,
        }
    metadata = state_record["metadata"]
    identity = {"state_id": state_record["state_id"], "max_budget": max_budget}
    record: Dict[str, Any] = {
        "record_type": "voi_target_multi_cost_v1",
        "voi_id": f"voi:{hash_dict(identity)[:24]}",
        "state_id": state_record["state_id"],
        "max_budget": int(max_budget),
        "remaining_budget": int(current.model_input["remaining_budget"]),
        "metadata": {
            key: metadata[key]
            for key in ("instance_id", "group_id", "dataset", "split", "model_id")
        },
        "acquired_probe_names": list(state_record["learner_input"]["acquired_probe_names"]),
        "legal_actions": legal + ["stop"],
        "current_set_size": int(current_set_size),
        "current_diagnostic_loss": float(current_loss),
        "current_entropy": float(current_entropy),
        "counterfactual_components": components,
        "eta_loss": float(eta_loss),
        "targets_by_cost_multiplier": targets_by_multiplier,
        "provenance": dict(provenance),
    }
    record["record_sha256"] = hash_dict(record)
    return record
