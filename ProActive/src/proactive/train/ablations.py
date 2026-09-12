"""Shared, fail-closed state transforms for predeclared Week 8 ablations."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

from proactive.train.state_data import ACTION_ORDER, OBSERVATION_FEATURES, PROBE_ORDER


FEATURE_ABLATIONS: Mapping[str, str | None] = {
    "no_answer_flip": "flip",
    "no_confidence_shift": "conf_shift",
    "no_semantic_match": "semantic_match",
    "no_relation_probe": None,
}


def validate_week8_ablation_authorization(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the exact owner-approved compute and split boundary."""

    execution = config.get("ablation_execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Week 8 ablation execution block is missing")
    authorization = execution.get("compute_authorization")
    expected = {
        "approved": True,
        "approved_on": "2026-09-11",
        "max_physical_gpus": 2,
        "seed": 42,
        "evidence_split": "val",
        "approved_combined_gpu_hours": 8.0,
        "training_timeout_minutes": 45,
        "frontier_timeout_minutes": 20,
        "heldout_shift_tuning_allowed": False,
    }
    if not isinstance(authorization, Mapping):
        raise ValueError("Week 8 ablation compute authorization is missing")
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise ValueError(f"Week 8 ablation authorization drift: {key}")
    if execution.get("split") != "val" or execution.get("locked_test_reopen_allowed") is not False:
        raise ValueError("Week 8 ablation split firewall drift")
    return authorization


def _ablation_state_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Ablation state is missing a non-empty state_id")
    return value if value.endswith("|ablation") else f"{value}|ablation"


def apply_state_record_ablation(
    record: Mapping[str, Any], ablation_id: str
) -> Dict[str, Any] | None:
    """Return a transformed partial state without mutating the source.

    ``None`` means the state is outside the declared no-relation support because
    it already contains relation evidence. This mirrors the tensor-manifest
    filter and prevents a policy target from being trained on forbidden input.
    """

    if ablation_id not in FEATURE_ABLATIONS:
        raise ValueError(f"Unknown feature ablation: {ablation_id}")
    if record.get("record_type") != "partial_state_v1":
        raise ValueError("Feature ablations require partial_state_v1 records")
    state = copy.deepcopy(dict(record))
    learner = state.get("learner_input")
    if not isinstance(learner, dict):
        raise ValueError("Feature-ablation state has no learner_input mapping")
    names = learner.get("acquired_probe_names")
    observations = learner.get("acquired_observations")
    action_mask = learner.get("action_mask")
    if not isinstance(names, list) or not isinstance(observations, list):
        raise ValueError("Feature-ablation state has malformed acquired evidence")
    if not isinstance(action_mask, dict) or set(action_mask) != set(ACTION_ORDER):
        raise ValueError("Feature-ablation state has malformed action_mask")
    if len(names) != len(observations):
        raise ValueError("Feature-ablation state has inconsistent acquired evidence")

    if ablation_id == "no_relation_probe":
        if "relation" in names:
            return None
        action_mask["relation"] = 0
    else:
        feature = FEATURE_ABLATIONS[ablation_id]
        assert feature in OBSERVATION_FEATURES
        for observation in observations:
            if not isinstance(observation, dict) or set(observation) != {
                "probe_id",
                *OBSERVATION_FEATURES,
            }:
                raise ValueError("Feature-ablation state has malformed observation")
            observation[feature] = 0

    state["state_id"] = _ablation_state_id(state.get("state_id"))
    return state


def apply_teacher_record_ablation(
    record: Mapping[str, Any], ablation_id: str
) -> Dict[str, Any]:
    """Transform cached future observations used during an ablated rollout."""

    if ablation_id not in FEATURE_ABLATIONS:
        raise ValueError(f"Unknown feature ablation: {ablation_id}")
    teacher = copy.deepcopy(dict(record))
    probes = teacher.get("probes")
    if not isinstance(probes, dict):
        raise ValueError("Feature-ablation teacher row has no probes mapping")
    feature = FEATURE_ABLATIONS[ablation_id]
    if feature is not None:
        for probe_name, payload in probes.items():
            if probe_name not in PROBE_ORDER or not isinstance(payload, dict):
                raise ValueError("Feature-ablation teacher row has malformed probes")
            if feature not in payload:
                raise ValueError(
                    f"Feature-ablation teacher probe {probe_name} lacks {feature}"
                )
            payload[feature] = 0
    return teacher
