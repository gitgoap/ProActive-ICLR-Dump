"""Strict Week 5+ tensor contract for leakage-safe partial evidence states.

The serialized Week 4 state keeps metadata and targets beside ``learner_input``
for auditing.  This module is the only supported bridge into neural models.  It
validates the complete record, extracts only the declared learner fields, and
returns metadata separately so dataset/model identity cannot be concatenated by
accident.

References: Plan §§5, 7, 16.2, 16.3, and 25.7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch


PROBE_ORDER: Tuple[str, ...] = (
    "blank",
    "blur",
    "crop",
    "brightness",
    "noise",
    "grounding",
    "relation",
)
ACTION_ORDER: Tuple[str, ...] = PROBE_ORDER + ("stop",)
CLEAN_FEATURES: Tuple[str, ...] = (
    "answer_prob",
    "token_entropy_mean",
    "token_margin_mean",
    "answer_len_tokens",
)
OBSERVATION_FEATURES: Tuple[str, ...] = (
    "flip",
    "conf_shift",
    "entropy_shift",
    "margin_shift",
    "exact_match",
    "semantic_match",
    "applicable",
)
PROBE_NUMERIC_FEATURES: Tuple[str, ...] = (
    "acquired",
    "applicable",
    "flip",
    "conf_shift",
    "entropy_shift",
    "margin_shift",
    "exact_match",
    "semantic_match",
    "cost",
    "severity",
)
SIX_WAY_LABELS: Tuple[str, ...] = (
    "visual",
    "language-prior",
    "alignment",
    "mixed",
    "unclear",
    "no-failure",
)
SOURCE_BITS: Tuple[str, ...] = ("visual", "language", "alignment")
SIGNATURE_FIELDS: Tuple[str, ...] = ("V", "L", "A")
FORBIDDEN_MAIN_INPUT_KEYS = {
    "dataset",
    "dataset_id",
    "model_id",
    "model_revision",
    "group_id",
    "instance_id",
    "correct",
    "clean_correct",
    "gold_answer",
    "teacher_bits",
    "teacher_label6",
    "teacher_signature",
    "benchmark_family",
}
DEFAULT_SEVERITIES: Mapping[str, float] = {
    "blank": 0.0,
    "blur": 8.0,
    "crop": 0.65,
    "brightness": 0.15,
    "noise": 25.0,
    "grounding": 0.0,
    "relation": 0.0,
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _binary(value: Any, field: str) -> float:
    if value not in (0, 1, False, True):
        raise ValueError(f"{field} must be binary, got {value!r}")
    return float(value)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _list(value: Any, field: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _reject_forbidden_keys(value: Any, path: str = "learner_input") -> None:
    if isinstance(value, Mapping):
        leaked = FORBIDDEN_MAIN_INPUT_KEYS & set(value)
        if leaked:
            raise ValueError(f"Forbidden learner keys at {path}: {sorted(leaked)}")
        for key, child in value.items():
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


@dataclass(frozen=True)
class VectorizedState:
    """One validated model input plus separately held audit metadata/targets."""

    model_input: Mapping[str, torch.Tensor]
    targets: Mapping[str, torch.Tensor]
    metadata: Mapping[str, str]
    state_id: str
    sampling_sources: Tuple[str, ...]


def vectorize_state(
    record: Mapping[str, Any],
    *,
    max_budget: int,
    severities: Mapping[str, float] = DEFAULT_SEVERITIES,
) -> VectorizedState:
    """Validate and tensorize one ``partial_state_v1`` record.

    ``remaining_budget`` is recomputed from ``max_budget``.  The Week 4 corpus
    was generated under the seven-probe full budget, while Weeks 5–7 train and
    evaluate multiple maximum budgets.  Trusting the serialized full-budget
    remainder would silently give the wrong policy state.
    """

    if record.get("record_type") != "partial_state_v1":
        raise ValueError("Expected record_type='partial_state_v1'")
    state_id = record.get("state_id")
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("State record is missing a non-empty state_id")
    if isinstance(max_budget, bool) or not isinstance(max_budget, int):
        raise ValueError("max_budget must be an integer")
    if max_budget < 0 or max_budget > len(PROBE_ORDER):
        raise ValueError(f"max_budget must be in [0, {len(PROBE_ORDER)}]")

    learner = _mapping(record.get("learner_input"), "learner_input")
    _reject_forbidden_keys(learner)
    expected_learner = {
        "clean_features",
        "acquired_probe_names",
        "acquired_observations",
        "remaining_budget",
        "action_mask",
    }
    if set(learner) != expected_learner:
        raise ValueError(
            "Learner-input schema mismatch: "
            f"expected {sorted(expected_learner)}, got {sorted(learner)}"
        )

    clean = _mapping(learner["clean_features"], "learner_input.clean_features")
    if set(clean) != set(CLEAN_FEATURES):
        raise ValueError(
            f"Clean feature schema mismatch: expected {list(CLEAN_FEATURES)}, "
            f"got {sorted(clean)}"
        )
    clean_tensor = torch.tensor(
        [_finite_number(clean[name], f"clean_features.{name}") for name in CLEAN_FEATURES],
        dtype=torch.float32,
    )

    names = _list(learner["acquired_probe_names"], "acquired_probe_names")
    observations = _list(learner["acquired_observations"], "acquired_observations")
    if len(names) != len(observations):
        raise ValueError("Acquired probe names/observations have different lengths")
    if len(names) != len(set(names)):
        raise ValueError("Acquired probe names contain duplicates")
    unknown = set(names) - set(PROBE_ORDER)
    if unknown:
        raise ValueError(f"Unknown acquired probes: {sorted(unknown)}")
    if len(names) > max_budget:
        raise ValueError(
            f"State cost {len(names)} exceeds requested maximum budget {max_budget}"
        )

    probe_numeric = torch.zeros(
        (len(PROBE_ORDER), len(PROBE_NUMERIC_FEATURES)), dtype=torch.float32
    )
    acquired_mask = torch.zeros(len(PROBE_ORDER), dtype=torch.bool)
    sequence_indices = torch.full((len(PROBE_ORDER),), -1, dtype=torch.long)
    observation_names: List[str] = []
    for sequence_position, raw_observation in enumerate(observations):
        observation = _mapping(
            raw_observation, f"acquired_observations[{sequence_position}]"
        )
        expected_observation = {"probe_id", *OBSERVATION_FEATURES}
        if set(observation) != expected_observation:
            raise ValueError(
                f"Probe observation schema mismatch at index {sequence_position}: "
                f"expected {sorted(expected_observation)}, got {sorted(observation)}"
            )
        probe_name = observation.get("probe_id")
        if probe_name not in PROBE_ORDER:
            raise ValueError(f"Unknown probe_id at observation {sequence_position}: {probe_name!r}")
        observation_names.append(str(probe_name))
        slot = PROBE_ORDER.index(str(probe_name))
        acquired_mask[slot] = True
        sequence_indices[sequence_position] = slot
        probe_numeric[slot] = torch.tensor(
            [
                1.0,
                _binary(observation["applicable"], f"{probe_name}.applicable"),
                _binary(observation["flip"], f"{probe_name}.flip"),
                _finite_number(observation["conf_shift"], f"{probe_name}.conf_shift"),
                _finite_number(observation["entropy_shift"], f"{probe_name}.entropy_shift"),
                _finite_number(observation["margin_shift"], f"{probe_name}.margin_shift"),
                _finite_number(observation["exact_match"], f"{probe_name}.exact_match"),
                _finite_number(observation["semantic_match"], f"{probe_name}.semantic_match"),
                1.0,
                _finite_number(severities.get(str(probe_name)), f"severity.{probe_name}"),
            ],
            dtype=torch.float32,
        )
    if observation_names != names:
        raise ValueError("Acquired observation order does not match acquired_probe_names")

    serialized_action_mask = _mapping(learner["action_mask"], "action_mask")
    if set(serialized_action_mask) != set(ACTION_ORDER):
        raise ValueError(
            f"Action-mask schema mismatch: expected {list(ACTION_ORDER)}, "
            f"got {sorted(serialized_action_mask)}"
        )
    # The serialized mask establishes applicability; budget is applied here.
    action_values: List[float] = []
    remaining_budget = max_budget - len(names)
    for action in PROBE_ORDER:
        serialized = _binary(serialized_action_mask[action], f"action_mask.{action}")
        expected_available = float(action not in names and serialized == 1.0 and remaining_budget >= 1)
        action_values.append(expected_available)
    if _binary(serialized_action_mask["stop"], "action_mask.stop") != 1.0:
        raise ValueError("STOP must always be legal")
    action_values.append(1.0)

    targets = _mapping(record.get("targets"), "targets")
    expected_targets = {
        "clean_correct",
        "teacher_signature",
        "teacher_bits",
        "teacher_label6",
    }
    if set(targets) != expected_targets:
        raise ValueError(
            f"Target schema mismatch: expected {sorted(expected_targets)}, got {sorted(targets)}"
        )
    bits = _mapping(targets["teacher_bits"], "targets.teacher_bits")
    signature = _mapping(targets["teacher_signature"], "targets.teacher_signature")
    if set(bits) != set(SOURCE_BITS):
        raise ValueError("Teacher-bit schema mismatch")
    if set(signature) != set(SIGNATURE_FIELDS):
        raise ValueError("Teacher-signature schema mismatch")
    label = targets.get("teacher_label6")
    if label not in SIX_WAY_LABELS:
        raise ValueError(f"Unknown six-way label: {label!r}")

    metadata = _mapping(record.get("metadata"), "metadata")
    required_metadata = {
        "instance_id",
        "group_id",
        "dataset",
        "split",
        "model_id",
        "model_revision",
    }
    if set(metadata) != required_metadata:
        raise ValueError("State metadata schema mismatch")
    metadata_strings: Dict[str, str] = {}
    for key in sorted(required_metadata):
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"metadata.{key} must be a non-empty string")
        metadata_strings[key] = value

    sampling = _mapping(record.get("sampling"), "sampling")
    sources = sampling.get("sources")
    if not isinstance(sources, list) or not all(isinstance(value, str) for value in sources):
        raise ValueError("sampling.sources must be a list of strings")

    return VectorizedState(
        model_input={
            "clean_features": clean_tensor,
            "probe_numeric": probe_numeric,
            "acquired_mask": acquired_mask,
            "sequence_indices": sequence_indices,
            "remaining_budget": torch.tensor(remaining_budget, dtype=torch.long),
            "max_budget": torch.tensor(max_budget, dtype=torch.long),
            "action_mask": torch.tensor(action_values, dtype=torch.bool),
        },
        targets={
            "source_bits": torch.tensor(
                [_binary(bits[name], f"teacher_bits.{name}") for name in SOURCE_BITS],
                dtype=torch.float32,
            ),
            "six_way": torch.tensor(SIX_WAY_LABELS.index(str(label)), dtype=torch.long),
            "signature": torch.tensor(
                [_finite_number(signature[name], f"teacher_signature.{name}") for name in SIGNATURE_FIELDS],
                dtype=torch.float32,
            ),
        },
        metadata=metadata_strings,
        state_id=state_id,
        sampling_sources=tuple(sources),
    )


def collate_vectorized_states(states: Sequence[VectorizedState]) -> Dict[str, Any]:
    if not states:
        raise ValueError("Cannot collate an empty state batch")
    input_keys = tuple(states[0].model_input)
    target_keys = tuple(states[0].targets)
    for state in states:
        if tuple(state.model_input) != input_keys or tuple(state.targets) != target_keys:
            raise ValueError("Inconsistent vectorized-state schema in batch")
    return {
        "model_input": {
            key: torch.stack([state.model_input[key] for state in states])
            for key in input_keys
        },
        "targets": {
            key: torch.stack([state.targets[key] for state in states])
            for key in target_keys
        },
        "metadata": [dict(state.metadata) for state in states],
        "state_id": [state.state_id for state in states],
        "sampling_sources": [state.sampling_sources for state in states],
    }


@dataclass
class FeatureNormalizer:
    """Train-only z-score statistics for continuous learner features."""

    clean_mean: torch.Tensor
    clean_std: torch.Tensor
    probe_mean: torch.Tensor
    probe_std: torch.Tensor

    @classmethod
    def fit(cls, batches: Iterable[Mapping[str, torch.Tensor]]) -> "FeatureNormalizer":
        clean_chunks: List[torch.Tensor] = []
        probe_chunks: List[torch.Tensor] = []
        for model_input in batches:
            clean = model_input["clean_features"].detach().cpu().float()
            probe = model_input["probe_numeric"].detach().cpu().float()
            mask = model_input["acquired_mask"].detach().cpu().bool()
            if clean.ndim != 2 or probe.ndim != 3 or mask.shape != probe.shape[:2]:
                raise ValueError("Malformed tensors while fitting feature normalizer")
            clean_chunks.append(clean)
            if mask.any():
                probe_chunks.append(probe[mask])
        if not clean_chunks or not probe_chunks:
            raise ValueError("Training data must contain clean and acquired-probe features")
        return cls.fit_components(
            clean_features=torch.cat(clean_chunks, dim=0),
            acquired_probe_features=torch.cat(probe_chunks, dim=0),
        )

    @classmethod
    def fit_components(
        cls,
        *,
        clean_features: torch.Tensor,
        acquired_probe_features: torch.Tensor,
    ) -> "FeatureNormalizer":
        """Fit train-only statistics from explicitly de-duplicated components."""

        clean_all = clean_features.detach().cpu().float()
        probe_all = acquired_probe_features.detach().cpu().float()
        if clean_all.ndim != 2 or clean_all.shape[1] != len(CLEAN_FEATURES):
            raise ValueError("clean_features must have shape [N, 4]")
        if probe_all.ndim != 2 or probe_all.shape[1] != len(PROBE_NUMERIC_FEATURES):
            raise ValueError("acquired_probe_features must have shape [M, 10]")
        if clean_all.shape[0] == 0 or probe_all.shape[0] == 0:
            raise ValueError("Normalizer components cannot be empty")
        if not torch.isfinite(clean_all).all() or not torch.isfinite(probe_all).all():
            raise ValueError("Normalizer components must be finite")
        clean_std = clean_all.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        probe_std = probe_all.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        return cls(
            clean_mean=clean_all.mean(dim=0),
            clean_std=clean_std,
            probe_mean=probe_all.mean(dim=0),
            probe_std=probe_std,
        )

    def transform(self, model_input: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {key: value for key, value in model_input.items()}
        clean = model_input["clean_features"]
        probe = model_input["probe_numeric"]
        mask = model_input["acquired_mask"].bool()
        clean_mean = self.clean_mean.to(device=clean.device, dtype=clean.dtype)
        clean_std = self.clean_std.to(device=clean.device, dtype=clean.dtype)
        probe_mean = self.probe_mean.to(device=probe.device, dtype=probe.dtype)
        probe_std = self.probe_std.to(device=probe.device, dtype=probe.dtype)
        output["clean_features"] = (clean - clean_mean) / clean_std
        normalized_probe = torch.zeros_like(probe)
        if mask.any():
            normalized_probe[mask] = (probe[mask] - probe_mean) / probe_std
        output["probe_numeric"] = normalized_probe
        return output

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "clean_mean": self.clean_mean.cpu(),
            "clean_std": self.clean_std.cpu(),
            "probe_mean": self.probe_mean.cpu(),
            "probe_std": self.probe_std.cpu(),
        }

    @classmethod
    def from_state_dict(cls, values: Mapping[str, torch.Tensor]) -> "FeatureNormalizer":
        expected = {"clean_mean", "clean_std", "probe_mean", "probe_std"}
        if set(values) != expected:
            raise ValueError("Feature-normalizer state schema mismatch")
        return cls(**{key: values[key].detach().cpu().float() for key in expected})


def permute_sequence_indices(
    sequence_indices: torch.Tensor,
    acquired_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Randomly permute only acquired sequence positions, independently per row."""

    if sequence_indices.ndim != 2 or acquired_mask.ndim != 2:
        raise ValueError("Expected batched sequence_indices and acquired_mask")
    output = sequence_indices.clone()
    lengths = acquired_mask.sum(dim=1).tolist()
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length > 1:
            permutation = torch.randperm(length, generator=generator, device="cpu")
            values = output[row, :length].detach().cpu()[permutation]
            output[row, :length] = values.to(output.device)
    return output
