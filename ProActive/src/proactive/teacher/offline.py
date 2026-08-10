"""Offline Week 4 label and partial-state construction.

This module contains only deterministic CPU work.  It deliberately separates
metadata/targets from ``learner_input`` so dataset IDs, model IDs, gold labels,
and unacquired probe observations cannot enter the diagnostic encoder.

References: Plan sections 6, 14, 16.2, 16.3, and 25.6.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from proactive.features.evidence_state import MAX_PROBES, ProbeAction, ProbeObservation
from proactive.teacher.label_computation import (
    LabelThresholds,
    compute_teacher_labels,
    validate_mandatory_probes,
)
from proactive.utils.hashing import hash_dict


CORE_PROBES: Tuple[str, ...] = (
    "blank",
    "blur",
    "crop",
    "brightness",
    "noise",
    "grounding",
)
RELATION_PROBE = "relation"

FIXED_BASELINES: Mapping[str, Tuple[str, ...]] = OrderedDict(
    (
        (
            "blank_first",
            ("blank", "grounding", "blur", "crop", "brightness", "noise", "relation"),
        ),
        (
            "visual_first",
            ("blur", "crop", "brightness", "noise", "blank", "grounding", "relation"),
        ),
        (
            "grounding_first",
            ("grounding", "blank", "blur", "crop", "brightness", "noise", "relation"),
        ),
        (
            "relation_first",
            ("relation", "grounding", "blank", "blur", "crop", "brightness", "noise"),
        ),
    )
)

PROBE_FEATURE_KEYS: Tuple[str, ...] = (
    "flip",
    "conf_shift",
    "entropy_shift",
    "margin_shift",
    "exact_match",
    "semantic_match",
    "applicable",
)


def teacher_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    """Return the unique model-instance identity for a teacher record."""
    model_id = record.get("model_id")
    instance_id = record.get("instance_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Teacher record has no non-empty model_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("Teacher record has no non-empty instance_id")
    return model_id, instance_id


def stable_shard_id(instance_id: str, model_id: str, num_shards: int) -> int:
    """Assign a model-instance pair to a stable, row-order-independent shard."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    digest = hashlib.sha256(f"{model_id}|{instance_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def validate_resume_teacher_records(
    records: Iterable[Mapping[str, Any]],
    model_id: str,
    selected_ids: Set[str],
    manifest_sha256: str,
    frozen_config_sha256: str,
    shard_id: int,
    num_shards: int,
) -> Set[Tuple[str, str]]:
    """Validate existing shard rows before any resumable append occurs."""
    completed: Set[Tuple[str, str]] = set()
    for row_number, row in enumerate(records, start=1):
        key = teacher_key(row)
        if key in completed:
            raise ValueError(f"Duplicate teacher key at row {row_number}: {key}")
        if row.get("record_type") != "teacher_cache":
            raise ValueError(f"Unexpected record_type at row {row_number}")
        if key[0] != model_id or key[1] not in selected_ids:
            raise ValueError(f"Existing row is outside current selection: {key}")
        if row.get("source_manifest_sha256") != manifest_sha256:
            raise ValueError(f"Manifest hash drift at row {row_number}")
        if row.get("frozen_probe_config_sha256") != frozen_config_sha256:
            raise ValueError(f"Frozen probe config hash drift at row {row_number}")
        if row.get("shard_id") != shard_id or row.get("num_shards") != num_shards:
            raise ValueError(f"Shard metadata drift at row {row_number}")
        if stable_shard_id(key[1], key[0], num_shards) != shard_id:
            raise ValueError(f"Wrong deterministic shard at row {row_number}")
        if row.get("valid") is not True:
            raise ValueError(f"Invalid teacher row cannot be resumed: {key}")
        probe_observations_from_record(row)
        completed.add(key)
    return completed


def legal_probe_names(record: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return the expected legal probe names for a teacher record."""
    names = list(CORE_PROBES)
    if record.get("relation_applicable") is True:
        names.append(RELATION_PROBE)
    return tuple(names)


def thresholds_from_mapping(values: Mapping[str, Any]) -> LabelThresholds:
    """Create label thresholds while rejecting unknown or missing values."""
    defaults = asdict(LabelThresholds())
    expected = set(defaults)
    actual = set(values)
    unknown = actual - expected
    if unknown:
        raise ValueError(f"Unknown label threshold fields: {sorted(unknown)}")
    # eps is a numerical stabilizer, not a selected scientific threshold, and
    # therefore need not be duplicated in the frozen Week 3 YAML.
    required = expected - {"eps"}
    missing = required - actual
    if missing:
        raise ValueError(f"Missing frozen label threshold fields: {sorted(missing)}")
    return LabelThresholds(**{**defaults, **dict(values)})


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean, got {value!r}")
    return value


def probe_observations_from_record(
    record: Mapping[str, Any],
) -> Dict[ProbeAction, ProbeObservation]:
    """Reconstruct typed observations from a serialized teacher row."""
    probes = record.get("probes")
    if not isinstance(probes, Mapping):
        raise ValueError("Teacher record probes must be a mapping")

    expected = set(legal_probe_names(record))
    actual = set(probes)
    if actual != expected:
        raise ValueError(
            f"Probe set mismatch for {teacher_key(record)}: expected "
            f"{sorted(expected)}, got {sorted(actual)}"
        )

    observations: Dict[ProbeAction, ProbeObservation] = {}
    for name in sorted(actual):
        payload = probes[name]
        if not isinstance(payload, Mapping):
            raise ValueError(f"Probe '{name}' payload must be a mapping")
        required_fields = {
            "raw_answer",
            "norm_answer",
            "flip",
            "conf_shift",
            "entropy_shift",
            "margin_shift",
            "exact_match",
            "semantic_match",
            "applicable",
            "valid",
        }
        missing = required_fields - set(payload)
        if missing:
            raise ValueError(f"Probe '{name}' is missing fields: {sorted(missing)}")
        if not isinstance(payload["raw_answer"], str) or not isinstance(
            payload["norm_answer"], str
        ):
            raise ValueError(f"Probe '{name}' answers must be strings")
        if not payload["norm_answer"]:
            raise ValueError(f"Probe '{name}' normalized answer must be non-empty")
        probe_id = ProbeAction(name)
        observations[probe_id] = ProbeObservation(
            probe_id=probe_id,
            raw_answer=payload["raw_answer"],
            norm_answer=payload["norm_answer"],
            flip=_require_bool(payload["flip"], f"{name}.flip"),
            conf_shift=_require_number(payload.get("conf_shift"), f"{name}.conf_shift"),
            entropy_shift=_require_number(
                payload.get("entropy_shift"), f"{name}.entropy_shift"
            ),
            margin_shift=_require_number(
                payload.get("margin_shift"), f"{name}.margin_shift"
            ),
            exact_match=_require_number(payload.get("exact_match"), f"{name}.exact_match"),
            semantic_match=_require_number(
                payload.get("semantic_match"), f"{name}.semantic_match"
            ),
            applicable=_require_bool(payload["applicable"], f"{name}.applicable"),
            severity=payload.get("severity"),
            latency_ms=payload.get("latency_ms"),
            prompt_hash=payload.get("prompt_hash"),
            image_transform_hash=payload.get("image_transform_hash"),
            generation_config_hash=payload.get("generation_config_hash"),
            valid=_require_bool(payload["valid"], f"{name}.valid"),
            invalid_reason=payload.get("invalid_reason"),
            parse_status=str(payload.get("parse_status", "ok")),
            score_method=str(payload.get("score_method", "generation_logits")),
        )

    validate_mandatory_probes(
        observations, relation_applicable=record.get("relation_applicable") is True
    )
    return observations


def build_label_record(
    teacher_record: Mapping[str, Any],
    thresholds: LabelThresholds,
    source_teacher_file_sha256: str,
) -> Dict[str, Any]:
    """Recompute and serialize deterministic Week 4 labels.

    Invalid teacher rows are rejected; zero-valued stand-ins are never promoted
    to labels.
    """
    key = teacher_key(teacher_record)
    if teacher_record.get("valid") is not True:
        raise ValueError(f"Cannot label invalid teacher record {key}")
    clean = teacher_record.get("clean")
    if not isinstance(clean, Mapping) or clean.get("valid") is not True:
        raise ValueError(f"Cannot label teacher record with invalid clean scores {key}")
    clean_correct_value = clean.get("correct")
    if clean_correct_value not in (0, 1, False, True):
        raise ValueError(f"clean.correct must be binary for {key}")

    observations = probe_observations_from_record(teacher_record)
    labels = compute_teacher_labels(
        probe_observations=observations,
        clean_answer_prob=_require_number(clean.get("answer_prob"), "clean.answer_prob"),
        clean_correct=bool(clean_correct_value),
        relation_applicable=teacher_record.get("relation_applicable") is True,
        swap_invariance=teacher_record.get("swap_invariance"),
        benchmark_family=teacher_record.get("benchmark_family") or None,
        thresholds=thresholds,
        strict_validation=True,
    )

    recomputed_signature = {
        "V": labels.teacher_signature.V,
        "L": labels.teacher_signature.L,
        "A": labels.teacher_signature.A,
    }
    recomputed_bits = {
        "visual": int(labels.source_bits.visual),
        "language": int(labels.source_bits.language),
        "alignment": int(labels.source_bits.alignment),
    }
    recomputed_label = labels.six_way_state.value

    # The GPU row already contains provisional labels.  Requiring exact
    # agreement detects configuration drift between generation and offline
    # construction.
    if dict(teacher_record.get("teacher_bits", {})) != recomputed_bits:
        raise ValueError(f"Embedded teacher bits disagree with recomputation for {key}")
    if teacher_record.get("teacher_label6") != recomputed_label:
        raise ValueError(f"Embedded six-way label disagrees with recomputation for {key}")
    embedded_signature = teacher_record.get("teacher_signature", {})
    for component, value in recomputed_signature.items():
        embedded = _require_number(
            embedded_signature.get(component), f"teacher_signature.{component}"
        )
        if abs(embedded - value) > 1e-12:
            raise ValueError(
                f"Embedded signature {component} disagrees with recomputation for {key}"
            )

    label_record: Dict[str, Any] = {
        "record_type": "teacher_labels",
        "instance_id": teacher_record["instance_id"],
        "group_id": teacher_record["group_id"],
        "dataset": teacher_record["dataset"],
        "split": teacher_record["split"],
        "model_id": teacher_record["model_id"],
        "model_revision": teacher_record["model_revision"],
        "relation_applicable": teacher_record.get("relation_applicable") is True,
        "clean_correct": int(bool(clean_correct_value)),
        "teacher_signature": recomputed_signature,
        "teacher_bits": recomputed_bits,
        "teacher_label6": recomputed_label,
        "benchmark_family": labels.benchmark_family or "",
        "source_teacher_file_sha256": source_teacher_file_sha256,
        "source_teacher_record_sha256": hash_dict(dict(teacher_record)),
    }
    label_record["label_record_sha256"] = hash_dict(label_record)
    return label_record


def _deterministic_rank(seed: int, key: Tuple[str, str], value: Sequence[str]) -> str:
    text = f"{seed}|{key[0]}|{key[1]}|{'|'.join(value)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_partial_subsets(
    teacher_record: Mapping[str, Any],
    seed: int = 42,
    random_subset_count: int = 16,
) -> List[Tuple[Tuple[str, ...], List[str]]]:
    """Build mandatory pre-policy Week 4 subsets for one teacher row.

    Policy-rollout and oracle-next-action states cannot exist before their
    respective Week 5/6 checkpoints and are therefore recorded as deferred in
    each serialized state rather than fabricated here.
    """
    if random_subset_count < 0:
        raise ValueError("random_subset_count must be non-negative")
    legal = legal_probe_names(teacher_record)
    legal_set = set(legal)
    key = teacher_key(teacher_record)
    sources: Dict[Tuple[str, ...], List[str]] = OrderedDict()

    def add(subset: Iterable[str], source: str) -> None:
        normalized = tuple(sorted(subset))
        if not set(normalized).issubset(legal_set):
            raise AssertionError(f"Illegal probe in sampled subset: {normalized}")
        sources.setdefault(normalized, []).append(source)

    add((), "empty")
    for probe in legal:
        add((probe,), f"singleton:{probe}")

    for baseline_name, full_order in FIXED_BASELINES.items():
        order = [probe for probe in full_order if probe in legal_set]
        for length in range(1, len(order) + 1):
            add(order[:length], f"prefix:{baseline_name}:{length}")

    if random_subset_count:
        by_size: Dict[int, List[Tuple[str, ...]]] = {}
        for size in range(1, min(4, len(legal)) + 1):
            combinations = list(itertools.combinations(sorted(legal), size))
            combinations.sort(key=lambda value: _deterministic_rank(seed, key, value))
            by_size[size] = combinations

        cursors = {size: 0 for size in by_size}
        sizes = sorted(by_size)
        for draw in range(random_subset_count):
            size = sizes[draw % len(sizes)]
            candidates = by_size[size]
            if cursors[size] >= len(candidates):
                raise ValueError(
                    f"Cannot draw {random_subset_count} deterministic unique random subsets "
                    f"for size {size} from {len(legal)} legal probes"
                )
            subset = candidates[cursors[size]]
            cursors[size] += 1
            add(subset, f"random:{draw:02d}:size{size}")

    return [(subset, provenance) for subset, provenance in sources.items()]


def _clean_learner_features(teacher_record: Mapping[str, Any]) -> Dict[str, float]:
    clean = teacher_record.get("clean")
    if not isinstance(clean, Mapping) or clean.get("valid") is not True:
        raise ValueError(f"Invalid clean features for {teacher_key(teacher_record)}")
    return {
        "answer_prob": _require_number(clean.get("answer_prob"), "clean.answer_prob"),
        "token_entropy_mean": _require_number(
            clean.get("token_entropy_mean"), "clean.token_entropy_mean"
        ),
        "token_margin_mean": _require_number(
            clean.get("token_margin_mean"), "clean.token_margin_mean"
        ),
        "answer_len_tokens": _require_number(
            clean.get("answer_len_tokens"), "clean.answer_len_tokens"
        ),
    }


def _probe_learner_features(name: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("valid") is not True:
        raise ValueError(f"Cannot expose invalid acquired probe '{name}'")
    features: Dict[str, Any] = {"probe_id": name}
    for field in PROBE_FEATURE_KEYS:
        value = payload.get(field)
        if field in {"flip", "applicable"}:
            if not isinstance(value, bool):
                raise ValueError(f"Probe feature {name}.{field} must be boolean")
            features[field] = int(value)
        else:
            features[field] = _require_number(value, f"{name}.{field}")
    return features


def build_state_records(
    teacher_record: Mapping[str, Any],
    label_record: Mapping[str, Any],
    source_teacher_file_sha256: str,
    source_label_file_sha256: str,
    seed: int = 42,
    random_subset_count: int = 16,
) -> List[Dict[str, Any]]:
    """Serialize leakage-safe partial states for one teacher/label pair."""
    if teacher_key(teacher_record) != teacher_key(label_record):
        raise ValueError("Teacher and label identities do not match")
    for field in ("group_id", "dataset", "split", "model_revision"):
        if teacher_record.get(field) != label_record.get(field):
            raise ValueError(f"Teacher/label {field} mismatch for {teacher_key(teacher_record)}")

    probes = teacher_record.get("probes")
    if not isinstance(probes, Mapping):
        raise ValueError("Teacher probes must be a mapping")
    # Validate the complete record before hiding observations.
    probe_observations_from_record(teacher_record)
    legal = legal_probe_names(teacher_record)

    output: List[Dict[str, Any]] = []
    for subset, state_sources in sample_partial_subsets(
        teacher_record, seed=seed, random_subset_count=random_subset_count
    ):
        acquired = [
            _probe_learner_features(name, probes[name]) for name in subset
        ]
        acquired_set = set(subset)
        action_mask = {
            probe: int(probe not in acquired_set) for probe in legal
        }
        if RELATION_PROBE not in action_mask:
            action_mask[RELATION_PROBE] = 0
        action_mask[ProbeAction.STOP.value] = 1

        state_identity = {
            "model_id": teacher_record["model_id"],
            "instance_id": teacher_record["instance_id"],
            "acquired_probe_names": list(subset),
            "seed": seed,
        }
        state: Dict[str, Any] = {
            "record_type": "partial_state_v1",
            "state_id": f"state:{hash_dict(state_identity)[:24]}",
            "metadata": {
                "instance_id": teacher_record["instance_id"],
                "group_id": teacher_record["group_id"],
                "dataset": teacher_record["dataset"],
                "split": teacher_record["split"],
                "model_id": teacher_record["model_id"],
                "model_revision": teacher_record["model_revision"],
            },
            "learner_input": {
                "clean_features": _clean_learner_features(teacher_record),
                "acquired_probe_names": list(subset),
                "acquired_observations": acquired,
                "remaining_budget": MAX_PROBES - len(subset),
                "action_mask": action_mask,
            },
            "targets": {
                "clean_correct": label_record["clean_correct"],
                "teacher_signature": dict(label_record["teacher_signature"]),
                "teacher_bits": dict(label_record["teacher_bits"]),
                "teacher_label6": label_record["teacher_label6"],
            },
            "sampling": {
                "seed": seed,
                "sources": state_sources,
                "phase": "pre_policy_week4",
                "deferred_sources": ["policy_rollout", "oracle_next_action"],
            },
            "source_teacher_file_sha256": source_teacher_file_sha256,
            "source_label_file_sha256": source_label_file_sha256,
            "source_teacher_record_sha256": label_record[
                "source_teacher_record_sha256"
            ],
            "source_label_record_sha256": label_record["label_record_sha256"],
        }
        state["state_record_sha256"] = hash_dict(state)
        output.append(state)
    return output
