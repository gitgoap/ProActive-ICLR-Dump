"""Strict Week 4 artifact and leakage validation (Plan section 25.6)."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

from proactive.teacher.offline import (
    CORE_PROBES,
    FIXED_BASELINES,
    RELATION_PROBE,
    build_label_record,
    legal_probe_names,
    teacher_key,
)
from proactive.utils.io import file_sha256, iter_jsonl
from proactive.features.normalization import normalize_answer


IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SIX_WAY_LABELS = {
    "no-failure",
    "visual",
    "language-prior",
    "alignment",
    "mixed",
    "unclear",
}
STATE_CLEAN_FEATURES = {
    "answer_prob",
    "token_entropy_mean",
    "token_margin_mean",
    "answer_len_tokens",
}
STATE_PROBE_FEATURES = {
    "probe_id",
    "flip",
    "conf_shift",
    "entropy_shift",
    "margin_shift",
    "exact_match",
    "semantic_match",
    "applicable",
}
FORBIDDEN_LEARNER_KEYS = {
    "dataset",
    "dataset_id",
    "model",
    "model_id",
    "model_revision",
    "instance_id",
    "group_id",
    "split",
    "gold_answer",
    "correct",
    "clean_correct",
    "teacher_bits",
    "teacher_signature",
    "teacher_label6",
    "raw_answer",
    "norm_answer",
    "image_path",
    "prompt_text",
}

ArtifactRow = Tuple[Dict[str, Any], Path, str]


def teacher_answer_contract_errors(
    row: Mapping[str, Any], source: Mapping[str, Any], key: Tuple[str, str]
) -> List[str]:
    """Validate dataset answer metadata and deterministic binary correctness."""

    errors: List[str] = []
    if source.get("dataset") == "vizwiz":
        for field in (
            "gold_answer",
            "answer_contract_version",
            "answer_match_mode",
            "reference_answers",
            "vizwiz_gold_policy",
            "vizwiz_answer_counts",
            "vizwiz_tied_top_answer_count",
        ):
            if row.get(field) != source.get(field):
                errors.append(f"Teacher/manifest {field} mismatch for {key}")
        expected_correct = int(
            normalize_answer(
                str(row.get("clean", {}).get("raw_answer", "")), "vizwiz"
            )
            == source.get("gold_answer")
        )
        if row.get("clean", {}).get("correct") != expected_correct:
            errors.append(f"VizWiz clean correctness is stale for {key}")
        return errors
    if source.get("dataset") != "hallusionbench":
        return errors
    for field in (
        "answer_type",
        "answer_contract_version",
        "answer_match_mode",
        "gold_answer",
        "benchmark_gold_answer",
        "gt_answer_details",
        "reference_answers",
    ):
        if row.get(field) != source.get(field):
            errors.append(f"Teacher/manifest {field} mismatch for {key}")
    if source.get("answer_type") == "binary":
        expected_correct = int(
            normalize_answer(
                str(row.get("clean", {}).get("raw_answer", "")),
                "hallusionbench",
            )
            == source.get("gold_answer")
        )
        if row.get("clean", {}).get("correct") != expected_correct:
            errors.append(f"HallusionBench binary clean correctness is stale for {key}")
    return errors


def collect_artifact_rows(path: Path, prefix: str) -> Tuple[List[ArtifactRow], List[Dict[str, Any]]]:
    """Read artifact rows and return per-file checksum metadata."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob(f"{prefix}*.jsonl")) or sorted(path.glob("*.jsonl"))
        files = [
            file_path
            for file_path in files
            if not file_path.name.endswith(".failures.jsonl")
        ]
    else:
        raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError(f"No JSONL artifacts found under {path}")

    rows: List[ArtifactRow] = []
    file_manifest: List[Dict[str, Any]] = []
    for file_path in files:
        sha = file_sha256(file_path)
        count = 0
        for row in iter_jsonl(file_path):
            rows.append((row, file_path, sha))
            count += 1
        file_manifest.append(
            {"path": str(file_path), "sha256": sha, "row_count": count}
        )
    return rows, file_manifest


def _find_forbidden_keys(value: Any, path: str = "learner_input") -> List[str]:
    findings: List[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in FORBIDDEN_LEARNER_KEYS:
                findings.append(f"{path}.{key}")
            findings.extend(_find_forbidden_keys(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            findings.extend(_find_forbidden_keys(nested, f"{path}[{index}]"))
    return findings


def validate_week4_artifacts(
    manifest_records: Sequence[Mapping[str, Any]],
    teacher_rows: Sequence[ArtifactRow],
    label_rows: Sequence[ArtifactRow],
    state_rows: Sequence[ArtifactRow],
    required_model_ids: Set[str],
    thresholds: Any,
    max_sixway_fraction: float,
    min_bit_count_per_dataset_model: int,
    require_complete: bool,
) -> Dict[str, Any]:
    """Validate teacher, label, and partial-state artifacts fail closed."""
    errors: List[str] = []
    warnings: List[str] = []
    manifest_index: Dict[str, Mapping[str, Any]] = {}
    group_splits: Dict[str, Set[str]] = defaultdict(set)
    for row in manifest_records:
        instance_id = row.get("instance_id")
        if not instance_id or instance_id in manifest_index:
            errors.append(f"Duplicate or missing manifest instance_id: {instance_id!r}")
            continue
        manifest_index[str(instance_id)] = row
        group_splits[str(row.get("group_id"))].add(str(row.get("split")))
    for group_id, splits in group_splits.items():
        if len(splits) != 1:
            errors.append(f"Manifest group {group_id} leaks across splits: {sorted(splits)}")

    teachers: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    teacher_source_sha: Dict[Tuple[str, str], str] = {}
    probe_keys: Set[Tuple[str, str, str]] = set()
    model_counts: Counter[str] = Counter()
    expected_forward_passes = 0
    for row, path, source_sha in teacher_rows:
        try:
            key = teacher_key(row)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if key in teachers:
            errors.append(f"Duplicate model-instance teacher key: {key}")
            continue
        teachers[key] = row
        teacher_source_sha[key] = source_sha
        model_counts[key[0]] += 1
        source = manifest_index.get(key[1])
        if source is None:
            errors.append(f"Teacher key absent from grouped manifest: {key}")
        else:
            for field in ("group_id", "dataset", "split"):
                if row.get(field) != source.get(field):
                    errors.append(f"Teacher/manifest {field} mismatch for {key}")
            errors.extend(teacher_answer_contract_errors(row, source, key))
        if row.get("record_type") != "teacher_cache":
            errors.append(f"Teacher row has wrong record_type: {key}")
        if row.get("valid") is not True:
            errors.append(f"Teacher row is invalid: {key}")
        if not IMMUTABLE_REVISION_RE.fullmatch(str(row.get("model_revision", ""))):
            errors.append(f"Teacher row has unpinned model revision: {key}")
        expected_names = set(legal_probe_names(row))
        probes = row.get("probes")
        if not isinstance(probes, Mapping) or set(probes) != expected_names:
            errors.append(f"Teacher legal probe set mismatch: {key}")
            continue
        expected_forward_passes += 1 + len(expected_names)
        for probe_name in expected_names:
            probe_key = (key[0], key[1], probe_name)
            if probe_key in probe_keys:
                errors.append(f"Duplicate instance-model-probe key: {probe_key}")
            probe_keys.add(probe_key)
            probe = probes[probe_name]
            if probe.get("valid") is not True or probe.get("applicable") is not True:
                errors.append(f"Legal probe is invalid/inapplicable: {probe_key}")

    if require_complete:
        for model_id in required_model_ids:
            model_keys = {instance_id for mid, instance_id in teachers if mid == model_id}
            expected_keys = set(manifest_index)
            if model_keys != expected_keys:
                errors.append(
                    f"Core cache incomplete for {model_id}: expected {len(expected_keys)}, "
                    f"found {len(model_keys)}, missing {len(expected_keys - model_keys)}, "
                    f"extra {len(model_keys - expected_keys)}"
                )
    missing_core_models = required_model_ids - set(model_counts)
    if missing_core_models:
        errors.append(f"Missing core model caches: {sorted(missing_core_models)}")

    labels: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    label_source_sha: Dict[Tuple[str, str], str] = {}
    class_counts: Counter[str] = Counter()
    slice_class_counts: Counter[Tuple[str, str, str]] = Counter()
    slice_bit_counts: Counter[Tuple[str, str, str, int]] = Counter()
    slice_totals: Counter[Tuple[str, str]] = Counter()
    for row, path, source_sha in label_rows:
        try:
            key = teacher_key(row)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if key in labels:
            errors.append(f"Duplicate model-instance label key: {key}")
            continue
        labels[key] = row
        label_source_sha[key] = source_sha
        teacher = teachers.get(key)
        if teacher is None:
            errors.append(f"Label has no teacher row: {key}")
            continue
        if row.get("source_teacher_file_sha256") != teacher_source_sha[key]:
            errors.append(f"Label teacher-file hash mismatch: {key}")
        try:
            expected = build_label_record(teacher, thresholds, teacher_source_sha[key])
        except ValueError as exc:
            errors.append(f"Label recomputation failed for {key}: {exc}")
            continue
        for field in (
            "group_id",
            "dataset",
            "split",
            "model_revision",
            "clean_correct",
            "teacher_signature",
            "teacher_bits",
            "teacher_label6",
            "label_record_sha256",
        ):
            if row.get(field) != expected.get(field):
                errors.append(f"Label field {field} is nondeterministic/stale: {key}")
        label_name = str(row.get("teacher_label6"))
        if label_name not in SIX_WAY_LABELS:
            errors.append(f"Invalid six-way label for {key}: {label_name}")
            continue
        bits = row.get("teacher_bits", {})
        for bit in ("visual", "language", "alignment"):
            value = bits.get(bit)
            if value not in (0, 1):
                errors.append(f"Invalid source bit {bit} for {key}: {value}")
        # Label-balance inspection is a selection diagnostic. It is restricted
        # to train/val so calibration/test distributions remain locked.
        if row.get("split") in {"train", "val"}:
            dataset = str(row.get("dataset"))
            model_id = key[0]
            class_counts[label_name] += 1
            slice_class_counts[(dataset, model_id, label_name)] += 1
            slice_totals[(dataset, model_id)] += 1
            for bit in ("visual", "language", "alignment"):
                value = bits.get(bit)
                if value in (0, 1):
                    slice_bit_counts[(dataset, model_id, bit, int(value))] += 1

    if set(labels) != set(teachers):
        errors.append(
            f"Teacher/label key mismatch: missing labels={len(set(teachers) - set(labels))}, "
            f"orphan labels={len(set(labels) - set(teachers))}"
        )

    selection_label_count = sum(class_counts.values())
    if selection_label_count:
        dominant_label, dominant_count = class_counts.most_common(1)[0]
        dominant_fraction = dominant_count / selection_label_count
        if dominant_fraction > max_sixway_fraction:
            errors.append(
                f"Dominant label collapse: {dominant_label}={dominant_fraction:.4f} "
                f"> {max_sixway_fraction:.4f}"
            )
    else:
        dominant_label, dominant_fraction = None, None

    for slice_key, total in slice_totals.items():
        for bit in ("visual", "language", "alignment"):
            positives = slice_bit_counts[(slice_key[0], slice_key[1], bit, 1)]
            negatives = total - positives
            if min(positives, negatives) < min_bit_count_per_dataset_model:
                message = (
                    f"Unusable bit balance for {slice_key}/{bit}: "
                    f"positive={positives}, negative={negatives}, "
                    f"minimum={min_bit_count_per_dataset_model}"
                )
                if require_complete:
                    errors.append(message)
                else:
                    warnings.append(message)

    states_by_teacher: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    state_ids: Set[str] = set()
    for row, path, _ in state_rows:
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or not state_id:
            errors.append(f"State has missing state_id in {path}")
            continue
        if state_id in state_ids:
            errors.append(f"Duplicate state_id: {state_id}")
            continue
        state_ids.add(state_id)
        metadata = row.get("metadata")
        learner_input = row.get("learner_input")
        targets = row.get("targets")
        sampling = row.get("sampling")
        if not all(isinstance(item, Mapping) for item in (metadata, learner_input, targets, sampling)):
            errors.append(f"Malformed state blocks: {state_id}")
            continue
        try:
            key = teacher_key(metadata)
        except ValueError as exc:
            errors.append(f"State {state_id}: {exc}")
            continue
        states_by_teacher[key].append(row)
        teacher = teachers.get(key)
        label = labels.get(key)
        if teacher is None or label is None:
            errors.append(f"State has no teacher/label pair: {state_id}")
            continue
        for field in ("group_id", "dataset", "split", "model_revision"):
            if metadata.get(field) != teacher.get(field):
                errors.append(f"State metadata {field} mismatch: {state_id}")
        if row.get("source_teacher_file_sha256") != teacher_source_sha[key]:
            errors.append(f"State teacher-file hash mismatch: {state_id}")
        if row.get("source_label_file_sha256") != label_source_sha[key]:
            errors.append(f"State label-file hash mismatch: {state_id}")
        forbidden = _find_forbidden_keys(learner_input)
        if forbidden:
            errors.append(f"Forbidden learner features in {state_id}: {forbidden}")
        clean_features = learner_input.get("clean_features")
        if not isinstance(clean_features, Mapping) or set(clean_features) != STATE_CLEAN_FEATURES:
            errors.append(f"State clean feature schema mismatch: {state_id}")
        acquired_names = learner_input.get("acquired_probe_names")
        observations = learner_input.get("acquired_observations")
        if not isinstance(acquired_names, list) or not isinstance(observations, list):
            errors.append(f"State acquired fields malformed: {state_id}")
            continue
        observation_names = []
        for observation in observations:
            if not isinstance(observation, Mapping) or set(observation) != STATE_PROBE_FEATURES:
                errors.append(f"State probe feature schema mismatch: {state_id}")
                continue
            observation_names.append(observation.get("probe_id"))
        if observation_names != acquired_names or len(acquired_names) != len(set(acquired_names)):
            errors.append(f"State acquired observation/name mismatch: {state_id}")
        legal = set(legal_probe_names(teacher))
        if not set(acquired_names).issubset(legal):
            errors.append(f"State contains illegal/unacquired probe evidence: {state_id}")
        expected_mask = {probe: int(probe not in set(acquired_names)) for probe in legal}
        if RELATION_PROBE not in expected_mask:
            expected_mask[RELATION_PROBE] = 0
        expected_mask["stop"] = 1
        if learner_input.get("action_mask") != expected_mask:
            errors.append(f"State action mask mismatch: {state_id}")
        if targets.get("teacher_label6") != label.get("teacher_label6"):
            errors.append(f"State target mismatch: {state_id}")
        if sampling.get("phase") != "pre_policy_week4":
            errors.append(f"State sampling phase mismatch: {state_id}")
        if set(sampling.get("deferred_sources", [])) != {
            "policy_rollout",
            "oracle_next_action",
        }:
            errors.append(f"State deferred-source declaration mismatch: {state_id}")

    if set(states_by_teacher) != set(teachers):
        errors.append(
            f"Teacher/state key mismatch: missing states={len(set(teachers) - set(states_by_teacher))}, "
            f"orphan states={len(set(states_by_teacher) - set(teachers))}"
        )
    for key, teacher in teachers.items():
        rows = states_by_teacher.get(key, [])
        source_tags = {
            source
            for row in rows
            for source in row.get("sampling", {}).get("sources", [])
        }
        if "empty" not in source_tags:
            errors.append(f"Missing empty state for {key}")
        for probe in legal_probe_names(teacher):
            if f"singleton:{probe}" not in source_tags:
                errors.append(f"Missing singleton {probe} for {key}")
        for baseline, order in FIXED_BASELINES.items():
            legal_order = [probe for probe in order if probe in legal_probe_names(teacher)]
            for length in range(1, len(legal_order) + 1):
                if f"prefix:{baseline}:{length}" not in source_tags:
                    errors.append(f"Missing {baseline} prefix {length} for {key}")
        random_tags = {tag for tag in source_tags if tag.startswith("random:")}
        if len(random_tags) != 16:
            errors.append(f"Expected 16 random subset draws for {key}, found {len(random_tags)}")

    class_distribution_rows: List[Dict[str, Any]] = []
    bit_distribution_rows: List[Dict[str, Any]] = []
    for (dataset, model_id), total in sorted(slice_totals.items()):
        for label_name in sorted(SIX_WAY_LABELS):
            count = slice_class_counts[(dataset, model_id, label_name)]
            class_distribution_rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "teacher_label6": label_name,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                }
            )
        for bit in ("visual", "language", "alignment"):
            positives = slice_bit_counts[(dataset, model_id, bit, 1)]
            bit_distribution_rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "bit": bit,
                    "positive_count": positives,
                    "negative_count": total - positives,
                    "positive_fraction": positives / total if total else 0.0,
                }
            )

    return {
        "is_valid": not errors,
        "require_complete": require_complete,
        "manifest_records": len(manifest_index),
        "teacher_records": len(teachers),
        "label_records": len(labels),
        "state_records": len(state_ids),
        "unique_probe_records": len(probe_keys),
        "expected_forward_passes": expected_forward_passes,
        "model_teacher_counts": dict(sorted(model_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "class_balance_splits": ["train", "val"],
        "class_balance_record_count": selection_label_count,
        "dominant_label": dominant_label,
        "dominant_fraction": dominant_fraction,
        "class_distribution_rows": class_distribution_rows,
        "bit_distribution_rows": bit_distribution_rows,
        "errors": errors[:500],
        "error_count": len(errors),
        "warnings": warnings[:500],
        "warning_count": len(warnings),
    }


def validate_audit_packet(
    audit_dir: Path,
    expected_count: int,
    require_all_images: bool = True,
    required_model_ids: Set[str] | None = None,
    required_datasets: Set[str] | None = None,
) -> Dict[str, Any]:
    """Verify packet size, blinding columns, IDs, images, and checksums."""
    errors: List[str] = []
    blinded_path = audit_dir / "human_audit_blinded.csv"
    private_path = audit_dir / "human_audit_private_key.jsonl"
    manifest_path = audit_dir / "human_audit_manifest.json"
    if not all(path.exists() for path in (blinded_path, private_path, manifest_path)):
        return {"is_valid": False, "errors": ["Audit packet artifacts are missing"]}
    with open(blinded_path, "r", encoding="utf-8", newline="") as handle:
        blinded = list(csv.DictReader(handle))
    private = list(iter_jsonl(private_path))
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    forbidden_columns = {
        "instance_id",
        "dataset",
        "model_id",
        "model_revision",
        "teacher_signature",
        "teacher_bits",
        "teacher_label6",
        "source_image_path",
    }
    present_forbidden = forbidden_columns & set(blinded[0] if blinded else {})
    if present_forbidden:
        errors.append(f"Blinded CSV exposes private columns: {sorted(present_forbidden)}")
    if len(blinded) != expected_count or len(private) != expected_count:
        errors.append(
            f"Audit row count mismatch: blinded={len(blinded)}, private={len(private)}, expected={expected_count}"
        )
    blinded_ids = {row.get("audit_id") for row in blinded}
    private_ids = {row.get("audit_id") for row in private}
    if blinded_ids != private_ids or len(blinded_ids) != len(blinded):
        errors.append("Audit IDs are duplicated or differ between blinded/private files")
    image_count = 0
    for row in blinded:
        image_path = audit_dir / str(row.get("image_file", ""))
        if image_path.exists():
            image_count += 1
    if require_all_images and image_count != expected_count:
        errors.append(f"Audit materialized images: found {image_count}, expected {expected_count}")
    if require_all_images and manifest.get("materialized_image_count") != expected_count:
        errors.append(
            "Audit manifest does not certify the complete materialized image set"
        )
    for relative, sha in manifest.get("artifact_sha256", {}).items():
        path = audit_dir / relative
        if not path.exists() or file_sha256(path) != sha:
            errors.append(f"Audit checksum mismatch: {relative}")
    if manifest.get("is_interim"):
        errors.append("Audit packet is marked interim due to incomplete model coverage")
    observed_models = {str(row.get("model_id")) for row in private}
    observed_datasets = {str(row.get("dataset")) for row in private}
    if required_model_ids and not required_model_ids.issubset(observed_models):
        errors.append(
            f"Audit model coverage is incomplete: missing {sorted(required_model_ids - observed_models)}"
        )
    if required_datasets and not required_datasets.issubset(observed_datasets):
        errors.append(
            f"Audit dataset coverage is incomplete: missing {sorted(required_datasets - observed_datasets)}"
        )
    return {
        "is_valid": not errors,
        "blinded_rows": len(blinded),
        "private_rows": len(private),
        "materialized_images": image_count,
        "observed_model_count": len(observed_models),
        "observed_dataset_count": len(observed_datasets),
        "errors": errors,
    }
