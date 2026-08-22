#!/usr/bin/env python3
"""Migrate Week 4 teacher caches to HallusionBench answer contract v1.

The migration is deliberately conservative:

* non-HallusionBench manifest rows and model observations must be unchanged;
* binary HallusionBench generations are reused, then their normalization,
  clean correctness, answer-comparison features, and labels are recomputed;
* every open-ended HallusionBench row is removed from both the valid cache and
  failure ledger because its prompt/answer domain changed and must be rerun;
* source artifacts are never overwritten, and every migrated row records
  source-file and source-record hashes.

This makes reuse independent of model success/failure and prevents selective
post-hoc exclusion of difficult examples.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from proactive.data.hallusion_contract import (  # noqa: E402
    HALLUSION_BINARY,
    HALLUSION_OPEN_ENDED,
)
from proactive.data.manifests import load_manifest, validate_manifest  # noqa: E402
from proactive.features.normalization import normalize_answer  # noqa: E402
from proactive.features.semantic import (  # noqa: E402
    SemanticMatcher,
    compute_semantic_match,
)
from proactive.prompts.templates import parse_grounding_output  # noqa: E402
from proactive.teacher.label_computation import compute_teacher_labels  # noqa: E402
from proactive.teacher.offline import (  # noqa: E402
    probe_observations_from_record,
    stable_shard_id,
    thresholds_from_mapping,
    validate_resume_teacher_records,
)
from proactive.utils.hashing import hash_dict  # noqa: E402
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_jsonl  # noqa: E402


MIGRATION_SCHEMA_VERSION = 2
TEACHER_FILE_RE = re.compile(
    r"^teacher_(?P<model_name>.+)_all_all_shard(?P<shard>\d+)-of-(?P<num_shards>\d+)\.jsonl$"
)
ANSWER_FIELDS = (
    "gold_answer",
    "answer_type",
    "answer_contract_version",
    "answer_match_mode",
    "benchmark_gold_answer",
    "gt_answer_details",
    "reference_answers",
)
HALLUSION_INVARIANT_FIELDS = (
    "instance_id",
    "group_id",
    "dataset",
    "image_id",
    "question_id",
    "image_path",
    "question",
    "relation_applicable",
    "category",
    "subcategory",
    "split",
)
VIZWIZ_CONTRACT_FIELDS = (
    "gold_answer",
    "answer_contract_version",
    "answer_match_mode",
    "reference_answers",
    "vizwiz_gold_policy",
    "vizwiz_answer_counts",
    "vizwiz_tied_top_answer_count",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old_manifest", required=True)
    parser.add_argument("--new_manifest", required=True)
    parser.add_argument("--config", default="configs/experiments/teacher_core.yaml")
    parser.add_argument("--input_dir", default="outputs/teacher_core")
    parser.add_argument("--output_dir", default="outputs/teacher_core_contract_v1")
    parser.add_argument(
        "--report",
        default="outputs/week4_reports/hallusion_answer_contract_migration.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _by_id(rows: Iterable[Mapping[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        instance_id = str(row.get("instance_id", ""))
        if not instance_id:
            raise ValueError(f"{label} contains a row without instance_id")
        if instance_id in result:
            raise ValueError(f"{label} contains duplicate instance_id {instance_id}")
        result[instance_id] = dict(row)
    return result


def validate_manifest_transition(
    old_rows: Iterable[Mapping[str, Any]],
    new_rows: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], set[str]]:
    """Validate only declared HallusionBench/VizWiz contract changes."""

    old_by_id = _by_id(old_rows, "old manifest")
    new_by_id = _by_id(new_rows, "new manifest")
    if set(old_by_id) != set(new_by_id):
        missing = sorted(set(old_by_id) - set(new_by_id))
        added = sorted(set(new_by_id) - set(old_by_id))
        raise ValueError(
            f"Manifest identity drift: removed={len(missing)}, added={len(added)}"
        )

    open_ids: set[str] = set()
    for instance_id, old in old_by_id.items():
        new = new_by_id[instance_id]
        if old.get("dataset") == "vizwiz":
            allowed = set(VIZWIZ_CONTRACT_FIELDS)
            changed_fields = sorted(
                key
                for key in set(old).union(new)
                if old.get(key) != new.get(key)
            )
            unexpected = [field for field in changed_fields if field not in allowed]
            if unexpected:
                details = "; ".join(
                    f"{field}: old={old.get(field)!r}, new={new.get(field)!r}"
                    for field in unexpected[:5]
                )
                raise ValueError(
                    f"Unexpected VizWiz manifest drift for {instance_id}: {details}"
                )
            if (
                new.get("vizwiz_gold_policy")
                != "normalized_majority_source_order_tiebreak_v1"
                or new.get("answer_match_mode") != "normalized_exact"
                or new.get("answer_contract_version") != 1
            ):
                raise ValueError(f"Invalid VizWiz answer contract for {instance_id}")
            continue
        if old.get("dataset") != "hallusionbench":
            if old != new:
                changed_fields = sorted(
                    key
                    for key in set(old).union(new)
                    if old.get(key) != new.get(key)
                )
                details = "; ".join(
                    f"{field}: old={old.get(field)!r}, new={new.get(field)!r}"
                    for field in changed_fields[:5]
                )
                raise ValueError(
                    "Non-HallusionBench manifest row changed: "
                    f"{instance_id}; fields={changed_fields}; {details}"
                )
            continue
        for field in HALLUSION_INVARIANT_FIELDS:
            if old.get(field) != new.get(field):
                raise ValueError(
                    f"HallusionBench invariant field changed for {instance_id}: {field}"
                )
        if str(old.get("gold_answer", "")) != str(
            new.get("benchmark_gold_answer", "")
        ):
            raise ValueError(
                f"HallusionBench benchmark indicator drift for {instance_id}"
            )
        if new.get("answer_contract_version") != 1:
            raise ValueError(f"Missing answer contract v1 for {instance_id}")
        answer_type = new.get("answer_type")
        if answer_type == HALLUSION_OPEN_ENDED:
            open_ids.add(instance_id)
        elif answer_type != HALLUSION_BINARY:
            raise ValueError(f"Invalid answer type for {instance_id}: {answer_type!r}")

    if len(open_ids) != 14:
        raise ValueError(
            f"Expected exactly 14 open-ended HallusionBench rows, found {len(open_ids)}"
        )
    return old_by_id, new_by_id, open_ids


def _update_answer_features(
    row: MutableMapping[str, Any], manifest_row: Mapping[str, Any]
) -> None:
    """Update binary HallusionBench normalization from existing raw answers."""

    for field in ANSWER_FIELDS:
        row[field] = copy.deepcopy(manifest_row.get(field))
    clean = row.get("clean")
    if not isinstance(clean, MutableMapping):
        raise ValueError(f"Teacher row {row.get('instance_id')} has no clean mapping")
    clean_norm = normalize_answer(str(clean.get("raw_answer", "")), "hallusionbench")
    clean["norm_answer"] = clean_norm
    clean["correct"] = int(clean_norm == manifest_row["gold_answer"])

    probes = row.get("probes")
    if not isinstance(probes, MutableMapping):
        raise ValueError(f"Teacher row {row.get('instance_id')} has no probes mapping")
    for probe_name, payload in probes.items():
        if not isinstance(payload, MutableMapping):
            raise ValueError(
                f"Teacher row {row.get('instance_id')} has invalid probe {probe_name}"
            )
        if payload.get("valid") is not True:
            # Invalid observations remain fail-closed and will never be used as
            # labels.  The uniform grounding recovery replaces the invalid
            # grounding observation later.
            continue
        if probe_name == "grounding":
            parsed = parse_grounding_output(
                str(payload.get("raw_answer", "")), "hallusionbench"
            )
            if not parsed.is_valid:
                raise ValueError(
                    f"Valid grounding row no longer parses for {row.get('instance_id')}: "
                    f"{parsed.invalid_reason}"
                )
            probe_norm = parsed.norm_final_answer
            payload["parse_status"] = parsed.parse_status
        else:
            probe_norm = normalize_answer(
                str(payload.get("raw_answer", "")), "hallusionbench"
            )
        payload["norm_answer"] = probe_norm
        matches = probe_norm == clean_norm
        payload["flip"] = not matches
        payload["exact_match"] = float(matches)
        payload["semantic_match"] = float(matches)


def _recompute_labels(row: MutableMapping[str, Any], thresholds: Any) -> None:
    if row.get("valid") is not True:
        return
    observations = probe_observations_from_record(row)
    clean = row["clean"]
    labels = compute_teacher_labels(
        probe_observations=observations,
        clean_answer_prob=float(clean["answer_prob"]),
        clean_correct=bool(clean["correct"]),
        relation_applicable=row.get("relation_applicable") is True,
        swap_invariance=row.get("swap_invariance"),
        benchmark_family=row.get("benchmark_family") or None,
        thresholds=thresholds,
        strict_validation=True,
    )
    row["teacher_signature"] = {
        "V": labels.teacher_signature.V,
        "L": labels.teacher_signature.L,
        "A": labels.teacher_signature.A,
    }
    row["teacher_bits"] = {
        "visual": int(labels.source_bits.visual),
        "language": int(labels.source_bits.language),
        "alignment": int(labels.source_bits.alignment),
    }
    row["teacher_label6"] = labels.six_way_state.value
    row["benchmark_family"] = labels.benchmark_family or ""


def _update_vizwiz_answer_contract(
    row: MutableMapping[str, Any],
    manifest_row: Mapping[str, Any],
    *,
    semantic_threshold: float = 0.82,
    embedding_fn: Callable[[str, str], float] | None = None,
) -> None:
    """Recompute VizWiz answer metadata and parser-dependent probe features."""

    for field in VIZWIZ_CONTRACT_FIELDS:
        row[field] = copy.deepcopy(manifest_row.get(field))
    clean = row.get("clean")
    if not isinstance(clean, MutableMapping):
        raise ValueError(f"Teacher row {row.get('instance_id')} has no clean mapping")
    old_clean_norm = str(clean.get("norm_answer", ""))
    clean_norm = normalize_answer(str(clean.get("raw_answer", "")), "vizwiz")
    clean["norm_answer"] = clean_norm
    clean["correct"] = int(clean_norm == manifest_row["gold_answer"])

    probes = row.get("probes")
    if not isinstance(probes, MutableMapping):
        raise ValueError(f"Teacher row {row.get('instance_id')} has no probes mapping")
    for probe_name, payload in probes.items():
        if not isinstance(payload, MutableMapping):
            raise ValueError(
                f"Teacher row {row.get('instance_id')} has invalid probe {probe_name}"
            )
        if payload.get("valid") is not True:
            continue
        old_probe_norm = str(payload.get("norm_answer", ""))
        if probe_name == "grounding":
            parsed = parse_grounding_output(
                str(payload.get("raw_answer", "")), "vizwiz"
            )
            if not parsed.is_valid:
                raise ValueError(
                    f"Valid VizWiz grounding row no longer parses for "
                    f"{row.get('instance_id')}: {parsed.invalid_reason}"
                )
            probe_norm = parsed.norm_final_answer
            payload["parse_status"] = parsed.parse_status
        else:
            probe_norm = normalize_answer(
                str(payload.get("raw_answer", "")), "vizwiz"
            )
        payload["norm_answer"] = probe_norm
        exact = probe_norm == clean_norm
        payload["flip"] = not exact
        payload["exact_match"] = float(exact)
        if exact:
            payload["semantic_match"] = 1.0
        elif probe_norm != old_probe_norm or clean_norm != old_clean_norm:
            payload["semantic_match"] = compute_semantic_match(
                pred_answer=probe_norm,
                target_answer=clean_norm,
                dataset="vizwiz",
                threshold=semantic_threshold,
                embedding_fn=embedding_fn,
            )


def migrate_teacher_row(
    source: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    *,
    new_manifest_sha256: str,
    source_path: Path,
    source_file_sha256: str,
    thresholds: Any,
    semantic_threshold: float = 0.82,
    embedding_fn: Callable[[str, str], float] | None = None,
) -> Dict[str, Any]:
    """Migrate one non-open teacher/invalid-teacher record."""

    row = copy.deepcopy(dict(source))
    for field in ("group_id", "dataset", "split", "image_path", "question"):
        if row.get(field) != manifest_row.get(field):
            raise ValueError(
                f"Teacher/manifest {field} drift for {row.get('instance_id')}"
            )
    if manifest_row.get("dataset") == "hallusionbench":
        if manifest_row.get("answer_type") != HALLUSION_BINARY:
            raise ValueError("Open-ended rows must be regenerated, not migrated")
        _update_answer_features(row, manifest_row)
        _recompute_labels(row, thresholds)
    elif manifest_row.get("dataset") == "vizwiz":
        _update_vizwiz_answer_contract(
            row,
            manifest_row,
            semantic_threshold=semantic_threshold,
            embedding_fn=embedding_fn,
        )
        _recompute_labels(row, thresholds)
    row["source_manifest_sha256"] = new_manifest_sha256
    row["answer_contract_migration"] = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "policy": (
            "reuse_safe_recompute_gold_and_parser_dependent_"
            "drop_open_ended"
        ),
        "source_path": str(source_path),
        "source_file_sha256": source_file_sha256,
        "source_record_sha256": hash_dict(dict(source)),
    }
    return row


def _validate_source_row(
    row: Mapping[str, Any], *, old_manifest_sha256: str, path: Path
) -> None:
    if row.get("source_manifest_sha256") != old_manifest_sha256:
        raise ValueError(
            f"Source manifest hash mismatch in {path} for {row.get('instance_id')}"
        )


def _validate_current_grounding_parser(rows: Iterable[Mapping[str, Any]]) -> None:
    """Require migrated valid rows to satisfy the current resume parser."""

    for row in rows:
        grounding = row.get("probes", {}).get("grounding")
        if not isinstance(grounding, Mapping) or grounding.get("valid") is not True:
            raise ValueError(
                "Migrated valid row has no valid grounding observation: "
                f"{row.get('instance_id')}"
            )
        parsed = parse_grounding_output(
            str(grounding.get("raw_answer", "")),
            str(row.get("dataset", "")),
            answer_type=row.get("answer_type"),
        )
        if not parsed.is_valid:
            raise ValueError(
                "Current grounding parser rejects migrated valid row "
                f"{row.get('instance_id')}: {parsed.invalid_reason}"
            )
        if parsed.norm_final_answer != grounding.get("norm_answer"):
            raise ValueError(
                "Grounding parser drift remains after migration for "
                f"{row.get('instance_id')}: saved={grounding.get('norm_answer')!r}, "
                f"current={parsed.norm_final_answer!r}"
            )


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in iter_jsonl(path)]


def _migrate_shard(
    *,
    teacher_path: Path,
    output_dir: Path,
    new_by_id: Mapping[str, Mapping[str, Any]],
    open_ids: set[str],
    old_manifest_sha256: str,
    new_manifest_sha256: str,
    frozen_sha256: str,
    thresholds: Any,
    overwrite: bool,
    semantic_threshold: float = 0.82,
    embedding_fn: Callable[[str, str], float] | None = None,
) -> Dict[str, Any]:
    match = TEACHER_FILE_RE.fullmatch(teacher_path.name)
    if match is None:
        raise ValueError(f"Unexpected teacher filename: {teacher_path.name}")
    shard_id = int(match.group("shard"))
    num_shards = int(match.group("num_shards"))
    failure_path = teacher_path.with_suffix(".failures.jsonl")
    teacher_sha = file_sha256(teacher_path)
    failure_sha = file_sha256(failure_path) if failure_path.exists() else ""
    source_valid = _load_rows(teacher_path)
    source_failures = _load_rows(failure_path)
    identity_rows = source_valid + source_failures
    if not identity_rows:
        raise ValueError(f"Empty teacher shard: {teacher_path}")
    model_ids = {str(row.get("model_id", "")) for row in identity_rows}
    if len(model_ids) != 1 or "" in model_ids:
        raise ValueError(f"Ambiguous model identity in {teacher_path}")
    model_id = model_ids.pop()
    selected_ids = {
        instance_id
        for instance_id in new_by_id
        if stable_shard_id(instance_id, model_id, num_shards) == shard_id
    }

    seen: set[Tuple[str, str]] = set()
    migrated_valid: List[Dict[str, Any]] = []
    migrated_failures: List[Dict[str, Any]] = []
    dropped_open: set[str] = set()
    binary_correctness_changes = 0
    answer_feature_changes = 0
    vizwiz_correctness_changes = 0
    vizwiz_probe_feature_changes = 0

    for source in source_valid:
        _validate_source_row(
            source, old_manifest_sha256=old_manifest_sha256, path=teacher_path
        )
        instance_id = str(source.get("instance_id", ""))
        key = (model_id, instance_id)
        if key in seen or instance_id not in selected_ids:
            raise ValueError(f"Duplicate/out-of-scope valid source row: {key}")
        seen.add(key)
        if instance_id in open_ids:
            dropped_open.add(instance_id)
            continue
        before_correct = source.get("clean", {}).get("correct")
        before_features = hash_dict(dict(source.get("probes", {})))
        migrated = migrate_teacher_row(
            source,
            new_by_id[instance_id],
            new_manifest_sha256=new_manifest_sha256,
            source_path=teacher_path,
            source_file_sha256=teacher_sha,
            thresholds=thresholds,
            semantic_threshold=semantic_threshold,
            embedding_fn=embedding_fn,
        )
        if migrated.get("clean", {}).get("correct") != before_correct:
            if source.get("dataset") == "vizwiz":
                vizwiz_correctness_changes += 1
            else:
                binary_correctness_changes += 1
        if hash_dict(dict(migrated.get("probes", {}))) != before_features:
            if source.get("dataset") == "vizwiz":
                vizwiz_probe_feature_changes += 1
            else:
                answer_feature_changes += 1
        migrated_valid.append(migrated)

    for source_failure in source_failures:
        _validate_source_row(
            source_failure,
            old_manifest_sha256=old_manifest_sha256,
            path=failure_path,
        )
        instance_id = str(source_failure.get("instance_id", ""))
        key = (model_id, instance_id)
        if key in seen or instance_id not in selected_ids:
            raise ValueError(f"Duplicate/out-of-scope failure source row: {key}")
        seen.add(key)
        if instance_id in open_ids:
            dropped_open.add(instance_id)
            continue
        invalid = source_failure.get("invalid_teacher_record")
        if not isinstance(invalid, Mapping):
            raise ValueError(f"Failure has no recoverable teacher record: {key}")
        migrated_invalid = migrate_teacher_row(
            invalid,
            new_by_id[instance_id],
            new_manifest_sha256=new_manifest_sha256,
            source_path=failure_path,
            source_file_sha256=failure_sha,
            thresholds=thresholds,
            semantic_threshold=semantic_threshold,
            embedding_fn=embedding_fn,
        )
        failure = copy.deepcopy(source_failure)
        failure["invalid_teacher_record"] = migrated_invalid
        failure["source_manifest_sha256"] = new_manifest_sha256
        failure["answer_contract_migration"] = {
            "schema_version": MIGRATION_SCHEMA_VERSION,
            "policy": (
                "reuse_safe_recompute_gold_and_parser_dependent_"
                "drop_open_ended"
            ),
            "source_path": str(failure_path),
            "source_file_sha256": failure_sha,
            "source_record_sha256": hash_dict(dict(source_failure)),
        }
        migrated_failures.append(failure)

    if {key[1] for key in seen} != selected_ids:
        missing = selected_ids - {key[1] for key in seen}
        raise ValueError(
            f"Source shard coverage mismatch for {teacher_path}: missing={len(missing)}"
        )

    _validate_current_grounding_parser(migrated_valid)

    output_path = output_dir / teacher_path.name
    output_failure_path = output_path.with_suffix(".failures.jsonl")
    write_jsonl(migrated_valid, output_path, overwrite=overwrite)
    write_jsonl(migrated_failures, output_failure_path, overwrite=overwrite)

    completed = validate_resume_teacher_records(
        migrated_valid,
        model_id=model_id,
        selected_ids=selected_ids,
        manifest_sha256=new_manifest_sha256,
        frozen_config_sha256=frozen_sha256,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    failure_keys = {
        (str(row["model_id"]), str(row["instance_id"]))
        for row in migrated_failures
    }
    if completed.intersection(failure_keys):
        raise ValueError(f"Valid/failure overlap after migration: {teacher_path}")
    expected_pending = selected_ids - {key[1] for key in completed}
    actual_pending = {key[1] for key in failure_keys}.union(dropped_open)
    if expected_pending != actual_pending:
        raise ValueError(f"Pending-set mismatch after migration: {teacher_path}")

    return {
        "model_id": model_id,
        "model_name": match.group("model_name"),
        "shard_id": shard_id,
        "num_shards": num_shards,
        "source_teacher_path": str(teacher_path),
        "source_teacher_sha256": teacher_sha,
        "source_failure_path": str(failure_path),
        "source_failure_sha256": failure_sha,
        "output_teacher_path": str(output_path),
        "output_teacher_sha256": file_sha256(output_path),
        "output_failure_path": str(output_failure_path),
        "output_failure_sha256": file_sha256(output_failure_path),
        "selected_count": len(selected_ids),
        "valid_count": len(migrated_valid),
        "failure_count": len(migrated_failures),
        "dropped_open_ended_count": len(dropped_open),
        "pending_count": len(expected_pending),
        "binary_clean_correctness_changes": binary_correctness_changes,
        "binary_probe_feature_changes": answer_feature_changes,
        "vizwiz_clean_correctness_changes": vizwiz_correctness_changes,
        "vizwiz_probe_feature_changes": vizwiz_probe_feature_changes,
    }


def main() -> None:
    args = _parse_args()
    old_manifest_path = Path(args.old_manifest)
    new_manifest_path = Path(args.new_manifest)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)

    old_rows = load_manifest(old_manifest_path)
    new_rows = load_manifest(new_manifest_path)
    manifest_errors = validate_manifest(new_rows)
    if manifest_errors:
        raise SystemExit(f"New manifest validation failed: {manifest_errors[:10]}")
    _, new_by_id, open_ids = validate_manifest_transition(old_rows, new_rows)

    experiment = _read_yaml(Path(args.config))
    frozen_path = Path(str(experiment["frozen_probe_config"]))
    frozen = _read_yaml(frozen_path)
    thresholds = thresholds_from_mapping(frozen["label_thresholds"])
    semantic_config = frozen.get("semantic_matching")
    if not isinstance(semantic_config, Mapping):
        raise SystemExit("Frozen config has no semantic_matching mapping")
    semantic_model = str(semantic_config.get("embedding_model", ""))
    semantic_revision = str(semantic_config.get("embedding_revision", ""))
    semantic_threshold = float(semantic_config.get("threshold"))
    if not semantic_model or not semantic_revision:
        raise SystemExit("Frozen semantic model/revision provenance is incomplete")
    semantic_matcher = SemanticMatcher(
        model_name_or_path=semantic_model,
        revision=semantic_revision,
        device="cpu",
    )
    if not semantic_matcher.is_available:
        raise SystemExit(
            "Frozen semantic matcher is unavailable for VizWiz migration: "
            f"{semantic_matcher.load_error}"
        )

    @lru_cache(maxsize=None)
    def cached_similarity(left: str, right: str) -> float:
        return semantic_matcher.similarity(left, right)

    frozen_sha = file_sha256(frozen_path)
    old_manifest_sha = file_sha256(old_manifest_path)
    new_manifest_sha = file_sha256(new_manifest_path)
    if old_manifest_sha == new_manifest_sha:
        raise SystemExit("Old and new manifest hashes are identical; nothing to migrate")

    teacher_files = sorted(
        path
        for path in input_dir.glob("teacher_*_all_all_shard??-of-??.jsonl")
        if not path.name.endswith(".failures.jsonl")
    )
    if not teacher_files:
        raise SystemExit(f"No teacher shards found in {input_dir}")

    shard_reports = [
        _migrate_shard(
            teacher_path=path,
            output_dir=output_dir,
            new_by_id=new_by_id,
            open_ids=open_ids,
            old_manifest_sha256=old_manifest_sha,
            new_manifest_sha256=new_manifest_sha,
            frozen_sha256=frozen_sha,
            thresholds=thresholds,
            overwrite=args.overwrite,
            semantic_threshold=semantic_threshold,
            embedding_fn=cached_similarity,
        )
        for path in teacher_files
    ]
    by_model: Dict[str, Counter[str]] = defaultdict(Counter)
    for shard in shard_reports:
        model = shard["model_id"]
        for target, source in (
            ("selected_count", "selected_count"),
            ("valid_count", "valid_count"),
            ("failure_count", "failure_count"),
            ("dropped_open_ended_count", "dropped_open_ended_count"),
            ("pending_count", "pending_count"),
            ("binary_clean_correctness_changes", "binary_clean_correctness_changes"),
            ("binary_probe_feature_changes", "binary_probe_feature_changes"),
            ("vizwiz_clean_correctness_changes", "vizwiz_clean_correctness_changes"),
            ("vizwiz_probe_feature_changes", "vizwiz_probe_feature_changes"),
        ):
            by_model[model][target] += int(shard[source])
    for model_id, counts in by_model.items():
        if counts["selected_count"] != len(new_rows):
            raise ValueError(f"Model coverage mismatch after migration: {model_id}")
        if counts["dropped_open_ended_count"] != 14:
            raise ValueError(f"Model did not invalidate exactly 14 open rows: {model_id}")

    report = {
        "is_valid": True,
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "policy": (
            "retain_all_951_regenerate_14_open_ended_and_"
            "deterministically_relabel_and_refresh_vizwiz_parser_features"
        ),
        "old_manifest_path": str(old_manifest_path),
        "old_manifest_sha256": old_manifest_sha,
        "new_manifest_path": str(new_manifest_path),
        "new_manifest_sha256": new_manifest_sha,
        "frozen_probe_config_path": str(frozen_path),
        "frozen_probe_config_sha256": frozen_sha,
        "manifest_record_count": len(new_rows),
        "hallusionbench_open_ended_count": len(open_ids),
        "hallusionbench_open_ended_ids": sorted(open_ids),
        "models": {model: dict(counts) for model, counts in sorted(by_model.items())},
        "shards": shard_reports,
    }
    write_json(report, report_path, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
