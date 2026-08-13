#!/usr/bin/env python3
"""Run Week 4 readiness, daily teacher-progress, or full completion checks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.audits.week4_validation import (
    IMMUTABLE_REVISION_RE,
    collect_artifact_rows,
    validate_audit_packet,
    validate_week4_artifacts,
)
from proactive.data.manifests import load_manifest, validate_manifest
from proactive.teacher.offline import (
    legal_probe_names,
    probe_observations_from_record,
    stable_shard_id,
    teacher_key,
    thresholds_from_mapping,
)
from proactive.utils.io import file_sha256, write_json


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validate_week4")


REQUIRED_WEEK4_FILES = (
    "src/proactive/teacher/offline.py",
    "src/proactive/audits/human_audit.py",
    "src/proactive/audits/week4_validation.py",
    "scripts/run_teacher.py",
    "scripts/build_labels.py",
    "scripts/sample_states.py",
    "scripts/export_human_audit.py",
    "scripts/validate_week4.py",
    "configs/experiments/teacher_core.yaml",
    "doc/docs/WEEK_04_REQUIREMENTS.md",
    "tests/test_week4_pipeline.py",
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _model_configs(experiment: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        key: _load_yaml(Path(str(path)))
        for key, path in experiment["models"].items()
    }


def readiness_report(
    repo_root: Path,
    experiment: Mapping[str, Any],
    model_configs: Mapping[str, Mapping[str, Any]],
    manifest_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    errors: List[str] = []
    for relative in REQUIRED_WEEK4_FILES:
        path = repo_root / relative
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"Missing or empty Week 4 prerequisite: {relative}")
    manifest_errors = validate_manifest(manifest_records)
    errors.extend(manifest_errors)
    if any(row.get("split") not in {"train", "val", "cal", "test"} for row in manifest_records):
        errors.append("Every manifest row must have a train/val/cal/test split")
    group_splits: Dict[str, Set[str]] = {}
    for row in manifest_records:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
    leaks = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaks:
        errors.append(f"Grouped manifest split leakage in {len(leaks)} groups")

    if experiment.get("metadata", {}).get("approval_status") != "APPROVED":
        errors.append("Week 4 experiment config still requires owner approval")
    compute_authorization = experiment.get("compute_authorization")
    if not isinstance(compute_authorization, dict):
        errors.append("Missing Week 4 compute_authorization mapping")
        compute_authorization = {}
    if compute_authorization.get("staged_checks_approved") is not True:
        errors.append("Week 4 staged checks are not approved")
    staged_max = compute_authorization.get("staged_max_examples")
    if (
        not isinstance(staged_max, int)
        or isinstance(staged_max, bool)
        or staged_max <= 0
    ):
        errors.append("staged_max_examples must be a positive integer")
    staged_datasets = compute_authorization.get("staged_full_dataset_allowlist")
    active_datasets = {str(name) for name in experiment.get("active_datasets", [])}
    if not isinstance(staged_datasets, list) or not staged_datasets:
        errors.append("staged_full_dataset_allowlist must be a non-empty list")
    elif not {str(name) for name in staged_datasets}.issubset(active_datasets):
        errors.append("Staged full-dataset allowlist contains an inactive dataset")
    if not isinstance(compute_authorization.get("full_core_approved"), bool):
        errors.append("full_core_approved must be boolean")
    core_keys = set(experiment["core_model_keys"])
    for key in core_keys:
        revision = str(model_configs[key].get("model_revision", ""))
        if not IMMUTABLE_REVISION_RE.fullmatch(revision):
            errors.append(f"Core model {key} revision is not an immutable 40-hex commit: {revision!r}")
    frozen = _load_yaml(Path(str(experiment["frozen_probe_config"])))
    if frozen.get("metadata", {}).get("status") != "FROZEN":
        errors.append("Week 3 probe configuration is not frozen")
    try:
        thresholds_from_mapping(frozen["label_thresholds"])
    except (KeyError, ValueError) as exc:
        errors.append(f"Frozen label thresholds are invalid: {exc}")

    dataset_counts = Counter(row["dataset"] for row in manifest_records)
    relation_rows = sum(bool(row.get("relation_applicable")) for row in manifest_records)
    required_models = len(core_keys)
    expected_teacher_rows = len(manifest_records) * required_models
    expected_forward_passes = required_models * (
        len(manifest_records) * 7 + relation_rows
    )
    return {
        "is_valid": not errors,
        "approval_status": experiment.get("metadata", {}).get("approval_status"),
        "compute_authorization": compute_authorization,
        "manifest_records": len(manifest_records),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "relation_applicable_records": relation_rows,
        "required_core_models": sorted(core_keys),
        "expected_teacher_rows": expected_teacher_rows,
        "expected_forward_passes": expected_forward_passes,
        "errors": errors,
    }


def teacher_progress_report(
    manifest_records: List[Dict[str, Any]],
    teacher_rows: List[Tuple[Dict[str, Any], Path, str]],
    required_model_ids: Set[str],
    manifest_sha256: str,
) -> Dict[str, Any]:
    errors: List[str] = []
    manifest_index = {row["instance_id"]: row for row in manifest_records}
    keys: Set[Tuple[str, str]] = set()
    probe_keys: Set[Tuple[str, str, str]] = set()
    counts: Counter[str] = Counter()
    split_counts: Counter[Tuple[str, str]] = Counter()
    for row, path, _ in teacher_rows:
        try:
            key = teacher_key(row)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if key in keys:
            errors.append(f"Duplicate teacher key: {key}")
            continue
        keys.add(key)
        counts[key[0]] += 1
        split_counts[(key[0], str(row.get("split")))] += 1
        source = manifest_index.get(key[1])
        if source is None:
            errors.append(f"Teacher row absent from manifest: {key}")
            continue
        for field in ("group_id", "dataset", "split"):
            if row.get(field) != source.get(field):
                errors.append(f"Teacher/manifest {field} mismatch: {key}")
        if row.get("valid") is not True:
            errors.append(f"Invalid teacher row: {key}")
        if not IMMUTABLE_REVISION_RE.fullmatch(str(row.get("model_revision", ""))):
            errors.append(f"Unpinned teacher model revision: {key}")
        if row.get("source_manifest_sha256") != manifest_sha256:
            errors.append(f"Teacher manifest hash drift: {key}")
        for hash_field in (
            "frozen_probe_config_sha256",
            "experiment_config_sha256",
            "model_config_sha256",
        ):
            if not re.fullmatch(r"[0-9a-fA-F]{64}", str(row.get(hash_field, ""))):
                errors.append(f"Missing/invalid {hash_field}: {key}")
        shard_id = row.get("shard_id")
        num_shards = row.get("num_shards")
        if (
            not isinstance(shard_id, int)
            or isinstance(shard_id, bool)
            or not isinstance(num_shards, int)
            or isinstance(num_shards, bool)
            or num_shards <= 0
            or not 0 <= shard_id < num_shards
        ):
            errors.append(f"Invalid shard metadata: {key}")
        elif stable_shard_id(key[1], key[0], num_shards) != shard_id:
            errors.append(f"Teacher row assigned to wrong deterministic shard: {key}")
        try:
            probe_observations_from_record(row)
        except ValueError as exc:
            errors.append(f"Teacher probe validation failed for {key}: {exc}")
            continue
        for probe in legal_probe_names(row):
            probe_key = (key[0], key[1], probe)
            if probe_key in probe_keys:
                errors.append(f"Duplicate instance-model-probe key: {probe_key}")
            probe_keys.add(probe_key)

    expected_per_model = len(manifest_index)
    completion = {
        model_id: {
            "completed": counts[model_id],
            "expected": expected_per_model,
            "fraction": counts[model_id] / expected_per_model if expected_per_model else 0.0,
        }
        for model_id in sorted(required_model_ids | set(counts))
    }
    return {
        "is_valid": not errors and bool(keys),
        "teacher_records": len(keys),
        "unique_probe_records": len(probe_keys),
        "completion_by_model": completion,
        "split_counts": {
            f"{model}|{split}": count
            for (model, split), count in sorted(split_counts.items())
        },
        "errors": errors[:500],
        "error_count": len(errors),
    }


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + "_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["readiness", "teacher_progress", "full"], default="readiness")
    parser.add_argument("--config", default="configs/experiments/teacher_core.yaml")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--teacher_path", default="outputs/teacher_core")
    parser.add_argument("--labels_path", default="outputs/labels_core")
    parser.add_argument("--states_path", default="outputs/states_v1")
    parser.add_argument("--audit_dir", default="outputs/human_audit")
    parser.add_argument("--output_dir", default="outputs/week4_reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Accepted for script-contract parity")
    parser.add_argument("--device", default="cpu", help="Accepted for script-contract parity")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    experiment = _load_yaml(Path(args.config))
    configs = _model_configs(experiment)
    manifest_records = load_manifest(args.manifest_path)
    core_keys = list(experiment["core_model_keys"])
    required_model_ids = {str(configs[key]["model_id"]) for key in core_keys}
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"week4_{args.mode}_report.json"
    if output_path.exists() and not (args.resume or args.overwrite):
        raise SystemExit(f"Report exists: {output_path}. Use --resume or --overwrite.")

    readiness = readiness_report(repo_root, experiment, configs, manifest_records)
    if args.mode == "readiness":
        report = readiness
        report["mode"] = args.mode
        if not args.dry_run:
            write_json(report, output_path, overwrite=output_path.exists())
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["is_valid"] else 1)

    teacher_rows, teacher_manifest = collect_artifact_rows(Path(args.teacher_path), "teacher_")
    progress = teacher_progress_report(
        manifest_records,
        teacher_rows,
        required_model_ids,
        file_sha256(args.manifest_path),
    )
    if args.mode == "teacher_progress":
        report = {
            "mode": args.mode,
            "is_valid": progress["is_valid"],
            "readiness": readiness,
            "teacher_progress": progress,
            "teacher_files": teacher_manifest,
        }
        if not args.dry_run:
            write_json(report, output_path, overwrite=output_path.exists())
            write_json(
                {"files": teacher_manifest, "summary": progress},
                output_dir / "teacher_manifest.json",
                overwrite=(output_dir / "teacher_manifest.json").exists(),
            )
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["is_valid"] else 1)

    label_rows, label_manifest = collect_artifact_rows(Path(args.labels_path), "labels_")
    state_rows, state_manifest = collect_artifact_rows(Path(args.states_path), "states_")
    frozen = _load_yaml(Path(str(experiment["frozen_probe_config"])))
    thresholds = thresholds_from_mapping(frozen["label_thresholds"])
    gates = experiment["completion_gates"]
    artifact_report = validate_week4_artifacts(
        manifest_records=manifest_records,
        teacher_rows=teacher_rows,
        label_rows=label_rows,
        state_rows=state_rows,
        required_model_ids=required_model_ids,
        thresholds=thresholds,
        max_sixway_fraction=float(gates["max_sixway_fraction"]),
        min_bit_count_per_dataset_model=int(gates["min_bit_count_per_dataset_model"]),
        require_complete=True,
    )
    audit_report = validate_audit_packet(
        Path(args.audit_dir),
        int(experiment["human_audit"]["total_examples"]),
        required_model_ids={
            str(configs[key]["model_id"])
            for key in experiment["human_audit"]["required_model_keys"]
        },
        required_datasets=set(experiment["active_datasets"]),
    )

    catch_up_key = experiment["catch_up"]["model_key"]
    catch_up_model_id = configs[catch_up_key]["model_id"]
    catch_up_complete = artifact_report["model_teacher_counts"].get(catch_up_model_id, 0) == len(manifest_records)
    catch_up_document = Path(str(experiment["catch_up"]["documentation"]))
    catch_up_documented = False
    if catch_up_document.exists():
        text = catch_up_document.read_text(encoding="utf-8").upper()
        catch_up_documented = "SCHEDULED" in text or "IN PROGRESS" in text or "COMPLETE" in text
    catch_up_valid = catch_up_complete or catch_up_documented

    final_valid = (
        readiness["is_valid"]
        and progress["is_valid"]
        and artifact_report["is_valid"]
        and audit_report["is_valid"]
        and catch_up_valid
    )
    report = {
        "mode": "full",
        "is_valid": final_valid,
        "readiness": readiness,
        "teacher_progress": progress,
        "artifacts": artifact_report,
        "human_audit": audit_report,
        "catch_up": {
            "model_id": catch_up_model_id,
            "cache_complete": catch_up_complete,
            "documented": catch_up_documented,
            "is_valid": catch_up_valid,
        },
    }
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            artifact_report["class_distribution_rows"],
            output_dir / "label_distribution_by_dataset_model.csv",
        )
        _write_csv(
            artifact_report["bit_distribution_rows"],
            output_dir / "bit_distribution_by_dataset_model.csv",
        )
        write_json(
            {"files": teacher_manifest, "summary": progress},
            output_dir / "teacher_manifest.json",
            overwrite=(output_dir / "teacher_manifest.json").exists(),
        )
        write_json(
            {"files": label_manifest, "row_count": artifact_report["label_records"]},
            output_dir / "label_manifest.json",
            overwrite=(output_dir / "label_manifest.json").exists(),
        )
        write_json(
            {"files": state_manifest, "row_count": artifact_report["state_records"]},
            output_dir / "state_manifest.json",
            overwrite=(output_dir / "state_manifest.json").exists(),
        )
        write_json(report, output_path, overwrite=output_path.exists())
    printable = dict(report)
    printable["artifacts"] = {
        key: value
        for key, value in artifact_report.items()
        if key not in {"class_distribution_rows", "bit_distribution_rows"}
    }
    print(json.dumps(printable, indent=2))
    raise SystemExit(0 if final_valid else 1)


if __name__ == "__main__":
    main()
