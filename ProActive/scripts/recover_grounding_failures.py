#!/usr/bin/env python3
"""Recover only fail-closed grounding rows with one uniform concise retry.

This script never edits its accepted source cache.  It copies every valid
source row into a separate output shard, retries every remaining
*format/truncation* grounding failure with the same concise
describe-then-answer prompt, recomputes the affected labels, and retains an
atomic failure ledger.  Sources may be either the 1,024-token grounding-refresh
cache or a fail-closed teacher cache such as the Week 8 held-out shift run.
Rows cannot be selected by model prediction, gold label, dataset, or desired
outcome: the retry trigger is exclusively a malformed mandatory grounding
observation in the source failure ledger.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from refresh_grounding_cache import _rebuild_teacher_row  # noqa: E402
from run_teacher import (  # noqa: E402
    IMMUTABLE_REVISION_RE,
    _acquire_output_lock,
    _enforce_compute_authorization,
    _load_adapter,
    _load_yaml,
    _release_output_lock,
    _resolve_model_config,
)
from proactive.data.manifests import load_manifest, validate_manifest  # noqa: E402
from proactive.features.semantic import SemanticMatcher  # noqa: E402
from proactive.probes.probe_runner import _run_grounding_probe  # noqa: E402
from proactive.prompts.templates import (  # noqa: E402
    make_concise_grounding_retry_prompt,
)
from proactive.teacher.cache_builder import _load_image_safely  # noqa: E402
from proactive.teacher.offline import (  # noqa: E402
    stable_shard_id,
    thresholds_from_mapping,
    validate_resume_teacher_records,
)
from proactive.utils.hashing import hash_dict  # noqa: E402
from proactive.utils.io import (  # noqa: E402
    append_jsonl,
    file_sha256,
    iter_jsonl,
    write_jsonl,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("recover_grounding_failures")

RECOVERY_SCHEMA_VERSION = 1
RECOVERY_POLICY_ID = "concise_describe_then_answer_retry_v1"
RECOVERABLE_MESSAGES = {
    "Missing FINAL_ANSWER tag and unable to resolve valid final answer",
    "FINAL_ANSWER tag has no answer content",
    "Empty or unknown free-form output",
    "Empty generation output",
}
RECOVERABLE_MESSAGE_PREFIXES = (
    "Multiple-choice answer is not one option letter:",
    "Binary dataset answer not in valid domain:",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "cal", "test", "shift"),
        default="all",
    )
    parser.add_argument(
        "--input_dir", default="outputs/teacher_core_contract_v1_grounding1024"
    )
    parser.add_argument(
        "--output_dir", default="outputs/teacher_core_contract_v1_recovered"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--source_max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _source_paths(
    input_dir: Path,
    model_name: str,
    shard_id: int,
    num_shards: int,
    split: str = "all",
) -> Tuple[Path, Path]:
    name = (
        f"teacher_{model_name}_all_{split}_"
        f"shard{shard_id:02d}-of-{num_shards:02d}.jsonl"
    )
    teacher_path = input_dir / name
    return teacher_path, teacher_path.with_suffix(".failures.jsonl")


def _load_source_failures(
    path: Path,
    *,
    model_id: str,
    selected_ids: set[str],
    source_max_new_tokens: int,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    failures: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return failures
    for row in iter_jsonl(path):
        key = (str(row.get("model_id", "")), str(row.get("instance_id", "")))
        record_type = row.get("record_type")
        if record_type not in {"grounding_refresh_failure", "teacher_failure"}:
            raise ValueError(f"Invalid source grounding failure: {key}")
        if row.get("schema_version") != 1:
            raise ValueError(f"Unsupported source failure schema: {key}")
        if key in failures or key[0] != model_id or key[1] not in selected_ids:
            raise ValueError(f"Duplicate or out-of-scope source failure: {key}")
        if record_type == "grounding_refresh_failure":
            if row.get("uniform_max_new_tokens") != source_max_new_tokens:
                raise ValueError(f"Source failure token-cap drift: {key}")
            observation = row.get("grounding_observation")
            trigger_message = row.get("error_message")
        else:
            base = row.get("invalid_teacher_record")
            if not isinstance(base, Mapping):
                raise ValueError(f"Teacher failure lacks invalid record: {key}")
            if (
                str(base.get("model_id", "")) != model_id
                or str(base.get("instance_id", "")) != key[1]
            ):
                raise ValueError(f"Teacher failure base identity drift: {key}")
            probes = base.get("probes")
            observation = probes.get("grounding") if isinstance(probes, Mapping) else None
            trigger_message = (
                observation.get("invalid_reason")
                if isinstance(observation, Mapping)
                else None
            )
        if not isinstance(trigger_message, str) or not (
            trigger_message in RECOVERABLE_MESSAGES
            or trigger_message.startswith(RECOVERABLE_MESSAGE_PREFIXES)
        ):
            raise ValueError(f"Non-format failure is not retryable by this policy: {key}")
        if not isinstance(observation, Mapping) or observation.get("valid") is not False:
            raise ValueError(f"Source failure lacks an invalid grounding observation: {key}")
        if observation.get("parse_status") not in {"empty", "malformed"}:
            raise ValueError(f"Source failure is not a parse/format failure: {key}")
        if record_type == "grounding_refresh_failure":
            for field in (
                "source_kind",
                "source_path",
                "source_file_sha256",
                "source_record_sha256",
            ):
                if not row.get(field):
                    raise ValueError(f"Source failure lacks provenance {field}: {key}")
        failures[key] = row
    return failures


def _find_failure_base(
    failure: Mapping[str, Any], *, model_id: str, instance_id: str
) -> Dict[str, Any]:
    """Resolve and hash-check the pre-refresh teacher record for one failure."""
    if failure.get("record_type") == "teacher_failure":
        base = failure.get("invalid_teacher_record")
        if not isinstance(base, dict):
            raise ValueError(f"Teacher failure lacks invalid record: {instance_id}")
        if (
            str(base.get("model_id", "")) != model_id
            or str(base.get("instance_id", "")) != instance_id
        ):
            raise ValueError(f"Teacher failure base identity drift: {instance_id}")
        clean = base.get("clean")
        probes = base.get("probes")
        if not isinstance(clean, Mapping) or clean.get("valid") is not True:
            raise ValueError(f"Failure base has invalid clean observation: {instance_id}")
        if not isinstance(probes, Mapping) or "grounding" not in probes:
            raise ValueError(f"Failure base lacks grounding observation: {instance_id}")
        return base

    source_path = Path(str(failure["source_path"]))
    if not source_path.exists():
        raise FileNotFoundError(f"Missing failure provenance source: {source_path}")
    if file_sha256(source_path) != failure["source_file_sha256"]:
        raise ValueError(f"Failure provenance file hash drift: {source_path}")

    source_kind = failure["source_kind"]
    matches = []
    for row in iter_jsonl(source_path):
        if (
            str(row.get("model_id", "")) == model_id
            and str(row.get("instance_id", "")) == instance_id
        ):
            if source_kind == "valid_teacher":
                matches.append(row)
            elif source_kind == "failure_ledger":
                base = row.get("invalid_teacher_record")
                if isinstance(base, dict):
                    matches.append(base)
            else:
                raise ValueError(f"Unsupported failure provenance kind: {source_kind}")
    if len(matches) != 1:
        raise ValueError(
            f"Expected one recoverable provenance row for {instance_id}; "
            f"found {len(matches)}"
        )
    base = matches[0]
    if hash_dict(base) != failure["source_record_sha256"]:
        raise ValueError(f"Failure provenance record hash drift: {instance_id}")
    if base.get("valid") is True:
        return base
    clean = base.get("clean")
    probes = base.get("probes")
    if not isinstance(clean, Mapping) or clean.get("valid") is not True:
        raise ValueError(f"Failure base has invalid clean observation: {instance_id}")
    if not isinstance(probes, Mapping) or "grounding" not in probes:
        raise ValueError(f"Failure base lacks grounding observation: {instance_id}")
    return base


def _immediate_provenance(
    *, source_kind: str, path: Path, file_sha: str, row: Mapping[str, Any]
) -> Dict[str, str]:
    return {
        "source_kind": source_kind,
        "source_path": str(path),
        "source_file_sha256": file_sha,
        "source_record_sha256": hash_dict(dict(row)),
    }


def _annotate_copy(
    row: Mapping[str, Any], provenance: Mapping[str, str]
) -> Dict[str, Any]:
    copied = copy.deepcopy(dict(row))
    copied["grounding_recovery"] = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "policy_id": RECOVERY_POLICY_ID,
        "status": "not_required",
        **dict(provenance),
    }
    return copied


def _annotate_recovery(
    row: Mapping[str, Any],
    *,
    provenance: Mapping[str, str],
    source_failure: Mapping[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    recovered = copy.deepcopy(dict(row))
    recovered["grounding_recovery"] = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "policy_id": RECOVERY_POLICY_ID,
        "status": "recovered",
        "trigger_error": source_failure["error_message"],
        "retry_max_new_tokens": max_new_tokens,
        **dict(provenance),
    }
    return recovered


def _validate_output_rows(
    path: Path,
    *,
    model_id: str,
    selected_ids: set[str],
    manifest_sha: str,
    frozen_sha: str,
    shard_id: int,
    num_shards: int,
    valid_source_keys: set[Tuple[str, str]],
    failure_source_keys: set[Tuple[str, str]],
    provenance: Mapping[Tuple[str, str], Mapping[str, str]],
) -> set[Tuple[str, str]]:
    rows = list(iter_jsonl(path))
    completed = validate_resume_teacher_records(
        rows, model_id, selected_ids, manifest_sha, frozen_sha, shard_id, num_shards
    )
    for row in rows:
        key = (model_id, str(row["instance_id"]))
        recovery = row.get("grounding_recovery")
        if not isinstance(recovery, Mapping):
            raise ValueError(f"Missing grounding_recovery metadata: {key}")
        if recovery.get("schema_version") != RECOVERY_SCHEMA_VERSION:
            raise ValueError(f"Grounding recovery schema drift: {key}")
        if recovery.get("policy_id") != RECOVERY_POLICY_ID:
            raise ValueError(f"Grounding recovery policy drift: {key}")
        expected_status = "not_required" if key in valid_source_keys else "recovered"
        if key not in valid_source_keys and key not in failure_source_keys:
            raise ValueError(f"Output key absent from recovery source partition: {key}")
        if recovery.get("status") != expected_status:
            raise ValueError(f"Grounding recovery status drift: {key}")
        for field, expected in provenance[key].items():
            if recovery.get(field) != expected:
                raise ValueError(f"Grounding recovery provenance drift {field}: {key}")
    return completed


def _load_retry_ledger(
    path: Path,
    *,
    model_id: str,
    failure_source_keys: set[Tuple[str, str]],
    provenance: Mapping[Tuple[str, str], Mapping[str, str]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in iter_jsonl(path):
        key = (str(row.get("model_id", "")), str(row.get("instance_id", "")))
        if row.get("record_type") != "grounding_recovery_failure":
            raise ValueError(f"Invalid recovery failure record: {key}")
        if row.get("schema_version") != RECOVERY_SCHEMA_VERSION:
            raise ValueError(f"Recovery failure schema drift: {key}")
        if row.get("policy_id") != RECOVERY_POLICY_ID:
            raise ValueError(f"Recovery failure policy drift: {key}")
        if key in rows or key[0] != model_id or key not in failure_source_keys:
            raise ValueError(f"Duplicate or out-of-scope recovery failure: {key}")
        for field, expected in provenance[key].items():
            if row.get(field) != expected:
                raise ValueError(f"Recovery failure provenance drift {field}: {key}")
        rows[key] = row
    return rows


def _write_retry_ledger(
    path: Path, rows: Mapping[Tuple[str, str], Mapping[str, Any]]
) -> None:
    write_jsonl([dict(rows[key]) for key in sorted(rows)], path, overwrite=True)


def main() -> None:
    args = _parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("--shard_id must satisfy 0 <= shard_id < num_shards")
    if not 32 <= args.max_new_tokens <= 512:
        raise SystemExit("--max_new_tokens must be between 32 and 512")
    if args.source_max_new_tokens < args.max_new_tokens:
        raise SystemExit("Source token cap must be at least the concise retry cap")

    experiment = _load_yaml(Path(args.config))
    model_config_path = _resolve_model_config(experiment, args.model, None)
    model_config = _load_yaml(model_config_path)
    revision = str(model_config.get("model_revision", ""))
    if not IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise SystemExit("Grounding recovery requires an immutable model revision")

    frozen_path = Path(str(experiment["frozen_probe_config"]))
    frozen = _load_yaml(frozen_path)
    thresholds = thresholds_from_mapping(frozen["label_thresholds"])
    semantic = dict(frozen["semantic_matching"])

    manifest_path = Path(args.manifest_path)
    records = load_manifest(manifest_path)
    manifest_errors = validate_manifest(records)
    if manifest_errors:
        raise SystemExit(f"Manifest validation failed: {manifest_errors[:5]}")
    model_id = str(model_config["model_id"])
    selected = [
        row
        for row in records
        if (args.split == "all" or row.get("split") == args.split)
        and stable_shard_id(str(row["instance_id"]), model_id, args.num_shards)
        == args.shard_id
    ]
    selected_ids = {str(row["instance_id"]) for row in selected}
    manifest_sha = file_sha256(manifest_path)
    frozen_sha = file_sha256(frozen_path)
    model_name = str(model_config["model_name"])
    teacher_path, failure_path = _source_paths(
        Path(args.input_dir),
        model_name,
        args.shard_id,
        args.num_shards,
        args.split,
    )
    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing source teacher shard: {teacher_path}")

    teacher_rows = list(iter_jsonl(teacher_path))
    valid_keys = validate_resume_teacher_records(
        teacher_rows,
        model_id,
        selected_ids,
        manifest_sha,
        frozen_sha,
        args.shard_id,
        args.num_shards,
    )
    source_failures = _load_source_failures(
        failure_path,
        model_id=model_id,
        selected_ids=selected_ids,
        source_max_new_tokens=args.source_max_new_tokens,
    )
    failure_keys = set(source_failures)
    expected_keys = {(model_id, instance_id) for instance_id in selected_ids}
    if valid_keys.intersection(failure_keys):
        raise ValueError("Source valid/failure partitions overlap")
    if valid_keys.union(failure_keys) != expected_keys:
        raise ValueError(
            "Source coverage mismatch: "
            f"valid={len(valid_keys)} failures={len(failure_keys)} "
            f"expected={len(expected_keys)}"
        )

    # Authorization is based on the deterministic failure-triggered scope.
    args.limit = len(failure_keys)
    args.dataset = "all"
    _enforce_compute_authorization(experiment, args)

    teacher_sha = file_sha256(teacher_path)
    failure_sha = file_sha256(failure_path) if failure_path.exists() else ""
    valid_by_key = {
        (model_id, str(row["instance_id"])): row for row in teacher_rows
    }
    failure_bases: Dict[Tuple[str, str], Dict[str, Any]] = {}
    provenance: Dict[Tuple[str, str], Dict[str, str]] = {}
    for key, row in valid_by_key.items():
        provenance[key] = _immediate_provenance(
            source_kind=(
                "grounding1024_valid_teacher"
                if args.split == "all"
                else "teacher_valid"
            ),
            path=teacher_path,
            file_sha=teacher_sha,
            row=row,
        )
    for key, failure in source_failures.items():
        failure_bases[key] = _find_failure_base(
            failure, model_id=key[0], instance_id=key[1]
        )
        provenance[key] = _immediate_provenance(
            source_kind=(
                "grounding1024_failure"
                if failure.get("record_type") == "grounding_refresh_failure"
                else "teacher_failure"
            ),
            path=failure_path,
            file_sha=failure_sha,
            row=failure,
        )

    output_path = Path(args.output_dir) / teacher_path.name
    output_failure_path = output_path.with_suffix(".failures.jsonl")
    logger.info(
        "Model=%s shard=%s/%s valid_source=%s retry_scope=%s",
        model_name,
        args.shard_id,
        args.num_shards,
        len(valid_keys),
        len(failure_keys),
    )
    logger.info("Recovery policy=%s", RECOVERY_POLICY_ID)
    logger.info("Output=%s", output_path)
    if args.dry_run:
        return

    initialize = not output_path.exists() or args.overwrite
    if output_path.exists() and not (args.resume or args.overwrite):
        raise SystemExit(f"Output exists: {output_path}; use --resume or --overwrite")
    if initialize:
        copied = [
            _annotate_copy(valid_by_key[key], provenance[key])
            for key in sorted(valid_keys)
        ]
        write_jsonl(copied, output_path, overwrite=True)
        write_jsonl([], output_failure_path, overwrite=True)

    completed = _validate_output_rows(
        output_path,
        model_id=model_id,
        selected_ids=selected_ids,
        manifest_sha=manifest_sha,
        frozen_sha=frozen_sha,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        valid_source_keys=valid_keys,
        failure_source_keys=failure_keys,
        provenance=provenance,
    )
    retry_ledger = _load_retry_ledger(
        output_failure_path,
        model_id=model_id,
        failure_source_keys=failure_keys,
        provenance=provenance,
    )
    for key in set(retry_ledger).intersection(completed):
        del retry_ledger[key]
    pending = [
        row
        for row in selected
        if (model_id, str(row["instance_id"])) in failure_keys
        and (model_id, str(row["instance_id"])) not in completed
    ]
    logger.info("Completed=%s pending retries=%s", len(completed), len(pending))
    if not pending:
        _write_retry_ledger(output_failure_path, retry_ledger)
        return

    lock = _acquire_output_lock(output_path)
    adapter = None
    failures = 0
    original_limit = None
    started = time.monotonic()
    try:
        matcher = SemanticMatcher(
            model_name_or_path=semantic["embedding_model"],
            revision=semantic["embedding_revision"],
            device=args.device,
        )
        if not matcher.is_available:
            raise RuntimeError(f"Pinned semantic matcher failed: {matcher.load_error}")
        adapter = _load_adapter(model_config, args.device)
        adapter.load_model()
        if adapter.get_model_revision() != revision:
            raise RuntimeError("Adapter revision drift")
        original_limit = adapter.generation_config.get("max_new_tokens")
        adapter.generation_config["max_new_tokens"] = args.max_new_tokens

        for index, manifest_row in enumerate(pending, start=1):
            key = (model_id, str(manifest_row["instance_id"]))
            base = failure_bases[key]
            source_failure = source_failures[key]
            observation = None
            logger.info("[%s/%s] %s", index, len(pending), key[1])
            try:
                clean = base["clean"]
                image = _load_image_safely(base["image_path"], base["dataset"])
                retry_prompt = make_concise_grounding_retry_prompt(
                    base["question"], base["dataset"], answer_type=base.get("answer_type")
                )
                observation = _run_grounding_probe(
                    adapter=adapter,
                    image=image,
                    question=base["question"],
                    dataset=base["dataset"],
                    clean_norm_answer=clean["norm_answer"],
                    clean_prob=float(clean["answer_prob"]),
                    clean_entropy=float(clean["token_entropy_mean"]),
                    clean_margin=float(clean["token_margin_mean"]),
                    score_method=base["score_method"],
                    semantic_threshold=float(semantic["threshold"]),
                    embedding_fn=matcher.similarity,
                    answer_type=base.get("answer_type"),
                    normalizer_type=base.get("normalizer_type"),
                    prompt_text_override=retry_prompt,
                )
                if not observation.valid:
                    raise ValueError(
                        observation.invalid_reason or "invalid grounding recovery"
                    )
                base_for_recovery = copy.deepcopy(base)
                if not base_for_recovery.get("benchmark_family"):
                    base_for_recovery["benchmark_family"] = (
                        manifest_row.get("category")
                        or manifest_row.get("pope_split")
                        or ""
                    )
                rebuild_provenance = (
                    {
                        field: source_failure[field]
                        for field in (
                            "source_kind",
                            "source_path",
                            "source_file_sha256",
                            "source_record_sha256",
                        )
                    }
                    if source_failure.get("record_type") == "grounding_refresh_failure"
                    else provenance[key]
                )
                rebuilt = _rebuild_teacher_row(
                    base_for_recovery,
                    observation.to_dict(),
                    rebuild_provenance,
                    thresholds,
                    args.max_new_tokens,
                )
                recovered = _annotate_recovery(
                    rebuilt,
                    provenance=provenance[key],
                    source_failure=source_failure,
                    max_new_tokens=args.max_new_tokens,
                )
                append_jsonl(recovered, output_path)
                retry_ledger.pop(key, None)
                _write_retry_ledger(output_failure_path, retry_ledger)
            except Exception as exc:
                failures += 1
                previous = retry_ledger.get(key, {})
                retry_ledger[key] = {
                    "record_type": "grounding_recovery_failure",
                    "schema_version": RECOVERY_SCHEMA_VERSION,
                    "policy_id": RECOVERY_POLICY_ID,
                    "instance_id": key[1],
                    "dataset": base.get("dataset"),
                    "model_id": model_id,
                    "model_revision": revision,
                    "retry_max_new_tokens": args.max_new_tokens,
                    "attempt_count": int(previous.get("attempt_count", 0)) + 1,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "grounding_observation": (
                        observation.to_dict() if observation is not None else None
                    ),
                    **provenance[key],
                }
                _write_retry_ledger(output_failure_path, retry_ledger)
                logger.exception("Grounding recovery failed for %s: %s", key[1], exc)
    finally:
        if adapter is not None:
            if original_limit is not None:
                adapter.generation_config["max_new_tokens"] = original_limit
            adapter.unload_model()
        _release_output_lock(lock)

    completed = _validate_output_rows(
        output_path,
        model_id=model_id,
        selected_ids=selected_ids,
        manifest_sha=manifest_sha,
        frozen_sha=frozen_sha,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        valid_source_keys=valid_keys,
        failure_source_keys=failure_keys,
        provenance=provenance,
    )
    unresolved = failure_keys - completed
    logger.info(
        "Recovery finished in %.1fs; new failures=%s unresolved=%s completed=%s/%s",
        time.monotonic() - started,
        failures,
        len(unresolved),
        len(completed),
        len(expected_keys),
    )
    if failures or unresolved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
