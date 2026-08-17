#!/usr/bin/env python3
"""Build a complete teacher cache with one uniform longer grounding pass.

The original teacher cache is never modified.  Every model-instance row is
reconstructed from either its valid teacher row or its retained fail-closed
record, the grounding observation is regenerated with one uniform token cap,
and labels are recomputed before the row is appended to a separate output
directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from run_teacher import (  # noqa: E402
    IMMUTABLE_REVISION_RE,
    _acquire_output_lock,
    _enforce_compute_authorization,
    _load_adapter,
    _load_failure_ledger,
    _load_yaml,
    _release_output_lock,
    _resolve_model_config,
    _validate_existing_rows,
)
from proactive.data.manifests import load_manifest, validate_manifest  # noqa: E402
from proactive.features.semantic import SemanticMatcher  # noqa: E402
from proactive.probes.probe_runner import _run_grounding_probe  # noqa: E402
from proactive.teacher.cache_builder import _load_image_safely  # noqa: E402
from proactive.teacher.label_computation import compute_teacher_labels  # noqa: E402
from proactive.teacher.offline import (  # noqa: E402
    probe_observations_from_record,
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
logger = logging.getLogger("refresh_grounding_cache")
REFRESH_SCHEMA_VERSION = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_dir", default="outputs/teacher_core")
    parser.add_argument("--output_dir", default="outputs/teacher_core_grounding512")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _source_paths(input_dir: Path, model_name: str, shard_id: int, num_shards: int) -> Tuple[Path, Path]:
    name = f"teacher_{model_name}_all_all_shard{shard_id:02d}-of-{num_shards:02d}.jsonl"
    teacher_path = input_dir / name
    return teacher_path, teacher_path.with_suffix(".failures.jsonl")


def _load_base_rows(
    *,
    teacher_path: Path,
    failure_path: Path,
    model_id: str,
    selected_ids: set[str],
    manifest_sha256: str,
    frozen_sha256: str,
    shard_id: int,
    num_shards: int,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], Dict[str, str]]]:
    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing source teacher shard: {teacher_path}")
    _validate_existing_rows(
        teacher_path,
        model_id,
        selected_ids,
        manifest_sha256,
        frozen_sha256,
        shard_id,
        num_shards,
    )
    teacher_sha = file_sha256(teacher_path)
    bases: Dict[Tuple[str, str], Dict[str, Any]] = {}
    provenance: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in iter_jsonl(teacher_path):
        key = (model_id, str(row["instance_id"]))
        bases[key] = row
        provenance[key] = {
            "source_kind": "valid_teacher",
            "source_path": str(teacher_path),
            "source_file_sha256": teacher_sha,
            "source_record_sha256": hash_dict(row),
        }

    failures = _load_failure_ledger(
        failure_path,
        model_id=model_id,
        selected_ids=selected_ids,
        manifest_sha256=manifest_sha256,
        frozen_config_sha256=frozen_sha256,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    failure_sha = file_sha256(failure_path) if failure_path.exists() else ""
    for key, failure in failures.items():
        if key in bases:
            raise ValueError(f"Key occurs in teacher and failure inputs: {key}")
        base = failure.get("invalid_teacher_record")
        if not isinstance(base, dict):
            raise ValueError(f"Failure has no recoverable teacher record: {key}")
        clean = base.get("clean")
        probes = base.get("probes")
        if not isinstance(clean, Mapping) or clean.get("valid") is not True:
            raise ValueError(f"Failure has invalid clean baseline: {key}")
        if not isinstance(probes, Mapping) or "grounding" not in probes:
            raise ValueError(f"Failure has no grounding observation: {key}")
        bases[key] = base
        provenance[key] = {
            "source_kind": "failure_ledger",
            "source_path": str(failure_path),
            "source_file_sha256": failure_sha,
            "source_record_sha256": hash_dict(base),
        }

    expected = {(model_id, instance_id) for instance_id in selected_ids}
    missing = expected - set(bases)
    extra = set(bases) - expected
    if missing or extra:
        raise ValueError(
            f"Source coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    return bases, provenance


def _rebuild_teacher_row(
    base: Mapping[str, Any],
    grounding: Mapping[str, Any],
    source_provenance: Mapping[str, str],
    thresholds,
    max_new_tokens: int,
) -> Dict[str, Any]:
    row = copy.deepcopy(dict(base))
    row["probes"]["grounding"] = dict(grounding)
    row["valid"] = True
    row["invalid_reason"] = None
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
    row["grounding_refresh"] = {
        "schema_version": REFRESH_SCHEMA_VERSION,
        "uniform_max_new_tokens": max_new_tokens,
        **dict(source_provenance),
        "effective_generation_config_sha256": grounding.get("generation_config_hash"),
    }
    return row


def _validate_refresh_rows(
    path: Path,
    *,
    model_id: str,
    selected_ids: set[str],
    manifest_sha: str,
    frozen_sha: str,
    shard_id: int,
    num_shards: int,
    max_new_tokens: int,
    provenance: Mapping[Tuple[str, str], Mapping[str, str]],
) -> set[Tuple[str, str]]:
    rows = list(iter_jsonl(path))
    completed = validate_resume_teacher_records(
        rows, model_id, selected_ids, manifest_sha, frozen_sha, shard_id, num_shards
    )
    for row in rows:
        key = (model_id, str(row["instance_id"]))
        refresh = row.get("grounding_refresh")
        if not isinstance(refresh, Mapping):
            raise ValueError(f"Missing grounding_refresh provenance for {key}")
        if refresh.get("schema_version") != REFRESH_SCHEMA_VERSION:
            raise ValueError(f"Grounding refresh schema drift for {key}")
        if refresh.get("uniform_max_new_tokens") != max_new_tokens:
            raise ValueError(f"Grounding token-cap drift for {key}")
        expected = provenance[key]
        for field in ("source_kind", "source_path", "source_file_sha256", "source_record_sha256"):
            if refresh.get(field) != expected[field]:
                raise ValueError(f"Grounding source provenance drift for {key}: {field}")
    return completed


def _load_refresh_failures(
    path: Path,
    *,
    model_id: str,
    selected_ids: set[str],
    max_new_tokens: int,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    failures: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return failures
    for row in iter_jsonl(path):
        key = (str(row.get("model_id", "")), str(row.get("instance_id", "")))
        if row.get("record_type") != "grounding_refresh_failure":
            raise ValueError(f"Invalid refresh failure record: {key}")
        if key in failures or key[0] != model_id or key[1] not in selected_ids:
            raise ValueError(f"Duplicate or out-of-scope refresh failure: {key}")
        if row.get("uniform_max_new_tokens") != max_new_tokens:
            raise ValueError(f"Refresh failure token-cap drift: {key}")
        failures[key] = row
    return failures


def _write_refresh_failures(
    path: Path, failures: Mapping[Tuple[str, str], Mapping[str, Any]]
) -> None:
    write_jsonl([dict(failures[key]) for key in sorted(failures)], path, overwrite=True)


def main() -> None:
    args = _parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("--shard_id must satisfy 0 <= shard_id < num_shards")
    if args.max_new_tokens < 256 or args.max_new_tokens > 1024:
        raise SystemExit("--max_new_tokens must be between 256 and 1024")

    experiment = _load_yaml(Path(args.config))
    _enforce_compute_authorization(experiment, args)
    model_config_path = _resolve_model_config(experiment, args.model, None)
    model_config = _load_yaml(model_config_path)
    revision = str(model_config.get("model_revision", ""))
    if not IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise SystemExit("Grounding refresh requires an immutable model revision")
    configured_limit = int(model_config.get("generation_config", {}).get("max_new_tokens", 0))
    if args.max_new_tokens <= configured_limit:
        raise SystemExit(
            "Grounding refresh token cap must exceed the base model cap "
            f"({configured_limit})"
        )

    frozen_path = Path(str(experiment["frozen_probe_config"]))
    frozen = _load_yaml(frozen_path)
    thresholds = thresholds_from_mapping(frozen["label_thresholds"])
    semantic = dict(frozen["semantic_matching"])

    manifest_path = Path(args.manifest_path)
    records = load_manifest(manifest_path)
    errors = validate_manifest(records)
    if errors:
        raise SystemExit(f"Manifest validation failed: {errors[:5]}")
    model_id = str(model_config["model_id"])
    selected = [
        row for row in records
        if stable_shard_id(str(row["instance_id"]), model_id, args.num_shards)
        == args.shard_id
    ]
    selected_ids = {str(row["instance_id"]) for row in selected}
    manifest_sha = file_sha256(manifest_path)
    frozen_sha = file_sha256(frozen_path)
    model_name = str(model_config["model_name"])
    teacher_path, failure_path = _source_paths(
        Path(args.input_dir), model_name, args.shard_id, args.num_shards
    )
    bases, provenance = _load_base_rows(
        teacher_path=teacher_path,
        failure_path=failure_path,
        model_id=model_id,
        selected_ids=selected_ids,
        manifest_sha256=manifest_sha,
        frozen_sha256=frozen_sha,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    output_path = Path(args.output_dir) / teacher_path.name
    refresh_failure_path = output_path.with_suffix(".failures.jsonl")
    completed: set[Tuple[str, str]] = set()
    if output_path.exists():
        if args.overwrite:
            write_jsonl([], output_path, overwrite=True)
            write_jsonl([], refresh_failure_path, overwrite=True)
        elif args.resume:
            completed = _validate_refresh_rows(
                output_path,
                model_id=model_id,
                selected_ids=selected_ids,
                manifest_sha=manifest_sha,
                frozen_sha=frozen_sha,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                max_new_tokens=args.max_new_tokens,
                provenance=provenance,
            )
        else:
            raise SystemExit(f"Output exists: {output_path}; use --resume or --overwrite")
    refresh_failures = _load_refresh_failures(
        refresh_failure_path,
        model_id=model_id,
        selected_ids=selected_ids,
        max_new_tokens=args.max_new_tokens,
    )
    stale = set(refresh_failures).intersection(completed)
    if stale and not args.dry_run:
        for key in stale:
            del refresh_failures[key]
        _write_refresh_failures(refresh_failure_path, refresh_failures)
    pending = [row for row in selected if (model_id, str(row["instance_id"])) not in completed]
    logger.info("Model=%s shard=%s/%s completed=%s pending=%s", model_name, args.shard_id, args.num_shards, len(completed), len(pending))
    logger.info("One grounding pass per pending row; max_new_tokens=%s", args.max_new_tokens)
    logger.info("Output=%s", output_path)
    if args.dry_run or not pending:
        return

    lock = _acquire_output_lock(output_path)
    adapter = None
    failures = 0
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
            base = bases[key]
            observation = None
            logger.info("[%s/%s] %s", index, len(pending), key[1])
            try:
                clean = base["clean"]
                image = _load_image_safely(base["image_path"], base["dataset"])
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
                )
                if not observation.valid:
                    raise ValueError(observation.invalid_reason or "invalid grounding refresh")
                base_for_refresh = copy.deepcopy(base)
                if not base_for_refresh.get("benchmark_family"):
                    base_for_refresh["benchmark_family"] = (
                        manifest_row.get("category") or manifest_row.get("pope_split") or ""
                    )
                rebuilt = _rebuild_teacher_row(
                    base_for_refresh, observation.to_dict(), provenance[key], thresholds,
                    args.max_new_tokens
                )
                append_jsonl(rebuilt, output_path)
                if key in refresh_failures:
                    del refresh_failures[key]
                    _write_refresh_failures(refresh_failure_path, refresh_failures)
            except Exception as exc:
                failures += 1
                previous = refresh_failures.get(key, {})
                refresh_failures[key] = {
                    "record_type": "grounding_refresh_failure",
                    "schema_version": REFRESH_SCHEMA_VERSION,
                    "instance_id": key[1],
                    "dataset": base.get("dataset"),
                    "model_id": model_id,
                    "model_revision": revision,
                    "uniform_max_new_tokens": args.max_new_tokens,
                    "attempt_count": int(previous.get("attempt_count", 0)) + 1,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "grounding_observation": (
                        observation.to_dict() if observation is not None else None
                    ),
                    **provenance[key],
                }
                _write_refresh_failures(refresh_failure_path, refresh_failures)
                logger.exception("Grounding refresh failed for %s: %s", key[1], exc)
        adapter.generation_config["max_new_tokens"] = original_limit
    finally:
        if adapter is not None:
            adapter.unload_model()
        _release_output_lock(lock)
    logger.info("Refresh finished in %.1fs with %s failures", time.monotonic() - started, failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
