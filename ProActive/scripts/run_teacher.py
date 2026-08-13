#!/usr/bin/env python3
"""Generate deterministic, resumable Week 4 teacher-cache shards.

The script runs every legal canonical probe independently from the original
input.  It refuses unpinned model revisions for full runs and validates every
existing row before resuming, preventing the append-duplication failure seen in
the Week 3 pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.data.manifests import load_manifest, validate_manifest
from proactive.features.semantic import SemanticMatcher
from proactive.models.base_adapter import MLLMAdapter
from proactive.teacher.cache_builder import process_instance
from proactive.teacher.offline import (
    legal_probe_names,
    probe_observations_from_record,
    thresholds_from_mapping,
    stable_shard_id,
    validate_resume_teacher_records,
)
from proactive.utils.io import append_jsonl, file_sha256, iter_jsonl, write_jsonl


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_teacher")

IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
VALID_SPLITS = {"train", "val", "cal", "test"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _resolve_model_config(
    experiment: Mapping[str, Any], model: str | None, model_config: str | None
) -> Path:
    if model_config:
        return Path(model_config)
    if not model:
        raise ValueError("Provide --model or --model_config")
    model_map = experiment.get("models")
    if not isinstance(model_map, Mapping) or model not in model_map:
        raise ValueError(
            f"Unknown --model '{model}'. Configured models: "
            f"{sorted(model_map) if isinstance(model_map, Mapping) else []}"
        )
    return Path(str(model_map[model]))


def _load_adapter(model_config: Mapping[str, Any], device: str) -> MLLMAdapter:
    adapter_path = str(model_config["adapter_class"])
    module_path, class_name = adapter_path.rsplit(".", 1)
    adapter_class = getattr(importlib.import_module(module_path), class_name)
    generation_config = {
        key: value
        for key, value in dict(model_config.get("generation_config", {})).items()
        if key not in {"output_logits", "return_dict_in_generate"}
    }
    return adapter_class(
        model_path=model_config["model_path"],
        model_revision=model_config["model_revision"],
        generation_config=generation_config,
        dtype=model_config.get("dtype", "auto"),
        device=device,
    )


def _acquire_output_lock(output_path: Path):
    """Hold a process lock so two jobs cannot append to the same shard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    handle = open(lock_path, "a+b")
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:  # Local Windows safeguard; production GPU execution is Linux.
            import msvcrt

            if lock_path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except (BlockingIOError, OSError) as exc:
        handle.close()
        raise RuntimeError(f"Another process is writing shard {output_path}") from exc
    return handle


def _release_output_lock(handle) -> None:
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        else:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def deterministic_limit(
    records: List[Dict[str, Any]], limit: int | None, seed: int
) -> List[Dict[str, Any]]:
    if limit is None:
        return records
    if limit <= 0:
        raise ValueError("--limit must be positive")
    return sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{seed}|{row['instance_id']}".encode("utf-8")
        ).hexdigest(),
    )[: min(limit, len(records))]


def _validate_existing_rows(
    path: Path,
    model_id: str,
    selected_ids: Set[str],
    manifest_sha256: str,
    frozen_config_sha256: str,
    shard_id: int,
    num_shards: int,
) -> Set[Tuple[str, str]]:
    return validate_resume_teacher_records(
        iter_jsonl(path),
        model_id=model_id,
        selected_ids=selected_ids,
        manifest_sha256=manifest_sha256,
        frozen_config_sha256=frozen_config_sha256,
        shard_id=shard_id,
        num_shards=num_shards,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Week 4 experiment YAML")
    parser.add_argument("--manifest_path", required=True, help="Grouped manifest JSONL")
    parser.add_argument("--model", help="Model key from the experiment config")
    parser.add_argument("--model_config", help="Explicit model YAML path")
    parser.add_argument("--dataset", default="all", help="Dataset filter or 'all'")
    parser.add_argument(
        "--split", choices=["all", "train", "val", "cal", "test"], default="all"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output_dir", default="outputs/teacher_core")
    parser.add_argument("--out", help="Explicit output JSONL path")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--allow_unapproved_smoke",
        action="store_true",
        help="Allow draft config/unpinned revision only for dry-run or <=10 rows",
    )
    return parser.parse_args()


def _enforce_compute_authorization(
    experiment: Dict[str, Any], args: argparse.Namespace
) -> None:
    """Fail closed when a requested GPU scope exceeds owner authorization."""

    if args.dry_run:
        return
    authorization = experiment.get("compute_authorization")
    if not isinstance(authorization, dict):
        raise SystemExit("Missing compute_authorization in Week 4 experiment config")
    if authorization.get("full_core_approved") is True:
        return
    if authorization.get("staged_checks_approved") is not True:
        raise SystemExit("Week 4 staged GPU checks are not owner-approved")

    max_examples = authorization.get("staged_max_examples")
    if (
        isinstance(max_examples, int)
        and not isinstance(max_examples, bool)
        and max_examples > 0
        and args.limit is not None
        and args.limit <= max_examples
    ):
        return

    allowlist = authorization.get("staged_full_dataset_allowlist", [])
    if (
        isinstance(allowlist, list)
        and args.limit is None
        and args.dataset in {str(name) for name in allowlist}
    ):
        return

    raise SystemExit(
        "Full Week 4 core generation is not owner-approved. Allowed staged "
        f"scope: --limit <= {max_examples} or one complete dataset from "
        f"{allowlist}."
    )


def main() -> None:
    args = _parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("--shard_id must satisfy 0 <= shard_id < num_shards")
    smoke_exception = args.dry_run or (args.limit is not None and args.limit <= 10)
    if args.allow_unapproved_smoke and not smoke_exception:
        raise SystemExit("--allow_unapproved_smoke is restricted to dry-run or --limit <= 10")

    config_path = Path(args.config)
    experiment = _load_yaml(config_path)
    approval_status = experiment.get("metadata", {}).get("approval_status")
    if approval_status != "APPROVED" and not args.allow_unapproved_smoke:
        raise SystemExit(
            "Week 4 config is not APPROVED. Only a dry-run/<=10-row smoke may use "
            "--allow_unapproved_smoke."
        )
    if approval_status == "APPROVED":
        _enforce_compute_authorization(experiment, args)

    model_config_path = _resolve_model_config(experiment, args.model, args.model_config)
    model_config = _load_yaml(model_config_path)
    revision = str(model_config.get("model_revision", ""))
    if not IMMUTABLE_REVISION_RE.fullmatch(revision) and not args.allow_unapproved_smoke:
        raise SystemExit(
            f"Model revision must be an immutable 40-hex commit, got {revision!r}"
        )

    frozen_path = Path(str(experiment["frozen_probe_config"]))
    frozen = _load_yaml(frozen_path)
    if frozen.get("metadata", {}).get("status") != "FROZEN":
        raise SystemExit("Frozen probe config metadata.status must be FROZEN")
    thresholds = thresholds_from_mapping(frozen["label_thresholds"])
    severities = dict(frozen["probe_severities"])
    semantic = dict(frozen["semantic_matching"])

    manifest_path = Path(args.manifest_path)
    records = load_manifest(manifest_path)
    manifest_errors = validate_manifest(records)
    if manifest_errors:
        raise SystemExit(f"Manifest validation failed: {manifest_errors[:5]}")
    if any(row.get("split") not in VALID_SPLITS for row in records):
        raise SystemExit("Every manifest row must have a train/val/cal/test split")
    if args.dataset != "all":
        records = [row for row in records if row.get("dataset") == args.dataset]
    if args.split != "all":
        records = [row for row in records if row.get("split") == args.split]
    if not records:
        raise SystemExit("No manifest records remain after dataset/split filtering")

    all_ids = [str(row["instance_id"]) for row in records]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("Filtered manifest contains duplicate instance_id values")
    records = deterministic_limit(records, args.limit, args.seed)

    model_id = str(model_config["model_id"])
    records = [
        row
        for row in records
        if stable_shard_id(row["instance_id"], model_id, args.num_shards)
        == args.shard_id
    ]
    selected_ids = {str(row["instance_id"]) for row in records}

    model_name = str(model_config["model_name"])
    scope = f"{args.dataset}_{args.split}"
    default_name = (
        f"teacher_{model_name}_{scope}_shard{args.shard_id:02d}-of-"
        f"{args.num_shards:02d}.jsonl"
    )
    output_path = Path(args.out) if args.out else Path(args.output_dir) / default_name
    manifest_sha256 = file_sha256(manifest_path)
    frozen_sha256 = file_sha256(frozen_path)
    config_sha256 = file_sha256(config_path)
    model_config_sha256 = file_sha256(model_config_path)

    completed: Set[Tuple[str, str]] = set()
    if output_path.exists():
        if args.overwrite:
            write_jsonl([], output_path, overwrite=True)
        elif args.resume:
            try:
                completed = _validate_existing_rows(
                    output_path,
                    model_id,
                    selected_ids,
                    manifest_sha256,
                    frozen_sha256,
                    args.shard_id,
                    args.num_shards,
                )
            except (ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Unsafe resume refused: {exc}") from exc
        else:
            raise SystemExit(
                f"Output exists: {output_path}. Use --resume or explicit --overwrite."
            )

    pending = [row for row in records if (model_id, row["instance_id"]) not in completed]
    expected_probe_passes = sum(len(legal_probe_names(row)) + 1 for row in pending)
    logger.info("Model: %s (%s)", model_name, model_id)
    logger.info("Shard: %s/%s", args.shard_id, args.num_shards)
    logger.info("Selected rows: %s; completed: %s; pending: %s", len(records), len(completed), len(pending))
    logger.info("Expected remaining forward passes (clean + legal probes): %s", expected_probe_passes)
    logger.info("Output: %s", output_path)
    if args.dry_run or not pending:
        return

    lock_handle = _acquire_output_lock(output_path)
    adapter = None
    started = time.monotonic()
    failures = 0
    try:
        needs_semantic = any(
            str(row.get("dataset", "")).lower() in {"vizwiz", "gqa"}
            for row in pending
        )
        matcher = None
        if needs_semantic:
            matcher = SemanticMatcher(
                model_name_or_path=semantic["embedding_model"],
                revision=semantic["embedding_revision"],
                device=args.device if args.device.startswith("cuda") else "cpu",
            )
            if not matcher.is_available:
                raise RuntimeError(
                    f"Pinned semantic matcher failed to load: {matcher.load_error}"
                )

        adapter = _load_adapter(model_config, args.device)
        logger.info("Loading model on %s", args.device)
        adapter.load_model()
        actual_revision = adapter.get_model_revision()
        if actual_revision != revision:
            raise RuntimeError(
                f"Adapter revision drift: configured {revision}, loaded {actual_revision}"
            )

        for index, row in enumerate(pending, start=1):
            logger.info("[%s/%s] %s", index, len(pending), row["instance_id"])
            try:
                result = process_instance(
                    record=row,
                    adapter=adapter,
                    dataset_name=row["dataset"],
                    model_id=model_id,
                    model_revision=actual_revision,
                    severities=severities,
                    global_seed=args.seed,
                    semantic_matcher=matcher,
                    semantic_threshold=float(semantic["threshold"]),
                    label_thresholds=thresholds,
                )
                if result.get("valid") is not True:
                    raise ValueError(result.get("invalid_reason") or "invalid teacher row")
                probe_observations_from_record(result)
                result.update(
                    {
                        "record_type": "teacher_cache",
                        "source_manifest_sha256": manifest_sha256,
                        "frozen_probe_config_sha256": frozen_sha256,
                        "experiment_config_sha256": config_sha256,
                        "model_config_sha256": model_config_sha256,
                        "seed": args.seed,
                        "shard_id": args.shard_id,
                        "num_shards": args.num_shards,
                    }
                )
                append_jsonl(result, output_path)
            except Exception as exc:  # preserve shard and continue collecting failures
                failures += 1
                logger.exception("Teacher generation failed for %s: %s", row["instance_id"], exc)
    finally:
        if adapter is not None:
            adapter.unload_model()
        _release_output_lock(lock_handle)

    elapsed = time.monotonic() - started
    logger.info("Shard finished in %.1fs with %s failed rows", elapsed, failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
