#!/usr/bin/env python3
"""
Pilot teacher-cache generation script.

Runs clean inference + all probes for a deterministic train/val pilot subset,
producing a teacher-cache JSONL file.

Safety & Contract Enforcement:
1. --manifest_path is MANDATORY.
2. Refuses to run if any 'cal' or 'test' records are passed.
3. Uses deterministic stratified sampling from train and val splits.
4. Uses uniform scoring method (generation_logits or teacher_forced) per instance.
5. Supports --pilot_mode canonical | severity_grid.
6. Enforces non-zero exit code on GPU failures.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.data.manifests import load_manifest
from proactive.features.semantic import SemanticMatcher
from proactive.models.base_adapter import MLLMAdapter
from proactive.probes.image_transforms import PILOT_SEVERITIES
from proactive.teacher.cache_builder import process_instance, process_severity_grid_instance
from proactive.utils.io import append_jsonl, ensure_dir, iter_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("pilot_cache")


def stratified_sample(
    records: List[Dict[str, Any]], limit: int, seed: int
) -> List[Dict[str, Any]]:
    """Return exactly ``min(limit, len(records))`` deterministic balanced rows."""
    if limit <= 0:
        raise ValueError("--limit must be a positive integer")

    instance_ids = [r.get("instance_id") for r in records]
    if any(not instance_id for instance_id in instance_ids):
        raise ValueError("Every manifest row must have a non-empty instance_id")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Manifest contains duplicate instance_id values")

    import random
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (
            record.get("split", "train"),
            record.get("category", record.get("pope_split", "default")),
        )
        groups.setdefault(key, []).append(record)

    ordered_groups = []
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        ordered_groups.append(group)

    target = min(limit, len(records))
    per_group = max(1, limit // len(ordered_groups))
    sampled: List[Dict[str, Any]] = []
    positions: List[int] = []
    for group in ordered_groups:
        taken = min(per_group, len(group))
        sampled.extend(group[:taken])
        positions.append(taken)

    # Preserve the legacy pilot subset whenever it already filled the target;
    # otherwise extend that exact subset deterministically.  This lets valid
    # 97/79-row caches resume to 100 rather than forcing expensive regeneration.
    if len(sampled) >= target:
        rng.shuffle(sampled)
        return sampled[:target]

    while len(sampled) < target:
        made_progress = False
        for group_index, group in enumerate(ordered_groups):
            cursor = positions[group_index]
            if cursor < len(group):
                sampled.append(group[cursor])
                positions[group_index] += 1
                made_progress = True
                if len(sampled) == target:
                    break
        if not made_progress:
            break

    rng.shuffle(sampled)
    if len(sampled) != target:
        raise RuntimeError(f"Sampling under-filled: selected {len(sampled)} of {target}")
    return sampled


def pilot_record_key(record: Dict[str, Any], mode: str) -> Tuple[Any, ...]:
    """Build the unique resume key for a canonical or severity-pilot row."""
    instance_id = record.get("instance_id")
    if not instance_id:
        raise ValueError("Pilot record is missing instance_id")
    if mode == "canonical":
        return (instance_id,)
    probe = record.get("pilot_severity_probe")
    severity = record.get("pilot_severity_value")
    if not probe or not isinstance(severity, (int, float)) or not math.isfinite(severity):
        raise ValueError(
            f"Severity record {instance_id} is missing a finite probe/severity key"
        )
    return (instance_id, probe, float(severity))


def read_existing_pilot_keys(
    output_file: Path,
    mode: str,
    model_id: str,
    dataset_name: str,
) -> Set[Tuple[Any, ...]]:
    """Validate an existing cache and return unique keys; duplicates fail closed."""
    keys: Set[Tuple[Any, ...]] = set()
    for row_number, record in enumerate(iter_jsonl(output_file), start=1):
        if record.get("model_id") != model_id or record.get("dataset") != dataset_name:
            raise ValueError(
                f"Existing row {row_number} belongs to model={record.get('model_id')} "
                f"dataset={record.get('dataset')}, expected {model_id}/{dataset_name}"
            )
        key = pilot_record_key(record, mode)
        if key in keys:
            raise ValueError(
                f"Duplicate pilot key in {output_file} at row {row_number}: {key}. "
                "Regenerate with --overwrite."
            )
        keys.add(key)
    return keys


def load_adapter(model_config: dict, device: str) -> MLLMAdapter:
    """Dynamically load the correct adapter from config."""
    adapter_class_path = model_config["adapter_class"]
    module_path, class_name = adapter_class_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    AdapterClass = getattr(module, class_name)

    gen_config = model_config.get("generation_config", {})
    gen_config_clean = {
        k: v for k, v in gen_config.items()
        if k not in ("output_logits", "return_dict_in_generate")
    }

    adapter = AdapterClass(
        model_path=model_config["model_path"],
        model_revision=model_config.get("model_revision", "main"),
        generation_config=gen_config_clean,
        dtype=model_config.get("dtype", "auto"),
        device=device,
    )
    return adapter


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic pilot teacher cache generation (Contract 5)."
    )
    parser.add_argument(
        "--manifest_path", type=str, required=True,
        help="MANDATORY path to grouped dataset manifest JSONL file.",
    )
    parser.add_argument(
        "--model_config", type=str, required=True,
        help="Path to model YAML config.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/pilot_cache",
        help="Output directory for cache JSONL files.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to load model on.",
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max examples to process (stratified across train/val).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
        "--pilot_mode", choices=["canonical", "severity_grid"], default="canonical",
        help="Mode: canonical default or severity_grid for pilot inspection.",
    )
    parser.add_argument(
        "--shard_id", type=int, default=0,
        help="Shard index for parallel processing.",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of shards.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", action="store_true",
        help="Resume using canonical or composite severity keys; duplicates fail closed.",
    )
    resume_group.add_argument(
        "--overwrite", action="store_true",
        help="Atomically replace an existing output before generation.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print execution plan without loading model.",
    )
    args = parser.parse_args()

    # Load manifest and enforce strict split safety
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    raw_records = load_manifest(manifest_path)
    logger.info(f"Loaded {len(raw_records)} records from manifest {manifest_path}")

    # Check for calibration / test leakage
    forbidden = [r for r in raw_records if r.get("split") in ("cal", "test")]
    if forbidden:
        logger.warning(
            f"Manifest contains {len(forbidden)} cal/test records. Filtering strictly to train and val."
        )

    allowed_records = [
        r for r in raw_records if r.get("split") in ("train", "val")
    ]

    if not allowed_records:
        logger.error("FATAL: No train or val records found in manifest! Pilot cannot proceed.")
        sys.exit(1)

    try:
        records = stratified_sample(allowed_records, args.limit, args.seed)
    except (ValueError, RuntimeError) as exc:
        logger.error(f"FATAL: {exc}")
        sys.exit(1)
    logger.info(f"Selected {len(records)} train/val records for pilot (seed={args.seed})")

    if args.num_shards > 1:
        records = [
            r for i, r in enumerate(records) if i % args.num_shards == args.shard_id
        ]
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(records)} records")

    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    model_id = model_config["model_id"]
    model_name = model_config["model_name"]
    dataset_name = records[0]["dataset"] if records else "dataset"

    output_dir = ensure_dir(args.output_dir)
    mode_suffix = "_severity_pilot" if args.pilot_mode == "severity_grid" else ""
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    output_file = output_dir / f"{model_name}_{dataset_name}{mode_suffix}{shard_suffix}.jsonl"

    existing_keys: Set[Tuple[Any, ...]] = set()
    if output_file.exists() and not (args.resume or args.overwrite):
        logger.error(
            f"FATAL: Output already exists: {output_file}. "
            "Use --resume after validation or --overwrite to regenerate it."
        )
        sys.exit(1)
    if args.resume and output_file.exists():
        try:
            existing_keys = read_existing_pilot_keys(
                output_file, args.pilot_mode, model_id, dataset_name
            )
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error(f"FATAL: Cannot safely resume: {exc}")
            sys.exit(1)

        selected_ids = {r["instance_id"] for r in records}
        unexpected_ids = {key[0] for key in existing_keys} - selected_ids
        if unexpected_ids:
            logger.error(
                "FATAL: Existing cache contains IDs outside the current seed/limit/shard "
                f"selection ({len(unexpected_ids)} IDs). Use matching arguments or --overwrite."
            )
            sys.exit(1)
        logger.info(f"Resuming: {len(existing_keys)} unique output records already completed")

        if args.pilot_mode == "canonical":
            records = [r for r in records if (r["instance_id"],) not in existing_keys]
        else:
            def has_incomplete_grid(record: Dict[str, Any]) -> bool:
                return any(
                    (record["instance_id"], probe, float(severity)) not in existing_keys
                    for probe, values in PILOT_SEVERITIES.items()
                    for severity in values
                )

            records = [r for r in records if has_incomplete_grid(r)]

    if not records:
        logger.info("No records to process. Done.")
        return

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(f"Model: {model_id}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Examples: {len(records)}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Splits present: {set(r.get('split') for r in records)}")
        if args.pilot_mode == "severity_grid":
            remaining_rows = sum(
                (r["instance_id"], probe, float(severity)) not in existing_keys
                for r in records
                for probe, values in PILOT_SEVERITIES.items()
                for severity in values
            )
            logger.info(f"Remaining severity rows: {remaining_rows}")
        return

    # Initialize semantic matcher only after all CPU-only preflight checks pass.
    semantic_matcher = None
    if dataset_name.lower() in ("vizwiz", "gqa"):
        logger.info("Initializing pinned SemanticMatcher for freeform dataset...")
        semantic_matcher = SemanticMatcher(device=args.device if "cuda" in args.device else "cpu")
        if not semantic_matcher.is_available:
            logger.error(
                f"FATAL: SemanticMatcher failed to load for free-form dataset '{dataset_name}': {semantic_matcher.load_error}"
            )
            sys.exit(1)

    logger.info(f"Loading model '{model_id}' on {args.device}...")
    adapter = load_adapter(model_config, args.device)
    adapter.load_model()
    model_revision = adapter.get_model_revision()
    logger.info(f"Model loaded. Revision: {model_revision}")

    if args.overwrite:
        write_jsonl([], output_file, overwrite=output_file.exists())
        existing_keys.clear()

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for idx, record in enumerate(records):
        inst_id = record["instance_id"]
        logger.info(f"[{idx + 1}/{len(records)}] Processing {inst_id}...")

        try:
            if args.pilot_mode == "severity_grid":
                severity_rows = process_severity_grid_instance(
                    record=record,
                    adapter=adapter,
                    dataset_name=dataset_name,
                    model_id=model_id,
                    model_revision=model_revision,
                    global_seed=args.seed,
                    semantic_matcher=semantic_matcher,
                )
                for res in severity_rows:
                    key = pilot_record_key(res, args.pilot_mode)
                    if key in existing_keys:
                        continue
                    append_jsonl(res, output_file)
                    existing_keys.add(key)
            else:
                res = process_instance(
                    record=record,
                    adapter=adapter,
                    dataset_name=dataset_name,
                    model_id=model_id,
                    model_revision=model_revision,
                    global_seed=args.seed,
                    semantic_matcher=semantic_matcher,
                )
                key = pilot_record_key(res, args.pilot_mode)
                if key in existing_keys:
                    raise RuntimeError(f"Refusing to append duplicate key: {key}")
                append_jsonl(res, output_file)
                existing_keys.add(key)

            success_count += 1
        except Exception as e:
            logger.error(f"Failed on {inst_id}: {e}", exc_info=True)
            fail_count += 1

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"PILOT RUN COMPLETE: {success_count} succeeded, {fail_count} failed in {elapsed:.1f}s")
    logger.info(f"Output file: {output_file}")

    adapter.unload_model()

    if fail_count > 0:
        logger.error(f"FATAL: {fail_count} instances failed during pilot cache generation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
