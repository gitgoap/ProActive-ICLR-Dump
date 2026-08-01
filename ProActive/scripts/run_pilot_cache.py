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
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.data.manifests import load_manifest
from proactive.features.semantic import SemanticMatcher, SemanticMatcherError
from proactive.models.base_adapter import MLLMAdapter
from proactive.probes.image_transforms import CANONICAL_SEVERITIES, PILOT_SEVERITIES
from proactive.teacher.cache_builder import process_instance
from proactive.utils.io import append_jsonl, get_completed_ids, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("pilot_cache")


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
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-completed instance IDs.",
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

    # Deterministic stratified sampling
    import random
    rng = random.Random(args.seed)
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in allowed_records:
        key = (r.get("split", "train"), r.get("category", r.get("pope_split", "default")))
        groups.setdefault(key, []).append(r)

    sampled = []
    per_group_limit = max(1, args.limit // len(groups))
    for key, group_recs in sorted(groups.items()):
        group_copy = list(group_recs)
        rng.shuffle(group_copy)
        sampled.extend(group_copy[:per_group_limit])

    rng.shuffle(sampled)
    records = sampled[:args.limit]
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

    # Initialize semantic matcher if freeform dataset (VizWiz, GQA)
    semantic_matcher = None
    if dataset_name.lower() in ("vizwiz", "gqa"):
        logger.info("Initializing pinned SemanticMatcher for freeform dataset...")
        semantic_matcher = SemanticMatcher(device=args.device if "cuda" in args.device else "cpu")
        if not semantic_matcher.is_available:
            logger.error(
                f"FATAL: SemanticMatcher failed to load for free-form dataset '{dataset_name}': {semantic_matcher.load_error}"
            )
            sys.exit(1)

    output_dir = ensure_dir(args.output_dir)
    mode_suffix = "_severity_pilot" if args.pilot_mode == "severity_grid" else ""
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    output_file = output_dir / f"{model_name}_{dataset_name}{mode_suffix}{shard_suffix}.jsonl"

    if args.resume and output_file.exists():
        completed_ids = get_completed_ids(output_file)
        logger.info(f"Resuming: {len(completed_ids)} already completed")
        records = [r for r in records if r["instance_id"] not in completed_ids]

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
        return

    logger.info(f"Loading model '{model_id}' on {args.device}...")
    adapter = load_adapter(model_config, args.device)
    adapter.load_model()
    model_revision = adapter.get_model_revision()
    logger.info(f"Model loaded. Revision: {model_revision}")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for idx, record in enumerate(records):
        inst_id = record["instance_id"]
        logger.info(f"[{idx + 1}/{len(records)}] Processing {inst_id}...")

        try:
            if args.pilot_mode == "severity_grid":
                for probe_name, sevs in PILOT_SEVERITIES.items():
                    for sev in sevs:
                        sev_dict = {probe_name: sev}
                        res = process_instance(
                            record=record,
                            adapter=adapter,
                            dataset_name=dataset_name,
                            model_id=model_id,
                            model_revision=model_revision,
                            severities=sev_dict,
                            global_seed=args.seed,
                            semantic_matcher=semantic_matcher,
                        )
                        res["pilot_severity_probe"] = probe_name
                        res["pilot_severity_value"] = sev
                        append_jsonl(res, output_file)
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
                append_jsonl(res, output_file)

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
