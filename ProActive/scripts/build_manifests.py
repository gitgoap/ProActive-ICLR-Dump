#!/usr/bin/env python3
"""
Build grouped split manifests for all configured datasets.

Usage:
    python scripts/build_manifests.py --config_dir configs/data --output_dir outputs/manifests --seed 42
    python scripts/build_manifests.py --config_dir configs/data --output_dir outputs/manifests --dry_run

Supports: --config_dir, --output_dir, --seed, --limit, --dry_run, --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.data.loaders import load_dataset_config, get_loader
from proactive.data.splits import build_grouped_splits, validate_no_group_overlap, get_split_stats
from proactive.data.manifests import validate_manifest, save_manifest
from proactive.utils.io import ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build grouped split manifests for ProActive datasets."
    )
    parser.add_argument(
        "--config_dir", type=str, default="configs/data",
        help="Directory containing dataset YAML configs."
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Root directory for datasets (e.g. data or /home/aman/ProActive/data)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/manifests",
        help="Output directory for manifest JSONL files."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split assignment."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit records per dataset (for testing)."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load and validate but don't write files."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing manifest files."
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Specific datasets to build (default: all in config_dir)."
    )
    args = parser.parse_args()

    import os
    if args.data_root:
        os.environ["PROACTIVE_DATA_ROOT"] = str(Path(args.data_root).resolve())

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)

    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        sys.exit(1)

    # Find all dataset configs
    config_files = sorted(config_dir.glob("*.yaml"))
    if args.datasets:
        config_files = [
            f for f in config_files
            if f.stem in args.datasets
        ]

    if not config_files:
        logger.error(f"No config files found in {config_dir}")
        sys.exit(1)

    logger.info(f"Found {len(config_files)} dataset configs")

    all_records = []
    dataset_stats = {}

    for config_path in config_files:
        dataset_name = config_path.stem
        logger.info(f"--- Loading: {dataset_name} ---")

        try:
            config = load_dataset_config(config_path)
        except Exception as e:
            logger.error(f"Failed to parse config {config_path}: {e}")
            continue

        # Skip held-out datasets from main manifest building
        role = config.get("role", "core")
        if role == "held_out":
            logger.info(f"Skipping held-out dataset: {dataset_name}")
            continue

        # Skip datasets with pending construction
        if config.get("construction_status") == "pending":
            logger.warning(
                f"Skipping {dataset_name}: construction_status=pending"
            )
            continue

        loader_name = config.get("loader", dataset_name)
        try:
            loader_fn = get_loader(loader_name)
        except ValueError as e:
            logger.error(str(e))
            continue

        try:
            records = loader_fn(config, limit=args.limit)
        except FileNotFoundError as e:
            logger.warning(f"Data not found for {dataset_name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            continue

        logger.info(f"  Loaded {len(records)} records")

        # Validate manifest schema
        errors = validate_manifest(records)
        if errors:
            for err in errors[:10]:
                logger.error(f"  Validation: {err}")
            logger.error(f"  ({len(errors)} total errors)")
            continue

        # Assign grouped splits
        records = build_grouped_splits(
            records, seed=args.seed
        )

        # Validate no group overlap
        valid, violations = validate_no_group_overlap(records)
        if not valid:
            for v in violations[:5]:
                logger.error(f"  Split violation: {v}")
            logger.error("FATAL: Group overlap detected!")
            sys.exit(1)

        # Stats
        stats = get_split_stats(records)
        dataset_stats[dataset_name] = stats
        logger.info(f"  Split stats: {json.dumps(stats)}")

        all_records.extend(records)

    logger.info(f"\n=== Total: {len(all_records)} records across "
                f"{len(dataset_stats)} datasets ===")

    if args.dry_run:
        logger.info("Dry run complete. No files written.")
        # Print summary
        for ds, stats in dataset_stats.items():
            logger.info(f"  {ds}: {stats}")
        return

    # Write manifests
    ensure_dir(output_dir)

    # Per-dataset manifests
    for dataset_name in dataset_stats:
        ds_records = [r for r in all_records if r["dataset"] == dataset_name]
        out_path = output_dir / f"manifest_{dataset_name}.jsonl"
        manifest_hash = save_manifest(
            ds_records, out_path, overwrite=args.overwrite
        )
        logger.info(
            f"  Wrote {out_path} ({len(ds_records)} records, "
            f"hash={manifest_hash[:12]})"
        )

    # Combined manifest
    combined_path = output_dir / "manifest_combined.jsonl"
    combined_hash = save_manifest(
        all_records, combined_path, overwrite=args.overwrite
    )
    logger.info(
        f"  Wrote {combined_path} ({len(all_records)} records, "
        f"hash={combined_hash[:12]})"
    )

    # Write stats summary
    stats_path = output_dir / "manifest_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": args.seed, "datasets": dataset_stats},
            f, indent=2,
        )
    logger.info(f"  Wrote {stats_path}")

    logger.info("Manifest building complete.")


if __name__ == "__main__":
    main()
