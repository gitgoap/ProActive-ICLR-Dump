#!/usr/bin/env python3
"""Build frozen PRE-HAL and IllusionBench Week 8 shift manifests."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.data.heldout import load_illusionbench_release, load_prehal_release
from proactive.data.loaders import load_dataset_config, resolve_data_path
from proactive.data.manifests import save_manifest, validate_manifest
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json, write_jsonl


LOGGER = logging.getLogger("build_heldout_manifests")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_dir", default="configs/data")
    parser.add_argument("--data_root")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("prehal", "illusionbench"),
        default=["prehal", "illusionbench"],
    )
    parser.add_argument("--output_dir", default="outputs/week8_data/manifests")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.data_root:
        import os

        os.environ["PROACTIVE_DATA_ROOT"] = str(Path(args.data_root).resolve())

    output_dir = Path(args.output_dir)
    combined = []
    dataset_reports: Dict[str, Any] = {}
    for dataset in args.datasets:
        config_path = Path(args.config_dir) / f"{dataset}.yaml"
        config = load_dataset_config(config_path)
        if config.get("role") != "held_out" or config.get("dataset_name") != dataset:
            raise SystemExit(f"Held-out config contract mismatch: {config_path}")
        root = resolve_data_path(config)
        result = (
            load_prehal_release(root, config, limit=args.limit)
            if dataset == "prehal"
            else load_illusionbench_release(root, config, limit=args.limit)
        )
        errors = validate_manifest(result.records)
        if errors:
            raise SystemExit(
                f"{dataset} manifest validation failed with {len(errors)} errors: {errors[:5]}"
            )
        if args.limit is None and len(result.records) != int(config["sample_cap"]):
            raise SystemExit(
                f"{dataset} held-out cap mismatch: expected {config['sample_cap']}, "
                f"found {len(result.records)}"
            )
        LOGGER.info(
            "%s: selected=%s exclusions=%s",
            dataset,
            len(result.records),
            len(result.exclusions),
        )
        combined.extend(result.records)
        dataset_reports[dataset] = {
            **result.audit,
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
        }
        if args.dry_run:
            continue
        manifest_path = output_dir / f"manifest_{dataset}_shift.jsonl"
        exclusion_path = output_dir / f"{dataset}_exclusions.jsonl"
        audit_path = output_dir / f"{dataset}_release_audit.json"
        manifest_hash = save_manifest(
            result.records,
            manifest_path,
            overwrite=args.overwrite and manifest_path.exists(),
        )
        write_jsonl(
            result.exclusions,
            exclusion_path,
            overwrite=args.overwrite and exclusion_path.exists(),
        )
        audit: Dict[str, Any] = {
            **dataset_reports[dataset],
            "manifest_path": str(manifest_path),
            "manifest_content_sha256": manifest_hash,
            "manifest_file_sha256": file_sha256(manifest_path),
            "exclusion_path": str(exclusion_path),
            "exclusion_file_sha256": file_sha256(exclusion_path),
        }
        audit["report_sha256"] = hash_dict(audit)
        write_json(audit, audit_path, overwrite=args.overwrite and audit_path.exists())

    identities = [row["instance_id"] for row in combined]
    if len(identities) != len(set(identities)):
        raise SystemExit("Duplicate instance_id across held-out datasets")
    summary: Dict[str, Any] = {
        "format_version": "week8_heldout_manifest_bundle_v1",
        "status": "PILOT" if args.limit is not None else "FROZEN",
        "selection_uses_model_outputs": False,
        "target_domain_calibration_used": False,
        "datasets": dataset_reports,
        "combined_rows": len(combined),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    combined_path = output_dir / "manifest_heldout_shift.jsonl"
    combined_hash = save_manifest(
        combined,
        combined_path,
        overwrite=args.overwrite and combined_path.exists(),
    )
    summary["combined_manifest_path"] = str(combined_path)
    summary["combined_manifest_content_sha256"] = combined_hash
    summary["combined_manifest_file_sha256"] = file_sha256(combined_path)
    summary["report_sha256"] = hash_dict(summary)
    summary_path = output_dir / "heldout_manifest_bundle.json"
    write_json(
        summary,
        summary_path,
        overwrite=args.overwrite and summary_path.exists(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

