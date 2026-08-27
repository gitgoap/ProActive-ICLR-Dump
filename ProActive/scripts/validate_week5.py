#!/usr/bin/env python3
"""Week 5 readiness/full gate with artifact and split provenance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from proactive.train.checkpoints import validate_freeze_manifest
from proactive.train.vectorized import VECTOR_MANIFEST_VERSION
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("readiness", "full"), required=True)
    parser.add_argument("--config", default="configs/experiments/diag_bakeoff.yaml")
    parser.add_argument("--manifest_path", default="outputs/week5_data/vectorized_manifest.json")
    parser.add_argument("--week4_report", default="outputs/week4_reports/final/week4_full_report.json")
    parser.add_argument("--freeze_manifest", default="outputs/week5_reports/week5_encoder_freeze.json")
    parser.add_argument("--output_dir", default="outputs/week5_reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    if args.device != "cpu" or args.limit is not None:
        raise SystemExit("Week 5 validation is complete and CPU-only")
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    errors = []
    warnings = []
    if config.get("metadata", {}).get("approval_status") != "APPROVED":
        errors.append("Week 5 experiment settings are not APPROVED")
    week4_path = Path(args.week4_report)
    if not week4_path.exists():
        errors.append(f"Missing Week 4 full validation report: {week4_path}")
    else:
        week4 = _read(week4_path)
        if week4.get("is_valid") is not True:
            errors.append("Week 4 prerequisite report is not valid")
    source_manifest_path = Path(config["source_state_manifest"])
    if not source_manifest_path.exists():
        errors.append(f"Missing Week 4 state manifest: {source_manifest_path}")
    vector_path = Path(args.manifest_path)
    vector = None
    if vector_path.exists():
        vector = _read(vector_path)
        if vector.get("format_version") != VECTOR_MANIFEST_VERSION:
            errors.append("Vectorized manifest version mismatch")
        if vector.get("config_sha256") != file_sha256(config_path):
            errors.append("Vectorized manifest/config hash mismatch")
        if source_manifest_path.exists() and vector.get("source_state_manifest_sha256") != file_sha256(source_manifest_path):
            errors.append("Vectorized manifest/source-state hash mismatch")
        for entry in vector.get("files", []):
            artifact = vector_path.parent / entry["path"]
            if not artifact.exists() or file_sha256(artifact) != entry.get("sha256"):
                errors.append(f"Vectorized artifact drift: {artifact}")
    elif args.mode == "full":
        errors.append(f"Missing vectorized manifest: {vector_path}")
    if args.mode == "full":
        if vector is not None and vector.get("status") != "COMPLETE":
            errors.append("Full Week 5 requires COMPLETE vectorized states")
        freeze_path = Path(args.freeze_manifest)
        if not freeze_path.exists():
            errors.append(f"Missing Week 5 freeze manifest: {freeze_path}")
        else:
            try:
                validate_freeze_manifest(freeze_path, require_policy=False)
            except (ValueError, FileNotFoundError) as exc:
                errors.append(str(exc))
    result: Dict[str, Any] = {
        "format_version": "week5_validation_v1",
        "mode": args.mode,
        "is_valid": not errors,
        "config_approval": config.get("metadata", {}).get("approval_status"),
        "vector_status": vector.get("status") if vector else "MISSING",
        "split_counts": vector.get("split_counts") if vector else {},
        "errors": errors,
        "warnings": warnings,
    }
    result["report_sha256"] = hash_dict(result)
    output_path = Path(args.output_dir) / f"week5_{args.mode}_validation.json"
    if not args.dry_run:
        write_json(result, output_path, overwrite=(args.overwrite or args.resume) and output_path.exists())
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
