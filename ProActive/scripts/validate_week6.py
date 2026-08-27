#!/usr/bin/env python3
"""Week 6 readiness/full gate for targets, policies, and baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from proactive.train.checkpoints import (
    CHECKPOINT_VERSION,
    POLICY_CHECKPOINT_VERSION,
    load_checkpoint,
    validate_freeze_manifest,
)
from proactive.train.policy import VOI_MANIFEST_VERSION
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("readiness", "full"), required=True)
    parser.add_argument("--config", default="configs/experiments/policy_train.yaml")
    parser.add_argument("--manifest_path", default="outputs/week6_voi/voi_manifest.json")
    parser.add_argument("--week5_freeze", default="outputs/week5_reports/week5_encoder_freeze.json")
    parser.add_argument("--selection_report", default="outputs/week6_reports/week6_policy_selection.json")
    parser.add_argument("--full_frontier_report", default=None)
    parser.add_argument("--architecture_frontiers", nargs="*", default=[])
    parser.add_argument("--uncertainty_checkpoint", default=None)
    parser.add_argument("--scalar_checkpoint", default=None)
    parser.add_argument("--distilled_checkpoint", default=None)
    parser.add_argument("--output_dir", default="outputs/week6_reports")
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
        raise SystemExit("Week 6 validation is complete and CPU-only")
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    errors = []
    if config.get("metadata", {}).get("approval_status") != "APPROVED":
        errors.append("Week 6 settings are not APPROVED")
    week5_path = Path(args.week5_freeze)
    try:
        validate_freeze_manifest(week5_path, require_policy=False)
    except (ValueError, FileNotFoundError) as exc:
        errors.append(f"Week 5 prerequisite: {exc}")
    voi_path = Path(args.manifest_path)
    voi = None
    if voi_path.exists():
        voi = _read(voi_path)
        unsigned = {key: value for key, value in voi.items() if key != "manifest_sha256"}
        if voi.get("format_version") != VOI_MANIFEST_VERSION or voi.get("manifest_sha256") != hash_dict(unsigned):
            errors.append("VOI manifest version/self-hash mismatch")
        if voi.get("calibration_used") is not False or voi.get("test_used") is not False:
            errors.append("VOI manifest violates split provenance")
    elif args.mode == "full":
        errors.append(f"Missing VOI manifest: {voi_path}")
    if args.mode == "full":
        if voi is not None and voi.get("status") != "COMPLETE":
            errors.append("Full Week 6 requires COMPLETE VOI targets")
        selection_path = Path(args.selection_report)
        if not selection_path.exists():
            errors.append(f"Missing policy selection: {selection_path}")
        else:
            selection = _read(selection_path)
            unsigned = {key: value for key, value in selection.items() if key != "report_sha256"}
            if selection.get("report_sha256") != hash_dict(unsigned) or selection.get("completion_gate_passed") is not True:
                errors.append("Week 6 policy selection/completion gate is invalid")
            expected_seeds = sorted(int(value) for value in config["seeds"])
            for candidate in selection.get("candidate_scores", []):
                observed = sorted(int(item["seed"]) for item in candidate.get("seed_scores", []))
                if observed != expected_seeds:
                    errors.append(
                        f"Policy selection seed coverage mismatch for lambda={candidate.get('multiplier')}"
                    )
        required_conditions = {
            "scalar_confidence",
            "clean_only_learned",
            "one_pass_distilled",
            "random",
            "blank_first",
            "visual_first",
            "grounding_first",
            "relation_first",
            "dataset_specific_fixed",
            "uncertainty_greedy",
            "full_teacher",
            "oracle_next",
            "oracle_best_subset",
        }
        if not args.full_frontier_report:
            errors.append("Missing --full_frontier_report with all mandatory controls")
        else:
            try:
                frontier = _read(Path(args.full_frontier_report))
                unsigned = {key: value for key, value in frontier.items() if key != "report_sha256"}
                if frontier.get("report_sha256") != hash_dict(unsigned):
                    errors.append("Full frontier self-hash mismatch")
                if frontier.get("phase") != "validation" or frontier.get("limit") is not None:
                    errors.append("Full frontier is not a complete validation run")
                if not frontier.get("oracle_next_included") or not frontier.get("oracle_subset_included"):
                    errors.append("Full frontier omitted a mandatory oracle")
                observed = {row["condition"] for row in frontier.get("frontier_rows", [])}
                missing_controls = sorted(required_conditions - observed)
                if missing_controls:
                    errors.append(f"Full frontier missing controls: {missing_controls}")
            except (FileNotFoundError, ValueError, KeyError) as exc:
                errors.append(f"Full frontier: {exc}")
        expected_encoders = {
            str(config["policy_architectures"]["main"]),
            *[str(value) for value in config["policy_architectures"]["comparisons"]],
        }
        observed_encoders = set()
        for raw_path in args.architecture_frontiers:
            try:
                report = _read(Path(raw_path))
                unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
                if report.get("report_sha256") != hash_dict(unsigned):
                    errors.append(f"Architecture frontier self-hash mismatch: {raw_path}")
                    continue
                if report.get("phase") != "validation" or report.get("limit") is not None:
                    errors.append(f"Architecture frontier is not full validation: {raw_path}")
                    continue
                observed_encoders.add(str(report.get("encoder_name")))
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"Architecture frontier: {exc}")
        missing_encoders = sorted(expected_encoders - observed_encoders)
        if missing_encoders:
            errors.append(f"Missing learned-policy architecture frontiers: {missing_encoders}")
        controls = {
            "uncertainty": (args.uncertainty_checkpoint, POLICY_CHECKPOINT_VERSION),
            "scalar": (args.scalar_checkpoint, CHECKPOINT_VERSION),
            "distilled": (args.distilled_checkpoint, CHECKPOINT_VERSION),
        }
        for name, (raw_path, version) in controls.items():
            if not raw_path:
                errors.append(f"Missing --{name}_checkpoint argument")
                continue
            try:
                checkpoint = load_checkpoint(raw_path, expected_version=version, map_location="cpu")
                if not checkpoint.get("scientifically_valid"):
                    errors.append(f"{name} checkpoint is not scientifically valid")
            except (ValueError, FileNotFoundError) as exc:
                errors.append(f"{name} checkpoint: {exc}")
    result: Dict[str, Any] = {
        "format_version": "week6_validation_v1",
        "mode": args.mode,
        "is_valid": not errors,
        "config_approval": config.get("metadata", {}).get("approval_status"),
        "voi_status": voi.get("status") if voi else "MISSING",
        "voi_rows": voi.get("row_count", 0) if voi else 0,
        "errors": errors,
    }
    result["report_sha256"] = hash_dict(result)
    output_path = Path(args.output_dir) / f"week6_{args.mode}_validation.json"
    if not args.dry_run:
        write_json(result, output_path, overwrite=(args.overwrite or args.resume) and output_path.exists())
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
