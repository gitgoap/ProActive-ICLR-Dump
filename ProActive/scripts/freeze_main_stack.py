#!/usr/bin/env python3
"""Freeze encoder, heads, policy, configs, and controls before calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from proactive.train.checkpoints import (
    POLICY_CHECKPOINT_VERSION,
    load_checkpoint,
    validate_freeze_manifest,
    write_freeze_manifest,
)
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/calibrate_aps.yaml")
    parser.add_argument("--policy_config", default="configs/experiments/policy_train.yaml")
    parser.add_argument("--week5_freeze", default="outputs/week5_reports/week5_encoder_freeze.json")
    parser.add_argument("--selection_report", default="outputs/week6_reports/week6_policy_selection.json")
    parser.add_argument("--uncertainty_checkpoint", required=True)
    parser.add_argument("--scalar_checkpoint", required=True)
    parser.add_argument("--distilled_checkpoint", required=True)
    parser.add_argument("--dataset_schedule", required=True)
    parser.add_argument("--output_dir", default="outputs/week7_frozen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--approve_freeze", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    if args.device != "cpu" or args.limit is not None:
        raise SystemExit("Main-stack freeze is CPU-only and refuses --limit")
    config_path = Path(args.config)
    policy_config_path = Path(args.policy_config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("metadata", {}).get("approval_status") != "APPROVED" and not args.dry_run:
        raise SystemExit("Week 7 settings are not APPROVED")
    week5_path = Path(args.week5_freeze)
    week5 = validate_freeze_manifest(week5_path, require_policy=False)
    selection_path = Path(args.selection_report)
    selection = _read(selection_path)
    unsigned = {key: value for key, value in selection.items() if key != "report_sha256"}
    if selection.get("report_sha256") != hash_dict(unsigned) or selection.get("status") != "SELECTED":
        raise SystemExit("Week 6 selection report is not a valid selected artifact")
    if selection.get("calibration_used") is not False or selection.get("test_used") is not False:
        raise SystemExit("Week 6 selection provenance violated the split firewall")
    diagnostic_path = Path(week5["artifacts"]["diagnostic_checkpoint"]["path"])
    policy_path = Path(selection["selected_policy_path"])
    if file_sha256(policy_path) != selection.get("selected_policy_sha256"):
        raise SystemExit("Selected policy artifact drift")
    policy = load_checkpoint(policy_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu")
    if policy.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
        raise SystemExit("Selected policy was not trained with the frozen primary diagnostic")
    controls = {
        "uncertainty_baseline": Path(args.uncertainty_checkpoint),
        "scalar_baseline": Path(args.scalar_checkpoint),
        "distilled_baseline": Path(args.distilled_checkpoint),
    }
    for name, path in controls.items():
        checkpoint = load_checkpoint(
            path,
            expected_version=(POLICY_CHECKPOINT_VERSION if name == "uncertainty_baseline" else "proactive_diagnostic_checkpoint_v1"),
            map_location="cpu",
        )
        if not checkpoint.get("scientifically_valid"):
            raise SystemExit(f"Cannot freeze scientifically invalid {name}")
        if checkpoint.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
            raise SystemExit(f"{name} was built from a different diagnostic checkpoint")
    dataset_schedule_path = Path(args.dataset_schedule)
    dataset_schedule = _read(dataset_schedule_path)
    unsigned_schedule = {
        key: value for key, value in dataset_schedule.items() if key != "report_sha256"
    }
    if (
        dataset_schedule.get("report_sha256") != hash_dict(unsigned_schedule)
        or dataset_schedule.get("format_version") != "dataset_fixed_schedule_v1"
        or dataset_schedule.get("fit_split") != "val"
        or dataset_schedule.get("calibration_used") is not False
        or dataset_schedule.get("test_used") is not False
    ):
        raise SystemExit("Dataset-specific schedule is not a valid validation-only artifact")
    required_budgets = {int(value) for value in config["budgets"]}
    observed_budgets = {int(value) for value in dataset_schedule.get("budgets", [])}
    if observed_budgets != required_budgets:
        raise SystemExit(
            "Dataset-specific schedule must cover every final Week 7 budget: "
            f"expected={sorted(required_budgets)}, observed={sorted(observed_budgets)}"
        )
    output_path = Path(args.output_dir) / "main_stack_freeze.json"
    preview = {
        "diagnostic_checkpoint": str(diagnostic_path),
        "policy_checkpoint": str(policy_path),
        "selection_report": str(selection_path),
        "controls": {key: str(value) for key, value in controls.items()},
        "dataset_schedule": str(dataset_schedule_path),
        "output": str(output_path),
        "owner_approval_required": True,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return
    if not args.approve_freeze:
        raise SystemExit("Main-stack freeze requires explicit --approve_freeze after owner review")
    write_freeze_manifest(
        diagnostic_checkpoint=diagnostic_path,
        policy_checkpoint=policy_path,
        selection_report=selection_path,
        config_paths={
            "week7_config": config_path,
            "policy_config": policy_config_path,
            "week5_freeze": week5_path,
            "selected_frontier": Path(selection["selected_frontier_path"]),
        },
        additional_artifacts={**controls, "dataset_fixed_schedule": dataset_schedule_path},
        output_path=output_path,
        approved_by_owner=True,
        overwrite=(args.overwrite or args.resume) and output_path.exists(),
    )
    print(json.dumps(_read(output_path), indent=2))


if __name__ == "__main__":
    main()
