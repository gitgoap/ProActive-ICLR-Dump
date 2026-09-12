#!/usr/bin/env python3
"""Freeze one predeclared Week 8 diagnostic ablation before policy training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.train.ablations import validate_week8_ablation_authorization
from proactive.train.checkpoints import CHECKPOINT_VERSION, load_checkpoint, write_freeze_manifest
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week8_config", default="configs/experiments/week8_evaluation.yaml")
    parser.add_argument("--ablation_id", required=True)
    parser.add_argument("--diag_config", required=True)
    parser.add_argument("--policy_config", required=True)
    parser.add_argument("--vector_manifest", required=True)
    parser.add_argument("--diagnostic_checkpoint", required=True)
    parser.add_argument(
        "--clean_checkpoint",
        default=None,
        help="Optional matched clean control; not required for policy-only ablation evaluation",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--approve_freeze", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    week8_path = Path(args.week8_config)
    week8 = yaml.safe_load(week8_path.read_text(encoding="utf-8"))
    if week8.get("metadata", {}).get("approval_status") != "APPROVED":
        raise SystemExit("Week 8 configuration is not owner-approved")
    try:
        validate_week8_ablation_authorization(week8)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.ablation_id not in week8["ablation_execution"]["mandatory"]:
        raise SystemExit(f"Ablation is not predeclared: {args.ablation_id}")
    diag_config = Path(args.diag_config)
    policy_config = Path(args.policy_config)
    vector_path = Path(args.vector_manifest)
    diagnostic_path = Path(args.diagnostic_checkpoint)
    clean_path = Path(args.clean_checkpoint) if args.clean_checkpoint else None
    diagnostic = load_checkpoint(
        diagnostic_path, expected_version=CHECKPOINT_VERSION, map_location="cpu"
    )
    clean = (
        load_checkpoint(clean_path, expected_version=CHECKPOINT_VERSION, map_location="cpu")
        if clean_path is not None
        else None
    )
    vector_sha = file_sha256(vector_path)
    checkpoint_items = [("diagnostic", diagnostic, diagnostic_path)]
    if clean is not None and clean_path is not None:
        checkpoint_items.append(("clean", clean, clean_path))
    for name, checkpoint, path in checkpoint_items:
        if checkpoint.get("stage") != "week5_diagnostic":
            raise SystemExit(f"{name} checkpoint has the wrong stage")
        if checkpoint.get("source_manifest_sha256") != vector_sha:
            raise SystemExit(f"{name} checkpoint/vector-manifest mismatch")
        if checkpoint.get("config_sha256") != file_sha256(diag_config):
            raise SystemExit(f"{name} checkpoint/diagnostic-config mismatch")
        if checkpoint.get("scientifically_valid") is not True:
            raise SystemExit(f"{name} checkpoint is not scientifically valid")
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint: {path}")
    if diagnostic.get("encoder_name") != "deep_sets":
        raise SystemExit("Week 8 trained ablations require a Deep Sets diagnostic")
    if clean is not None and clean.get("encoder_name") != "clean_mlp":
        raise SystemExit("Optional Week 8 clean control has the wrong encoder")
    output_dir = Path(args.output_dir)
    selection_path = output_dir / "predeclared_selection.json"
    freeze_path = output_dir / "week8_diagnostic_freeze.json"
    selection: Dict[str, Any] = {
        "format_version": "week8_predeclared_diagnostic_selection_v1",
        "status": "PREDECLARED_NOT_SELECTED_ON_TEST",
        "ablation_id": args.ablation_id,
        "seed": int(week8["seed"]),
        "selection_split": "none",
        "calibration_used": False,
        "test_used": False,
        "week8_config_sha256": file_sha256(week8_path),
        "vector_manifest_sha256": vector_sha,
        "diagnostic_checkpoint_sha256": file_sha256(diagnostic_path),
        "clean_checkpoint_sha256": file_sha256(clean_path) if clean_path is not None else None,
    }
    selection["report_sha256"] = hash_dict(selection)
    preview = {
        "is_valid": True,
        "ablation_id": args.ablation_id,
        "diagnostic_checkpoint": str(diagnostic_path),
        "clean_checkpoint": str(clean_path) if clean_path is not None else None,
        "output": str(freeze_path),
        "owner_approval_required": True,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return
    if not args.approve_freeze:
        raise SystemExit("Use --approve_freeze only after owner review")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (selection_path, freeze_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {path}; use --overwrite")
    write_json(selection, selection_path, overwrite=args.overwrite and selection_path.exists())
    write_freeze_manifest(
        diagnostic_checkpoint=diagnostic_path,
        policy_checkpoint=None,
        selection_report=selection_path,
        config_paths={
            "week8_config": week8_path,
            "diagnostic_config": diag_config,
            "policy_config": policy_config,
        },
        additional_artifacts={
            **(
                {"diagnostic_checkpoint_clean_mlp_standard": clean_path}
                if clean_path is not None
                else {}
            ),
            "vector_manifest": vector_path,
        },
        output_path=freeze_path,
        approved_by_owner=True,
        overwrite=args.overwrite and freeze_path.exists(),
    )
    print(freeze_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
