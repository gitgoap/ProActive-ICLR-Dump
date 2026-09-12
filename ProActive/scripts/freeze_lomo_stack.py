#!/usr/bin/env python3
"""Freeze a predeclared Deep-Sets LOMO diagnostic or complete fold stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.train.checkpoints import CHECKPOINT_VERSION, POLICY_CHECKPOINT_VERSION, load_checkpoint, write_freeze_manifest
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("diagnostic", "stack"), required=True)
    parser.add_argument("--fold_manifest", required=True)
    parser.add_argument("--diagnostic_checkpoint", required=True)
    parser.add_argument("--clean_checkpoint", required=True)
    parser.add_argument("--policy_checkpoint")
    parser.add_argument("--scalar_checkpoint")
    parser.add_argument("--diag_config", default="configs/experiments/diag_bakeoff.yaml")
    parser.add_argument("--policy_config", default="configs/experiments/policy_train.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--approve_freeze", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_path = Path(args.fold_manifest)
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    if fold.get("format_version") != "lomo_fold_v1" or fold.get("status") != "COMPLETE":
        raise SystemExit("LOMO fold is not complete")
    vector_item = next((item for item in fold["artifacts"] if item["kind"] == "vector_manifest"), None)
    if vector_item is None or file_sha256(vector_item["path"]) != vector_item["sha256"]:
        raise SystemExit("LOMO vector artifact drift")
    vector_sha = vector_item["sha256"]
    diagnostic_path = Path(args.diagnostic_checkpoint)
    clean_path = Path(args.clean_checkpoint)
    diagnostic = load_checkpoint(diagnostic_path, expected_version=CHECKPOINT_VERSION, map_location="cpu")
    clean = load_checkpoint(clean_path, expected_version=CHECKPOINT_VERSION, map_location="cpu")
    if diagnostic.get("encoder_name") != "deep_sets" or clean.get("encoder_name") != "clean_mlp":
        raise SystemExit("LOMO architecture is predeclared as Deep Sets plus clean-only MLP")
    if diagnostic.get("source_manifest_sha256") != vector_sha or clean.get("source_manifest_sha256") != vector_sha:
        raise SystemExit("LOMO checkpoint/fold-vector mismatch")
    output_dir = Path(args.output_dir)
    selection_path = output_dir / "lomo_predeclared_selection.json"
    freeze_path = output_dir / f"lomo_{args.stage}_freeze.json"
    selection: Dict[str, Any] = {
        "format_version": "lomo_predeclared_selection_v1",
        "status": "PREDECLARED_NOT_SELECTED_ON_HELDOUT",
        "heldout_model_id": fold["heldout_model_id"],
        "encoder": "deep_sets",
        "seed": int(diagnostic["seed"]),
        "selection_split": "none_in_fold",
        "heldout_test_used": False,
        "diagnostic_checkpoint_sha256": file_sha256(diagnostic_path),
        "clean_checkpoint_sha256": file_sha256(clean_path),
        "fold_manifest_sha256": file_sha256(fold_path),
    }
    selection["report_sha256"] = hash_dict(selection)
    preview: Dict[str, Any] = {
        "stage": args.stage,
        "fold": str(fold_path),
        "diagnostic": str(diagnostic_path),
        "clean": str(clean_path),
        "output": str(freeze_path),
        "owner_approval_required": True,
    }
    policy_path = scalar_path = None
    if args.stage == "stack":
        if not args.policy_checkpoint or not args.scalar_checkpoint:
            raise SystemExit("Stack freeze requires --policy_checkpoint and --scalar_checkpoint")
        policy_path = Path(args.policy_checkpoint)
        scalar_path = Path(args.scalar_checkpoint)
        policy = load_checkpoint(policy_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu")
        scalar = load_checkpoint(scalar_path, expected_version=CHECKPOINT_VERSION, map_location="cpu")
        if policy.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
            raise SystemExit("LOMO policy/diagnostic mismatch")
        if policy.get("vector_manifest_sha256") != vector_sha:
            raise SystemExit("LOMO policy/vector mismatch")
        if scalar.get("source_manifest_sha256") != vector_sha:
            raise SystemExit("LOMO scalar/vector mismatch")
        if scalar.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
            raise SystemExit("LOMO scalar/diagnostic mismatch")
        preview.update({"policy": str(policy_path), "scalar": str(scalar_path)})
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return
    if not args.approve_freeze:
        raise SystemExit("LOMO freeze requires explicit --approve_freeze")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(selection, selection_path, overwrite=args.overwrite and selection_path.exists())
    additional = {"clean_checkpoint": clean_path, "fold_manifest": fold_path}
    if scalar_path is not None:
        additional["scalar_baseline"] = scalar_path
    write_freeze_manifest(
        diagnostic_checkpoint=diagnostic_path,
        policy_checkpoint=policy_path,
        selection_report=selection_path,
        config_paths={
            "diagnostic_config": Path(args.diag_config),
            "policy_config": Path(args.policy_config),
        },
        additional_artifacts=additional,
        output_path=freeze_path,
        approved_by_owner=True,
        overwrite=args.overwrite and freeze_path.exists(),
    )
    print(freeze_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
