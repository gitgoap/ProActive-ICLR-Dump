#!/usr/bin/env python3
"""Run the frozen learned policy on the calibration split only."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import yaml

from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.voi import ActionConditionedVOIHead, FrozenDiagnosticPolicy
from proactive.policy.rollout import learned_policy_rollout
from proactive.train.checkpoints import (
    POLICY_CHECKPOINT_VERSION,
    load_checkpoint,
    validate_freeze_manifest,
)
from proactive.train.state_data import FeatureNormalizer, SIX_WAY_LABELS
from proactive.train.vectorized import load_vectorized_split, project_model_input
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_jsonl


LOGGER = logging.getLogger("build_policy_trajectories")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/calibrate_aps.yaml")
    parser.add_argument("--manifest_path", required=True, help="Week 5 vectorized manifest")
    parser.add_argument("--freeze_manifest", default=None)
    parser.add_argument("--teacher_path", default="outputs/teacher_core_contract_v1_recovered")
    parser.add_argument("--output_dir", default="outputs/week7_calibration/trajectories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_unapproved_pilot", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def _teacher_index(directory: Path) -> tuple[Dict[tuple[str, str], Mapping[str, Any]], list[Dict[str, Any]]]:
    result: Dict[tuple[str, str], Mapping[str, Any]] = {}
    files = []
    for path in sorted(directory.glob("teacher_*.jsonl")):
        if path.name.endswith(".failures.jsonl"):
            continue
        rows = 0
        for record in iter_jsonl(path):
            if record.get("split") != "cal":
                continue
            key = (record.get("model_id"), record.get("instance_id"))
            if key in result:
                raise SystemExit(f"Duplicate calibration teacher identity: {key}")
            if record.get("valid") is not True:
                raise SystemExit(f"Invalid teacher row in frozen source: {key}")
            result[key] = record
            rows += 1
        files.append({"path": str(path), "sha256": file_sha256(path), "calibration_rows": rows})
    if not result:
        raise SystemExit(f"No calibration teachers found in {directory}")
    return result, files


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config_path = Path(args.config)
    vector_manifest_path = Path(args.manifest_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    approval = config.get("metadata", {}).get("approval_status")
    pilot = args.limit is not None and args.limit <= 100
    if approval != "APPROVED" and not args.dry_run and not (pilot and args.allow_unapproved_pilot):
        raise SystemExit("Week 7 settings are not APPROVED; only dry-run or explicit <=100-row pilot is allowed")
    if config["splits"]["calibration"] != "cal" or config["splits"]["locked_test"] != "test":
        raise SystemExit("Week 7 split firewall config drift")
    freeze_path = Path(args.freeze_manifest or config["stack_freeze_manifest"])
    freeze = validate_freeze_manifest(freeze_path, require_policy=True)
    diagnostic_path = Path(freeze["artifacts"]["diagnostic_checkpoint"]["path"])
    policy_path = Path(freeze["artifacts"]["policy_checkpoint"]["path"])
    diagnostic = load_checkpoint(diagnostic_path, map_location="cpu")
    policy = load_checkpoint(policy_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu")
    if policy.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
        raise SystemExit("Frozen policy/diagnostic checkpoint mismatch")
    if policy.get("vector_manifest_sha256") != file_sha256(vector_manifest_path):
        raise SystemExit("Frozen policy/vectorized-manifest mismatch")
    budgets = [int(value) for value in config["budgets"]]
    output_dir = Path(args.output_dir)
    output_manifest = output_dir / "calibration_trajectory_manifest.json"
    expected = {
        "format_version": "week7_calibration_trajectories_v1",
        "config_sha256": file_sha256(config_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "vector_manifest_sha256": file_sha256(vector_manifest_path),
        "diagnostic_checkpoint_sha256": file_sha256(diagnostic_path),
        "policy_checkpoint_sha256": file_sha256(policy_path),
        "limit": args.limit,
    }
    if output_manifest.exists() and args.resume:
        existing = _read_json(output_manifest)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"Calibration trajectory resume refused: {key} drift")
        for entry in existing.get("files", []):
            path = output_dir / entry["path"]
            if not path.exists() or file_sha256(path) != entry.get("sha256"):
                raise SystemExit(f"Calibration trajectory artifact drift: {path}")
            if sum(1 for _ in iter_jsonl(path)) != int(entry.get("row_count", -1)):
                raise SystemExit(f"Calibration trajectory row-count drift: {path}")
        unsigned = {key: value for key, value in existing.items() if key != "manifest_sha256"}
        if existing.get("manifest_sha256") != hash_dict(unsigned):
            raise SystemExit("Calibration trajectory manifest self-hash mismatch")
        print(json.dumps(existing, indent=2))
        return
    if output_manifest.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_manifest}; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({**expected, "split": "cal", "budgets": budgets, "output": str(output_manifest)}, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA requested but unavailable: {args.device}")
    diagnostic_model = build_diagnostic_model(diagnostic["encoder_name"], diagnostic["architecture"])
    diagnostic_model.load_state_dict(diagnostic["model_state_dict"])
    policy_architecture = policy["policy_architecture"]
    head = ActionConditionedVOIHead(
        state_dim=int(diagnostic["architecture"]["state_dim"]),
        action_embedding_dim=int(policy_architecture["action_embedding_dim"]),
        budget_embedding_dim=int(policy_architecture["budget_embedding_dim"]),
        hidden_dim=int(policy_architecture["hidden_dim"]),
        dropout=float(policy_architecture["dropout"]),
        max_budget=max(budgets),
    )
    head.load_state_dict(policy["model_state_dict"])
    model = FrozenDiagnosticPolicy(diagnostic_model, head).to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(diagnostic["normalizer"])
    base = load_vectorized_split(vector_manifest_path, "cal")
    empty_indices = torch.nonzero(base.model_input["acquired_mask"].sum(dim=1) == 0, as_tuple=False).flatten().tolist()
    identities = [
        (base.audit_metadata["model_id"][index], base.audit_metadata["instance_id"][index])
        for index in empty_indices
    ]
    if len(identities) != len(set(identities)):
        raise SystemExit("Calibration empty-state identities are duplicated")
    if args.limit is not None:
        empty_indices = empty_indices[: args.limit]
    teacher, teacher_files = _teacher_index(Path(args.teacher_path))
    missing = [
        (base.audit_metadata["model_id"][index], base.audit_metadata["instance_id"][index])
        for index in empty_indices
        if (base.audit_metadata["model_id"][index], base.audit_metadata["instance_id"][index]) not in teacher
    ]
    if missing:
        raise SystemExit(f"Missing calibration teacher rows: {missing[:3]} (total={len(missing)})")
    files = []
    total_rows = 0
    for budget in budgets:
        records = []
        action_counts: Counter[str] = Counter()
        for index in empty_indices:
            row = base[index]
            key = (row["audit_metadata"]["model_id"], row["audit_metadata"]["instance_id"])
            initial = project_model_input(row["model_input"], budget)
            trajectory = learned_policy_rollout(
                policy_model=model,
                initial_model_input=initial,
                teacher_record=teacher[key],
                normalizer=normalizer,
                device=device,
            )
            action_counts.update(trajectory["actions"])
            record: Dict[str, Any] = {
                "record_type": "frozen_policy_trajectory_v1",
                "state_id": row["state_id"],
                "metadata": dict(row["audit_metadata"]),
                "budget": budget,
                "actions": trajectory["actions"],
                "acquisition_cost": trajectory["acquisition_cost"],
                "bit_probabilities": trajectory["bit_probabilities"],
                "six_way_probabilities": trajectory["six_way_probabilities"],
                "signature_prediction": trajectory["signature_prediction"],
                "targets": {
                    "source_bits": row["targets"]["source_bits"].tolist(),
                    "six_way": int(row["targets"]["six_way"]),
                    "signature": row["targets"]["signature"].tolist(),
                },
                "freeze_manifest_sha256": file_sha256(freeze_path),
            }
            record["record_sha256"] = hash_dict(record)
            records.append(record)
        name = f"calibration_trajectories_budget{budget}.jsonl"
        path = output_dir / name
        write_jsonl(records, path, overwrite=(args.overwrite or args.resume) and path.exists())
        files.append(
            {
                "path": name,
                "budget": budget,
                "row_count": len(records),
                "sha256": file_sha256(path),
                "action_counts": dict(action_counts),
            }
        )
        total_rows += len(records)
        LOGGER.info("Budget %s: wrote %s calibration trajectories", budget, len(records))
    result: Dict[str, Any] = {
        **expected,
        "status": "PILOT" if args.limit is not None else "COMPLETE",
        "split": "cal",
        "calibration_only": True,
        "test_used": False,
        "budgets": budgets,
        "base_model_instances": len(empty_indices),
        "row_count": total_rows,
        "six_way_labels": list(SIX_WAY_LABELS),
        "teacher_files": teacher_files,
        "files": files,
    }
    result["manifest_sha256"] = hash_dict(result)
    write_json(result, output_manifest, overwrite=args.overwrite and output_manifest.exists())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
