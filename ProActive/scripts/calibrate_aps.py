#!/usr/bin/env python3
"""Fit final per-budget APS thresholds after the main stack is frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from proactive.conformal.aps import evaluate_sets, fit_aps, prediction_sets
from proactive.conformal.contracts import validate_final_aps_report
from proactive.train.checkpoints import validate_freeze_manifest
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/calibrate_aps.yaml")
    parser.add_argument("--manifest_path", required=True, help="Calibration trajectory manifest")
    parser.add_argument("--freeze_manifest", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    if args.device != "cpu":
        raise SystemExit("APS calibration is CPU-only")
    if args.limit is not None:
        raise SystemExit("Final APS refuses --limit; calibrate on the complete calibration split")
    config_path = Path(args.config)
    trajectory_manifest_path = Path(args.manifest_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("metadata", {}).get("approval_status") != "APPROVED" and not args.dry_run:
        raise SystemExit("Week 7 calibration settings are not APPROVED")
    freeze_path = Path(args.freeze_manifest or config["stack_freeze_manifest"])
    validate_freeze_manifest(freeze_path, require_policy=True)
    trajectory_manifest = _read_json(trajectory_manifest_path)
    unsigned_trajectory = {
        key: value for key, value in trajectory_manifest.items() if key != "manifest_sha256"
    }
    if trajectory_manifest.get("manifest_sha256") != hash_dict(unsigned_trajectory):
        raise SystemExit("Calibration trajectory manifest self-hash mismatch")
    if trajectory_manifest.get("status") != "COMPLETE" or trajectory_manifest.get("split") != "cal":
        raise SystemExit("Final APS requires complete calibration-only trajectories")
    if trajectory_manifest.get("test_used") is not False:
        raise SystemExit("Calibration trajectory provenance indicates test access")
    if trajectory_manifest.get("freeze_manifest_sha256") != file_sha256(freeze_path):
        raise SystemExit("Calibration trajectories/freeze hash mismatch")
    budgets = [int(value) for value in config["budgets"]]
    coverages = [float(value) for value in config["coverages"]]
    entries = {int(entry["budget"]): entry for entry in trajectory_manifest.get("files", [])}
    if set(entries) != set(budgets):
        raise SystemExit("Calibration trajectory budget coverage mismatch")
    output_dir = Path(args.output_dir or config["outputs"]["calibration_dir"])
    output_path = output_dir / "aps_thresholds.json"
    expected = {
        "format_version": "week7_final_aps_v1",
        "status": "FINAL_FROZEN",
        "fit_split": "cal",
        "test_used": False,
        "config_sha256": file_sha256(config_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "trajectory_manifest_sha256": file_sha256(trajectory_manifest_path),
    }
    if output_path.exists() and args.resume:
        existing = _read_json(output_path)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"APS resume refused: {key} drift")
        try:
            validate_final_aps_report(existing, freeze_manifest_path=freeze_path)
        except ValueError as exc:
            raise SystemExit(f"APS resume refused: {exc}") from exc
        print(json.dumps(existing, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({**expected, "budgets": budgets, "coverages": coverages, "output": str(output_path)}, indent=2))
        return
    thresholds: Dict[str, Any] = {}
    for budget in budgets:
        entry = entries[budget]
        path = trajectory_manifest_path.parent / entry["path"]
        if not path.exists() or file_sha256(path) != entry.get("sha256"):
            raise SystemExit(f"Calibration trajectory artifact drift: {path}")
        rows = list(iter_jsonl(path))
        if len(rows) != int(entry["row_count"]):
            raise SystemExit(f"Calibration trajectory row-count mismatch: {path}")
        seen_states = set()
        for row in rows:
            expected_hash = row.get("record_sha256")
            unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
            if expected_hash != hash_dict(unsigned):
                raise SystemExit(f"Calibration trajectory row self-hash mismatch: {path}")
            if row.get("record_type") != "frozen_policy_trajectory_v1":
                raise SystemExit(f"Calibration trajectory schema mismatch: {path}")
            if row.get("metadata", {}).get("split") != "cal" or int(row.get("budget", -1)) != budget:
                raise SystemExit(f"Calibration trajectory split/budget mismatch: {path}")
            state_id = row.get("state_id")
            if state_id in seen_states:
                raise SystemExit(f"Duplicate calibration trajectory state: {state_id}")
            seen_states.add(state_id)
        probabilities = [row["six_way_probabilities"] for row in rows]
        labels = [int(row["targets"]["six_way"]) for row in rows]
        thresholds[str(budget)] = {}
        for coverage in coverages:
            fitted = fit_aps(probabilities, labels, alpha=1.0 - coverage)
            sets = prediction_sets(probabilities, fitted.threshold)
            thresholds[str(budget)][format(coverage, ".12g")] = {
                "alpha": fitted.alpha,
                "target_coverage": fitted.target_coverage,
                "threshold": fitted.threshold,
                "calibration_count": fitted.calibration_count,
                "calibration_metrics": evaluate_sets(sets, labels),
            }
    result: Dict[str, Any] = {
        **expected,
        "budgets": budgets,
        "coverages": coverages,
        "thresholds": thresholds,
    }
    result["thresholds_sha256"] = hash_dict(result)
    write_json(result, output_path, overwrite=args.overwrite and output_path.exists())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
