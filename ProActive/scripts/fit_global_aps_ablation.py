#!/usr/bin/env python3
"""Fit one source-calibration APS threshold shared across all budgets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.conformal.aps import evaluate_sets, fit_aps, prediction_sets
from proactive.train.checkpoints import validate_freeze_manifest
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_manifest", required=True)
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.90, 0.95])
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.trajectory_manifest)
    freeze_path = Path(args.freeze_manifest)
    validate_freeze_manifest(freeze_path, require_policy=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("manifest_sha256") != hash_dict(unsigned)
        or manifest.get("status") != "COMPLETE"
        or manifest.get("split") != "cal"
        or manifest.get("test_used") is not False
        or manifest.get("freeze_manifest_sha256") != file_sha256(freeze_path)
    ):
        raise SystemExit("Global APS requires complete, source-only calibration trajectories")
    probabilities = []
    labels = []
    budgets = []
    for entry in manifest["files"]:
        path = manifest_path.parent / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise SystemExit(f"Calibration trajectory drift: {path}")
        rows = list(iter_jsonl(path))
        if len(rows) != int(entry["row_count"]):
            raise SystemExit(f"Calibration row-count drift: {path}")
        for row in rows:
            probabilities.append(row["six_way_probabilities"])
            labels.append(int(row["targets"]["six_way"]))
            budgets.append(int(row["budget"]))
    thresholds: Dict[str, Any] = {}
    for coverage in args.coverages:
        fitted = fit_aps(probabilities, labels, alpha=1.0 - coverage)
        sets = prediction_sets(probabilities, fitted.threshold)
        thresholds[format(coverage, ".12g")] = {
            "target_coverage": coverage,
            "threshold": fitted.threshold,
            "calibration_count": fitted.calibration_count,
            "calibration_metrics": evaluate_sets(sets, labels),
        }
    result: Dict[str, Any] = {
        "format_version": "week8_global_aps_v1",
        "status": "FROZEN_ABLATION",
        "fit_split": "cal",
        "test_used": False,
        "pooling": "all_budget_trajectories",
        "budgets": sorted(set(budgets)),
        "coverages": list(args.coverages),
        "thresholds": thresholds,
        "trajectory_manifest_sha256": file_sha256(manifest_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
    }
    result["report_sha256"] = hash_dict(result)
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}")
    write_json(result, output_path, overwrite=args.overwrite and output_path.exists())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
