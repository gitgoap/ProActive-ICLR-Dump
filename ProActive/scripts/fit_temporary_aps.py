#!/usr/bin/env python3
"""Fit Week 5 temporary APS thresholds on validation predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import yaml
from torch.utils.data import DataLoader

from proactive.conformal.aps import evaluate_sets, fit_aps, prediction_sets
from proactive.networks.diagnostic import build_diagnostic_model
from proactive.train.checkpoints import load_checkpoint
from proactive.train.diagnostic import predict
from proactive.train.state_data import FeatureNormalizer
from proactive.train.vectorized import BudgetProjectedDataset, load_vectorized_split
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/diag_bakeoff.yaml")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/week5_reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("config_sha256") != file_sha256(config_path):
        raise SystemExit("Temporary APS refused: checkpoint/config hash drift")
    if checkpoint.get("source_manifest_sha256") != file_sha256(manifest_path):
        raise SystemExit("Temporary APS refused: checkpoint/vectorized-manifest drift")
    if checkpoint.get("stage") != "week5_diagnostic":
        raise SystemExit("Temporary APS requires a Week 5 diagnostic checkpoint")
    if not checkpoint.get("scientifically_valid") and args.limit is None:
        raise SystemExit("Temporary APS requires a scientifically valid non-pilot checkpoint")
    output_path = Path(args.output_dir) / (
        f"temporary_aps_{checkpoint['encoder_name']}_{checkpoint['gru_condition']}_seed{checkpoint['seed']}.json"
    )
    budgets = [int(value) for value in config["temporary_aps"].get("budgets", checkpoint["budgets"])]
    coverages = [float(value) for value in config["temporary_aps"]["coverages"]]
    expected = {
        "format_version": "temporary_validation_aps_v1",
        "status": "TEMPORARY_NOT_FINAL_CALIBRATION",
        "fit_split": "val",
        "calibration_split_used": False,
        "test_split_used": False,
        "config_sha256": file_sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "budgets": budgets,
        "coverages": coverages,
        "limit": args.limit,
    }
    if output_path.exists() and args.resume:
        with open(output_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"Temporary APS resume refused: {key} drift")
        unsigned = {key: value for key, value in existing.items() if key != "report_sha256"}
        if existing.get("report_sha256") != hash_dict(unsigned):
            raise SystemExit("Temporary APS resume refused: report self-hash mismatch")
        print(json.dumps(existing, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({"split": "val", "checkpoint": str(checkpoint_path), "output": str(output_path)}, indent=2))
        return
    device = torch.device(args.device)
    model = build_diagnostic_model(checkpoint["encoder_name"], checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    base = load_vectorized_split(manifest_path, "val")
    data = BudgetProjectedDataset(base, budgets, limit=args.limit)
    loader = DataLoader(data, batch_size=512, shuffle=False)
    predictions = predict(model, loader, normalizer=normalizer, device=device)
    thresholds: Dict[str, Any] = {}
    for budget in budgets:
        indices = [index for index, value in enumerate(predictions["max_budget"]) if value == budget]
        if not indices:
            raise SystemExit(f"No validation predictions for budget {budget}")
        probabilities = predictions["six_way_probabilities"][indices]
        labels = predictions["six_way_targets"][indices]
        thresholds[str(budget)] = {}
        for coverage in coverages:
            fitted = fit_aps(probabilities, labels, alpha=1.0 - coverage)
            sets = prediction_sets(probabilities, fitted.threshold)
            thresholds[str(budget)][format(coverage, ".12g")] = {
                "alpha": fitted.alpha,
                "target_coverage": fitted.target_coverage,
                "threshold": fitted.threshold,
                "calibration_count": fitted.calibration_count,
                "validation_metrics": evaluate_sets(sets, labels),
            }
    report: Dict[str, Any] = {**expected, "thresholds": thresholds}
    report["report_sha256"] = hash_dict(report)
    write_json(report, output_path, overwrite=args.overwrite and output_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
