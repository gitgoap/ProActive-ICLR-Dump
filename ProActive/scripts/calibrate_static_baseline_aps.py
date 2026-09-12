#!/usr/bin/env python3
"""Fit APS for a frozen zero-acquisition baseline on val or cal only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from proactive.conformal.aps import evaluate_sets, fit_aps, prediction_sets
from proactive.networks.controls import ScalarConfidenceDiagnostic
from proactive.networks.diagnostic import build_diagnostic_model
from proactive.train.checkpoints import load_checkpoint, validate_freeze_manifest
from proactive.train.diagnostic import predict
from proactive.train.state_data import FeatureNormalizer
from proactive.train.vectorized import BudgetProjectedDataset, empty_state_view, load_vectorized_split
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval_config", required=True)
    parser.add_argument("--manifest_path", required=True, help="Week 5 vectorized manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase", choices=("validation", "calibration"), required=True)
    parser.add_argument("--freeze_manifest", default=None)
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.90, 0.95])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_unapproved_pilot", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def _model(checkpoint: Dict[str, Any]) -> tuple[str, torch.nn.Module]:
    stage = checkpoint.get("stage")
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise SystemExit("Static baseline checkpoint lacks architecture")
    if stage == "week5_diagnostic" and checkpoint.get("encoder_name") == "clean_mlp":
        name = "clean_only_learned"
        model = build_diagnostic_model("clean_mlp", architecture)
    elif stage == "week6_scalar_baseline":
        name = "scalar_confidence"
        model = ScalarConfidenceDiagnostic(
            state_dim=int(architecture["state_dim"]),
            dropout=float(architecture["dropout"]),
        )
    elif stage == "week6_distilled_baseline":
        name = "one_pass_distilled"
        model = build_diagnostic_model("clean_mlp", architecture)
    else:
        raise SystemExit(f"Unsupported static baseline checkpoint stage: {stage!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return name, model


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.phase == "calibration" and args.limit is not None:
        raise SystemExit("Final baseline calibration refuses --limit")
    budgets = sorted(set(args.budgets))
    coverages = sorted(set(args.coverages))
    if not budgets or budgets[0] < 1 or budgets[-1] > 7:
        raise SystemExit("Budgets must lie in [1, 7]")
    if not coverages or any(not 0.0 < value < 1.0 for value in coverages):
        raise SystemExit("Coverages must lie in (0, 1)")
    approval_path = Path(args.approval_config)
    with open(approval_path, "r", encoding="utf-8") as handle:
        approval = yaml.safe_load(handle)
    pilot = args.limit is not None and args.limit <= 100
    if (
        approval.get("metadata", {}).get("approval_status") != "APPROVED"
        and not args.dry_run
        and not (pilot and args.allow_unapproved_pilot)
    ):
        raise SystemExit("Experiment settings are not APPROVED")
    manifest_path = Path(args.manifest_path)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("source_manifest_sha256") != file_sha256(manifest_path):
        raise SystemExit("Static baseline/vectorized-manifest mismatch")
    if args.limit is None and checkpoint.get("scientifically_valid") is not True:
        raise SystemExit("Complete APS requires a scientifically valid checkpoint")
    baseline, model = _model(checkpoint)
    split = "val" if args.phase == "validation" else "cal"
    freeze_path = Path(args.freeze_manifest) if args.freeze_manifest else None
    freeze_sha = None
    if args.phase == "calibration":
        if freeze_path is None:
            raise SystemExit("Final baseline calibration requires --freeze_manifest")
        freeze = validate_freeze_manifest(freeze_path, require_policy=True)
        expected_name = {
            "scalar_confidence": "scalar_baseline",
            "one_pass_distilled": "distilled_baseline",
        }.get(baseline)
        if expected_name is not None:
            item = freeze["artifacts"].get(expected_name)
            if not item or item.get("sha256") != file_sha256(checkpoint_path):
                raise SystemExit(f"{baseline} checkpoint is not bound into the main freeze")
        else:
            nested_item = freeze["artifacts"].get("week5_freeze")
            if nested_item:
                nested = validate_freeze_manifest(nested_item["path"], require_policy=False)
                item = nested["artifacts"].get("diagnostic_checkpoint_clean_mlp_standard")
            else:
                # A leakage-safe LOMO fold has its matched clean checkpoint
                # directly in the fold freeze rather than inside the global
                # Week 5 selection freeze.
                item = freeze["artifacts"].get("clean_checkpoint")
            if not item or item.get("sha256") != file_sha256(checkpoint_path):
                raise SystemExit("Clean-only checkpoint is not bound into the active freeze")
        freeze_sha = file_sha256(freeze_path)
    output_path = Path(args.output_dir) / f"static_aps_{baseline}_{args.phase}.json"
    expected = {
        "format_version": "static_baseline_aps_v1",
        "status": "TEMPORARY_VALIDATION" if args.phase == "validation" else "FINAL_FROZEN",
        "baseline": baseline,
        "fit_split": split,
        "calibration_used": args.phase == "calibration",
        "test_used": False,
        "approval_config_sha256": file_sha256(approval_path),
        "vector_manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "freeze_manifest_sha256": freeze_sha,
        "budgets": budgets,
        "coverages": coverages,
        "limit": args.limit,
    }
    if output_path.exists() and args.resume:
        existing = _read(output_path)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"Static APS resume refused: {key} drift")
        unsigned = {key: value for key, value in existing.items() if key != "report_sha256"}
        if existing.get("report_sha256") != hash_dict(unsigned):
            raise SystemExit("Static APS report self-hash mismatch")
        print(json.dumps(existing, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({**expected, "output": str(output_path)}, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA requested but unavailable: {args.device}")
    model.to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    base = load_vectorized_split(manifest_path, split)
    data = BudgetProjectedDataset(empty_state_view(base), budgets, limit=args.limit)
    loader = DataLoader(data, batch_size=min(512, len(data)), shuffle=False, num_workers=0)
    predictions = predict(model, loader, normalizer=normalizer, device=device)
    thresholds: Dict[str, Any] = {}
    for budget in budgets:
        indices = [index for index, value in enumerate(predictions["max_budget"]) if value == budget]
        if not indices:
            raise SystemExit(f"No {split} predictions at budget {budget}")
        probability = predictions["six_way_probabilities"][indices]
        labels = predictions["six_way_targets"][indices]
        thresholds[str(budget)] = {}
        for coverage in coverages:
            fitted = fit_aps(probability, labels, alpha=1.0 - coverage)
            sets = prediction_sets(probability, fitted.threshold)
            thresholds[str(budget)][format(coverage, ".12g")] = {
                "alpha": fitted.alpha,
                "target_coverage": fitted.target_coverage,
                "threshold": fitted.threshold,
                "fit_count": fitted.calibration_count,
                "fit_metrics": evaluate_sets(sets, labels),
            }
    result: Dict[str, Any] = {**expected, "thresholds": thresholds}
    result["report_sha256"] = hash_dict(result)
    write_json(result, output_path, overwrite=args.overwrite and output_path.exists())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
