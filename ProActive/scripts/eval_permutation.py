#!/usr/bin/env python3
"""Run the Week 5 pilot or Week 7 locked permutation protocol."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import yaml

from proactive.conformal.aps import prediction_sets
from proactive.eval.permutation import (
    calibration_ranges,
    distinct_permutations,
    state_permutation_metrics,
    summarize_state_metrics,
)
from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.voi import ActionConditionedVOIHead
from proactive.policy.controller import select_action
from proactive.train.checkpoints import (
    POLICY_CHECKPOINT_VERSION,
    load_checkpoint,
    validate_freeze_manifest,
)
from proactive.train.state_data import FeatureNormalizer
from proactive.train.vectorized import load_vectorized_split, project_model_input
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aps_report", required=True)
    parser.add_argument("--policy_checkpoint", default=None)
    parser.add_argument("--freeze_manifest", default=None)
    parser.add_argument("--phase", choices=("week5", "week7"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
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


def _threshold(report: Mapping[str, Any], budget: int, coverage: float) -> float:
    value = report.get("thresholds", {}).get(str(budget), {}).get(format(coverage, ".12g"))
    if not isinstance(value, Mapping):
        raise SystemExit(f"APS report lacks coverage={coverage}, budget={budget}")
    return float(value["threshold"])


def _csv(rows: list[Dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot write an empty permutation CSV")
    import csv

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config_path = Path(args.config)
    manifest_path = Path(args.manifest_path)
    checkpoint_path = Path(args.checkpoint)
    aps_path = Path(args.aps_report)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    aps = _read_json(aps_path)
    split = "val" if args.phase == "week5" else "test"
    if args.phase == "week5":
        if aps.get("format_version") != "temporary_validation_aps_v1" or aps.get("fit_split") != "val":
            raise SystemExit("Week 5 permutation pilot requires validation-temporary APS")
        if aps.get("checkpoint_sha256") != file_sha256(checkpoint_path):
            raise SystemExit("Week 5 permutation checkpoint/APS drift")
        subset_sizes = [2, 3, 4]
        maximum_permutations = 10
        tolerance = 1.0e-6
        minimum_states = 1 if args.limit is not None else 100
        policy_checkpoint = None
        freeze_sha = None
    else:
        freeze_path = Path(args.freeze_manifest or config["stack_freeze_manifest"])
        freeze = validate_freeze_manifest(freeze_path, require_policy=True)
        primary_checkpoint = freeze["artifacts"]["diagnostic_checkpoint"]["sha256"] == file_sha256(checkpoint_path)
        if not primary_checkpoint:
            week5_item = freeze["artifacts"].get("week5_freeze")
            if not isinstance(week5_item, Mapping):
                raise SystemExit("Main freeze does not preserve the Week 5 comparison freeze")
            week5 = validate_freeze_manifest(week5_item["path"], require_policy=False)
            if not any(
                name.startswith("diagnostic_checkpoint") and item.get("sha256") == file_sha256(checkpoint_path)
                for name, item in week5["artifacts"].items()
            ):
                raise SystemExit("Week 7 comparison checkpoint was not frozen before test")
        if aps.get("format_version") != "week7_final_aps_v1" or aps.get("freeze_manifest_sha256") != file_sha256(freeze_path):
            raise SystemExit("Week 7 permutation requires final frozen APS")
        if primary_checkpoint:
            policy_path = Path(args.policy_checkpoint or freeze["artifacts"]["policy_checkpoint"]["path"])
            policy_checkpoint = load_checkpoint(
                policy_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu"
            )
            if freeze["artifacts"]["policy_checkpoint"]["sha256"] != file_sha256(policy_path):
                raise SystemExit("Week 7 permutation policy checkpoint is not frozen")
        else:
            if args.policy_checkpoint is not None:
                raise SystemExit("A non-primary diagnostic comparison cannot use the primary policy")
            policy_checkpoint = None
        freeze_sha = file_sha256(freeze_path)
        subset_sizes = [int(value) for value in config["permutation"]["subset_sizes"]]
        maximum_permutations = int(config["permutation"]["maximum_permutations_per_state"])
        tolerance = float(config["permutation"]["floating_tolerance"])
        minimum_states = int(config["permutation"]["minimum_states"])
    evaluation_budget = 4
    coverage = 0.90
    threshold = _threshold(aps, evaluation_budget, coverage)
    output_dir = Path(args.output_dir)
    stem = f"permutation_{args.phase}_{checkpoint['encoder_name']}_{checkpoint.get('gru_condition', 'standard')}_seed{checkpoint['seed']}"
    report_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    expected = {
        "format_version": "permutation_evaluation_v1",
        "phase": args.phase,
        "split": split,
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "aps_sha256": file_sha256(aps_path),
        "freeze_manifest_sha256": freeze_sha,
        "limit": args.limit,
    }
    if report_path.exists() and args.resume:
        existing = _read_json(report_path)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"Permutation resume refused: {key} drift")
        if not csv_path.exists() or file_sha256(csv_path) != existing.get("csv_sha256"):
            raise SystemExit("Permutation CSV drift")
        figure_path = Path(existing.get("figure_path", ""))
        if not figure_path.exists() or file_sha256(figure_path) != existing.get("figure_sha256"):
            raise SystemExit("Permutation figure drift")
        print(json.dumps(existing, indent=2))
        return
    if (report_path.exists() or csv_path.exists()) and not (args.overwrite or args.resume):
        raise SystemExit(f"Permutation outputs exist for {stem}; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({**expected, "subset_sizes": subset_sizes, "minimum_states": minimum_states, "output": str(report_path)}, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA requested but unavailable: {args.device}")
    model = build_diagnostic_model(checkpoint["encoder_name"], checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    policy_head = None
    if policy_checkpoint is not None:
        policy_architecture = policy_checkpoint["policy_architecture"]
        policy_head = ActionConditionedVOIHead(
            state_dim=int(checkpoint["architecture"]["state_dim"]),
            action_embedding_dim=int(policy_architecture["action_embedding_dim"]),
            budget_embedding_dim=int(policy_architecture["budget_embedding_dim"]),
            hidden_dim=int(policy_architecture["hidden_dim"]),
            dropout=float(policy_architecture["dropout"]),
            max_budget=int(checkpoint["architecture"]["max_budget"]),
        )
        policy_head.load_state_dict(policy_checkpoint["model_state_dict"])
        policy_head.to(device).eval()
    base = load_vectorized_split(manifest_path, split)
    candidates = [
        index
        for index, count in enumerate(base.model_input["acquired_mask"].sum(dim=1).tolist())
        if int(count) in subset_sizes and int(count) <= evaluation_budget
    ]
    candidates.sort(key=lambda index: hash_dict({"seed": args.seed, "state_id": base.state_ids[index]}))
    if args.limit is not None:
        candidates = candidates[: args.limit]
    elif len(candidates) > minimum_states:
        candidates = candidates[:minimum_states]
    if len(candidates) < minimum_states:
        raise SystemExit(f"Permutation protocol needs {minimum_states} states; found {len(candidates)}")
    rows: list[Dict[str, Any]] = []
    calibration_by_size: Dict[int, Dict[int, Dict[str, list[float]]]] = {}
    for index in candidates:
        row = base[index]
        model_input = project_model_input(row["model_input"], evaluation_budget)
        acquired_count = int(model_input["acquired_mask"].sum())
        canonical = model_input["sequence_indices"][:acquired_count].tolist()
        permutations = distinct_permutations(
            canonical, maximum=maximum_permutations, seed_key=f"{args.seed}|{row['state_id']}"
        )
        batch = {key: value.unsqueeze(0).repeat(len(permutations), *([1] * value.ndim)).to(device) for key, value in model_input.items()}
        for permutation_index, permutation in enumerate(permutations):
            batch["sequence_indices"][permutation_index, :acquired_count] = torch.tensor(
                permutation, dtype=torch.long, device=device
            )
        with torch.no_grad():
            output = model(normalizer.transform(batch))
            six = torch.softmax(output.six_way_logits, dim=-1).cpu().numpy()
            bits = torch.sigmoid(output.bit_logits).cpu().numpy()
            hidden = output.hidden.cpu().numpy()
            # Prediction-set drift is meaningful for every frozen encoder.  In
            # Week 7 the common primary-stack APS threshold is deliberately
            # reused for the GRU comparisons so only arrival order changes.
            sets = prediction_sets(six, threshold)
            actions = None
            if policy_head is not None:
                values = policy_head(output.hidden, batch["remaining_budget"]).cpu().numpy()
                legal_masks = batch["action_mask"].cpu().numpy()
                actions = [select_action(value, legal) for value, legal in zip(values, legal_masks)]
        metrics = state_permutation_metrics(
            six_way_probabilities=six,
            bit_probabilities=bits,
            hidden_states=hidden,
            prediction_sets=sets,
            actions=actions,
        )
        metric_row: Dict[str, Any] = {
            "state_id": row["state_id"],
            "dataset": row["audit_metadata"]["dataset"],
            "model_id": row["audit_metadata"]["model_id"],
            "subset_size": acquired_count,
            **metrics,
        }
        rows.append(metric_row)
        true_label = int(row["targets"]["six_way"])
        by_position = calibration_by_size.setdefault(acquired_count, {})
        if sets is not None:
            for position, prediction_set in enumerate(sets):
                values = by_position.setdefault(position, {"covered": [], "size": []})
                values["covered"].append(float(true_label in prediction_set))
                values["size"].append(float(len(prediction_set)))
    calibration = {}
    for size, by_position in sorted(calibration_by_size.items()):
        if not by_position:
            continue
        coverage_values = [float(np.mean(by_position[index]["covered"])) for index in sorted(by_position)]
        size_values = [float(np.mean(by_position[index]["size"])) for index in sorted(by_position)]
        calibration[str(size)] = {
            **calibration_ranges(coverage_values, size_values),
            "coverage_by_permutation": coverage_values,
            "average_set_size_by_permutation": size_values,
        }
    write_text(_csv(rows), csv_path, overwrite=(args.overwrite or args.resume) and csv_path.exists())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path = output_dir / f"{stem}.png"
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    fields = [
        ("js_drift", "JS drift"),
        ("bit_l1_drift", "Source-bit L1 drift"),
        ("hidden_relative_l2_drift", "Hidden relative L2 drift"),
    ]
    for axis, (field, title) in zip(axes, fields):
        for size in sorted(set(int(row["subset_size"]) for row in rows)):
            values = [float(row[field]) for row in rows if int(row["subset_size"]) == size]
            axis.hist(values, bins=30, alpha=0.45, label=f"|S|={size}")
        axis.set_title(title)
        axis.set_xlabel("drift")
        axis.set_ylabel("states")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    invariant_encoder = checkpoint["encoder_name"] in {"deep_sets", "masked_slot_mlp"}
    maximum_observed = max(
        max(float(row["js_drift"]), float(row["bit_l1_drift"]), float(row["hidden_relative_l2_drift"]))
        for row in rows
    )
    result: Dict[str, Any] = {
        **expected,
        "is_valid": not invariant_encoder or maximum_observed <= tolerance,
        "encoder_name": checkpoint["encoder_name"],
        "gru_condition": checkpoint.get("gru_condition", "standard"),
        "evaluation_budget": evaluation_budget,
        "coverage": coverage,
        "primary_frozen_stack": policy_checkpoint is not None or args.phase == "week5",
        "prediction_set_drift_applicable": True,
        "action_drift_applicable": policy_checkpoint is not None,
        "subset_sizes": subset_sizes,
        "maximum_permutations_per_state": maximum_permutations,
        "floating_tolerance": tolerance,
        "summary": summarize_state_metrics(rows),
        "calibration_drift_by_subset_size": calibration,
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
        "figure_path": str(figure_path),
        "figure_sha256": file_sha256(figure_path),
    }
    result["report_sha256"] = hash_dict(result)
    write_json(result, report_path, overwrite=(args.overwrite or args.resume) and report_path.exists())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
