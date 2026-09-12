#!/usr/bin/env python3
"""Evaluate exactly one frozen policy stack on core validation evidence only."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import yaml

from proactive.conformal.aps import fit_aps, prediction_sets
from proactive.eval.frontier import matched_budget, pareto_flags, summarize_trajectories
from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.voi import ActionConditionedVOIHead, FrozenDiagnosticPolicy
from proactive.policy.rollout import learned_policy_rollout
from proactive.train.ablations import (
    FEATURE_ABLATIONS,
    apply_teacher_record_ablation,
    validate_week8_ablation_authorization,
)
from proactive.train.checkpoints import (
    POLICY_CHECKPOINT_VERSION,
    load_checkpoint,
    validate_freeze_manifest,
)
from proactive.train.state_data import FeatureNormalizer
from proactive.train.vectorized import load_vectorized_split, project_model_input
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_jsonl, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week8_config", default="configs/experiments/week8_evaluation.yaml")
    parser.add_argument("--ablation_id", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--diagnostic_checkpoint", required=True)
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--aps_report", required=True)
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--teacher_path", default="outputs/teacher_core_contract_v1_recovered")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 3, 4, 7])
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.90, 0.95])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def _threshold(report: Mapping[str, Any], budget: int, coverage: float) -> float:
    item = report.get("thresholds", {}).get(str(budget), {}).get(format(coverage, ".12g"))
    if not isinstance(item, Mapping):
        raise SystemExit(f"APS report lacks coverage={coverage}, budget={budget}")
    return float(item["threshold"])


def _teacher_index(directory: Path) -> tuple[Dict[tuple[str, str], Mapping[str, Any]], list[Dict[str, Any]]]:
    result: Dict[tuple[str, str], Mapping[str, Any]] = {}
    files = []
    for path in sorted(directory.glob("teacher_*.jsonl")):
        if path.name.endswith(".failures.jsonl"):
            continue
        selected = 0
        for row in iter_jsonl(path):
            if row.get("split") != "val":
                continue
            key = (row.get("model_id"), row.get("instance_id"))
            if key in result:
                raise SystemExit(f"Duplicate validation teacher identity: {key}")
            if row.get("valid") is not True:
                raise SystemExit(f"Invalid validation teacher row: {key}")
            result[key] = row
            selected += 1
        files.append({"path": str(path), "sha256": file_sha256(path), "selected_rows": selected})
    if not result:
        raise SystemExit("No core validation teacher rows found")
    return result, files


def _trajectory(row: Mapping[str, Any], condition: str, budget: int, value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "condition": condition,
        "maximum_budget": budget,
        "state_id": row["state_id"],
        "metadata": dict(row["audit_metadata"]),
        "actions": list(value["actions"]),
        "acquisition_cost": int(value["acquisition_cost"]),
        "bit_probabilities": value["bit_probabilities"],
        "six_way_probabilities": value["six_way_probabilities"],
        "signature_prediction": value["signature_prediction"],
        "prediction_set": None,
        "targets": {
            "source_bits": row["targets"]["source_bits"].tolist(),
            "six_way": int(row["targets"]["six_way"]),
            "signature": row["targets"]["signature"].tolist(),
        },
    }


def _csv(rows: list[Dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise SystemExit("Week 8 ablations are approved only for seed 42")
    week8_path = Path(args.week8_config)
    manifest_path = Path(args.manifest_path)
    diagnostic_path = Path(args.diagnostic_checkpoint)
    policy_path = Path(args.policy_checkpoint)
    aps_path = Path(args.aps_report)
    freeze_path = Path(args.freeze_manifest)
    week8 = yaml.safe_load(week8_path.read_text(encoding="utf-8"))
    try:
        validate_week8_ablation_authorization(week8)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.ablation_id != "reference" and args.ablation_id not in week8["ablation_execution"]["mandatory"]:
        raise SystemExit(f"Ablation is not predeclared: {args.ablation_id}")
    if len(args.budgets) != len(set(args.budgets)) or any(value < 1 or value > 7 for value in args.budgets):
        raise SystemExit("Budgets must be unique integers in [1, 7]")
    if not args.coverages or any(not 0 < value < 1 for value in args.coverages):
        raise SystemExit("Coverages must lie in (0, 1)")

    diagnostic = load_checkpoint(diagnostic_path, map_location="cpu")
    policy = load_checkpoint(policy_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu")
    aps = _read_json(aps_path)
    freeze = validate_freeze_manifest(freeze_path, require_policy=False)
    frozen_hashes = {
        item.get("sha256") for item in freeze.get("artifacts", {}).values() if isinstance(item, Mapping)
    }
    for item in freeze.get("artifacts", {}).values():
        if not isinstance(item, Mapping):
            continue
        nested_path = Path(str(item.get("path", "")))
        if not nested_path.is_file() or nested_path.suffix.lower() != ".json":
            continue
        try:
            nested = validate_freeze_manifest(nested_path, require_policy=False)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        frozen_hashes.update(
            artifact.get("sha256")
            for artifact in nested.get("artifacts", {}).values()
            if isinstance(artifact, Mapping)
        )
    for name, path in (("diagnostic", diagnostic_path), ("policy", policy_path), ("vector manifest", manifest_path)):
        if file_sha256(path) not in frozen_hashes:
            raise SystemExit(f"{name} is not hash-bound in the supplied freeze")
    if diagnostic.get("source_manifest_sha256") != file_sha256(manifest_path):
        raise SystemExit("Diagnostic/vector-manifest mismatch")
    if policy.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
        raise SystemExit("Policy/diagnostic mismatch")
    if policy.get("vector_manifest_sha256") != file_sha256(manifest_path):
        raise SystemExit("Policy/vector-manifest mismatch")
    if diagnostic.get("scientifically_valid") is not True or policy.get("scientifically_valid") is not True:
        raise SystemExit("Ablation evaluation requires complete scientific checkpoints")
    if aps.get("format_version") != "temporary_validation_aps_v1" or aps.get("fit_split") != "val":
        raise SystemExit("Ablation evaluation requires validation-only temporary APS")
    if aps.get("calibration_split_used") is not False or aps.get("test_split_used") is not False:
        raise SystemExit("Ablation APS has unsafe split provenance")
    if aps.get("checkpoint_sha256") != file_sha256(diagnostic_path):
        raise SystemExit("APS/diagnostic mismatch")

    expected = {
        "format_version": "week8_ablation_frontier_v1",
        "status": "COMPLETE",
        "ablation_id": args.ablation_id,
        "evaluation_split": "val",
        "calibration_used": False,
        "test_used": False,
        "heldout_shift_used": False,
        "post_test_tuning_used": False,
        "seed": args.seed,
        "week8_config_sha256": file_sha256(week8_path),
        "manifest_sha256": file_sha256(manifest_path),
        "diagnostic_checkpoint_sha256": file_sha256(diagnostic_path),
        "policy_checkpoint_sha256": file_sha256(policy_path),
        "aps_sha256": file_sha256(aps_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "budgets": list(args.budgets),
        "coverages": list(args.coverages),
    }
    output_dir = Path(args.output_dir)
    report_path = output_dir / "frontier_validation.json"
    csv_path = output_dir / "frontier_validation.csv"
    trajectory_path = output_dir / "trajectories_validation.jsonl"
    if report_path.exists() and args.resume:
        existing = _read_json(report_path)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"Ablation frontier resume refused: {key} drift")
        if file_sha256(csv_path) != existing.get("csv_sha256") or file_sha256(trajectory_path) != existing.get("trajectory_sha256"):
            raise SystemExit("Ablation frontier resume artifact drift")
        print(json.dumps(existing, indent=2))
        return
    if report_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {report_path}; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({**expected, "output": str(report_path)}, indent=2))
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA requested but unavailable: {args.device}")
    diagnostic_model = build_diagnostic_model(diagnostic["encoder_name"], diagnostic["architecture"])
    diagnostic_model.load_state_dict(diagnostic["model_state_dict"])
    architecture = policy["policy_architecture"]
    head = ActionConditionedVOIHead(
        state_dim=int(diagnostic["architecture"]["state_dim"]),
        action_embedding_dim=int(architecture["action_embedding_dim"]),
        budget_embedding_dim=int(architecture["budget_embedding_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        dropout=float(architecture["dropout"]),
        max_budget=int(diagnostic["architecture"]["max_budget"]),
    )
    head.load_state_dict(policy["model_state_dict"])
    model = FrozenDiagnosticPolicy(diagnostic_model, head).to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(diagnostic["normalizer"])
    base = load_vectorized_split(manifest_path, "val")
    indices = torch.nonzero(base.model_input["acquired_mask"].sum(dim=1) == 0, as_tuple=False).flatten().tolist()
    indices.sort(key=lambda index: hash_dict({"seed": args.seed, "state_id": base.state_ids[index]}))
    teacher, teacher_files = _teacher_index(Path(args.teacher_path))
    grouped: Dict[tuple[str, int], list[Dict[str, Any]]] = defaultdict(list)
    feature_ablation = args.ablation_id if args.ablation_id in FEATURE_ABLATIONS else None
    for index in indices:
        row = base[index]
        identity = (row["audit_metadata"]["model_id"], row["audit_metadata"]["instance_id"])
        source_teacher = teacher.get(identity)
        if source_teacher is None:
            raise SystemExit(f"Missing validation teacher row for {identity}")
        if feature_ablation is not None:
            source_teacher = apply_teacher_record_ablation(source_teacher, feature_ablation)
        for budget in args.budgets:
            initial = project_model_input(row["model_input"], budget)
            learned = learned_policy_rollout(
                policy_model=model,
                initial_model_input=initial,
                teacher_record=source_teacher,
                normalizer=normalizer,
                device=device,
            )
            grouped[("proactive", budget)].append(_trajectory(row, "proactive", budget, learned))
            full = learned_policy_rollout(
                policy_model=model,
                initial_model_input=initial,
                teacher_record=source_teacher,
                normalizer=normalizer,
                device=device,
                force_full_budget=True,
            )
            grouped[("proactive_no_stop", budget)].append(
                _trajectory(row, "proactive_no_stop", budget, full)
            )

    global_thresholds: Dict[str, float] = {}
    for coverage in args.coverages:
        probabilities = [
            row["six_way_probabilities"]
            for budget in args.budgets
            for row in grouped[("proactive", budget)]
        ]
        labels = [
            int(row["targets"]["six_way"])
            for budget in args.budgets
            for row in grouped[("proactive", budget)]
        ]
        global_thresholds[format(coverage, ".12g")] = float(
            fit_aps(probabilities, labels, alpha=1.0 - coverage).threshold
        )

    summary_rows: list[Dict[str, Any]] = []
    trajectory_rows: list[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}
    for condition in ("proactive", "proactive_no_stop", "proactive_global_aps"):
        source_condition = "proactive" if condition == "proactive_global_aps" else condition
        for budget in args.budgets:
            source_rows = grouped[(source_condition, budget)]
            for coverage in args.coverages:
                threshold = (
                    global_thresholds[format(coverage, ".12g")]
                    if condition == "proactive_global_aps"
                    else _threshold(aps, budget, coverage)
                )
                rows = []
                for source in source_rows:
                    row = {**source, "condition": condition}
                    row["prediction_set"] = prediction_sets(
                        [row["six_way_probabilities"]], threshold
                    )[0]
                    rows.append(row)
                    trajectory_rows.append({**row, "target_coverage": coverage})
                matched_budget(rows, budget)
                summary = summarize_trajectories(rows)
                summaries[f"{condition}|{budget}|{format(coverage, '.12g')}"] = summary
                set_metrics = summary["prediction_sets"]
                summary_rows.append(
                    {
                        "condition": condition,
                        "maximum_budget": budget,
                        "target_coverage": coverage,
                        "row_count": summary["row_count"],
                        "mean_acquisition_cost": summary["mean_acquisition_cost"],
                        "source_bit_macro_f1": summary["diagnostic"]["source_bit_macro_f1"],
                        "source_bit_micro_f1": summary["diagnostic"]["source_bit_micro_f1"],
                        "six_way_macro_f1": summary["diagnostic"]["six_way_macro_f1"],
                        "coverage": set_metrics["coverage"],
                        "average_set_size": set_metrics["average_set_size"],
                    }
                )
    for coverage in args.coverages:
        selected = [index for index, row in enumerate(summary_rows) if row["target_coverage"] == coverage]
        flags = pareto_flags([summary_rows[index] for index in selected])
        for index, flag in zip(selected, flags):
            summary_rows[index]["pareto_nondominated"] = flag

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        trajectory_rows,
        trajectory_path,
        overwrite=(args.overwrite or args.resume) and trajectory_path.exists(),
    )
    write_text(
        _csv(summary_rows),
        csv_path,
        overwrite=(args.overwrite or args.resume) and csv_path.exists(),
    )
    report: Dict[str, Any] = {
        **expected,
        "base_model_instances": len(indices),
        "teacher_files": teacher_files,
        "global_aps_fit_split": "val",
        "global_aps_evaluation_split": "val",
        "global_aps_thresholds": global_thresholds,
        "summaries": summaries,
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": file_sha256(trajectory_path),
    }
    report["report_sha256"] = hash_dict(report)
    write_json(
        report,
        report_path,
        overwrite=(args.overwrite or args.resume) and report_path.exists(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
