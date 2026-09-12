#!/usr/bin/env python3
"""Evaluate learned and cached baselines at matched maximum budgets."""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from proactive.conformal.aps import prediction_sets
from proactive.conformal.contracts import authorize_locked_test
from proactive.eval.baselines import FixedSchedulePolicy, RandomPolicy
from proactive.eval.frontier import matched_budget, pareto_flags, summarize_trajectories
from proactive.networks.controls import ScalarConfidenceDiagnostic
from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.voi import ActionConditionedVOIHead, FrozenDiagnosticPolicy
from proactive.policy.rollout import (
    baseline_policy_rollout,
    diagnostic_snapshot,
    learned_policy_rollout,
    oracle_best_subset_rollout,
    oracle_next_rollout,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--diagnostic_checkpoint", required=True)
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--uncertainty_checkpoint", required=True)
    parser.add_argument("--scalar_checkpoint", required=True)
    parser.add_argument("--distilled_checkpoint", required=True)
    parser.add_argument("--clean_aps_report", required=True)
    parser.add_argument("--scalar_aps_report", required=True)
    parser.add_argument("--distilled_aps_report", required=True)
    parser.add_argument("--dataset_schedule_report", default=None)
    parser.add_argument("--aps_report", required=True)
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--teacher_path", default="outputs/teacher_core_contract_v1_recovered")
    parser.add_argument(
        "--phase",
        choices=("validation", "test", "shift"),
        required=True,
        help=(
            "shift applies the immutable Week 7 stack and calibration thresholds "
            "to held-out datasets without fitting anything on the target domain"
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Validation-only budget override. Use this once after policy selection to "
            "select the frozen dataset-specific schedule at every final Week 7 budget."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_oracle_subset", action="store_true")
    parser.add_argument("--skip_oracles", action="store_true")
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


def _teacher_index(directory: Path, split: str) -> tuple[Dict[tuple[str, str], Mapping[str, Any]], list[Dict[str, Any]]]:
    rows: Dict[tuple[str, str], Mapping[str, Any]] = {}
    files = []
    for path in sorted(directory.glob("teacher_*.jsonl")):
        if path.name.endswith(".failures.jsonl"):
            continue
        count = 0
        for record in iter_jsonl(path):
            if record.get("split") != split:
                continue
            key = (record.get("model_id"), record.get("instance_id"))
            if key in rows:
                raise SystemExit(f"Duplicate teacher identity: {key}")
            if record.get("valid") is not True:
                raise SystemExit(f"Invalid recovered teacher row: {key}")
            rows[key] = record
            count += 1
        files.append({"path": str(path), "sha256": file_sha256(path), "selected_rows": count})
    if not rows:
        raise SystemExit(f"No {split} teacher rows found")
    return rows, files


def _threshold(report: Mapping[str, Any], budget: int, coverage: float) -> float:
    item = report.get("thresholds", {}).get(str(budget), {}).get(format(coverage, ".12g"))
    if not isinstance(item, Mapping):
        raise SystemExit(f"APS report lacks coverage={coverage}, budget={budget}")
    return float(item["threshold"])


def _trajectory_record(
    row: Mapping[str, Any],
    condition: str,
    budget: int,
    trajectory: Mapping[str, Any],
    threshold: float | None,
) -> Dict[str, Any]:
    probability = trajectory["six_way_probabilities"]
    return {
        "condition": condition,
        "maximum_budget": budget,
        "state_id": row["state_id"],
        "metadata": dict(row["audit_metadata"]),
        "actions": list(trajectory["actions"]),
        "acquisition_cost": int(trajectory["acquisition_cost"]),
        "bit_probabilities": trajectory["bit_probabilities"],
        "six_way_probabilities": probability,
        "signature_prediction": trajectory["signature_prediction"],
        "prediction_set": (
            prediction_sets([probability], threshold)[0] if threshold is not None else None
        ),
        "targets": {
            "source_bits": row["targets"]["source_bits"].tolist(),
            "six_way": int(row["targets"]["six_way"]),
            "signature": row["targets"]["signature"].tolist(),
        },
    }


def _csv(rows: list[Dict[str, Any]]) -> str:
    import csv

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _signed_report(path: Path) -> Dict[str, Any]:
    value = _read_json(path)
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    if value.get("report_sha256") != hash_dict(unsigned):
        raise SystemExit(f"Report self-hash mismatch: {path}")
    return value


def _validate_static_aps(
    *,
    path: Path,
    baseline: str,
    checkpoint_path: Path,
    phase: str,
    freeze_path: Path,
) -> Dict[str, Any]:
    report = _signed_report(path)
    expected_split = "val" if phase == "validation" else "cal"
    expected_status = "TEMPORARY_VALIDATION" if phase == "validation" else "FINAL_FROZEN"
    if (
        report.get("format_version") != "static_baseline_aps_v1"
        or report.get("status") != expected_status
        or report.get("baseline") != baseline
        or report.get("fit_split") != expected_split
        or report.get("test_used") is not False
        or report.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or report.get("limit") is not None
    ):
        raise SystemExit(f"Static APS contract mismatch: {path}")
    if phase in {"test", "shift"} and report.get("freeze_manifest_sha256") != file_sha256(freeze_path):
        raise SystemExit(f"Final static APS/freeze mismatch: {path}")
    return report


def _source_macro_f1(rows: list[Dict[str, Any]]) -> float:
    if not rows:
        raise SystemExit("Cannot score an empty dataset-specific schedule slice")
    truth = np.asarray([row["targets"]["source_bits"] for row in rows], dtype=np.int64)
    prediction = (
        np.asarray([row["bit_probabilities"] for row in rows], dtype=np.float64) >= 0.5
    ).astype(np.int64)
    return float(f1_score(truth, prediction, average="macro", zero_division=0))


def _select_dataset_schedules(
    grouped: Mapping[tuple[str, int], list[Dict[str, Any]]],
    *,
    fixed: list[str],
    budgets: list[int],
) -> tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    datasets = sorted(
        {
            str(row["metadata"]["dataset"])
            for key, rows in grouped.items()
            if key[0] in fixed
            for row in rows
        }
    )
    mapping: Dict[str, Dict[str, str]] = {}
    evidence: Dict[str, Any] = {}
    for budget in budgets:
        mapping[str(budget)] = {}
        evidence[str(budget)] = {}
        for dataset in datasets:
            scores = {
                schedule: _source_macro_f1(
                    [
                        row
                        for row in grouped[(schedule, budget)]
                        if row["metadata"]["dataset"] == dataset
                    ]
                )
                for schedule in fixed
            }
            selected = max(fixed, key=lambda schedule: scores[schedule])
            mapping[str(budget)][dataset] = selected
            evidence[str(budget)][dataset] = {
                "selected_schedule": selected,
                "validation_source_bit_macro_f1": scores,
            }
    return mapping, evidence


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.phase in {"test", "shift"} and args.budgets is not None:
        raise SystemExit("Locked test/shift budgets come only from the frozen APS report")
    config_path = Path(args.config)
    manifest_path = Path(args.manifest_path)
    diagnostic_path = Path(args.diagnostic_checkpoint)
    policy_path = Path(args.policy_checkpoint)
    uncertainty_path = Path(args.uncertainty_checkpoint)
    scalar_path = Path(args.scalar_checkpoint)
    distilled_path = Path(args.distilled_checkpoint)
    clean_aps_path = Path(args.clean_aps_report)
    scalar_aps_path = Path(args.scalar_aps_report)
    distilled_aps_path = Path(args.distilled_aps_report)
    aps_path = Path(args.aps_report)
    freeze_path = Path(args.freeze_manifest)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    diagnostic = load_checkpoint(diagnostic_path, map_location="cpu")
    policy = load_checkpoint(policy_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu")
    uncertainty = load_checkpoint(
        uncertainty_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu"
    )
    scalar = load_checkpoint(scalar_path, map_location="cpu")
    distilled = load_checkpoint(distilled_path, map_location="cpu")
    aps = _read_json(aps_path)
    split = {"validation": "val", "test": "test", "shift": "shift"}[args.phase]
    freeze = validate_freeze_manifest(
        freeze_path, require_policy=args.phase in {"test", "shift"}
    )
    week5_freeze = freeze
    if args.phase in {"test", "shift"}:
        nested = freeze.get("artifacts", {}).get("week5_freeze")
        if not isinstance(nested, Mapping):
            raise SystemExit("Main freeze does not contain the Week 5 freeze contract")
        week5_freeze = validate_freeze_manifest(Path(nested["path"]), require_policy=False)
    clean_item = week5_freeze.get("artifacts", {}).get(
        "diagnostic_checkpoint_clean_mlp_standard"
    )
    if not isinstance(clean_item, Mapping):
        raise SystemExit("Week 5 freeze lacks the mandatory clean-only checkpoint")
    clean_path = Path(clean_item["path"])
    if file_sha256(clean_path) != clean_item.get("sha256"):
        raise SystemExit("Frozen clean-only checkpoint drift")
    clean_checkpoint = load_checkpoint(clean_path, map_location="cpu")
    if clean_checkpoint.get("encoder_name") != "clean_mlp":
        raise SystemExit("Frozen clean-only artifact has the wrong encoder")
    training_manifest_sha = policy.get("vector_manifest_sha256")
    if args.phase == "shift":
        if clean_checkpoint.get("source_manifest_sha256") != training_manifest_sha:
            raise SystemExit("Frozen clean-only/training-manifest mismatch")
    elif clean_checkpoint.get("source_manifest_sha256") != file_sha256(manifest_path):
        raise SystemExit("Clean-only/vectorized-manifest mismatch")
    dataset_schedule_path = Path(args.dataset_schedule_report) if args.dataset_schedule_report else None
    dataset_schedule = None
    if args.phase == "validation":
        if aps.get("format_version") != "temporary_validation_aps_v1" or aps.get("fit_split") != "val":
            raise SystemExit("Validation frontier requires validation-temporary APS")
        if not any(
            name.startswith("diagnostic_checkpoint") and item.get("sha256") == file_sha256(diagnostic_path)
            for name, item in freeze["artifacts"].items()
        ):
            raise SystemExit("Validation diagnostic checkpoint is not in the Week 5 freeze")
        budgets = (
            [int(value) for value in args.budgets]
            if args.budgets is not None
            else [int(value) for value in config["budgets"]]
        )
        if len(budgets) != len(set(budgets)) or any(budget < 1 or budget > 7 for budget in budgets):
            raise SystemExit("Validation budgets must be unique integers in [1, 7]")
        aps_budgets = {int(value) for value in aps.get("budgets", [])}
        if not set(budgets).issubset(aps_budgets):
            raise SystemExit("Validation frontier requested a budget absent from temporary APS")
        coverage = float(config["temporary_aps_target_coverage"])
        eta_loss = float(config["eta_loss"])
        expected_policy_config_sha = file_sha256(config_path)
    else:
        if args.phase == "test" and args.limit is not None:
            raise SystemExit("Locked test evaluation refuses --limit")
        if args.phase == "test" and args.skip_oracle_subset:
            raise SystemExit("Locked test requires the mandatory oracle-best-subset control")
        if args.phase == "test" and args.skip_oracles:
            raise SystemExit("Locked test requires both mandatory oracle controls")
        if args.phase == "test":
            try:
                authorize_locked_test(freeze_manifest_path=freeze_path, aps_report=aps)
            except ValueError as exc:
                raise SystemExit(f"Test evaluation lock refused: {exc}") from exc
        elif aps.get("status") != "FINAL_FROZEN" or aps.get("fit_split") != "cal":
            raise SystemExit("Shift evaluation requires the final core-calibration APS report")
        if freeze["artifacts"]["diagnostic_checkpoint"]["sha256"] != file_sha256(diagnostic_path):
            raise SystemExit("Test diagnostic checkpoint is not the frozen primary")
        if freeze["artifacts"]["policy_checkpoint"]["sha256"] != file_sha256(policy_path):
            raise SystemExit("Test policy checkpoint is not frozen")
        if freeze["artifacts"].get("uncertainty_baseline", {}).get("sha256") != file_sha256(uncertainty_path):
            raise SystemExit("Test uncertainty baseline is not frozen")
        if freeze["artifacts"].get("scalar_baseline", {}).get("sha256") != file_sha256(scalar_path):
            raise SystemExit("Test scalar baseline is not frozen")
        if freeze["artifacts"].get("distilled_baseline", {}).get("sha256") != file_sha256(distilled_path):
            raise SystemExit("Test distilled baseline is not frozen")
        budgets = [int(value) for value in aps.get("budgets", [])]
        if not budgets:
            raise SystemExit("Final APS report does not declare evaluation budgets")
        coverage = 0.90
        eta_loss = 0.25
        expected_policy_config_sha = policy["config_sha256"]
        if args.phase == "test" and dataset_schedule_path is None:
            raise SystemExit("Locked test requires --dataset_schedule_report selected on validation")
        if args.phase == "test":
            dataset_schedule = _signed_report(dataset_schedule_path)
            if (
                dataset_schedule.get("format_version") != "dataset_fixed_schedule_v1"
                or dataset_schedule.get("fit_split") != "val"
                or dataset_schedule.get("calibration_used") is not False
                or dataset_schedule.get("test_used") is not False
            ):
                raise SystemExit("Dataset-specific schedule is not a validation-only selection artifact")
            frozen_schedule = freeze["artifacts"].get("dataset_fixed_schedule")
            if not frozen_schedule or frozen_schedule.get("sha256") != file_sha256(dataset_schedule_path):
                raise SystemExit("Dataset-specific schedule is not bound into the main freeze")
    if policy.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
        raise SystemExit("Policy/diagnostic checkpoint mismatch")
    if uncertainty.get("stage") != "week6_uncertainty_baseline" or uncertainty.get("target_kind") != "entropy_reduction":
        raise SystemExit("Uncertainty baseline checkpoint contract mismatch")
    if uncertainty.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
        raise SystemExit("Uncertainty/diagnostic checkpoint mismatch")
    if args.phase != "shift" and policy.get("vector_manifest_sha256") != file_sha256(manifest_path):
        raise SystemExit("Policy/vectorized-manifest mismatch")
    if policy.get("config_sha256") != expected_policy_config_sha:
        raise SystemExit("Policy config hash mismatch")
    for name, checkpoint, path, expected_stage in (
        ("scalar", scalar, scalar_path, "week6_scalar_baseline"),
        ("distilled", distilled, distilled_path, "week6_distilled_baseline"),
    ):
        if checkpoint.get("stage") != expected_stage:
            raise SystemExit(f"{name} baseline checkpoint stage mismatch")
        if checkpoint.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
            raise SystemExit(f"{name} baseline/diagnostic checkpoint mismatch")
        expected_source_sha = training_manifest_sha if args.phase == "shift" else file_sha256(manifest_path)
        if checkpoint.get("source_manifest_sha256") != expected_source_sha:
            raise SystemExit(f"{name} baseline/vectorized-manifest mismatch")
        if args.limit is None and checkpoint.get("scientifically_valid") is not True:
            raise SystemExit(f"Full frontier requires a scientifically valid {name} baseline")
    static_aps = {
        "clean_only_learned": _validate_static_aps(
            path=clean_aps_path,
            baseline="clean_only_learned",
            checkpoint_path=clean_path,
            phase=args.phase,
            freeze_path=freeze_path,
        ),
        "scalar_confidence": _validate_static_aps(
            path=scalar_aps_path,
            baseline="scalar_confidence",
            checkpoint_path=scalar_path,
            phase=args.phase,
            freeze_path=freeze_path,
        ),
        "one_pass_distilled": _validate_static_aps(
            path=distilled_aps_path,
            baseline="one_pass_distilled",
            checkpoint_path=distilled_path,
            phase=args.phase,
            freeze_path=freeze_path,
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"frontier_{args.phase}.json"
    csv_path = output_dir / f"frontier_{args.phase}.csv"
    expected = {
        "format_version": "proactive_frontier_v1",
        "phase": args.phase,
        "split": split,
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "diagnostic_checkpoint_sha256": file_sha256(diagnostic_path),
        "encoder_name": diagnostic.get("encoder_name"),
        "gru_condition": diagnostic.get("gru_condition", "standard"),
        "clean_checkpoint_sha256": file_sha256(clean_path),
        "clean_checkpoint_path": str(clean_path),
        "policy_checkpoint_sha256": file_sha256(policy_path),
        "policy_checkpoint_path": str(policy_path),
        "uncertainty_checkpoint_sha256": file_sha256(uncertainty_path),
        "uncertainty_checkpoint_path": str(uncertainty_path),
        "scalar_checkpoint_sha256": file_sha256(scalar_path),
        "scalar_checkpoint_path": str(scalar_path),
        "distilled_checkpoint_sha256": file_sha256(distilled_path),
        "distilled_checkpoint_path": str(distilled_path),
        "clean_aps_sha256": file_sha256(clean_aps_path),
        "scalar_aps_sha256": file_sha256(scalar_aps_path),
        "distilled_aps_sha256": file_sha256(distilled_aps_path),
        "aps_sha256": file_sha256(aps_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "evaluation_budgets": budgets,
        "limit": args.limit,
        "oracle_subset_included": not args.skip_oracle_subset,
        "oracle_next_included": not args.skip_oracles,
        "target_domain_calibration_used": False if args.phase == "shift" else None,
        "coverage_claim": "empirical_shift_only" if args.phase == "shift" else None,
    }
    if dataset_schedule_path is not None:
        expected["dataset_schedule_sha256"] = file_sha256(dataset_schedule_path)
    if report_path.exists() and args.resume:
        existing = _read_json(report_path)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"Frontier resume refused: {key} drift")
        if not csv_path.exists() or file_sha256(csv_path) != existing.get("csv_sha256"):
            raise SystemExit("Frontier CSV drift")
        figure_path = Path(existing.get("figure_path", ""))
        if not figure_path.exists() or file_sha256(figure_path) != existing.get("figure_sha256"):
            raise SystemExit("Frontier figure drift")
        if existing.get("dataset_schedule_path"):
            schedule_path = Path(existing["dataset_schedule_path"])
            if not schedule_path.exists() or file_sha256(schedule_path) != existing.get("dataset_schedule_sha256"):
                raise SystemExit("Dataset-specific schedule drift")
        trajectory_path = Path(existing.get("trajectory_path", ""))
        if not trajectory_path.exists() or file_sha256(trajectory_path) != existing.get("trajectory_sha256"):
            raise SystemExit("Frontier trajectory-record drift")
        print(json.dumps(existing, indent=2))
        return
    if (report_path.exists() or csv_path.exists()) and not (args.resume or args.overwrite):
        raise SystemExit(f"Frontier outputs exist for {args.phase}; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({**expected, "budgets": budgets, "coverage": coverage, "output": str(report_path)}, indent=2))
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
        max_budget=int(diagnostic["architecture"]["max_budget"]),
    )
    head.load_state_dict(policy["model_state_dict"])
    learned_model = FrozenDiagnosticPolicy(diagnostic_model, head).to(device).eval()
    uncertainty_architecture = uncertainty["policy_architecture"]
    uncertainty_head = ActionConditionedVOIHead(
        state_dim=int(diagnostic["architecture"]["state_dim"]),
        action_embedding_dim=int(uncertainty_architecture["action_embedding_dim"]),
        budget_embedding_dim=int(uncertainty_architecture["budget_embedding_dim"]),
        hidden_dim=int(uncertainty_architecture["hidden_dim"]),
        dropout=float(uncertainty_architecture["dropout"]),
        max_budget=int(diagnostic["architecture"]["max_budget"]),
    )
    uncertainty_head.load_state_dict(uncertainty["model_state_dict"])
    uncertainty_model = FrozenDiagnosticPolicy(diagnostic_model, uncertainty_head).to(device).eval()
    clean_model = build_diagnostic_model("clean_mlp", clean_checkpoint["architecture"])
    clean_model.load_state_dict(clean_checkpoint["model_state_dict"])
    clean_model.to(device).eval()
    clean_normalizer = FeatureNormalizer.from_state_dict(clean_checkpoint["normalizer"])
    scalar_model = ScalarConfidenceDiagnostic(
        state_dim=int(scalar["architecture"]["state_dim"]),
        dropout=float(scalar["architecture"]["dropout"]),
    )
    scalar_model.load_state_dict(scalar["model_state_dict"])
    scalar_model.to(device).eval()
    distilled_model = build_diagnostic_model("clean_mlp", distilled["architecture"])
    distilled_model.load_state_dict(distilled["model_state_dict"])
    distilled_model.to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(diagnostic["normalizer"])
    base = load_vectorized_split(manifest_path, split)
    empty_indices = torch.nonzero(base.model_input["acquired_mask"].sum(dim=1) == 0, as_tuple=False).flatten().tolist()
    empty_indices.sort(key=lambda index: hash_dict({"seed": args.seed, "state_id": base.state_ids[index]}))
    if args.limit is not None:
        empty_indices = empty_indices[: args.limit]
    teacher, teacher_files = _teacher_index(Path(args.teacher_path), split)
    grouped: Dict[tuple[str, int], list[Dict[str, Any]]] = defaultdict(list)
    fixed = ["blank_first", "visual_first", "grounding_first", "relation_first"]
    for index in empty_indices:
        row = base[index]
        identity = (row["audit_metadata"]["model_id"], row["audit_metadata"]["instance_id"])
        if identity not in teacher:
            raise SystemExit(f"Missing teacher row for {identity}")
        source_teacher = teacher[identity]
        for budget in budgets:
            threshold = _threshold(aps, budget, coverage)
            initial = project_model_input(row["model_input"], budget)
            clean_snapshot = diagnostic_snapshot(
                diagnostic_model=diagnostic_model,
                model_input=initial,
                normalizer=normalizer,
                device=device,
            )
            clean_trajectory = {
                "actions": ["stop"],
                "acquisition_cost": 0,
                **clean_snapshot,
            }
            grouped[("diagnostic_no_acquisition", budget)].append(
                _trajectory_record(
                    row,
                    "diagnostic_no_acquisition",
                    budget,
                    clean_trajectory,
                    threshold,
                )
            )
            clean_only_snapshot = diagnostic_snapshot(
                diagnostic_model=clean_model,
                model_input=initial,
                normalizer=clean_normalizer,
                device=device,
            )
            grouped[("clean_only_learned", budget)].append(
                _trajectory_record(
                    row,
                    "clean_only_learned",
                    budget,
                    {"actions": ["stop"], "acquisition_cost": 0, **clean_only_snapshot},
                    _threshold(static_aps["clean_only_learned"], budget, coverage),
                )
            )
            for condition, baseline_model in (
                ("scalar_confidence", scalar_model),
                ("one_pass_distilled", distilled_model),
            ):
                snapshot = diagnostic_snapshot(
                    diagnostic_model=baseline_model,
                    model_input=initial,
                    normalizer=normalizer,
                    device=device,
                )
                grouped[(condition, budget)].append(
                    _trajectory_record(
                        row,
                        condition,
                        budget,
                        {"actions": ["stop"], "acquisition_cost": 0, **snapshot},
                        _threshold(static_aps[condition], budget, coverage),
                    )
                )
            learned = learned_policy_rollout(
                policy_model=learned_model,
                initial_model_input=initial,
                teacher_record=source_teacher,
                normalizer=normalizer,
                device=device,
            )
            grouped[("proactive", budget)].append(_trajectory_record(row, "proactive", budget, learned, threshold))
            no_stop = learned_policy_rollout(
                policy_model=learned_model,
                initial_model_input=initial,
                teacher_record=source_teacher,
                normalizer=normalizer,
                device=device,
                force_full_budget=True,
            )
            grouped[("proactive_no_stop", budget)].append(
                _trajectory_record(row, "proactive_no_stop", budget, no_stop, threshold)
            )
            uncertainty_trajectory = learned_policy_rollout(
                policy_model=uncertainty_model,
                initial_model_input=initial,
                teacher_record=source_teacher,
                normalizer=normalizer,
                device=device,
            )
            grouped[("uncertainty_greedy", budget)].append(
                _trajectory_record(
                    row,
                    "uncertainty_greedy",
                    budget,
                    uncertainty_trajectory,
                    threshold,
                )
            )
            random = baseline_policy_rollout(
                acquisition_policy=RandomPolicy(seed=args.seed),
                diagnostic_model=diagnostic_model,
                initial_model_input=initial,
                teacher_record=source_teacher,
                normalizer=normalizer,
                device=device,
                instance_id=row["audit_metadata"]["instance_id"],
            )
            grouped[("random", budget)].append(_trajectory_record(row, "random", budget, random, threshold))
            for schedule in fixed:
                trajectory = baseline_policy_rollout(
                    acquisition_policy=FixedSchedulePolicy(schedule),
                    diagnostic_model=diagnostic_model,
                    initial_model_input=initial,
                    teacher_record=source_teacher,
                    normalizer=normalizer,
                    device=device,
                    instance_id=row["audit_metadata"]["instance_id"],
                )
                grouped[(schedule, budget)].append(_trajectory_record(row, schedule, budget, trajectory, threshold))
            if dataset_schedule is not None:
                dataset = str(row["audit_metadata"]["dataset"])
                try:
                    schedule = str(dataset_schedule["mapping"][str(budget)][dataset])
                except KeyError as exc:
                    raise SystemExit(
                        f"Dataset-specific schedule lacks budget={budget}, dataset={dataset}"
                    ) from exc
                trajectory = baseline_policy_rollout(
                    acquisition_policy=FixedSchedulePolicy(schedule),
                    diagnostic_model=diagnostic_model,
                    initial_model_input=initial,
                    teacher_record=source_teacher,
                    normalizer=normalizer,
                    device=device,
                    instance_id=row["audit_metadata"]["instance_id"],
                )
                grouped[("dataset_specific_fixed", budget)].append(
                    _trajectory_record(row, "dataset_specific_fixed", budget, trajectory, threshold)
                )
            if not args.skip_oracles:
                oracle_next = oracle_next_rollout(
                    diagnostic_model=diagnostic_model,
                    initial_model_input=initial,
                    targets=row["targets"],
                    teacher_record=source_teacher,
                    normalizer=normalizer,
                    device=device,
                    aps_threshold=threshold,
                    eta_loss=eta_loss,
                    cost_multiplier=float(policy["cost_multiplier"]),
                    bit_pos_weight=diagnostic["bit_pos_weight"],
                    class_weight=diagnostic["class_weight"],
                    six_way_weight=0.5,
                    signature_weight=0.1,
                )
                grouped[("oracle_next", budget)].append(
                    _trajectory_record(row, "oracle_next", budget, oracle_next, threshold)
                )
            if not args.skip_oracles and not args.skip_oracle_subset:
                oracle_subset = oracle_best_subset_rollout(
                    diagnostic_model=diagnostic_model,
                    initial_model_input=initial,
                    targets=row["targets"],
                    teacher_record=source_teacher,
                    normalizer=normalizer,
                    device=device,
                    aps_threshold=threshold,
                    bit_pos_weight=diagnostic["bit_pos_weight"],
                    class_weight=diagnostic["class_weight"],
                    six_way_weight=0.5,
                    signature_weight=0.1,
                )
                grouped[("oracle_best_subset", budget)].append(_trajectory_record(row, "oracle_best_subset", budget, oracle_subset, threshold))
        full_budget = 7
        full_threshold = _threshold(aps, full_budget, coverage)
        full_initial = project_model_input(row["model_input"], full_budget)
        full = baseline_policy_rollout(
            acquisition_policy=FixedSchedulePolicy("blank_first"),
            diagnostic_model=diagnostic_model,
            initial_model_input=full_initial,
            teacher_record=source_teacher,
            normalizer=normalizer,
            device=device,
            instance_id=row["audit_metadata"]["instance_id"],
        )
        grouped[("full_teacher", full_budget)].append(
            _trajectory_record(row, "full_teacher", full_budget, full, full_threshold)
        )
    dataset_schedule_output = dataset_schedule_path
    if args.phase == "validation":
        mapping, evidence = _select_dataset_schedules(grouped, fixed=fixed, budgets=budgets)
        for budget in budgets:
            composite = []
            for schedule in fixed:
                composite.extend(
                    row
                    for row in grouped[(schedule, budget)]
                    if mapping[str(budget)][str(row["metadata"]["dataset"])] == schedule
                )
            grouped[("dataset_specific_fixed", budget)] = composite
        dataset_schedule_output = output_dir / "dataset_fixed_schedule_validation.json"
        schedule_report: Dict[str, Any] = {
            "format_version": "dataset_fixed_schedule_v1",
            "status": "SELECTED",
            "fit_split": "val",
            "selection_metric": "source_bit_macro_f1",
            "calibration_used": False,
            "test_used": False,
            "config_sha256": file_sha256(config_path),
            "manifest_sha256": file_sha256(manifest_path),
            "budgets": budgets,
            "mapping": mapping,
            "evidence": evidence,
        }
        schedule_report["report_sha256"] = hash_dict(schedule_report)
        write_json(
            schedule_report,
            dataset_schedule_output,
            overwrite=(args.overwrite or args.resume) and dataset_schedule_output.exists(),
        )
    evaluation_coverages = (
        [coverage]
        if args.phase == "validation"
        else [float(value) for value in aps["coverages"]]
    )
    coverage_grouped: Dict[tuple[str, int, float], list[Dict[str, Any]]] = {}
    for (condition, budget), rows in grouped.items():
        threshold_report = static_aps.get(condition, aps)
        for target_coverage in evaluation_coverages:
            threshold = _threshold(threshold_report, budget, target_coverage)
            calibrated_rows = []
            for row in rows:
                calibrated = dict(row)
                calibrated["prediction_set"] = prediction_sets(
                    [row["six_way_probabilities"]], threshold
                )[0]
                calibrated_rows.append(calibrated)
            coverage_grouped[(condition, budget, target_coverage)] = calibrated_rows
    summary_rows = []
    summaries: Dict[str, Any] = {}
    for (condition, budget, target_coverage), rows in sorted(coverage_grouped.items()):
        matched_budget(rows, budget)
        summary = summarize_trajectories(rows)
        set_metrics = summary["prediction_sets"]
        key = f"{condition}|{budget}|{format(target_coverage, '.12g')}"
        summaries[key] = summary
        summary_rows.append(
            {
                "condition": condition,
                "maximum_budget": budget,
                "target_coverage": target_coverage,
                "row_count": summary["row_count"],
                "mean_acquisition_cost": summary["mean_acquisition_cost"],
                "source_bit_macro_f1": summary["diagnostic"]["source_bit_macro_f1"],
                "source_bit_micro_f1": summary["diagnostic"]["source_bit_micro_f1"],
                "six_way_macro_f1": summary["diagnostic"]["six_way_macro_f1"],
                "coverage": set_metrics["coverage"] if set_metrics is not None else None,
                "average_set_size": set_metrics["average_set_size"] if set_metrics is not None else None,
            }
        )
    for target_coverage in evaluation_coverages:
        indices = [
            index
            for index, row in enumerate(summary_rows)
            if float(row["target_coverage"]) == target_coverage
        ]
        flags = pareto_flags([summary_rows[index] for index in indices])
        for index, flag in zip(indices, flags):
            summary_rows[index]["pareto_nondominated"] = flag
    trajectory_path = output_dir / f"trajectories_{args.phase}.jsonl"
    trajectory_rows = []
    for (condition, budget, target_coverage), rows in sorted(coverage_grouped.items()):
        for row in rows:
            trajectory_rows.append(
                {
                    **row,
                    "target_coverage": target_coverage,
                    "coverage_claim": (
                        "empirical_shift_only" if args.phase == "shift" else "core_calibrated"
                    ),
                }
            )
    write_jsonl(
        trajectory_rows,
        trajectory_path,
        overwrite=(args.overwrite or args.resume) and trajectory_path.exists(),
    )
    write_text(_csv(summary_rows), csv_path, overwrite=(args.overwrite or args.resume) and csv_path.exists())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path = output_dir / f"frontier_{args.phase}.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    plotted_conditions = ["proactive", "random", "dataset_specific_fixed", *fixed]
    for condition in plotted_conditions:
        condition_rows = sorted(
            [
                row
                for row in summary_rows
                if row["condition"] == condition and float(row["target_coverage"]) == 0.90
            ],
            key=lambda row: float(row["mean_acquisition_cost"]),
        )
        if not condition_rows:
            continue
        x = [row["mean_acquisition_cost"] for row in condition_rows]
        axes[0].plot(x, [row["source_bit_macro_f1"] for row in condition_rows], marker="o", label=condition)
        set_rows = [row for row in condition_rows if row["average_set_size"] is not None]
        if set_rows:
            axes[1].plot(
                [row["mean_acquisition_cost"] for row in set_rows],
                [row["average_set_size"] for row in set_rows],
                marker="o",
                label=condition,
            )
    axes[0].set(xlabel="Mean acquired probes", ylabel="Source-bit Macro-F1", title="Diagnostic frontier")
    axes[1].set(xlabel="Mean acquired probes", ylabel="APS average set size", title="Set-size frontier")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    learned_rows = {
        row["maximum_budget"]: row
        for row in summary_rows
        if row["condition"] == "proactive" and float(row["target_coverage"]) == 0.90
    }
    random_rows = {
        row["maximum_budget"]: row
        for row in summary_rows
        if row["condition"] == "random" and float(row["target_coverage"]) == 0.90
    }
    fixed_rows = [
        row
        for row in summary_rows
        if row["condition"] in fixed and float(row["target_coverage"]) == 0.90
    ]
    beats_random = any(
        learned_rows[budget]["source_bit_macro_f1"] > random_rows[budget]["source_bit_macro_f1"]
        for budget in learned_rows.keys() & random_rows.keys()
    )
    beats_fixed = any(
        learned_rows[row["maximum_budget"]]["source_bit_macro_f1"] > row["source_bit_macro_f1"]
        for row in fixed_rows
        if row["maximum_budget"] in learned_rows
    )
    result: Dict[str, Any] = {
        **expected,
        "is_valid": True,
        "coverages": evaluation_coverages,
        "budgets": budgets,
        "base_model_instances": len(empty_indices),
        "completion_indicators": {
            "proactive_beats_random_any_budget": beats_random,
            "proactive_beats_fixed_any_budget": beats_fixed,
        },
        "summaries": summaries,
        "frontier_rows": summary_rows,
        "teacher_files": teacher_files,
        "dataset_schedule_path": (
            str(dataset_schedule_output) if dataset_schedule_output is not None else None
        ),
        "dataset_schedule_sha256": (
            file_sha256(dataset_schedule_output) if dataset_schedule_output is not None else None
        ),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": file_sha256(trajectory_path),
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
