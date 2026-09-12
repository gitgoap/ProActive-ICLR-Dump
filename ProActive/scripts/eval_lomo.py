#!/usr/bin/env python3
"""Evaluate one frozen leave-one-model-out fold on its held-out test model."""

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

from proactive.conformal.aps import prediction_sets
from proactive.conformal.contracts import validate_final_aps_report
from proactive.eval.baselines import FixedSchedulePolicy, RandomPolicy
from proactive.eval.frontier import matched_budget, summarize_trajectories
from proactive.networks.controls import ScalarConfidenceDiagnostic
from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.voi import ActionConditionedVOIHead, FrozenDiagnosticPolicy
from proactive.policy.rollout import baseline_policy_rollout, diagnostic_snapshot, learned_policy_rollout
from proactive.train.checkpoints import CHECKPOINT_VERSION, POLICY_CHECKPOINT_VERSION, load_checkpoint, validate_freeze_manifest
from proactive.train.state_data import FeatureNormalizer
from proactive.train.vectorized import load_vectorized_split, project_model_input
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_jsonl, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold_manifest", required=True)
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--aps_report", required=True)
    parser.add_argument("--clean_aps_report", required=True)
    parser.add_argument("--scalar_aps_report", required=True)
    parser.add_argument("--teacher_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _teacher_index(path: Path) -> Dict[tuple[str, str], Mapping[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("teacher_*.jsonl"))
    result = {}
    for source in files:
        if source.name.endswith(".failures.jsonl"):
            continue
        for row in iter_jsonl(source):
            if row.get("split") != "test":
                continue
            key = (str(row["model_id"]), str(row["instance_id"]))
            if key in result:
                raise SystemExit(f"Duplicate held-out teacher identity: {key}")
            if row.get("valid") is not True:
                raise SystemExit(f"Invalid LOMO teacher row: {key}")
            result[key] = row
    return result


def _threshold(aps: Mapping[str, Any], budget: int, coverage: float) -> float:
    try:
        return float(aps["thresholds"][str(budget)][format(coverage, ".12g")]["threshold"])
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"LOMO APS lacks budget={budget}, coverage={coverage}") from exc


def _validate_static_aps(
    path: Path,
    *,
    baseline: str,
    checkpoint_path: Path,
    freeze_path: Path,
) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    if (
        value.get("format_version") != "static_baseline_aps_v1"
        or value.get("status") != "FINAL_FROZEN"
        or value.get("baseline") != baseline
        or value.get("fit_split") != "cal"
        or value.get("test_used") is not False
        or value.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or value.get("freeze_manifest_sha256") != file_sha256(freeze_path)
        or value.get("limit") is not None
        or value.get("report_sha256") != hash_dict(unsigned)
    ):
        raise SystemExit(f"LOMO static APS contract failed: {path}")
    return value


def _record(row: Mapping[str, Any], condition: str, budget: int, trajectory: Mapping[str, Any], threshold: float | None, coverage: float) -> Dict[str, Any]:
    probabilities = trajectory["six_way_probabilities"]
    return {
        "condition": condition,
        "maximum_budget": budget,
        "target_coverage": coverage,
        "state_id": row["state_id"],
        "metadata": dict(row["audit_metadata"]),
        "actions": list(trajectory["actions"]),
        "acquisition_cost": int(trajectory["acquisition_cost"]),
        "bit_probabilities": trajectory["bit_probabilities"],
        "six_way_probabilities": probabilities,
        "signature_prediction": trajectory["signature_prediction"],
        "prediction_set": prediction_sets([probabilities], threshold)[0] if threshold is not None else None,
        "targets": {
            "source_bits": row["targets"]["source_bits"].tolist(),
            "six_way": int(row["targets"]["six_way"]),
            "signature": row["targets"]["signature"].tolist(),
        },
        "coverage_claim": "lomo_empirical_using_source_model_calibration",
    }


def _csv(rows: list[Dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    args = parse_args()
    fold_path = Path(args.fold_manifest)
    freeze_path = Path(args.freeze_manifest)
    aps_path = Path(args.aps_report)
    clean_aps_path = Path(args.clean_aps_report)
    scalar_aps_path = Path(args.scalar_aps_report)
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    if fold.get("format_version") != "lomo_fold_v1" or fold.get("status") != "COMPLETE":
        raise SystemExit("LOMO fold contract is incomplete")
    vector_item = next(item for item in fold["artifacts"] if item["kind"] == "vector_manifest")
    vector_path = Path(vector_item["path"])
    if file_sha256(vector_path) != vector_item["sha256"]:
        raise SystemExit("LOMO vector manifest drift")
    freeze = validate_freeze_manifest(freeze_path, require_policy=True)
    diagnostic_path = Path(freeze["artifacts"]["diagnostic_checkpoint"]["path"])
    clean_path = Path(freeze["artifacts"]["clean_checkpoint"]["path"])
    policy_path = Path(freeze["artifacts"]["policy_checkpoint"]["path"])
    scalar_path = Path(freeze["artifacts"]["scalar_baseline"]["path"])
    diagnostic = load_checkpoint(diagnostic_path, map_location="cpu")
    clean = load_checkpoint(clean_path, map_location="cpu")
    policy = load_checkpoint(policy_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu")
    scalar = load_checkpoint(scalar_path, map_location="cpu")
    vector_sha = file_sha256(vector_path)
    if any(checkpoint.get("source_manifest_sha256") != vector_sha for checkpoint in (diagnostic, clean, scalar)):
        raise SystemExit("LOMO diagnostic/control vector mismatch")
    if policy.get("vector_manifest_sha256") != vector_sha or policy.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
        raise SystemExit("LOMO policy provenance mismatch")
    if fold["heldout_model_id"] == "" or fold["heldout_model_id"] is None:
        raise SystemExit("LOMO held-out model is missing")
    aps = json.loads(aps_path.read_text(encoding="utf-8"))
    try:
        validate_final_aps_report(aps, freeze_manifest_path=freeze_path)
    except ValueError as exc:
        raise SystemExit(f"LOMO APS contract failed: {exc}") from exc
    clean_aps = _validate_static_aps(
        clean_aps_path,
        baseline="clean_only_learned",
        checkpoint_path=clean_path,
        freeze_path=freeze_path,
    )
    scalar_aps = _validate_static_aps(
        scalar_aps_path,
        baseline="scalar_confidence",
        checkpoint_path=scalar_path,
        freeze_path=freeze_path,
    )
    budgets = [int(value) for value in aps["budgets"]]
    coverages = [float(value) for value in aps["coverages"]]
    output_dir = Path(args.output_dir)
    report_path = output_dir / "lomo.json"
    csv_path = output_dir / "lomo.csv"
    trajectory_path = output_dir / "lomo_trajectories.jsonl"
    expected = {
        "format_version": "lomo_evaluation_v1",
        "heldout_model_id": fold["heldout_model_id"],
        "fold_manifest_sha256": file_sha256(fold_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "aps_sha256": file_sha256(aps_path),
        "clean_aps_sha256": file_sha256(clean_aps_path),
        "scalar_aps_sha256": file_sha256(scalar_aps_path),
        "vector_manifest_sha256": vector_sha,
        "calibration_models_exclude_heldout": True,
        "test_rows_only_heldout_model": True,
    }
    if report_path.exists() and args.resume:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"LOMO resume refused: {key} drift")
        print(json.dumps(existing, indent=2))
        return
    if args.dry_run:
        print(json.dumps({**expected, "budgets": budgets, "coverages": coverages}, indent=2))
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA device is unavailable")
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
    learned = FrozenDiagnosticPolicy(diagnostic_model, head).to(device).eval()
    clean_model = build_diagnostic_model("clean_mlp", clean["architecture"])
    clean_model.load_state_dict(clean["model_state_dict"])
    clean_model.to(device).eval()
    scalar_model = ScalarConfidenceDiagnostic(
        state_dim=int(scalar["architecture"]["state_dim"]),
        dropout=float(scalar["architecture"]["dropout"]),
    )
    scalar_model.load_state_dict(scalar["model_state_dict"])
    scalar_model.to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(diagnostic["normalizer"])
    clean_normalizer = FeatureNormalizer.from_state_dict(clean["normalizer"])
    base = load_vectorized_split(vector_path, "test")
    empty = torch.nonzero(base.model_input["acquired_mask"].sum(dim=1) == 0, as_tuple=False).flatten().tolist()
    if any(base.audit_metadata["model_id"][index] != fold["heldout_model_id"] for index in empty):
        raise SystemExit("LOMO test split contains a source-model row")
    teachers = _teacher_index(Path(args.teacher_path))
    grouped: Dict[tuple[str, int, float], list[Dict[str, Any]]] = defaultdict(list)
    fixed_names = ("blank_first", "visual_first", "grounding_first", "relation_first")
    for index in empty:
        row = base[index]
        identity = (row["audit_metadata"]["model_id"], row["audit_metadata"]["instance_id"])
        teacher = teachers.get(identity)
        if teacher is None:
            raise SystemExit(f"Missing LOMO teacher row: {identity}")
        for budget in budgets:
            initial = project_model_input(row["model_input"], budget)
            proactive = learned_policy_rollout(
                policy_model=learned,
                initial_model_input=initial,
                teacher_record=teacher,
                normalizer=normalizer,
                device=device,
            )
            no_stop = learned_policy_rollout(
                policy_model=learned,
                initial_model_input=initial,
                teacher_record=teacher,
                normalizer=normalizer,
                device=device,
                force_full_budget=True,
            )
            clean_snapshot = diagnostic_snapshot(
                diagnostic_model=clean_model,
                model_input=initial,
                normalizer=clean_normalizer,
                device=device,
            )
            scalar_snapshot = diagnostic_snapshot(
                diagnostic_model=scalar_model,
                model_input=initial,
                normalizer=normalizer,
                device=device,
            )
            random = baseline_policy_rollout(
                acquisition_policy=RandomPolicy(seed=args.seed),
                diagnostic_model=diagnostic_model,
                initial_model_input=initial,
                teacher_record=teacher,
                normalizer=normalizer,
                device=device,
                instance_id=row["audit_metadata"]["instance_id"],
            )
            fixed = {
                name: baseline_policy_rollout(
                    acquisition_policy=FixedSchedulePolicy(name),
                    diagnostic_model=diagnostic_model,
                    initial_model_input=initial,
                    teacher_record=teacher,
                    normalizer=normalizer,
                    device=device,
                    instance_id=row["audit_metadata"]["instance_id"],
                )
                for name in fixed_names
            }
            for coverage in coverages:
                threshold = _threshold(aps, budget, coverage)
                clean_threshold = _threshold(clean_aps, budget, coverage)
                scalar_threshold = _threshold(scalar_aps, budget, coverage)
                grouped[("proactive", budget, coverage)].append(_record(row, "proactive", budget, proactive, threshold, coverage))
                grouped[("proactive_no_stop", budget, coverage)].append(
                    _record(row, "proactive_no_stop", budget, no_stop, threshold, coverage)
                )
                grouped[("clean_only_learned", budget, coverage)].append(
                    _record(row, "clean_only_learned", budget, {"actions": ["stop"], "acquisition_cost": 0, **clean_snapshot}, clean_threshold, coverage)
                )
                grouped[("scalar_confidence", budget, coverage)].append(
                    _record(row, "scalar_confidence", budget, {"actions": ["stop"], "acquisition_cost": 0, **scalar_snapshot}, scalar_threshold, coverage)
                )
                grouped[("random", budget, coverage)].append(_record(row, "random", budget, random, threshold, coverage))
                for name, trajectory in fixed.items():
                    grouped[(name, budget, coverage)].append(_record(row, name, budget, trajectory, threshold, coverage))

    trajectory_rows = [row for key in sorted(grouped) for row in grouped[key]]
    summary_rows = []
    summaries = {}
    for key, rows in sorted(grouped.items()):
        matched_budget(rows, key[1])
        summary = summarize_trajectories(rows)
        summaries["|".join(map(str, key))] = summary
        set_metrics = summary["prediction_sets"]
        summary_rows.append(
            {
                "heldout_model_id": fold["heldout_model_id"],
                "condition": key[0],
                "maximum_budget": key[1],
                "target_coverage": key[2],
                "row_count": summary["row_count"],
                "mean_acquisition_cost": summary["mean_acquisition_cost"],
                "source_bit_macro_f1": summary["diagnostic"]["source_bit_macro_f1"],
                "six_way_macro_f1": summary["diagnostic"]["six_way_macro_f1"],
                "empirical_coverage": set_metrics["coverage"] if set_metrics else None,
                "average_set_size": set_metrics["average_set_size"] if set_metrics else None,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    allow = args.overwrite or args.resume
    write_jsonl(trajectory_rows, trajectory_path, overwrite=allow and trajectory_path.exists())
    write_text(_csv(summary_rows), csv_path, overwrite=allow and csv_path.exists())
    result: Dict[str, Any] = {
        **expected,
        "is_valid": True,
        "heldout_test_instances": len(empty),
        "budgets": budgets,
        "coverages": coverages,
        "summaries": summaries,
        "rows": summary_rows,
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": file_sha256(trajectory_path),
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
    }
    result["report_sha256"] = hash_dict(result)
    write_json(result, report_path, overwrite=allow and report_path.exists())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
