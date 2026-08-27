#!/usr/bin/env python3
"""Select the validation cost multiplier and enforce the Week 6 gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict

import yaml

from proactive.train.checkpoints import POLICY_CHECKPOINT_VERSION, load_checkpoint
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/policy_train.yaml")
    parser.add_argument("--manifest_path", required=True, help="Week 5 vectorized manifest")
    parser.add_argument("--frontier_reports", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/week6_reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--approve_selection", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def _signed(path: Path) -> Dict[str, Any]:
    value = _read(path)
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    if value.get("report_sha256") != hash_dict(unsigned):
        raise SystemExit(f"Frontier report self-hash mismatch: {path}")
    return value


def main() -> None:
    args = parse_args()
    if args.device != "cpu" or args.limit is not None:
        raise SystemExit("Week 6 selection is complete and CPU-only; --limit is forbidden")
    config_path = Path(args.config)
    manifest_path = Path(args.manifest_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("metadata", {}).get("approval_status") != "APPROVED" and not args.dry_run:
        raise SystemExit("Week 6 settings are not APPROVED")
    grid = {float(value) for value in config["cost_multiplier_grid"]}
    budgets = {int(value) for value in config["budgets"]}
    required_seeds = {int(value) for value in config["seeds"]}
    primary_seed = int(config["primary_seed"])
    runs_by_multiplier: Dict[float, list[Dict[str, Any]]] = defaultdict(list)
    for raw_path in args.frontier_reports:
        path = Path(raw_path)
        report = _signed(path)
        if report.get("phase") != "validation" or report.get("split") != "val" or report.get("limit") is not None:
            raise SystemExit(f"Week 6 selection refuses non-full validation frontier: {path}")
        if report.get("config_sha256") != file_sha256(config_path):
            raise SystemExit(f"Frontier/policy config mismatch: {path}")
        if report.get("manifest_sha256") != file_sha256(manifest_path):
            raise SystemExit(f"Frontier/vector manifest mismatch: {path}")
        policy_path = Path(report["policy_checkpoint_path"])
        policy = load_checkpoint(policy_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu")
        if report.get("policy_checkpoint_sha256") != file_sha256(policy_path):
            raise SystemExit(f"Frontier/policy checkpoint drift: {path}")
        multiplier = float(policy["cost_multiplier"])
        if report.get("encoder_name") != config["policy_architectures"]["main"]:
            raise SystemExit(f"Policy selection received a non-main encoder frontier: {path}")
        seed = int(policy["seed"])
        if any(int(run["seed"]) == seed for run in runs_by_multiplier[multiplier]):
            raise SystemExit(f"Duplicate frontier for multiplier={multiplier}, seed={seed}")
        proactive = [
            row for row in report["frontier_rows"]
            if row["condition"] == "proactive"
            and int(row["maximum_budget"]) in budgets
            and float(row["target_coverage"])
            == float(config["temporary_aps_target_coverage"])
        ]
        if {int(row["maximum_budget"]) for row in proactive} != budgets:
            raise SystemExit(f"Frontier budget coverage mismatch: {path}")
        runs_by_multiplier[multiplier].append(
            {
                "seed": seed,
                "multiplier": multiplier,
                "report_path": path,
                "report": report,
                "policy_path": policy_path,
                "mean_macro_f1": mean(float(row["source_bit_macro_f1"]) for row in proactive),
                "mean_cost": mean(float(row["mean_acquisition_cost"]) for row in proactive),
                "mean_set_size": mean(float(row["average_set_size"]) for row in proactive),
            }
        )
    missing = sorted(grid - set(runs_by_multiplier))
    extra = sorted(set(runs_by_multiplier) - grid)
    seed_mismatch = {
        str(multiplier): {
            "missing": sorted(required_seeds - {int(run["seed"]) for run in runs}),
            "extra": sorted({int(run["seed"]) for run in runs} - required_seeds),
        }
        for multiplier, runs in sorted(runs_by_multiplier.items())
        if {int(run["seed"]) for run in runs} != required_seeds
    }
    if missing or extra or seed_mismatch:
        message = {
            "is_ready": False,
            "missing_cost_multipliers": missing,
            "extra_cost_multipliers": extra,
            "seed_mismatch": seed_mismatch,
        }
        if args.dry_run:
            print(json.dumps(message, indent=2))
            return
        raise SystemExit(json.dumps(message))
    candidates = []
    for multiplier, runs in sorted(runs_by_multiplier.items()):
        primary = next(run for run in runs if int(run["seed"]) == primary_seed)
        candidates.append(
            {
                "multiplier": multiplier,
                "primary_report_path": primary["report_path"],
                "primary_report": primary["report"],
                "primary_policy_path": primary["policy_path"],
                "mean_macro_f1": mean(float(run["mean_macro_f1"]) for run in runs),
                "mean_cost": mean(float(run["mean_cost"]) for run in runs),
                "mean_set_size": mean(float(run["mean_set_size"]) for run in runs),
                "seed_scores": [
                    {
                        "seed": int(run["seed"]),
                        "mean_macro_f1": float(run["mean_macro_f1"]),
                        "mean_cost": float(run["mean_cost"]),
                        "mean_set_size": float(run["mean_set_size"]),
                        "policy_path": str(run["policy_path"]),
                        "report_path": str(run["report_path"]),
                    }
                    for run in sorted(runs, key=lambda item: int(item["seed"]))
                ],
            }
        )
    selected = max(
        candidates,
        key=lambda item: (item["mean_macro_f1"], -item["mean_set_size"], -item["mean_cost"], -item["multiplier"]),
    )
    selected_runs = runs_by_multiplier[float(selected["multiplier"])]
    def aggregate(condition: str, budget: int, metric: str) -> float:
        values = []
        for run in selected_runs:
            row = next(
                item
                for item in run["report"]["frontier_rows"]
                if item["condition"] == condition
                and int(item["maximum_budget"]) == budget
                and float(item["target_coverage"])
                == float(config["temporary_aps_target_coverage"])
            )
            values.append(float(row[metric]))
        return mean(values)

    ordered_budgets = sorted(budgets)
    clean_condition = "clean_only_learned"
    if not all(
        any(row["condition"] == clean_condition for row in run["report"]["frontier_rows"])
        for run in selected_runs
    ):
        clean_condition = "diagnostic_no_acquisition"
    full_quality = mean(
        float(
            next(
                row
                for row in run["report"]["frontier_rows"]
                if row["condition"] == "full_teacher"
                and float(row["target_coverage"])
                == float(config["temporary_aps_target_coverage"])
            )["source_bit_macro_f1"]
        )
        for run in selected_runs
    )
    clean_quality = aggregate(
        clean_condition,
        max(ordered_budgets),
        "source_bit_macro_f1",
    )
    relative_full_gain = (full_quality - clean_quality) / max(abs(clean_quality), 1.0e-12)
    active_signal_passed = relative_full_gain >= float(
        config["active_signal_gate"]["minimum_relative_full_teacher_gain_over_clean"]
    )
    proactive = [
        {
            "maximum_budget": budget,
            "source_bit_macro_f1": aggregate("proactive", budget, "source_bit_macro_f1"),
            "average_set_size": aggregate("proactive", budget, "average_set_size"),
        }
        for budget in ordered_budgets
    ]
    budget_progress = all(
        right["source_bit_macro_f1"] >= left["source_bit_macro_f1"]
        or right["average_set_size"] <= left["average_set_size"]
        for left, right in zip(proactive, proactive[1:])
    )
    beats_random = any(
        aggregate("proactive", budget, "source_bit_macro_f1")
        > aggregate("random", budget, "source_bit_macro_f1")
        for budget in ordered_budgets
    )
    fixed_names = ["blank_first", "visual_first", "grounding_first", "relation_first"]
    beats_fixed = any(
        aggregate("proactive", budget, "source_bit_macro_f1")
        > aggregate(schedule, budget, "source_bit_macro_f1")
        for budget in ordered_budgets
        for schedule in fixed_names
    )
    completion = bool(
        beats_random
        and beats_fixed
        and active_signal_passed
        and budget_progress
    )
    result: Dict[str, Any] = {
        "format_version": "week6_policy_selection_v1",
        "status": "SELECTED" if completion else "GATE_FAILED",
        "fit_split": "train",
        "selection_split": "val",
        "calibration_used": False,
        "test_used": False,
        "selected_cost_multiplier": selected["multiplier"] if completion else None,
        "selected_seed": primary_seed if completion else None,
        "selected_policy_path": str(selected["primary_policy_path"]) if completion else None,
        "selected_policy_sha256": file_sha256(selected["primary_policy_path"]) if completion else None,
        "selected_frontier_path": str(selected["primary_report_path"]) if completion else None,
        "selected_frontier_sha256": file_sha256(selected["primary_report_path"]) if completion else None,
        "completion_gate_passed": completion,
        "gate_components": {
            "beats_random": beats_random,
            "beats_fixed": beats_fixed,
            "full_teacher_stronger_than_no_acquisition": bool(full_quality > clean_quality),
            "full_teacher_source_bit_macro_f1_mean": full_quality,
            "no_acquisition_source_bit_macro_f1_mean": clean_quality,
            "relative_full_teacher_gain_over_clean": relative_full_gain,
            "active_signal_gate_passed": active_signal_passed,
            "active_signal_gate_config": dict(config["active_signal_gate"]),
            "budget_progress": budget_progress,
        },
        "candidate_scores": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in candidate.items()
                if key != "primary_report"
            }
            for candidate in sorted(candidates, key=lambda item: item["multiplier"])
        ],
        "selection_rule": "max mean source-bit Macro-F1; then min mean set size; then min cost; then min lambda",
        "config_sha256": file_sha256(config_path),
        "vector_manifest_sha256": file_sha256(manifest_path),
    }
    result["report_sha256"] = hash_dict(result)
    output_path = Path(args.output_dir) / "week6_policy_selection.json"
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return
    write_json(result, output_path, overwrite=(args.overwrite or args.resume) and output_path.exists())
    if not completion:
        raise SystemExit("Week 6 completion gate failed; do not freeze Week 7")
    if not args.approve_selection:
        raise SystemExit(f"Selection report written to {output_path}; owner approval is still required")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
