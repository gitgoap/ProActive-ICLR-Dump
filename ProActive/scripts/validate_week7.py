#!/usr/bin/env python3
"""Week 7 freeze/calibration/locked-test completion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from proactive.conformal.contracts import validate_final_aps_report
from proactive.train.checkpoints import validate_freeze_manifest
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("readiness", "full"), required=True)
    parser.add_argument("--config", default="configs/experiments/calibrate_aps.yaml")
    parser.add_argument("--manifest_path", default="outputs/week7_calibration/aps_thresholds.json")
    parser.add_argument("--freeze_manifest", default="outputs/week7_frozen/main_stack_freeze.json")
    parser.add_argument("--frontier_report", default="outputs/week7_reports/frontier_test.json")
    parser.add_argument("--permutation_report", default="outputs/week7_reports/permutation_week7_deep_sets_standard_seed42.json")
    parser.add_argument("--gru_reports", nargs="*", default=[])
    parser.add_argument("--output_dir", default="outputs/week7_reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _signed(path: Path) -> Dict[str, Any]:
    value = _read(path)
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    if value.get("report_sha256") != hash_dict(unsigned):
        raise ValueError(f"Report self-hash mismatch: {path}")
    return value


def main() -> None:
    args = parse_args()
    if args.device != "cpu" or args.limit is not None:
        raise SystemExit("Week 7 validation is complete and CPU-only")
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    errors = []
    warnings = []
    if config.get("metadata", {}).get("approval_status") != "APPROVED":
        errors.append("Week 7 settings are not APPROVED")
    freeze_path = Path(args.freeze_manifest)
    freeze = None
    try:
        freeze = validate_freeze_manifest(freeze_path, require_policy=True)
    except (ValueError, FileNotFoundError) as exc:
        errors.append(f"Main-stack freeze: {exc}")
    if args.mode == "full" and not errors:
        aps_path = Path(args.manifest_path)
        frontier_path = Path(args.frontier_report)
        permutation_path = Path(args.permutation_report)
        try:
            week5_item = freeze["artifacts"].get("week5_freeze")
            if not isinstance(week5_item, dict):
                raise ValueError("Main freeze lacks the Week 5 freeze artifact")
            week5 = validate_freeze_manifest(week5_item["path"], require_policy=False)
            selection_item = week5["artifacts"].get("selection_report")
            if not isinstance(selection_item, dict):
                raise ValueError("Week 5 freeze lacks its selection report")
            week5_selection = _signed(Path(selection_item["path"]))
            if (
                week5_selection.get("status") != "SELECTED"
                or week5_selection.get("completion_gate_passed") is not True
                or week5_selection.get("set_transformer_gate") != "NOT_TRIGGERED"
                or week5_selection.get("shortcut_gate", {}).get("status") != "PASSED"
            ):
                errors.append("Frozen Week 5 encoder/shortcut/Set-Transformer decision is incomplete")
            raps_status = week5_selection.get("raps_gate", {}).get("status")
            if raps_status not in {"NOT_TRIGGERED", "TRIGGERED_APPENDIX_ABLATION"}:
                errors.append("Frozen validation-only APS/RAPS decision is missing")
            elif raps_status == "TRIGGERED_APPENDIX_ABLATION":
                warnings.append(
                    "Validation APS triggered the optional appendix RAPS ablation; "
                    "the locked APS main result remains unchanged"
                )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            errors.append(f"Frozen Week 5 decisions: {exc}")
        try:
            aps = _read(aps_path)
            validate_final_aps_report(aps, freeze_manifest_path=freeze_path)
            tolerance = float(config["undercoverage_tolerance_proposal"])
            for budget, coverages in aps["thresholds"].items():
                for coverage, item in coverages.items():
                    observed = float(item["calibration_metrics"]["coverage"])
                    target = float(item["target_coverage"])
                    if observed < target - tolerance:
                        errors.append(f"Calibration undercoverage at budget={budget}, coverage={coverage}")
        except (ValueError, FileNotFoundError, KeyError) as exc:
            errors.append(f"Final APS: {exc}")
        try:
            frontier = _signed(frontier_path)
            if frontier.get("phase") != "test" or frontier.get("split") != "test" or frontier.get("limit") is not None:
                errors.append("Frontier is not a complete locked-test report")
            if frontier.get("freeze_manifest_sha256") != file_sha256(freeze_path):
                errors.append("Frontier/freeze hash mismatch")
            if frontier.get("oracle_subset_included") is not True:
                errors.append("Locked frontier omitted the mandatory oracle-best-subset control")
            if frontier.get("oracle_next_included") is not True:
                errors.append("Locked frontier omitted the mandatory oracle-next control")
            rows = frontier["frontier_rows"]
            required_coverages = {float(value) for value in config["coverages"]}
            observed_coverage_pairs = {
                (int(row["maximum_budget"]), float(row["target_coverage"]))
                for row in rows
                if row["condition"] == "proactive"
            }
            required_coverage_pairs = {
                (int(budget), coverage)
                for budget in config["budgets"]
                for coverage in required_coverages
            }
            if observed_coverage_pairs != required_coverage_pairs:
                errors.append("Locked frontier lacks complete 90%/95% budget coverage")
            frontier_tolerance = float(config["undercoverage_tolerance_proposal"])
            for row in rows:
                if (
                    row["condition"] == "proactive"
                    and float(row["coverage"])
                    < float(row["target_coverage"]) - frontier_tolerance
                ):
                    errors.append(
                        "Locked-test undercoverage at "
                        f"budget={row['maximum_budget']}, target={row['target_coverage']}"
                    )
            learned = {
                int(row["maximum_budget"]): row
                for row in rows
                if row["condition"] == "proactive" and float(row["target_coverage"]) == 0.90
            }
            random = {
                int(row["maximum_budget"]): row
                for row in rows
                if row["condition"] == "random" and float(row["target_coverage"]) == 0.90
            }
            fixed_names = {"blank_first", "visual_first", "grounding_first", "relation_first"}
            fixed_wins = {
                row["condition"]
                for row in rows
                if row["condition"] in fixed_names
                and float(row["target_coverage"]) == 0.90
                and int(row["maximum_budget"]) in learned
                and learned[int(row["maximum_budget"])]["source_bit_macro_f1"] > row["source_bit_macro_f1"]
            }
            beats_random = any(
                learned[budget]["source_bit_macro_f1"] > random[budget]["source_bit_macro_f1"]
                for budget in learned.keys() & random.keys()
            )
            if not beats_random:
                errors.append("ProActive does not beat random on locked test")
            if len(fixed_wins) < 2:
                errors.append("ProActive does not beat at least two fixed schedules")
            required_conditions = {
                "scalar_confidence",
                "clean_only_learned",
                "one_pass_distilled",
                "dataset_specific_fixed",
                "uncertainty_greedy",
                "full_teacher",
                "oracle_next",
                "oracle_best_subset",
            }
            observed_conditions = {row["condition"] for row in rows}
            missing_conditions = sorted(required_conditions - observed_conditions)
            if missing_conditions:
                errors.append(f"Locked frontier missing controls: {missing_conditions}")
        except (ValueError, FileNotFoundError, KeyError) as exc:
            errors.append(f"Locked frontier: {exc}")
        try:
            permutation = _signed(permutation_path)
            if permutation.get("phase") != "week7" or permutation.get("split") != "test":
                errors.append("Primary permutation report is not locked test")
            if permutation.get("freeze_manifest_sha256") != file_sha256(freeze_path):
                errors.append("Primary permutation/freeze hash mismatch")
            if permutation.get("summary", {}).get("state_count", 0) < int(config["permutation"]["minimum_states"]):
                errors.append("Primary permutation protocol has too few states")
            if permutation.get("is_valid") is not True:
                errors.append("Primary invariant encoder exceeded drift tolerance")
        except (ValueError, FileNotFoundError, KeyError) as exc:
            errors.append(f"Primary permutation report: {exc}")
        if len(args.gru_reports) < 2:
            errors.append("Both canonical and random-permutation GRU reports are required")
        else:
            observed_conditions = set()
            for raw_path in args.gru_reports:
                try:
                    report = _signed(Path(raw_path))
                    if report.get("encoder_name") != "gru":
                        errors.append(f"Non-GRU comparison report: {raw_path}")
                    if report.get("phase") != "week7" or report.get("split") != "test":
                        errors.append(f"GRU report is not a locked Week 7 test run: {raw_path}")
                    if report.get("freeze_manifest_sha256") != file_sha256(freeze_path):
                        errors.append(f"GRU permutation/freeze hash mismatch: {raw_path}")
                    if report.get("summary", {}).get("state_count", 0) < int(config["permutation"]["minimum_states"]):
                        errors.append(f"GRU report has too few states: {raw_path}")
                    observed_conditions.add(report.get("gru_condition"))
                except (ValueError, FileNotFoundError) as exc:
                    errors.append(f"GRU permutation report: {exc}")
            if not {"canonical", "random_permutation"}.issubset(observed_conditions):
                errors.append("GRU canonical/random-permutation conditions are incomplete")
    output_dir = Path(args.output_dir)
    memo_path = output_dir / "week7_go_no_go.md"
    result: Dict[str, Any] = {
        "format_version": "week7_validation_v1",
        "mode": args.mode,
        "is_valid": not errors,
        "config_approval": config.get("metadata", {}).get("approval_status"),
        "errors": errors,
        "warnings": warnings,
        "go_no_go": "GO" if args.mode == "full" and not errors else "NO-GO",
        "memo_path": str(memo_path) if args.mode == "full" else None,
    }
    result["report_sha256"] = hash_dict(result)
    output_path = output_dir / f"week7_{args.mode}_validation.json"
    if not args.dry_run:
        write_json(result, output_path, overwrite=(args.overwrite or args.resume) and output_path.exists())
        if args.mode == "full":
            lines = [
                "# Week 7 Go/No-Go Memo",
                "",
                f"**Decision:** {result['go_no_go']}",
                "",
                "This decision was generated by the locked Week 7 validator after the stack freeze, "
                "final calibration, test frontier, and permutation checks.",
                "",
                "## Blocking findings",
                "",
            ]
            lines.extend(f"- {error}" for error in errors)
            if not errors:
                lines.append("- None.")
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in warnings)
            if not warnings:
                lines.append("- None.")
            lines.extend(
                [
                    "",
                    "## Provenance",
                    "",
                    f"- Validation report: `{output_path}`",
                    f"- Freeze manifest SHA-256: `{file_sha256(freeze_path) if freeze_path.exists() else 'MISSING'}`",
                    "- No post-test tuning is authorized by this memo.",
                    "",
                ]
            )
            write_text(
                "\n".join(lines),
                memo_path,
                overwrite=(args.overwrite or args.resume) and memo_path.exists(),
            )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
