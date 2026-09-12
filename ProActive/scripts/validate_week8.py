#!/usr/bin/env python3
"""Fail-closed Week 8 implementation, readiness, and completion gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


IMPLEMENTATION_FILES = (
    "configs/data/prehal.yaml",
    "configs/data/illusionbench.yaml",
    "configs/experiments/week8_evaluation.yaml",
    "src/proactive/data/heldout.py",
    "src/proactive/eval/statistics.py",
    "src/proactive/train/ablations.py",
    "scripts/setup_heldout_datasets.py",
    "scripts/build_heldout_manifests.py",
    "scripts/index_state_manifest.py",
    "scripts/eval_frontier.py",
    "scripts/build_lomo_fold.py",
    "scripts/freeze_lomo_stack.py",
    "scripts/eval_lomo.py",
    "scripts/build_feature_ablation_manifest.py",
    "scripts/materialize_week8_ablation_configs.py",
    "scripts/freeze_week8_diagnostic.py",
    "scripts/freeze_week8_ablation_stack.py",
    "scripts/eval_week8_ablation.py",
    "scripts/build_week8_auxiliary_ablation_evidence.py",
    "scripts/run_week8_ablations.sh",
    "scripts/compare_ablation_frontiers.py",
    "scripts/aggregate_week8_ablations.py",
    "scripts/fit_global_aps_ablation.py",
    "scripts/measure_latency.py",
    "scripts/analyze_week8.py",
    "scripts/prepare_human_annotation_packets.py",
    "scripts/merge_human_annotations.py",
    "scripts/analyze_human_audit.py",
    "tests/test_week8_heldout.py",
    "tests/test_week8_statistics.py",
    "tests/test_week8_ablation_models.py",
    "tests/test_week8_lomo_and_human.py",
    "tests/test_week8_integrity.py",
    "human_annotation/README.md",
    "human_annotation/ANNOTATOR_GUIDE.md",
    "human_annotation/ADJUDICATION_GUIDE.md",
    "WEEK_8_DATASET_SETUP_AND_EXECUTION.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("implementation", "readiness", "full"), required=True)
    parser.add_argument("--config", default="configs/experiments/week8_evaluation.yaml")
    parser.add_argument("--output_dir", default="outputs/week8_reports")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _read_signed(path: Path, errors: list[str]) -> Dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"Missing artifact: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError, OSError) as exc:
        errors.append(f"Unreadable artifact {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"Report must be a JSON object: {path}")
        return None
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    expected = value.get("report_sha256")
    if not isinstance(expected, str) or not expected:
        errors.append(f"Report self-hash missing or invalid: {path}")
        return None
    if expected != hash_dict(unsigned):
        errors.append(f"Report self-hash mismatch: {path}")
        return None
    return value


def main() -> None:
    import yaml

    from proactive.train.checkpoints import validate_freeze_manifest

    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    missing_code = [path for path in IMPLEMENTATION_FILES if not Path(path).is_file()]
    if missing_code:
        errors.append(f"Week 8 implementation files missing: {missing_code}")
    if args.mode in {"readiness", "full"}:
        week7_path = Path("outputs/week7_reports/week7_full_validation.json")
        week7 = _read_signed(week7_path, errors)
        if week7 is not None and (week7.get("is_valid") is not True or week7.get("go_no_go") != "GO"):
            errors.append("Week 7 full validation is not GO")
        try:
            validate_freeze_manifest(config["week7_freeze_manifest"], require_policy=True)
        except (ValueError, FileNotFoundError) as exc:
            errors.append(f"Week 7 freeze invalid: {exc}")
        if config.get("metadata", {}).get("approval_status") != "APPROVED":
            errors.append("Week 8 GPU settings are not owner-approved")
        setup = _read_signed(Path(config["outputs"]["data_dir"]) / "dataset_setup_report.json", errors)
        if setup is not None:
            if setup.get("is_valid") is not True:
                errors.append("Held-out dataset setup report is not valid")
            observed = set(setup.get("results", {}))
            expected = set(config["active_datasets"])
            if observed != expected:
                errors.append(
                    "Held-out dataset setup coverage mismatch: "
                    f"expected {sorted(expected)}, found {sorted(observed)}"
                )
        bundle = _read_signed(Path(config["outputs"]["data_dir"]) / "manifests" / "heldout_manifest_bundle.json", errors)
        if bundle is not None:
            if bundle.get("status") != "FROZEN" or int(bundle.get("combined_rows", -1)) != 1200:
                errors.append("Held-out manifest bundle must contain exactly 1,200 frozen rows")
            if bundle.get("selection_uses_model_outputs") is not False or bundle.get("target_domain_calibration_used") is not False:
                errors.append("Held-out selection/calibration provenance is unsafe")
            observed = set(bundle.get("datasets", {}))
            expected = set(config["active_datasets"])
            if observed != expected:
                errors.append(
                    "Held-out manifest dataset mismatch: "
                    f"expected {sorted(expected)}, found {sorted(observed)}"
                )
    if args.mode == "full":
        shift = _read_signed(Path(config["outputs"]["report_dir"]) / "shift" / "frontier_shift.json", errors)
        if shift is not None:
            if shift.get("phase") != "shift" or shift.get("coverage_claim") != "empirical_shift_only":
                errors.append("Shift report does not use the empirical-only claim contract")
            if shift.get("target_domain_calibration_used") is not False:
                errors.append("Shift report indicates target-domain calibration")
            if int(shift.get("base_model_instances", -1)) != 2400:
                errors.append("Shift report must contain 600 examples × 2 datasets × 2 core models")
        lomo_reports = []
        for model_key in config["evaluation"]["lomo_models"]:
            value = _read_signed(Path(config["outputs"]["report_dir"]) / "lomo" / model_key / "lomo.json", errors)
            if value is not None:
                lomo_reports.append(value)
        if len(lomo_reports) == len(config["evaluation"]["lomo_models"]):
            for report in lomo_reports:
                if report.get("is_valid") is not True:
                    errors.append(f"LOMO report is not valid for {report.get('heldout_model_id')}")
                    continue
                primary = [row for row in report["rows"] if row["condition"] == "proactive" and float(row["target_coverage"]) == 0.9]
                clean = {(row["maximum_budget"]): row for row in report["rows"] if row["condition"] == "clean_only_learned" and float(row["target_coverage"]) == 0.9}
                scalar = {(row["maximum_budget"]): row for row in report["rows"] if row["condition"] == "scalar_confidence" and float(row["target_coverage"]) == 0.9}
                budgets = {row["maximum_budget"] for row in primary}
                if not primary or not budgets.issubset(clean) or not budgets.issubset(scalar):
                    errors.append(f"LOMO control frontier is incomplete for {report['heldout_model_id']}")
                    continue
                calibrated = [*primary, *clean.values(), *scalar.values()]
                if any(
                    row.get("empirical_coverage") is None
                    or row.get("average_set_size") is None
                    for row in calibrated
                ):
                    errors.append(f"LOMO calibrated-set metrics are missing for {report['heldout_model_id']}")
                    continue
                if not any(row["source_bit_macro_f1"] > clean[row["maximum_budget"]]["source_bit_macro_f1"] and row["source_bit_macro_f1"] > scalar[row["maximum_budget"]]["source_bit_macro_f1"] for row in primary):
                    errors.append(f"LOMO does not beat clean/scalar for {report['heldout_model_id']}")
        ablations = _read_signed(Path(config["outputs"]["report_dir"]) / "ablations.json", errors)
        if ablations is not None and (ablations.get("is_valid") is not True or int(ablations.get("mandatory_count", 0)) != len(config["ablation_execution"]["mandatory"])):
            errors.append("Mandatory ablation bundle is incomplete")
        latency = _read_signed(Path(config["outputs"]["report_dir"]) / "latency_report.json", errors)
        if latency is not None and (latency.get("is_valid") is not True or int(latency.get("measured_examples", 0)) < 100 or latency.get("cuda_synchronized") is not True):
            errors.append("Fixed-hardware latency protocol is incomplete")
        audit = _read_signed(Path(config["outputs"]["report_dir"]) / "human_audit" / "human_audit_summary.json", errors)
        if audit is not None and audit.get("is_valid") is not True:
            errors.append("Human audit did not pass the predeclared gates")
        analysis = _read_signed(Path(config["outputs"]["report_dir"]) / "week8_analysis.json", errors)
        if analysis is not None:
            if int(analysis.get("bootstrap_resamples", 0)) < 1000:
                errors.append("Grouped bootstrap evidence is incomplete")
            if analysis.get("primary_comparison_protocol") != config["evaluation"].get("primary_comparisons"):
                errors.append("Primary paired-comparison protocol is missing or drifted")
    result: Dict[str, Any] = {
        "format_version": "week8_validation_v1",
        "mode": args.mode,
        "is_valid": not errors,
        "status": (
            "COMPLETE" if args.mode == "full" and not errors else
            "IMPLEMENTED, NOT VALIDATED" if args.mode == "implementation" and not errors else
            "PILOT VALIDATED" if args.mode == "readiness" and not errors else
            "NOT STARTED" if missing_code else "IMPLEMENTED, NOT VALIDATED"
        ),
        "errors": errors,
        "warnings": warnings,
        "config_sha256": file_sha256(config_path),
    }
    result["report_sha256"] = hash_dict(result)
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"week8_{args.mode}_validation.json"
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {output_path}; use --overwrite")
        write_json(result, output_path, overwrite=args.overwrite and output_path.exists())
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
