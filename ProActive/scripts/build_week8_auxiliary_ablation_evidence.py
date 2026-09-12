#!/usr/bin/env python3
"""Build signed evidence for non-training Week 8 ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation_id",
        required=True,
        choices=("pass_count_vs_latency", "gru_canonical_vs_permutation_augmentation"),
    )
    parser.add_argument("--reference_report", required=True)
    parser.add_argument("--comparison_report", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _signed(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    if value.get("report_sha256") != hash_dict(unsigned):
        raise SystemExit(f"Report self-hash mismatch: {path}")
    return value


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference_report)
    reference = _signed(reference_path)
    rows: list[Dict[str, Any]]
    artifacts = [{"role": "reference", "path": str(reference_path), "sha256": file_sha256(reference_path)}]
    if args.ablation_id == "pass_count_vs_latency":
        if reference.get("format_version") != "week8_latency_v1" or reference.get("is_valid") is not True:
            raise SystemExit("Pass-count evidence requires a valid Week 8 latency report")
        if reference.get("evaluation_split") != "val":
            raise SystemExit("Week 8 ablation latency evidence must use validation only")
        rows = [
            {
                "maximum_budget": int(reference["budget"]),
                "target_coverage": float(reference["target_coverage"]),
                "measured_examples": int(reference["measured_examples"]),
                "controller_latency_ms_mean": float(reference["controller_latency_ms"]["mean"]),
                "total_generation_latency_ms_mean": float(reference["total_generation_latency_ms_mean"]),
                "normalized_latency_mean": float(reference["normalized_latency_mean"]),
            }
        ]
    else:
        if args.comparison_report is None:
            raise SystemExit("GRU augmentation evidence requires --comparison_report")
        comparison_path = Path(args.comparison_report)
        comparison = _signed(comparison_path)
        artifacts.append({"role": "comparison", "path": str(comparison_path), "sha256": file_sha256(comparison_path)})
        for name, value, condition in (
            ("reference", reference, "canonical"),
            ("comparison", comparison, "random_permutation"),
        ):
            if (
                value.get("format_version") != "permutation_evaluation_v1"
                or value.get("is_valid") is not True
                or value.get("split") != "val"
                or value.get("encoder_name") != "gru"
                or value.get("gru_condition") != condition
            ):
                raise SystemExit(f"Invalid {name} GRU permutation report")
        if reference.get("manifest_sha256") != comparison.get("manifest_sha256"):
            raise SystemExit("GRU permutation reports use different validation manifests")
        rows = []
        for metric in ("bit_l1_drift", "hidden_relative_l2_drift", "js_drift", "set_disagreement"):
            left = reference["summary"][metric]
            right = comparison["summary"][metric]
            rows.append(
                {
                    "metric": metric,
                    "canonical_mean": float(left["mean"]),
                    "augmented_mean": float(right["mean"]),
                    "canonical_minus_augmented_mean": float(left["mean"]) - float(right["mean"]),
                    "canonical_max": float(left["max"]),
                    "augmented_max": float(right["max"]),
                }
            )
    report: Dict[str, Any] = {
        "format_version": "week8_ablation_evidence_v1",
        "status": "COMPLETE",
        "ablation_id": args.ablation_id,
        "evaluation_split": "val",
        "post_test_tuning_used": False,
        "artifacts": artifacts,
        "rows": rows,
    }
    report["report_sha256"] = hash_dict(report)
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}")
    write_json(report, output_path, overwrite=args.overwrite and output_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
