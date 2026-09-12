#!/usr/bin/env python3
"""Create a standard, hash-bound Week 8 ablation comparison report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation_id", required=True)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--comparison_csv", required=True)
    parser.add_argument("--reference_condition", default="proactive")
    parser.add_argument("--comparison_condition", default="proactive")
    parser.add_argument("--evaluation_split", choices=("val", "test", "shift"), required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read(path: Path, condition: str) -> Dict[tuple[int, float], Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        selected = [row for row in csv.DictReader(handle) if row.get("condition") == condition]
    result = {(int(row["maximum_budget"]), float(row["target_coverage"])): row for row in selected}
    if not result or len(result) != len(selected):
        raise SystemExit(f"Missing or duplicate frontier points in {path} for {condition}")
    return result


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference_csv)
    comparison_path = Path(args.comparison_csv)
    reference = _read(reference_path, args.reference_condition)
    comparison = _read(comparison_path, args.comparison_condition)
    if set(reference) != set(comparison):
        raise SystemExit("Reference/comparison frontier grids differ")
    rows = []
    for key in sorted(reference):
        left = reference[key]
        right = comparison[key]
        if int(left["row_count"]) != int(right["row_count"]):
            raise SystemExit(f"Ablation row-count mismatch at {key}")
        item: Dict[str, Any] = {
            "maximum_budget": key[0],
            "target_coverage": key[1],
            "row_count": int(left["row_count"]),
        }
        for metric in (
            "source_bit_macro_f1",
            "six_way_macro_f1",
            "coverage",
            "average_set_size",
            "mean_acquisition_cost",
        ):
            if left.get(metric, "") == "" or right.get(metric, "") == "":
                item[f"reference_{metric}"] = None
                item[f"comparison_{metric}"] = None
                item[f"reference_minus_comparison_{metric}"] = None
            else:
                left_value = float(left[metric])
                right_value = float(right[metric])
                item[f"reference_{metric}"] = left_value
                item[f"comparison_{metric}"] = right_value
                item[f"reference_minus_comparison_{metric}"] = left_value - right_value
        rows.append(item)
    report: Dict[str, Any] = {
        "format_version": "week8_ablation_evidence_v1",
        "status": "COMPLETE",
        "ablation_id": args.ablation_id,
        "evaluation_split": args.evaluation_split,
        "post_test_tuning_used": False,
        "reference_condition": args.reference_condition,
        "comparison_condition": args.comparison_condition,
        "reference_csv": str(reference_path),
        "reference_csv_sha256": file_sha256(reference_path),
        "comparison_csv": str(comparison_path),
        "comparison_csv_sha256": file_sha256(comparison_path),
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
