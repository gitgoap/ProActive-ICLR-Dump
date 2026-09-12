#!/usr/bin/env python3
"""Create grouped CIs, frozen slices, and qualitative cases for Week 8."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.eval.statistics import (
    METRICS,
    grouped_bootstrap,
    holm_bonferroni,
    paired_grouped_bootstrap_difference,
    stable_example_rank,
)
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/week8_evaluation.yaml")
    parser.add_argument("--trajectory_path", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/week8_reports")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _read_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    identities = set()
    for path in paths:
        for row in iter_jsonl(path):
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                raise SystemExit(f"Trajectory row without metadata: {path}")
            if row.get("coverage_claim") == "empirical_shift_only" and metadata.get("split") != "shift":
                raise SystemExit("Shift trajectory contains a non-shift record")
            identity = (
                row.get("condition"),
                row.get("maximum_budget"),
                row.get("target_coverage"),
                metadata.get("model_id"),
                metadata.get("instance_id"),
            )
            if identity in identities:
                raise SystemExit(f"Duplicate trajectory identity: {identity}")
            identities.add(identity)
            rows.append(dict(row))
    if not rows:
        raise SystemExit("No trajectory records found")
    return rows


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evaluation = config["evaluation"]
    seed = int(config.get("seed", 42) if args.seed is None else args.seed)
    resamples = int(evaluation["bootstrap_resamples"])
    confidence = float(evaluation["bootstrap_confidence"])
    if evaluation.get("bootstrap_group") != "group_id":
        raise SystemExit("Week 8 bootstrap must use group_id")
    paths = [Path(path) for path in args.trajectory_path]
    output_dir = Path(args.output_dir)
    report_path = output_dir / "week8_analysis.json"
    ci_path = output_dir / "confidence_intervals.csv"
    paired_path = output_dir / "paired_comparisons.csv"
    slice_path = output_dir / "slices.csv"
    qualitative_path = output_dir / "qualitative_examples.json"
    expected = {
        "format_version": "week8_analysis_v1",
        "config_sha256": file_sha256(config_path),
        "trajectory_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        "seed": seed,
        "bootstrap_resamples": resamples,
        "bootstrap_confidence": confidence,
        "bootstrap_group": "group_id",
        "target_domain_calibration_used": False,
        "primary_comparison_protocol": evaluation["primary_comparisons"],
    }
    if report_path.exists() and args.resume:
        with open(report_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"Week 8 analysis resume refused: {key} drift")
        for key in ("ci", "paired", "slices", "qualitative"):
            path = Path(existing[f"{key}_path"])
            if not path.exists() or file_sha256(path) != existing[f"{key}_sha256"]:
                raise SystemExit(f"Week 8 analysis artifact drift: {key}")
        print(json.dumps(existing, indent=2))
        return
    if report_path.exists() and not (args.overwrite or args.resume):
        raise SystemExit("Week 8 analysis exists; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({**expected, "output": str(report_path)}, indent=2))
        return

    rows = _read_rows(paths)
    grouped: Dict[tuple[str, int, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["condition"]),
                int(row["maximum_budget"]),
                float(row["target_coverage"]),
            )
        ].append(row)

    ci_rows = []
    for (condition, budget, coverage), group_rows in sorted(grouped.items()):
        for metric_name, metric in METRICS.items():
            result = grouped_bootstrap(
                group_rows,
                metric=metric,
                resamples=resamples,
                confidence=confidence,
                seed=seed,
            )
            ci_rows.append(
                {
                    "condition": condition,
                    "maximum_budget": budget,
                    "target_coverage": coverage,
                    "metric": metric_name,
                    **result,
                }
            )

    paired_rows = []
    declared_controls = ("random", "clean_only_learned", "scalar_confidence")
    for budget in sorted({key[1] for key in grouped if key[0] == "proactive"}):
        for coverage in sorted({key[2] for key in grouped if key[0] == "proactive" and key[1] == budget}):
            left = grouped[("proactive", budget, coverage)]
            for control in declared_controls:
                right = grouped.get((control, budget, coverage))
                if not right:
                    continue
                for metric_name in ("source_bit_macro_f1", "empirical_coverage", "average_set_size", "mean_acquisition_cost"):
                    result = paired_grouped_bootstrap_difference(
                        left,
                        right,
                        metric=METRICS[metric_name],
                        resamples=resamples,
                        confidence=confidence,
                        seed=seed,
                    )
                    paired_rows.append(
                        {
                            "left": "proactive",
                            "right": control,
                            "maximum_budget": budget,
                            "target_coverage": coverage,
                            "metric": metric_name,
                            **result,
                        }
                    )

    primary = evaluation["primary_comparisons"]
    if primary.get("multiplicity_correction") != "holm":
        raise SystemExit("Week 8 primary comparisons require Holm correction")
    primary_controls = list(primary["controls"])
    primary_indices = [
        index
        for index, row in enumerate(paired_rows)
        if row["right"] in primary_controls
        and int(row["maximum_budget"]) == int(primary["budget"])
        and float(row["target_coverage"]) == float(primary["coverage"])
        and row["metric"] == primary["metric"]
    ]
    if len(primary_indices) != len(primary_controls):
        raise SystemExit(
            "Primary paired-comparison family is incomplete: expected "
            f"{len(primary_controls)}, found {len(primary_indices)}"
        )
    corrected = holm_bonferroni(
        [paired_rows[index]["p_value_two_sided"] for index in primary_indices],
        alpha=float(primary["familywise_alpha"]),
    )
    for row in paired_rows:
        row["primary_comparison"] = False
        row["multiplicity_correction"] = "none"
        row["p_value_holm"] = None
        row["reject_familywise"] = None
    for index, correction in zip(primary_indices, corrected):
        paired_rows[index].update(
            {
                "primary_comparison": True,
                "multiplicity_correction": "holm",
                **correction,
            }
        )

    slice_rows = []
    for (condition, budget, coverage), group_rows in sorted(grouped.items()):
        if condition not in {"proactive", "random", "clean_only_learned", "scalar_confidence"}:
            continue
        for slice_kind, field in (("dataset", "dataset"), ("model", "model_id")):
            by_value: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in group_rows:
                by_value[str(row["metadata"][field])].append(row)
            for value, selected in sorted(by_value.items()):
                slice_rows.append(
                    {
                        "condition": condition,
                        "maximum_budget": budget,
                        "target_coverage": coverage,
                        "slice_kind": slice_kind,
                        "slice_value": value,
                        "row_count": len(selected),
                        "source_bit_macro_f1": METRICS["source_bit_macro_f1"](selected),
                        "six_way_macro_f1": METRICS["six_way_macro_f1"](selected),
                        "empirical_coverage": METRICS["empirical_coverage"](selected),
                        "average_set_size": METRICS["average_set_size"](selected),
                        "mean_acquisition_cost": METRICS["mean_acquisition_cost"](selected),
                    }
                )

    primary_rows = [
        row
        for row in rows
        if row["condition"] == "proactive"
        and int(row["maximum_budget"]) == max(evaluation["budgets"])
        and float(row["target_coverage"]) == 0.90
    ]
    positives = []
    negatives = []
    for row in primary_rows:
        target = int(row["targets"]["six_way"])
        predicted = max(
            range(len(row["six_way_probabilities"])),
            key=lambda index: float(row["six_way_probabilities"][index]),
        )
        compact = {
            "state_id": row["state_id"],
            "metadata": row["metadata"],
            "actions": row["actions"],
            "target_six_way": target,
            "predicted_six_way": predicted,
            "prediction_set": row["prediction_set"],
            "selection_rule": "frozen_outcome_then_sha256_rank_v1",
        }
        if predicted == target and target in row["prediction_set"]:
            positives.append(compact)
        else:
            negatives.append(compact)
    positives.sort(key=lambda row: stable_example_rank(seed, "positive", row))
    negatives.sort(key=lambda row: stable_example_rank(seed, "negative", row))
    qualitative = {
        "format_version": "week8_qualitative_v1",
        "selection_uses_model_tuning": False,
        "positive": positives[:10],
        "negative": negatives[:10],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    allow = args.overwrite or args.resume
    write_text(_csv(ci_rows), ci_path, overwrite=allow and ci_path.exists())
    write_text(_csv(paired_rows), paired_path, overwrite=allow and paired_path.exists())
    write_text(_csv(slice_rows), slice_path, overwrite=allow and slice_path.exists())
    write_json(qualitative, qualitative_path, overwrite=allow and qualitative_path.exists())
    result: Dict[str, Any] = {
        **expected,
        "trajectory_row_count": len(rows),
        "ci_path": str(ci_path),
        "ci_sha256": file_sha256(ci_path),
        "paired_path": str(paired_path),
        "paired_sha256": file_sha256(paired_path),
        "slices_path": str(slice_path),
        "slices_sha256": file_sha256(slice_path),
        "qualitative_path": str(qualitative_path),
        "qualitative_sha256": file_sha256(qualitative_path),
    }
    result["report_sha256"] = hash_dict(result)
    write_json(result, report_path, overwrite=allow and report_path.exists())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
