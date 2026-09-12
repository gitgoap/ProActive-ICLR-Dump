#!/usr/bin/env python3
"""Unblind a completed three-person audit and compute Week 8 agreement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.audits.human_audit import LABELS
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/week8_evaluation.yaml")
    parser.add_argument(
        "--merged_csv",
        default="outputs/human_annotation_merged/human_audit_merged_blinded.csv",
    )
    parser.add_argument(
        "--private_key",
        default="outputs/human_audit/human_audit_private_key.jsonl",
    )
    parser.add_argument("--output_dir", default="outputs/week8_reports/human_audit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _fleiss_kappa(rows: List[List[str]]) -> float:
    n_raters = 3
    counts = []
    for labels in rows:
        count = Counter(labels)
        counts.append([count.get(label, 0) for label in LABELS])
    p_i = [
        (sum(value * value for value in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in counts
    ]
    observed = sum(p_i) / len(p_i)
    total = len(rows) * n_raters
    category_proportions = [sum(row[index] for row in counts) / total for index in range(len(LABELS))]
    expected = sum(value * value for value in category_proportions)
    if math.isclose(1.0 - expected, 0.0):
        raise ValueError("Fleiss kappa is undefined because all ratings use one category")
    return float((observed - expected) / (1.0 - expected))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    merged_path = Path(args.merged_csv)
    private_path = Path(args.private_key)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    thresholds = config["human_audit"]
    with open(merged_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    private_rows = {row["audit_id"]: row for row in iter_jsonl(private_path)}
    if len(rows) != int(thresholds["row_count"]) or set(private_rows) != {row["audit_id"] for row in rows}:
        raise SystemExit("Merged audit and private key do not contain the same required rows")

    ratings: List[List[str]] = []
    adjudicated: List[str] = []
    majority_count = 0
    output_rows = []
    for row in rows:
        labels = [row[f"ann{index}_label6"].strip() for index in range(1, 4)]
        if any(label not in LABELS for label in labels):
            raise SystemExit(f"Incomplete/invalid annotation at {row['audit_id']}")
        counts = Counter(labels)
        winner, count = counts.most_common(1)[0]
        if count >= 2:
            final = winner
            majority_count += 1
        else:
            final = row.get("adjudicated_label6", "").strip()
            if final not in LABELS:
                raise SystemExit(
                    f"Three-way disagreement at {row['audit_id']} requires blinded adjudication"
                )
        teacher = str(private_rows[row["audit_id"]]["teacher_label6"])
        ratings.append(labels)
        adjudicated.append(final)
        output_rows.append(
            {
                "audit_id": row["audit_id"],
                "ann1_label6": labels[0],
                "ann2_label6": labels[1],
                "ann3_label6": labels[2],
                "adjudicated_label6": final,
                "teacher_label6": teacher,
                "rule_match": int(final == teacher),
                "teacher_singleton": int(teacher in {"visual", "language-prior", "alignment"}),
            }
        )
    kappa = _fleiss_kappa(ratings)
    majority_fraction = majority_count / len(rows)
    overall_match = sum(row["rule_match"] for row in output_rows) / len(output_rows)
    singleton_rows = [row for row in output_rows if row["teacher_singleton"]]
    if not singleton_rows:
        raise SystemExit("Audit contains no teacher singleton-label cases")
    singleton_match = sum(row["rule_match"] for row in singleton_rows) / len(singleton_rows)
    gates = {
        "fleiss_kappa": kappa >= float(thresholds["minimum_fleiss_kappa"]),
        "majority_agreement": majority_fraction >= float(thresholds["minimum_majority_agreement"]),
        "rule_match_overall": overall_match >= float(thresholds["minimum_rule_match_overall"]),
        "rule_match_singleton": singleton_match >= float(thresholds["minimum_rule_match_singleton"]),
    }
    result: Dict[str, Any] = {
        "format_version": "week8_human_audit_summary_v1",
        "is_valid": all(gates.values()),
        "interpretation": "behavioural_semantic_plausibility_not_causal_source",
        "annotator_count": 3,
        "row_count": len(rows),
        "fleiss_kappa_label6": kappa,
        "majority_agreement_fraction": majority_fraction,
        "rule_match_overall": overall_match,
        "rule_match_singleton": singleton_match,
        "singleton_row_count": len(singleton_rows),
        "gates": gates,
        "config_sha256": file_sha256(config_path),
        "merged_csv_sha256": file_sha256(merged_path),
        "private_key_sha256": file_sha256(private_path),
    }
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "human_audit_scored.csv"
    summary_path = output_dir / "human_audit_summary.json"
    if (detail_path.exists() or summary_path.exists()) and not args.overwrite:
        raise SystemExit(f"Human-audit results exist in {output_dir}; use --overwrite")
    with open(detail_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    result["detail_path"] = str(detail_path)
    result["detail_sha256"] = file_sha256(detail_path)
    result["report_sha256"] = hash_dict(result)
    write_json(result, summary_path, overwrite=args.overwrite and summary_path.exists())
    print(json.dumps(result, indent=2))
    if not result["is_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
