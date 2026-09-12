#!/usr/bin/env python3
"""Fail-closed aggregation of all predeclared mandatory Week 8 ablations."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/week8_evaluation.yaml")
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="ABLATION_ID=REPORT.json",
        help=(
            "One ablation evidence binding. Repeat --evidence once per ablation. "
            "Repeated values are accumulated rather than silently replacing earlier values."
        ),
    )
    parser.add_argument("--output_dir", default="outputs/week8_reports")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = list(config["ablation_execution"]["mandatory"])
    supplied: Dict[str, Path] = {}
    for value in args.evidence:
        if "=" not in value:
            raise SystemExit(f"Invalid --evidence value: {value}")
        name, raw_path = value.split("=", 1)
        if name in supplied:
            raise SystemExit(f"Duplicate ablation evidence: {name}")
        supplied[name] = Path(raw_path)
    unknown = sorted(set(supplied) - set(required))
    if unknown:
        raise SystemExit(f"Unknown ablation IDs: {unknown}")
    missing = sorted(set(required) - set(supplied))
    if args.dry_run:
        print(json.dumps({"required": required, "supplied": sorted(supplied), "missing": missing}, indent=2))
        return
    if missing:
        raise SystemExit("Mandatory Week 8 ablation evidence missing: " + ", ".join(missing))
    rows = []
    artifacts = []
    for name in required:
        path = supplied[name]
        value = json.loads(path.read_text(encoding="utf-8"))
        unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
        if (
            value.get("format_version") != "week8_ablation_evidence_v1"
            or value.get("status") != "COMPLETE"
            or value.get("ablation_id") != name
            or value.get("post_test_tuning_used") is not False
            or value.get("report_sha256") != hash_dict(unsigned)
        ):
            raise SystemExit(f"Invalid ablation evidence contract: {path}")
        artifacts.append({"ablation_id": name, "path": str(path), "sha256": file_sha256(path)})
        for item in value["rows"]:
            rows.append({"ablation_id": name, "evaluation_split": value["evaluation_split"], **item})
    stream = io.StringIO(newline="")
    if not rows:
        raise SystemExit("Mandatory Week 8 ablation evidence contains no rows")
    preferred = ["ablation_id", "evaluation_split"]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    writer = csv.DictWriter(stream, fieldnames=preferred + remaining)
    writer.writeheader()
    writer.writerows(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablations.csv"
    report_path = output_dir / "ablations.json"
    write_text(stream.getvalue(), csv_path, overwrite=args.overwrite and csv_path.exists())
    report: Dict[str, Any] = {
        "format_version": "week8_ablation_bundle_v1",
        "is_valid": True,
        "mandatory_count": len(required),
        "post_test_tuning_used": False,
        "evidence": artifacts,
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
    }
    report["report_sha256"] = hash_dict(report)
    write_json(report, report_path, overwrite=args.overwrite and report_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
