#!/usr/bin/env python3
"""Validate and merge three independent blinded annotation files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.audits.human_audit import LABELS
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json
from prepare_human_annotation_packets import ANNOTATION_FIELDS, BASE_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet_dir", default="outputs/human_annotation_packets")
    parser.add_argument("--output_dir", default="outputs/human_annotation_merged")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 180:
        raise SystemExit(f"{path} must have 180 rows, found {len(rows)}")
    result = {row["audit_id"]: row for row in rows}
    if len(result) != len(rows):
        raise SystemExit(f"Duplicate audit_id in {path}")
    return result


def _validate_annotation(row: Mapping[str, str], path: Path) -> None:
    for field in ANNOTATION_FIELDS:
        value = row.get(field, "").strip()
        if field == "label6":
            if value not in LABELS:
                raise SystemExit(f"Invalid label6 {value!r} in {path}")
        elif value not in {"0", "1"}:
            raise SystemExit(f"{field} must be 0 or 1 in {path}")


def main() -> None:
    args = parse_args()
    packet_dir = Path(args.packet_dir)
    sources = [packet_dir / f"annotator_{index}" / "annotations.csv" for index in range(1, 4)]
    tables = [_read(path) for path in sources]
    identities = set(tables[0])
    if any(set(table) != identities for table in tables[1:]):
        raise SystemExit("Annotator files do not contain identical audit IDs")
    merged: List[Dict[str, Any]] = []
    unanimous = majority = three_way = 0
    for audit_id in sorted(identities):
        source_rows = [table[audit_id] for table in tables]
        for path, row in zip(sources, source_rows):
            _validate_annotation(row, path)
        for field in BASE_FIELDS:
            values = {row[field] for row in source_rows}
            if len(values) != 1:
                raise SystemExit(f"Blinded source field changed: {audit_id}/{field}")
        row: Dict[str, Any] = {field: source_rows[0][field] for field in BASE_FIELDS}
        for index, source in enumerate(source_rows, start=1):
            for field in ANNOTATION_FIELDS:
                row[f"ann{index}_{field}"] = source[field].strip()
        counts = Counter(source["label6"].strip() for source in source_rows)
        winner, count = counts.most_common(1)[0]
        if count == 3:
            unanimous += 1
            row["adjudicated_label6"] = winner
        elif count == 2:
            majority += 1
            row["adjudicated_label6"] = winner
        else:
            three_way += 1
            row["adjudicated_label6"] = ""
        merged.append(row)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "human_audit_merged_blinded.csv"
    manifest_path = output_dir / "merge_manifest.json"
    if (csv_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise SystemExit(f"Output exists in {output_dir}; use --overwrite")
    summary = {
        "format_version": "human_audit_merge_v1",
        "source_files": [{"path": str(path), "sha256": file_sha256(path)} for path in sources],
        "row_count": len(merged),
        "unanimous_rows": unanimous,
        "two_of_three_rows": majority,
        "three_way_disagreement_rows": three_way,
        "requires_manual_adjudication": three_way > 0,
        "still_blinded": True,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    summary["merged_path"] = str(csv_path)
    summary["merged_sha256"] = file_sha256(csv_path)
    summary["report_sha256"] = hash_dict(summary)
    write_json(summary, manifest_path, overwrite=args.overwrite and manifest_path.exists())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
