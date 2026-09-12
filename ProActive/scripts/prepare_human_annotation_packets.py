#!/usr/bin/env python3
"""Split the blinded audit into three independent annotator packets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.audits.human_audit import LABELS
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json, write_text


BASE_FIELDS = (
    "audit_id",
    "image_file",
    "question",
    "gold_answer",
    "clean_answer",
    "probe_results_json",
)
ANNOTATION_FIELDS = (
    "clean_correct",
    "visual_fragile",
    "language_persistence",
    "alignment_instability",
    "label6",
    "insufficient_or_contradictory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_dir", default="outputs/human_audit")
    parser.add_argument("--guide_dir", default="human_annotation")
    parser.add_argument("--output_dir", default="outputs/human_annotation_packets")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*BASE_FIELDS, *ANNOTATION_FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    guide_dir = Path(args.guide_dir)
    output_dir = Path(args.output_dir)
    source_csv = audit_dir / "human_audit_blinded.csv"
    source_manifest = audit_dir / "human_audit_manifest.json"
    guide_path = guide_dir / "ANNOTATOR_GUIDE.md"
    for path in (source_csv, source_manifest, guide_path):
        if not path.is_file():
            raise SystemExit(f"Required input missing: {path}")
    with open(source_csv, "r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 180:
        raise SystemExit(f"Human audit must contain 180 rows, found {len(source_rows)}")
    if len({row["audit_id"] for row in source_rows}) != len(source_rows):
        raise SystemExit("Duplicate audit_id in blinded CSV")
    expected = {
        "format_version": "human_annotation_packets_v1",
        "source_csv": str(source_csv),
        "source_csv_sha256": file_sha256(source_csv),
        "source_manifest_sha256": file_sha256(source_manifest),
        "guide_sha256": file_sha256(guide_path),
        "annotator_count": 3,
        "row_count_per_annotator": len(source_rows),
        "labels": list(LABELS),
        "independent": True,
        "private_key_included": False,
    }
    manifest_path = output_dir / "packet_manifest.json"
    if manifest_path.exists() and args.resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"Annotation-packet resume refused: {key} drift")
        for relative, sha in existing.get("artifact_sha256", {}).items():
            path = output_dir / relative
            if not path.exists() or file_sha256(path) != sha:
                raise SystemExit(f"Annotation packet drift: {relative}")
        print(json.dumps(existing, indent=2))
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    if args.dry_run:
        print(json.dumps(expected, indent=2))
        return

    artifact_hashes: Dict[str, str] = {}
    for annotator in range(1, 4):
        annotator_dir = output_dir / f"annotator_{annotator}"
        images_dir = annotator_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for source in source_rows:
            row = {field: source[field] for field in BASE_FIELDS}
            row.update({field: "" for field in ANNOTATION_FIELDS})
            image_name = Path(source["image_file"]).name
            source_image = audit_dir / source["image_file"]
            destination = images_dir / image_name
            if not source_image.is_file():
                raise SystemExit(f"Audit image missing: {source_image}")
            shutil.copy2(source_image, destination)
            row["image_file"] = f"images/{image_name}"
            rows.append(row)
            relative = str(destination.relative_to(output_dir)).replace("\\", "/")
            artifact_hashes[relative] = file_sha256(destination)
        csv_path = annotator_dir / "annotations.csv"
        _write_csv(csv_path, rows)
        shutil.copy2(guide_path, annotator_dir / "README.md")
        for path in (csv_path, annotator_dir / "README.md"):
            relative = str(path.relative_to(output_dir)).replace("\\", "/")
            artifact_hashes[relative] = file_sha256(path)
    manifest: Dict[str, Any] = {**expected, "artifact_sha256": artifact_hashes}
    manifest["report_sha256"] = hash_dict(manifest)
    write_json(manifest, manifest_path, overwrite=args.overwrite and manifest_path.exists())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
