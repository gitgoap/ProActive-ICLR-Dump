#!/usr/bin/env python3
"""Export and calibrate the Week 3 free-form semantic-match audit."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.audits.schema_validator import validate_path
from proactive.features.semantic import (
    DEFAULT_FREEFORM_THRESHOLD,
    SemanticMatcher,
    calibrate_semantic_threshold_from_scores,
)
from proactive.utils.io import file_sha256, iter_jsonl


AUDIT_FIELDS = [
    "audit_id",
    "instance_id",
    "model_id",
    "split",
    "probe",
    "clean_answer",
    "probe_answer",
    "cosine_similarity",
    "current_threshold",
    "current_match",
    "human_match",
]


def select_evenly_across_scores(rows: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """Select a deterministic score-range audit rather than only easy cases."""
    if count <= 0:
        raise ValueError("sample_count must be positive")
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["cosine_similarity"]),
            row["model_id"],
            row["instance_id"],
            row["probe"],
        ),
    )
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indexes = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indexes]


def write_csv_atomic(rows: List[Dict[str, Any]], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent, suffix=".tmp", prefix=output_path.stem + "_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def write_json_atomic(payload: Dict[str, Any], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent, suffix=".tmp", prefix=output_path.stem + "_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def export_audit(args: argparse.Namespace) -> None:
    pilot_dir = Path(args.pilot_dir)
    schema_report = validate_path(pilot_dir)
    if not schema_report["is_valid"]:
        raise ValueError("Pilot cache must pass schema and duplicate validation before export")

    matcher = SemanticMatcher(device=args.device)
    if not matcher.is_available:
        raise RuntimeError(f"Semantic matcher unavailable: {matcher.load_error}")

    candidates: List[Dict[str, Any]] = []
    for file_path in sorted(pilot_dir.glob("*.jsonl")):
        for record in iter_jsonl(file_path):
            if record.get("dataset", "").lower() != "vizwiz":
                continue
            if "pilot_severity_probe" in record or record.get("split") not in ("train", "val"):
                continue
            clean_answer = record.get("clean", {}).get("norm_answer", "")
            for probe, observation in record.get("probes", {}).items():
                if not observation.get("applicable", True) or not observation.get("valid", True):
                    continue
                probe_answer = observation.get("norm_answer", "")
                if not clean_answer or not probe_answer or clean_answer == probe_answer:
                    continue
                similarity = matcher.similarity(clean_answer, probe_answer)
                candidates.append({
                    "audit_id": "",
                    "instance_id": record["instance_id"],
                    "model_id": record["model_id"],
                    "split": record["split"],
                    "probe": probe,
                    "clean_answer": clean_answer,
                    "probe_answer": probe_answer,
                    "cosine_similarity": f"{similarity:.8f}",
                    "current_threshold": f"{args.current_threshold:.4f}",
                    "current_match": int(similarity >= args.current_threshold),
                    "human_match": "",
                })

    if not candidates:
        raise ValueError("No non-exact valid VizWiz answer pairs were found")
    selected = select_evenly_across_scores(candidates, args.sample_count)
    for index, row in enumerate(selected, start=1):
        row["audit_id"] = f"semantic_{index:04d}"
    write_csv_atomic(selected, Path(args.output_csv), args.overwrite)
    print(f"Exported {len(selected)} semantic pairs to {args.output_csv}")


def calibrate_audit(args: argparse.Namespace) -> None:
    annotator = str(args.annotator or "").strip()
    if not annotator:
        raise ValueError("--annotator with the human annotator's full name is required")

    labeled_path = Path(args.labeled_csv)
    with open(labeled_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < args.min_labels:
        raise ValueError(f"Need at least {args.min_labels} labelled rows; found {len(rows)}")

    scored_pairs = []
    for row_number, row in enumerate(rows, start=2):
        label_text = str(row.get("human_match", "")).strip().lower()
        if label_text not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(f"Row {row_number}: human_match must be 0 or 1")
        label = label_text in {"1", "true", "yes"}
        similarity = float(row["cosine_similarity"])
        if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
            raise ValueError(f"Row {row_number}: invalid cosine_similarity {similarity}")
        if row.get("split") not in ("train", "val"):
            raise ValueError(f"Row {row_number}: forbidden calibration split {row.get('split')}")
        scored_pairs.append((similarity, label))

    threshold, metrics = calibrate_semantic_threshold_from_scores(
        scored_pairs, target_recall=args.target_recall
    )
    payload = {
        "is_valid": True,
        "annotator": annotator,
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(labeled_path),
        "source_sha256": file_sha256(labeled_path),
        "labeled_count": len(scored_pairs),
        "positive_count": sum(label for _, label in scored_pairs),
        "negative_count": sum(not label for _, label in scored_pairs),
        "target_recall": args.target_recall,
        "recommended_threshold": threshold,
        "metrics": metrics,
        "calibration_split": "train_val",
    }
    write_json_atomic(payload, Path(args.output_report), args.overwrite)
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot_dir", help="Export an audit from this validated pilot directory.")
    mode.add_argument("--labeled_csv", help="Calibrate from a completed audit CSV.")
    parser.add_argument(
        "--output_csv",
        default="outputs/pilot_reports/semantic_match_audit.csv",
    )
    parser.add_argument(
        "--output_report",
        default="outputs/pilot_reports/semantic_calibration_report.json",
    )
    parser.add_argument("--sample_count", type=int, default=50)
    parser.add_argument("--min_labels", type=int, default=50)
    parser.add_argument("--current_threshold", type=float, default=DEFAULT_FREEFORM_THRESHOLD)
    parser.add_argument("--target_recall", type=float, default=0.90)
    parser.add_argument(
        "--annotator",
        help="Full name of the human who labelled the audit; required with --labeled_csv.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.current_threshold <= 1.0:
        parser.error("--current_threshold must be in [0, 1]")
    if not 0.0 < args.target_recall <= 1.0:
        parser.error("--target_recall must be in (0, 1]")

    if args.pilot_dir:
        export_audit(args)
    else:
        calibrate_audit(args)


if __name__ == "__main__":
    main()
