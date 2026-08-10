#!/usr/bin/env python3
"""Recompute Week 4 signatures, source bits, and six-way labels on CPU."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.data.manifests import load_manifest
from proactive.teacher.offline import (
    build_label_record,
    teacher_key,
    thresholds_from_mapping,
)
from proactive.utils.io import file_sha256, iter_jsonl, write_jsonl


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_labels")


def _jsonl_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("teacher_*.jsonl"))
        if files:
            return files
        return sorted(path.glob("*.jsonl"))
    raise FileNotFoundError(path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _output_name(input_path: Path) -> str:
    name = input_path.name
    return f"labels_{name[len('teacher_'):] if name.startswith('teacher_') else name}"


def _validate_resume(
    output_path: Path, source_sha256: str, expected_keys: Set[Tuple[str, str]]
) -> bool:
    seen: Set[Tuple[str, str]] = set()
    for row_number, row in enumerate(iter_jsonl(output_path), start=1):
        key = teacher_key(row)
        if key in seen:
            raise ValueError(f"Duplicate label key at row {row_number}: {key}")
        if row.get("record_type") != "teacher_labels":
            raise ValueError(f"Unexpected label record_type at row {row_number}")
        if row.get("source_teacher_file_sha256") != source_sha256:
            raise ValueError(f"Teacher source hash drift at row {row_number}")
        seen.add(key)
    if seen != expected_keys:
        missing = expected_keys - seen
        extra = seen - expected_keys
        raise ValueError(
            f"Existing label file is incomplete or stale: missing={len(missing)}, extra={len(extra)}"
        )
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher_path", required=True, help="Teacher JSONL file or directory")
    parser.add_argument("--manifest_path", help="Optional grouped manifest for identity checks")
    parser.add_argument("--output_dir", default="outputs/labels_core")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu", help="Accepted for script-contract parity; CPU only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    experiment = _load_yaml(Path(args.config))
    frozen_path = Path(str(experiment["frozen_probe_config"]))
    frozen = _load_yaml(frozen_path)
    thresholds = thresholds_from_mapping(frozen["label_thresholds"])

    manifest_index: Dict[str, Mapping[str, Any]] = {}
    if args.manifest_path:
        for row in load_manifest(args.manifest_path):
            instance_id = row["instance_id"]
            if instance_id in manifest_index:
                raise SystemExit(f"Duplicate manifest instance_id: {instance_id}")
            manifest_index[instance_id] = row

    input_files = _jsonl_files(Path(args.teacher_path))
    if not input_files:
        raise SystemExit("No teacher JSONL files found")
    output_dir = Path(args.output_dir)
    global_keys: Set[Tuple[str, str]] = set()
    remaining = args.limit

    for teacher_file in input_files:
        teacher_rows: List[Dict[str, Any]] = []
        for row in iter_jsonl(teacher_file):
            if remaining is not None and remaining == 0:
                break
            key = teacher_key(row)
            if key in global_keys:
                raise SystemExit(f"Duplicate teacher key across shards: {key}")
            global_keys.add(key)
            if manifest_index:
                source = manifest_index.get(row["instance_id"])
                if source is None:
                    raise SystemExit(f"Teacher row absent from manifest: {key}")
                for field in ("group_id", "dataset", "split"):
                    if row.get(field) != source.get(field):
                        raise SystemExit(f"Teacher/manifest {field} mismatch: {key}")
            teacher_rows.append(row)
            if remaining is not None:
                remaining -= 1

        if not teacher_rows:
            continue
        source_sha256 = file_sha256(teacher_file)
        output_path = output_dir / _output_name(teacher_file)
        expected_keys = {teacher_key(row) for row in teacher_rows}
        logger.info("%s -> %s (%s rows)", teacher_file, output_path, len(teacher_rows))
        if output_path.exists():
            if args.resume:
                try:
                    _validate_resume(output_path, source_sha256, expected_keys)
                except (ValueError, json.JSONDecodeError) as exc:
                    raise SystemExit(f"Unsafe label resume refused: {exc}") from exc
                logger.info("Validated complete label output; skipping")
                continue
            if not args.overwrite:
                raise SystemExit(
                    f"Output exists: {output_path}. Use --resume or --overwrite."
                )
        if args.dry_run:
            continue
        labels = [
            build_label_record(row, thresholds, source_sha256) for row in teacher_rows
        ]
        write_jsonl(labels, output_path, overwrite=args.overwrite and output_path.exists())

    logger.info("Processed %s unique teacher records", len(global_keys))


if __name__ == "__main__":
    main()

