#!/usr/bin/env python3
"""Build deterministic leakage-safe Week 4 partial evidence states."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from proactive.data.manifests import load_manifest
from proactive.teacher.offline import build_state_records, teacher_key
from proactive.utils.io import file_sha256, iter_jsonl, write_jsonl


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sample_states")


def _jsonl_files(path: Path, prefix: str) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob(f"{prefix}*.jsonl"))
        if files:
            return files
        return sorted(path.glob("*.jsonl"))
    raise FileNotFoundError(path)


def _paired_label_file(teacher_file: Path, labels_path: Path) -> Path:
    if labels_path.is_file():
        return labels_path
    suffix = teacher_file.name[len("teacher_") :] if teacher_file.name.startswith("teacher_") else teacher_file.name
    candidate = labels_path / f"labels_{suffix}"
    if not candidate.exists():
        raise FileNotFoundError(f"No label shard paired with {teacher_file}: {candidate}")
    return candidate


def _output_name(teacher_file: Path) -> str:
    suffix = teacher_file.name[len("teacher_") :] if teacher_file.name.startswith("teacher_") else teacher_file.name
    return f"states_{suffix}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--manifest_path", help="Optional grouped manifest for split/group checks")
    parser.add_argument("--output_dir", default="outputs/states_v1")
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
    with open(args.config, "r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    if not isinstance(experiment, Mapping):
        raise SystemExit("Experiment config must be a mapping")
    random_count = int(experiment.get("state_sampling", {}).get("random_subsets", 16))
    if random_count != 16:
        raise SystemExit("Plan section 16.2 requires exactly 16 random pre-policy subsets")

    manifest_index: Dict[str, Mapping[str, Any]] = {}
    if args.manifest_path:
        for row in load_manifest(args.manifest_path):
            if row["instance_id"] in manifest_index:
                raise SystemExit(f"Duplicate manifest instance_id: {row['instance_id']}")
            manifest_index[row["instance_id"]] = row

    teacher_files = _jsonl_files(Path(args.teacher_path), "teacher_")
    if not teacher_files:
        raise SystemExit("No teacher JSONL files found")
    labels_path = Path(args.labels_path)
    output_dir = Path(args.output_dir)
    remaining = args.limit
    global_teacher_keys: Set[Tuple[str, str]] = set()

    for teacher_file in teacher_files:
        label_file = _paired_label_file(teacher_file, labels_path)
        labels: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for label in iter_jsonl(label_file):
            key = teacher_key(label)
            if key in labels:
                raise SystemExit(f"Duplicate label key in {label_file}: {key}")
            labels[key] = label

        teacher_sha256 = file_sha256(teacher_file)
        label_sha256 = file_sha256(label_file)
        state_rows: List[Dict[str, Any]] = []
        selected_teacher_keys: Set[Tuple[str, str]] = set()
        for teacher in iter_jsonl(teacher_file):
            if remaining is not None and remaining == 0:
                break
            key = teacher_key(teacher)
            if key in global_teacher_keys:
                raise SystemExit(f"Duplicate teacher key across shards: {key}")
            global_teacher_keys.add(key)
            selected_teacher_keys.add(key)
            label = labels.get(key)
            if label is None:
                raise SystemExit(f"Missing label for teacher record: {key}")
            if label.get("source_teacher_file_sha256") != teacher_sha256:
                raise SystemExit(f"Label source hash drift for {key}")
            if manifest_index:
                source = manifest_index.get(teacher["instance_id"])
                if source is None:
                    raise SystemExit(f"Teacher row absent from manifest: {key}")
                for field in ("group_id", "dataset", "split"):
                    if teacher.get(field) != source.get(field):
                        raise SystemExit(f"Teacher/manifest {field} mismatch: {key}")
            state_rows.extend(
                build_state_records(
                    teacher,
                    label,
                    source_teacher_file_sha256=teacher_sha256,
                    source_label_file_sha256=label_sha256,
                    seed=args.seed,
                    random_subset_count=random_count,
                )
            )
            if remaining is not None:
                remaining -= 1

        if not selected_teacher_keys:
            continue
        output_path = output_dir / _output_name(teacher_file)
        logger.info(
            "%s + %s -> %s (%s states from %s teachers)",
            teacher_file,
            label_file,
            output_path,
            len(state_rows),
            len(selected_teacher_keys),
        )
        if output_path.exists():
            if args.resume:
                existing_ids: Set[str] = set()
                for row_number, row in enumerate(iter_jsonl(output_path), start=1):
                    state_id = row.get("state_id")
                    if not isinstance(state_id, str) or not state_id:
                        raise SystemExit(f"Missing state_id at {output_path}:{row_number}")
                    if state_id in existing_ids:
                        raise SystemExit(f"Duplicate state_id in existing output: {state_id}")
                    if row.get("source_teacher_file_sha256") != teacher_sha256:
                        raise SystemExit(f"Teacher hash drift at {output_path}:{row_number}")
                    if row.get("source_label_file_sha256") != label_sha256:
                        raise SystemExit(f"Label hash drift at {output_path}:{row_number}")
                    existing_ids.add(state_id)
                expected_ids = {row["state_id"] for row in state_rows}
                if existing_ids != expected_ids:
                    raise SystemExit(
                        f"Unsafe state resume: expected {len(expected_ids)} IDs, found {len(existing_ids)}"
                    )
                logger.info("Validated complete state output; skipping")
                continue
            if not args.overwrite:
                raise SystemExit(
                    f"Output exists: {output_path}. Use --resume or --overwrite."
                )
        if not args.dry_run:
            write_jsonl(
                state_rows,
                output_path,
                overwrite=args.overwrite and output_path.exists(),
            )

    logger.info("Processed %s unique teacher records", len(global_teacher_keys))


if __name__ == "__main__":
    main()

