#!/usr/bin/env python3
"""Create a hash-bound state manifest for vectorization.

This is intentionally separate from state sampling: it verifies every JSONL
row before publishing an index consumed by ``prepare_week5_data.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json


ALLOWED_SPLITS = {"train", "val", "cal", "test", "shift"}


def _state_identity(
    row: Mapping[str, Any], path: Path, row_number: int
) -> tuple[str, str, str]:
    """Read identity from the canonical ``partial_state_v1`` metadata block."""
    if row.get("record_type") != "partial_state_v1":
        raise SystemExit(
            f"Unexpected state record type at {path}:{row_number}: "
            f"{row.get('record_type')!r}"
        )
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise SystemExit(f"Missing state metadata at {path}:{row_number}")
    values: Dict[str, str] = {}
    for field in ("split", "model_id", "instance_id"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value:
            raise SystemExit(
                f"Missing metadata.{field} at {path}:{row_number}"
            )
        values[field] = value
    if values["split"] not in ALLOWED_SPLITS:
        raise SystemExit(
            f"Unknown metadata.split at {path}:{row_number}: "
            f"{values['split']!r}"
        )
    return values["split"], values["model_id"], values["instance_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--required_split", choices=("train", "val", "cal", "test", "shift"))
    parser.add_argument("--expected_teacher_rows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dir = Path(args.state_dir)
    output_path = Path(args.output_path)
    files = sorted(state_dir.glob("states_*.jsonl"))
    if not files:
        raise SystemExit(f"No state JSONL files found in {state_dir}")

    global_ids: Set[str] = set()
    teacher_keys: Set[tuple[str, str]] = set()
    split_counts: Counter[str] = Counter()
    entries = []
    for path in files:
        local_ids: Set[str] = set()
        row_count = 0
        for row_number, row in enumerate(iter_jsonl(path), start=1):
            state_id = row.get("state_id")
            if not isinstance(state_id, str) or not state_id:
                raise SystemExit(f"Missing state_id at {path}:{row_number}")
            if state_id in local_ids or state_id in global_ids:
                raise SystemExit(f"Duplicate state_id: {state_id}")
            split, model_id, instance_id = _state_identity(row, path, row_number)
            if args.required_split and split != args.required_split:
                raise SystemExit(
                    f"Unexpected split at {path}:{row_number}: expected "
                    f"{args.required_split}, found {split}"
                )
            local_ids.add(state_id)
            global_ids.add(state_id)
            teacher_keys.add((model_id, instance_id))
            split_counts[str(split)] += 1
            row_count += 1
        entries.append(
            {"path": str(path), "row_count": row_count, "sha256": file_sha256(path)}
        )

    if args.expected_teacher_rows is not None and len(teacher_keys) != args.expected_teacher_rows:
        raise SystemExit(
            f"Teacher coverage mismatch: expected {args.expected_teacher_rows}, "
            f"found {len(teacher_keys)}"
        )
    report: Dict[str, Any] = {
        "format_version": "state_manifest_v2",
        "state_dir": str(state_dir),
        "files": entries,
        "row_count": len(global_ids),
        "teacher_row_count": len(teacher_keys),
        "split_counts": dict(sorted(split_counts.items())),
        "required_split": args.required_split,
    }
    report["report_sha256"] = hash_dict(report)

    if output_path.exists() and args.resume:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != report:
            raise SystemExit("State-manifest resume refused: source drift")
        print(json.dumps(existing, indent=2))
        return
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}; use --resume or --overwrite")
    if not args.dry_run:
        write_json(report, output_path, overwrite=args.overwrite and output_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
