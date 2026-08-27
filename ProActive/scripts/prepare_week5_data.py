#!/usr/bin/env python3
"""Vectorize Week 4 partial states into compact, hash-bound tensor shards."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

import torch
import yaml

from proactive.train.state_data import DEFAULT_SEVERITIES, vectorize_state
from proactive.train.vectorized import (
    VECTOR_MANIFEST_VERSION,
    atomic_torch_save,
    stack_vectorized_rows,
    validate_tensor_shard,
)
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json


LOGGER = logging.getLogger("prepare_week5_data")
ALLOWED_SPLITS = {"train", "val", "cal", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/diag_bakeoff.yaml")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--state_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise SystemExit(f"Expected JSON mapping: {path}")
    return value


def _validate_resume(
    path: Path,
    config_sha: str,
    source_manifest_sha: str,
    requested_limit: int | None,
) -> None:
    manifest = _load_json(path)
    if manifest.get("format_version") != VECTOR_MANIFEST_VERSION:
        raise SystemExit("Vectorized resume manifest version mismatch")
    if manifest.get("config_sha256") != config_sha:
        raise SystemExit("Vectorized resume refused: config hash drift")
    if manifest.get("source_state_manifest_sha256") != source_manifest_sha:
        raise SystemExit("Vectorized resume refused: source manifest hash drift")
    if manifest.get("limit") != requested_limit:
        raise SystemExit("Vectorized resume refused: --limit drift")
    for entry in manifest.get("files", []):
        artifact = path.parent / entry["path"]
        if not artifact.exists() or file_sha256(artifact) != entry.get("sha256"):
            raise SystemExit(f"Vectorized resume refused: artifact drift at {artifact}")
        shard = torch.load(artifact, map_location="cpu", weights_only=False)
        validate_tensor_shard(shard, str(entry["split"]))
    LOGGER.info("Validated complete vectorized output; skipping")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if args.device != "cpu":
        raise SystemExit("State vectorization is CPU-only; use --device cpu")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise SystemExit("Diagnostic config must be a mapping")
    manifest_path = Path(args.manifest_path or config["source_state_manifest"])
    state_path = Path(args.state_path or config["source_state_dir"])
    output_dir = Path(args.output_dir or config["outputs"]["vectorized_dir"])
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    config_sha = file_sha256(config_path)
    source_manifest_sha = file_sha256(manifest_path)
    source_manifest = _load_json(manifest_path)
    entries = source_manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("State manifest has no files")

    sources: List[Dict[str, Any]] = []
    for entry in entries:
        source = state_path / Path(entry["path"]).name
        if not source.exists():
            raise SystemExit(f"State source missing: {source}")
        actual_sha = file_sha256(source)
        if actual_sha != entry.get("sha256"):
            raise SystemExit(f"State source hash mismatch: {source}")
        sources.append(
            {
                "path": str(source),
                "sha256": actual_sha,
                "row_count": int(entry["row_count"]),
            }
        )
    LOGGER.info("Validated %s state files (%s source rows)", len(sources), sum(x["row_count"] for x in sources))
    if args.dry_run:
        print(json.dumps({
            "is_valid": True,
            "source_files": len(sources),
            "source_rows": sum(x["row_count"] for x in sources),
            "limit": args.limit,
            "output_dir": str(output_dir),
        }, indent=2))
        return

    final_manifest_path = output_dir / "vectorized_manifest.json"
    if final_manifest_path.exists() and args.resume:
        _validate_resume(final_manifest_path, config_sha, source_manifest_sha, args.limit)
        return
    if final_manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {final_manifest_path}; use --resume or --overwrite")

    severities = dict(DEFAULT_SEVERITIES)
    frozen_path = Path(config["frozen_probe_config"])
    with open(frozen_path, "r", encoding="utf-8") as handle:
        frozen = yaml.safe_load(handle)
    if not isinstance(frozen, Mapping):
        raise SystemExit("Frozen probe config must be a mapping")
    severities.update({key: float(value) for key, value in frozen["probe_severities"].items()})

    output_entries: List[Dict[str, Any]] = []
    global_state_ids: Set[str] = set()
    group_splits: Dict[str, str] = {}
    remaining = args.limit
    for source in sources:
        by_split: Dict[str, List[Any]] = defaultdict(list)
        observed_rows = 0
        for record in iter_jsonl(source["path"]):
            if remaining is not None and remaining == 0:
                break
            state_id = record.get("state_id")
            if state_id in global_state_ids:
                raise SystemExit(f"Duplicate state_id across source files: {state_id}")
            vector = vectorize_state(record, max_budget=7, severities=severities)
            split = vector.metadata["split"]
            if split not in ALLOWED_SPLITS:
                raise SystemExit(f"Unknown split {split!r} in {state_id}")
            group_id = vector.metadata["group_id"]
            previous = group_splits.setdefault(group_id, split)
            if previous != split:
                raise SystemExit(f"Grouped split leakage for {group_id}: {previous} versus {split}")
            global_state_ids.add(state_id)
            by_split[split].append(vector)
            observed_rows += 1
            if remaining is not None:
                remaining -= 1
        if args.limit is None and observed_rows != source["row_count"]:
            raise SystemExit(
                f"State row-count mismatch for {source['path']}: expected {source['row_count']}, found {observed_rows}"
            )
        for split, rows in sorted(by_split.items()):
            output_name = f"{Path(source['path']).stem}__{split}.pt"
            output_path = output_dir / output_name
            shard = stack_vectorized_rows(rows, split)
            atomic_torch_save(shard, output_path, overwrite=args.overwrite and output_path.exists())
            output_entries.append(
                {
                    "path": output_name,
                    "split": split,
                    "row_count": len(rows),
                    "sha256": file_sha256(output_path),
                    "source_path": source["path"],
                    "source_sha256": source["sha256"],
                }
            )
        if remaining is not None and remaining == 0:
            break

    manifest: Dict[str, Any] = {
        "format_version": VECTOR_MANIFEST_VERSION,
        "status": "PILOT" if args.limit is not None else "COMPLETE",
        "limit": args.limit,
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "source_state_manifest_path": str(manifest_path),
        "source_state_manifest_sha256": source_manifest_sha,
        "source_files": sources,
        "source_row_count": len(global_state_ids),
        "group_count": len(group_splits),
        "split_counts": {
            split: sum(entry["row_count"] for entry in output_entries if entry["split"] == split)
            for split in sorted(ALLOWED_SPLITS)
        },
        "files": output_entries,
    }
    manifest["manifest_sha256"] = hash_dict(manifest)
    write_json(manifest, final_manifest_path, overwrite=args.overwrite and final_manifest_path.exists())
    LOGGER.info("Wrote %s vectorized states to %s", len(global_state_ids), final_manifest_path)


if __name__ == "__main__":
    main()
