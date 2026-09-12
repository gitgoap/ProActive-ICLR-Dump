#!/usr/bin/env python3
"""Materialize a leakage-safe leave-one-model-out fold from cached artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import yaml

from proactive.train.vectorized import TENSOR_SHARD_VERSION, VECTOR_MANIFEST_VERSION, atomic_torch_save, validate_tensor_shard
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_jsonl


LOMO_BUILDER_REVISION = "metadata_identity_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--vector_manifest", default="outputs/week5_data/vectorized_manifest.json")
    parser.add_argument("--teacher_dir", default="outputs/teacher_core_contract_v1_recovered")
    parser.add_argument("--labels_dir", default="outputs/labels_core")
    parser.add_argument("--states_dir", default="outputs/states_v1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _keep(split: str, model_id: str, heldout_model_id: str) -> bool:
    if split in {"train", "val", "cal"}:
        return model_id != heldout_model_id
    if split == "test":
        return model_id == heldout_model_id
    return False


def _jsonl_files(directory: Path, prefix: str) -> List[Path]:
    return [
        path
        for path in sorted(directory.glob(f"{prefix}*.jsonl"))
        if not path.name.endswith(".failures.jsonl")
    ]


def _row_split_and_model(
    row: Mapping[str, Any], source: Path
) -> tuple[str, str]:
    """Read identity from teacher/label rows or canonical partial-state metadata."""
    identity: Mapping[str, Any]
    if row.get("record_type") == "partial_state_v1":
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise SystemExit(f"Missing partial-state metadata in {source}")
        identity = metadata
    else:
        identity = row
    split = identity.get("split")
    model_id = identity.get("model_id")
    if not isinstance(split, str) or not split:
        raise SystemExit(f"Missing split in {source}")
    if not isinstance(model_id, str) or not model_id:
        raise SystemExit(f"Missing model_id in {source}")
    return split, model_id


def _filter_jsonl(
    sources: Iterable[Path],
    destination: Path,
    *,
    heldout_model_id: str,
    overwrite: bool,
) -> Dict[str, Any]:
    rows = []
    split_counts: Counter[str] = Counter()
    observed_rows = 0
    observed_identities: Counter[str] = Counter()
    for source in sources:
        for row in iter_jsonl(source):
            split, model_id = _row_split_and_model(row, source)
            observed_rows += 1
            observed_identities[f"{split}|{model_id}"] += 1
            if _keep(split, model_id, heldout_model_id):
                rows.append(row)
                split_counts[split] += 1
    if not rows:
        raise SystemExit(
            f"LOMO filtering produced no rows for {destination.name}; "
            f"builder_revision={LOMO_BUILDER_REVISION}; "
            f"heldout_model_id={heldout_model_id!r}; "
            f"observed_rows={observed_rows}; "
            f"observed_identities={dict(sorted(observed_identities.items()))}"
        )
    write_jsonl(rows, destination, overwrite=overwrite and destination.exists())
    return {
        "path": str(destination),
        "sha256": file_sha256(destination),
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
    }


def _slice_shard(shard: Mapping[str, Any], indices: List[int]) -> Dict[str, Any]:
    split = str(shard["split"])
    return {
        "format_version": TENSOR_SHARD_VERSION,
        "split": split,
        "row_count": len(indices),
        "model_input": {key: value[indices].clone() for key, value in shard["model_input"].items()},
        "targets": {key: value[indices].clone() for key, value in shard["targets"].items()},
        "audit_metadata": {key: [value[index] for index in indices] for key, value in shard["audit_metadata"].items()},
        "state_id": [shard["state_id"][index] for index in indices],
        "sampling_sources": [shard["sampling_sources"][index] for index in indices],
    }


def main() -> None:
    args = parse_args()
    with open(args.model_config, "r", encoding="utf-8") as handle:
        heldout_model_id = str(yaml.safe_load(handle)["model_id"])
    print(f"LOMO builder revision: {LOMO_BUILDER_REVISION}")
    vector_manifest_path = Path(args.vector_manifest)
    with open(vector_manifest_path, "r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    if source_manifest.get("format_version") != VECTOR_MANIFEST_VERSION or source_manifest.get("status") != "COMPLETE":
        raise SystemExit("LOMO requires the complete core vectorized manifest")
    output_dir = Path(args.output_dir)
    report_path = output_dir / "lomo_fold_manifest.json"
    expected = {
        "format_version": "lomo_fold_v1",
        "builder_revision": LOMO_BUILDER_REVISION,
        "heldout_model_id": heldout_model_id,
        "heldout_model_config": str(args.model_config),
        "heldout_model_config_sha256": file_sha256(args.model_config),
        "source_vector_manifest": str(vector_manifest_path),
        "source_vector_manifest_sha256": file_sha256(vector_manifest_path),
        "selection_rule": "source_train_val_cal_exclude_model__test_only_model_v1",
        "model_identity_used_as_learner_feature": False,
    }
    if report_path.exists() and args.resume:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"LOMO resume refused: {key} drift")
        for artifact in existing.get("artifacts", []):
            path = Path(artifact["path"])
            if not path.exists() or file_sha256(path) != artifact["sha256"]:
                raise SystemExit(f"LOMO artifact drift: {path}")
        print(json.dumps(existing, indent=2))
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"LOMO output directory is not empty: {output_dir}")
    if args.dry_run:
        print(json.dumps(expected, indent=2))
        return

    vector_dir = output_dir / "vectorized"
    teacher_dir = output_dir / "teachers"
    labels_dir = output_dir / "labels"
    states_dir = output_dir / "states"
    for path in (vector_dir, teacher_dir, labels_dir, states_dir):
        path.mkdir(parents=True, exist_ok=True)

    entries = []
    split_counts: Counter[str] = Counter()
    for entry in source_manifest["files"]:
        source = vector_manifest_path.parent / entry["path"]
        if file_sha256(source) != entry["sha256"]:
            raise SystemExit(f"Source vector shard drift: {source}")
        shard = torch.load(source, map_location="cpu", weights_only=False)
        validate_tensor_shard(shard, str(entry["split"]))
        indices = [
            index
            for index, model_id in enumerate(shard["audit_metadata"]["model_id"])
            if _keep(str(shard["split"]), str(model_id), heldout_model_id)
        ]
        if not indices:
            continue
        sliced = _slice_shard(shard, indices)
        destination = vector_dir / source.name
        atomic_torch_save(sliced, destination, overwrite=args.overwrite and destination.exists())
        entries.append(
            {
                "path": destination.name,
                "split": sliced["split"],
                "row_count": len(indices),
                "sha256": file_sha256(destination),
                "source_path": str(source),
                "source_sha256": file_sha256(source),
            }
        )
        split_counts[sliced["split"]] += len(indices)
    required_splits = {"train", "val", "cal", "test"}
    if set(split_counts) != required_splits:
        raise SystemExit(f"LOMO vector split coverage mismatch: {dict(split_counts)}")
    vector_manifest: Dict[str, Any] = {
        "format_version": VECTOR_MANIFEST_VERSION,
        "status": "COMPLETE",
        "limit": None,
        "seed": source_manifest.get("seed"),
        "config_path": source_manifest.get("config_path"),
        "config_sha256": source_manifest.get("config_sha256"),
        "source_state_manifest_path": source_manifest.get("source_state_manifest_path"),
        "source_state_manifest_sha256": source_manifest.get("source_state_manifest_sha256"),
        "source_files": source_manifest.get("source_files", []),
        "source_row_count": sum(split_counts.values()),
        "group_count": len(
            {
                value
                for entry in entries
                for value in torch.load(vector_dir / entry["path"], map_location="cpu", weights_only=False)["audit_metadata"]["group_id"]
            }
        ),
        "split_counts": {split: int(split_counts[split]) for split in sorted(required_splits)},
        "files": entries,
        "lomo_heldout_model_id": heldout_model_id,
    }
    vector_manifest["manifest_sha256"] = hash_dict(vector_manifest)
    lomo_vector_path = vector_dir / "vectorized_manifest.json"
    write_json(vector_manifest, lomo_vector_path, overwrite=args.overwrite and lomo_vector_path.exists())

    artifacts = [
        {"kind": "vector_manifest", "path": str(lomo_vector_path), "sha256": file_sha256(lomo_vector_path)}
    ]
    for kind, source_dir, prefix, target_dir, target_name in (
        ("teacher", Path(args.teacher_dir), "teacher_", teacher_dir, "teacher_lomo.jsonl"),
        ("labels", Path(args.labels_dir), "labels_", labels_dir, "labels_lomo.jsonl"),
        ("states", Path(args.states_dir), "states_", states_dir, "states_lomo.jsonl"),
    ):
        sources = _jsonl_files(source_dir, prefix)
        if not sources:
            raise SystemExit(f"No {kind} sources found in {source_dir}")
        artifacts.append(
            {
                "kind": kind,
                **_filter_jsonl(
                    sources,
                    target_dir / target_name,
                    heldout_model_id=heldout_model_id,
                    overwrite=args.overwrite,
                ),
            }
        )
    report: Dict[str, Any] = {
        **expected,
        "status": "COMPLETE",
        "split_counts": dict(sorted(split_counts.items())),
        "artifacts": artifacts,
    }
    report["report_sha256"] = hash_dict(report)
    write_json(report, report_path, overwrite=args.overwrite and report_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
