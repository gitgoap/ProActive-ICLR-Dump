#!/usr/bin/env python3
"""Build hash-bound tensor manifests for the four evidence-feature ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from proactive.train.ablations import FEATURE_ABLATIONS
from proactive.train.state_data import PROBE_NUMERIC_FEATURES, PROBE_ORDER
from proactive.train.vectorized import VECTOR_MANIFEST_VERSION, atomic_torch_save, validate_tensor_shard
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", default="outputs/week5_data/vectorized_manifest.json")
    parser.add_argument("--ablation", required=True, choices=tuple(FEATURE_ABLATIONS))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _subset(shard: Dict[str, Any], indices: List[int]) -> Dict[str, Any]:
    return {
        "format_version": shard["format_version"],
        "split": shard["split"],
        "row_count": len(indices),
        "model_input": {key: value[indices].clone() for key, value in shard["model_input"].items()},
        "targets": {key: value[indices].clone() for key, value in shard["targets"].items()},
        "audit_metadata": {key: [value[index] for index in indices] for key, value in shard["audit_metadata"].items()},
        "state_id": [f"{shard['state_id'][index]}|ablation" for index in indices],
        "sampling_sources": [shard["sampling_sources"][index] for index in indices],
    }


def main() -> None:
    args = parse_args()
    source_manifest_path = Path(args.manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("format_version") != VECTOR_MANIFEST_VERSION or source_manifest.get("status") != "COMPLETE":
        raise SystemExit("Feature ablations require the complete core vector manifest")
    output_dir = Path(args.output_dir)
    output_manifest_path = output_dir / "vectorized_manifest.json"
    expected = {
        "format_version": VECTOR_MANIFEST_VERSION,
        "status": "COMPLETE",
        "limit": None,
        "ablation_id": args.ablation,
        "source_vector_manifest": str(source_manifest_path),
        "source_vector_manifest_sha256": file_sha256(source_manifest_path),
        "model_identity_used_as_input": False,
    }
    if output_manifest_path.exists() and args.resume:
        existing = json.loads(output_manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(f"Feature-ablation resume refused: {key} drift")
        for entry in existing["files"]:
            path = output_dir / entry["path"]
            if not path.exists() or file_sha256(path) != entry["sha256"]:
                raise SystemExit(f"Feature-ablation artifact drift: {path}")
        print(json.dumps(existing, indent=2))
        return
    if output_manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_manifest_path}")
    if args.dry_run:
        print(json.dumps(expected, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    split_counts: Dict[str, int] = {}
    relation_index = PROBE_ORDER.index("relation")
    feature_name = FEATURE_ABLATIONS[args.ablation]
    for entry in source_manifest["files"]:
        source = source_manifest_path.parent / entry["path"]
        if file_sha256(source) != entry["sha256"]:
            raise SystemExit(f"Source vector drift: {source}")
        shard = torch.load(source, map_location="cpu", weights_only=False)
        validate_tensor_shard(shard, entry["split"])
        if args.ablation == "no_relation_probe":
            keep = torch.nonzero(
                ~shard["model_input"]["acquired_mask"][:, relation_index].bool(),
                as_tuple=False,
            ).flatten().tolist()
            if not keep:
                continue
            output = _subset(shard, keep)
            output["model_input"]["probe_numeric"][:, relation_index, :] = 0
            output["model_input"]["acquired_mask"][:, relation_index] = False
            output["model_input"]["action_mask"][:, relation_index] = False
        else:
            output = _subset(shard, list(range(int(shard["row_count"]))))
            feature_index = PROBE_NUMERIC_FEATURES.index(str(feature_name))
            output["model_input"]["probe_numeric"][:, :, feature_index] = 0
        validate_tensor_shard(output, entry["split"])
        destination = output_dir / entry["path"]
        atomic_torch_save(output, destination, overwrite=args.overwrite and destination.exists())
        count = int(output["row_count"])
        split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + count
        entries.append(
            {
                **entry,
                "row_count": count,
                "sha256": file_sha256(destination),
                "source_path": str(source),
                "source_sha256": file_sha256(source),
            }
        )
    report: Dict[str, Any] = {
        **expected,
        "seed": source_manifest.get("seed"),
        "config_path": source_manifest.get("config_path"),
        "config_sha256": source_manifest.get("config_sha256"),
        "source_state_manifest_path": source_manifest.get("source_state_manifest_path"),
        "source_state_manifest_sha256": source_manifest.get("source_state_manifest_sha256"),
        "source_files": source_manifest.get("source_files", []),
        "source_row_count": sum(split_counts.values()),
        "group_count": source_manifest.get("group_count"),
        "split_counts": split_counts,
        "files": entries,
    }
    report["manifest_sha256"] = hash_dict(report)
    write_json(report, output_manifest_path, overwrite=args.overwrite and output_manifest_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
