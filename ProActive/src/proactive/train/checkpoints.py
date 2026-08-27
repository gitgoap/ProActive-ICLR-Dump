"""Hash-bound, atomic checkpoint and freeze contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from proactive.train.vectorized import atomic_torch_save
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


CHECKPOINT_VERSION = "proactive_diagnostic_checkpoint_v1"
POLICY_CHECKPOINT_VERSION = "proactive_policy_checkpoint_v1"
FREEZE_VERSION = "proactive_stack_freeze_v1"


def save_checkpoint(payload: Mapping[str, Any], path: str | Path, overwrite: bool = False) -> Path:
    value = dict(payload)
    version = value.get("format_version")
    if version not in {CHECKPOINT_VERSION, POLICY_CHECKPOINT_VERSION}:
        raise ValueError("Checkpoint has an unsupported format_version")
    return atomic_torch_save(value, path, overwrite=overwrite)


def load_checkpoint(
    path: str | Path,
    *,
    expected_version: str = CHECKPOINT_VERSION,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    value = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(value, dict) or value.get("format_version") != expected_version:
        raise ValueError(f"Checkpoint format mismatch: {path}")
    required = {
        "format_version",
        "stage",
        "config_sha256",
        "source_manifest_sha256",
        "seed",
        "model_state_dict",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    return value


def write_freeze_manifest(
    *,
    diagnostic_checkpoint: str | Path,
    policy_checkpoint: str | Path | None,
    selection_report: str | Path,
    config_paths: Mapping[str, str | Path],
    additional_artifacts: Mapping[str, str | Path] | None = None,
    output_path: str | Path,
    approved_by_owner: bool,
    overwrite: bool = False,
) -> Path:
    if not approved_by_owner:
        raise ValueError("Main-stack freeze requires explicit owner approval")
    paths = {
        "diagnostic_checkpoint": Path(diagnostic_checkpoint),
        "selection_report": Path(selection_report),
        **{name: Path(path) for name, path in config_paths.items()},
    }
    if additional_artifacts:
        overlap = set(paths) & set(additional_artifacts)
        if overlap:
            raise ValueError(f"Duplicate freeze artifact names: {sorted(overlap)}")
        paths.update({name: Path(path) for name, path in additional_artifacts.items()})
    if policy_checkpoint is not None:
        paths["policy_checkpoint"] = Path(policy_checkpoint)
    for name, path in paths.items():
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Freeze input {name} does not exist: {path}")
    value: Dict[str, Any] = {
        "format_version": FREEZE_VERSION,
        "status": "FROZEN",
        "owner_approved": True,
        "artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in sorted(paths.items())
        },
    }
    value["freeze_sha256"] = hash_dict(value)
    return write_json(value, output_path, overwrite=overwrite)


def validate_freeze_manifest(path: str | Path, require_policy: bool = True) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("format_version") != FREEZE_VERSION or value.get("status") != "FROZEN":
        raise ValueError("Freeze manifest is not final")
    if value.get("owner_approved") is not True:
        raise ValueError("Freeze manifest lacks owner approval")
    expected_hash = value.get("freeze_sha256")
    unsigned = {key: item for key, item in value.items() if key != "freeze_sha256"}
    if expected_hash != hash_dict(unsigned):
        raise ValueError("Freeze manifest self-hash mismatch")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Freeze manifest artifact block is missing")
    if require_policy and "policy_checkpoint" not in artifacts:
        raise ValueError("Freeze manifest does not include a policy checkpoint")
    for name, item in artifacts.items():
        artifact_path = Path(item["path"])
        if not artifact_path.exists() or file_sha256(artifact_path) != item.get("sha256"):
            raise ValueError(f"Frozen artifact drift: {name}")
    return value
