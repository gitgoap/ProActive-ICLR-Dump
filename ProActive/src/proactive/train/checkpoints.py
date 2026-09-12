"""Hash-bound, atomic checkpoint and freeze contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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


def validate_completed_training_report(
    report_path: str | Path,
    checkpoint_path: str | Path,
    *,
    expected_report: Mapping[str, Any],
    expected_checkpoint: Mapping[str, Any],
    expected_version: str,
) -> Dict[str, Any] | None:
    """Return a hash-valid completed report, or ``None`` when none exists.

    A final validation report is the completion marker for the diagnostic and
    policy trainers.  ``--resume`` must not continue training past an already
    completed early-stopping boundary: doing so can replace the selected
    checkpoint and invalidate every downstream hash-bound artifact.
    """

    report_path = Path(report_path)
    checkpoint_path = Path(checkpoint_path)
    if not report_path.exists():
        return None
    if not report_path.is_file():
        raise ValueError(f"Completed training report is not a file: {report_path}")
    if not checkpoint_path.is_file():
        raise ValueError(
            f"Completed training report exists but checkpoint is missing: {checkpoint_path}"
        )
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"Completed training report is not a mapping: {report_path}")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != hash_dict(unsigned):
        raise ValueError(f"Completed training report self-hash mismatch: {report_path}")
    for key, value in expected_report.items():
        if report.get(key) != value:
            raise ValueError(f"Completed training report {key} drift")
    if report.get("checkpoint_path") != str(checkpoint_path):
        raise ValueError("Completed training report checkpoint_path drift")
    if report.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise ValueError("Completed training report checkpoint_sha256 drift")

    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_version=expected_version,
        map_location="cpu",
    )
    for key, value in expected_checkpoint.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Completed training checkpoint {key} drift")
    if int(report.get("best_epoch", -2)) != int(checkpoint.get("best_epoch", -1)):
        raise ValueError("Completed training report best_epoch drift")
    return report


def validate_early_stopping_history(
    scores: Sequence[Sequence[float]],
    *,
    patience: int,
    minimize: bool,
    reported_best_epoch: int,
) -> None:
    """Reject histories that continued after their first stopping boundary."""

    if patience <= 0:
        raise ValueError("Early-stopping patience must be positive")
    if not scores:
        raise ValueError("Completed training history is empty")
    best: tuple[float, ...] | None = None
    best_epoch = -1
    without_improvement = 0
    stopping_epoch: int | None = None
    score_width: int | None = None
    for epoch, raw_score in enumerate(scores):
        score = tuple(float(value) for value in raw_score)
        if not score or any(not math.isfinite(value) for value in score):
            raise ValueError("Completed training history contains a non-finite score")
        if score_width is None:
            score_width = len(score)
        elif len(score) != score_width:
            raise ValueError("Completed training history score shape drift")
        improved = best is None or (score < best if minimize else score > best)
        if improved:
            best = score
            best_epoch = epoch
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= patience:
            stopping_epoch = epoch
            break
    if stopping_epoch is not None and stopping_epoch != len(scores) - 1:
        raise ValueError(
            "Completed training history continues after the early-stopping boundary"
        )
    if best_epoch != int(reported_best_epoch):
        raise ValueError("Completed training history best_epoch drift")


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


def freeze_includes_file(
    freeze: Mapping[str, Any],
    path: str | Path,
    *,
    artifact_name_prefix: str | None = None,
) -> bool:
    """Return whether ``path`` is hash-bound in an already validated freeze.

    ``write_freeze_manifest`` stores the selected diagnostic under
    ``diagnostic_checkpoint`` and comparison diagnostics under names such as
    ``diagnostic_checkpoint_gru_random_permutation``. Week 6 must accept any
    explicitly frozen diagnostic comparison while still rejecting arbitrary
    checkpoints.
    """

    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Freeze manifest artifact block is missing")
    expected_sha = file_sha256(path)
    return any(
        (artifact_name_prefix is None or str(name).startswith(artifact_name_prefix))
        and isinstance(item, Mapping)
        and item.get("sha256") == expected_sha
        for name, item in artifacts.items()
    )
