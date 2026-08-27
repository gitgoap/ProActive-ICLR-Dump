from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from proactive.train.checkpoints import (
    CHECKPOINT_VERSION,
    POLICY_CHECKPOINT_VERSION,
    save_checkpoint,
    validate_freeze_manifest,
    write_freeze_manifest,
)


def _checkpoint(path: Path, version: str) -> None:
    save_checkpoint(
        {
            "format_version": version,
            "stage": "test",
            "config_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "seed": 42,
            "model_state_dict": {"weight": torch.ones(1)},
        },
        path,
    )


def test_freeze_requires_approval_and_detects_artifact_drift(tmp_path: Path) -> None:
    diagnostic = tmp_path / "diagnostic.pt"
    policy = tmp_path / "policy.pt"
    selection = tmp_path / "selection.json"
    config = tmp_path / "config.yaml"
    freeze = tmp_path / "freeze.json"
    _checkpoint(diagnostic, CHECKPOINT_VERSION)
    _checkpoint(policy, POLICY_CHECKPOINT_VERSION)
    selection.write_text("{}\n", encoding="utf-8")
    config.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="approval"):
        write_freeze_manifest(
            diagnostic_checkpoint=diagnostic,
            policy_checkpoint=policy,
            selection_report=selection,
            config_paths={"config": config},
            output_path=freeze,
            approved_by_owner=False,
        )
    write_freeze_manifest(
        diagnostic_checkpoint=diagnostic,
        policy_checkpoint=policy,
        selection_report=selection,
        config_paths={"config": config},
        output_path=freeze,
        approved_by_owner=True,
    )
    validate_freeze_manifest(freeze)
    config.write_text("seed: 43\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drift"):
        validate_freeze_manifest(freeze)
