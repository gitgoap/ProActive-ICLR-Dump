from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from proactive.train.checkpoints import (
    CHECKPOINT_VERSION,
    POLICY_CHECKPOINT_VERSION,
    freeze_includes_file,
    save_checkpoint,
    validate_freeze_manifest,
    write_freeze_manifest,
)


def _checkpoint(path: Path, version: str, *, seed: int = 42, weight: float = 1.0) -> None:
    save_checkpoint(
        {
            "format_version": version,
            "stage": "test",
            "config_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "seed": seed,
            "model_state_dict": {"weight": torch.full((1,), weight)},
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


def test_freeze_includes_selected_and_comparison_diagnostics(tmp_path: Path) -> None:
    selected = tmp_path / "selected.pt"
    comparison = tmp_path / "comparison.pt"
    unrelated = tmp_path / "unrelated.pt"
    selection = tmp_path / "selection.json"
    config = tmp_path / "config.yaml"
    freeze_path = tmp_path / "freeze.json"
    _checkpoint(selected, CHECKPOINT_VERSION, seed=42, weight=1.0)
    _checkpoint(comparison, CHECKPOINT_VERSION, seed=43, weight=2.0)
    _checkpoint(unrelated, CHECKPOINT_VERSION, seed=44, weight=3.0)
    selection.write_text("{}\n", encoding="utf-8")
    config.write_text("seed: 42\n", encoding="utf-8")
    write_freeze_manifest(
        diagnostic_checkpoint=selected,
        policy_checkpoint=None,
        selection_report=selection,
        config_paths={"config": config},
        additional_artifacts={"diagnostic_checkpoint_gru_random": comparison},
        output_path=freeze_path,
        approved_by_owner=True,
    )
    freeze = validate_freeze_manifest(freeze_path, require_policy=False)
    assert freeze_includes_file(
        freeze,
        selected,
        artifact_name_prefix="diagnostic_checkpoint",
    )
    assert freeze_includes_file(
        freeze,
        comparison,
        artifact_name_prefix="diagnostic_checkpoint",
    )
    assert not freeze_includes_file(
        freeze,
        unrelated,
        artifact_name_prefix="diagnostic_checkpoint",
    )
    assert not freeze_includes_file(
        freeze,
        config,
        artifact_name_prefix="diagnostic_checkpoint",
    )
