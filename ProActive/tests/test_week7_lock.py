from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from proactive.conformal.contracts import authorize_locked_test
from proactive.train.checkpoints import (
    CHECKPOINT_VERSION,
    POLICY_CHECKPOINT_VERSION,
    save_checkpoint,
    write_freeze_manifest,
)
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256


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


def test_locked_test_requires_final_freeze_bound_calibration(tmp_path: Path) -> None:
    diagnostic = tmp_path / "diagnostic.pt"
    policy = tmp_path / "policy.pt"
    selection = tmp_path / "selection.json"
    config = tmp_path / "config.yaml"
    freeze = tmp_path / "freeze.json"
    _checkpoint(diagnostic, CHECKPOINT_VERSION)
    _checkpoint(policy, POLICY_CHECKPOINT_VERSION)
    selection.write_text("{}\n", encoding="utf-8")
    config.write_text("x: 1\n", encoding="utf-8")
    write_freeze_manifest(
        diagnostic_checkpoint=diagnostic,
        policy_checkpoint=policy,
        selection_report=selection,
        config_paths={"config": config},
        output_path=freeze,
        approved_by_owner=True,
    )
    report = {
        "format_version": "week7_final_aps_v1",
        "status": "FINAL_FROZEN",
        "fit_split": "cal",
        "test_used": False,
        "freeze_manifest_sha256": file_sha256(freeze),
        "thresholds": {"1": {"0.9": {"threshold": 0.9}}},
    }
    report["thresholds_sha256"] = hash_dict(report)
    authorize_locked_test(freeze_manifest_path=freeze, aps_report=report)
    leaked = dict(report)
    leaked["test_used"] = True
    leaked["thresholds_sha256"] = hash_dict({key: value for key, value in leaked.items() if key != "thresholds_sha256"})
    with pytest.raises(ValueError, match="split provenance"):
        authorize_locked_test(freeze_manifest_path=freeze, aps_report=leaked)


def test_locked_test_rejects_unrelated_freeze_hash(tmp_path: Path) -> None:
    report = {
        "format_version": "week7_final_aps_v1",
        "status": "FINAL_FROZEN",
        "fit_split": "cal",
        "test_used": False,
        "freeze_manifest_sha256": "0" * 64,
        "thresholds": {"1": {}},
    }
    report["thresholds_sha256"] = hash_dict(report)
    with pytest.raises((FileNotFoundError, ValueError)):
        authorize_locked_test(freeze_manifest_path=tmp_path / "missing.json", aps_report=report)
