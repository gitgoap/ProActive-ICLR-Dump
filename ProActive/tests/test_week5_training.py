from __future__ import annotations

import json

import pytest
import torch

from proactive.networks.diagnostic import DiagnosticOutput
from proactive.networks.losses import compute_training_weights, diagnostic_loss
from proactive.train.checkpoints import (
    CHECKPOINT_VERSION,
    save_checkpoint,
    validate_completed_training_report,
    validate_early_stopping_history,
)
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256


def test_diagnostic_loss_matches_declared_weighting() -> None:
    output = DiagnosticOutput(
        hidden=torch.zeros(2, 4),
        bit_logits=torch.zeros(2, 3),
        six_way_logits=torch.zeros(2, 6),
        signature=torch.zeros(2, 3),
    )
    targets = {
        "source_bits": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        "six_way": torch.tensor([0, 1]),
        "signature": torch.ones(2, 3),
    }
    loss = diagnostic_loss(
        output,
        targets,
        bit_pos_weight=torch.ones(3),
        class_weight=torch.ones(6),
        six_way_weight=0.5,
        signature_weight=0.1,
    )
    assert torch.allclose(loss.total, loss.source_bits + 0.5 * loss.six_way + 0.1 * loss.signature)


def test_class_weights_are_train_derived_and_fail_on_missing_class() -> None:
    bits = torch.tensor([[0, 0, 0], [1, 1, 1]] * 3)
    classes = torch.arange(6)
    bit_weight, class_weight = compute_training_weights(bits, classes)
    assert torch.equal(bit_weight, torch.ones(3))
    assert torch.equal(class_weight, torch.ones(6))
    with pytest.raises(ValueError, match="Every six-way class"):
        compute_training_weights(bits[:4], torch.tensor([0, 1, 2, 3]))


def test_completed_early_stopping_history_is_idempotent() -> None:
    validate_early_stopping_history(
        [(0.5, 0.2), (0.6, 0.1), (0.59, 0.3), (0.58, 0.4)],
        patience=2,
        minimize=False,
        reported_best_epoch=1,
    )


def test_history_cannot_continue_after_early_stopping_boundary() -> None:
    with pytest.raises(ValueError, match="continues after"):
        validate_early_stopping_history(
            [(0.5, 0.2), (0.6, 0.1), (0.59, 0.3), (0.58, 0.4), (0.7, 0.5)],
            patience=2,
            minimize=False,
            reported_best_epoch=4,
        )


def test_completed_history_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_early_stopping_history(
            [(0.5, 0.2), (float("nan"), 0.3)],
            patience=2,
            minimize=False,
            reported_best_epoch=0,
        )


def test_policy_early_stopping_uses_minimized_lexicographic_score() -> None:
    validate_early_stopping_history(
        [(0.8, -0.7), (0.7, -0.6), (0.7, -0.7), (0.71, -0.9)],
        patience=2,
        minimize=True,
        reported_best_epoch=2,
    )


def test_completed_training_report_binds_checkpoint_and_provenance(tmp_path) -> None:
    checkpoint_path = tmp_path / "model.best.pt"
    report_path = tmp_path / "model.validation.json"
    save_checkpoint(
        {
            "format_version": CHECKPOINT_VERSION,
            "stage": "test",
            "config_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "seed": 42,
            "model_state_dict": {},
            "best_epoch": 3,
        },
        checkpoint_path,
    )
    report = {
        "config_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "seed": 42,
        "best_epoch": 3,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    report["report_sha256"] = hash_dict(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = validate_completed_training_report(
        report_path,
        checkpoint_path,
        expected_report={"seed": 42},
        expected_checkpoint={"seed": 42},
        expected_version=CHECKPOINT_VERSION,
    )
    assert loaded == report

    report["checkpoint_sha256"] = "0" * 64
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hash_dict(unsigned)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_sha256 drift"):
        validate_completed_training_report(
            report_path,
            checkpoint_path,
            expected_report={"seed": 42},
            expected_checkpoint={"seed": 42},
            expected_version=CHECKPOINT_VERSION,
        )
