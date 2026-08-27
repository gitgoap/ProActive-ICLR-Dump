from __future__ import annotations

import pytest
import torch

from proactive.networks.diagnostic import DiagnosticOutput
from proactive.networks.losses import compute_training_weights, diagnostic_loss


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
