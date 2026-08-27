from __future__ import annotations

import torch

from proactive.networks.diagnostic import DiagnosticOutput
from proactive.train.distillation import distillation_loss


def _output(offset: float) -> DiagnosticOutput:
    return DiagnosticOutput(
        hidden=torch.zeros(2, 4),
        bit_logits=torch.full((2, 3), offset),
        six_way_logits=torch.full((2, 6), offset),
        signature=torch.full((2, 3), offset),
    )


def test_distillation_loss_is_lower_for_matching_teacher() -> None:
    teacher = _output(0.0)
    matching = distillation_loss(_output(0.0), teacher)
    shifted = distillation_loss(_output(2.0), teacher)
    assert matching < shifted
