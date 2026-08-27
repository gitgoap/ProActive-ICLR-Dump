"""Clean-only learned diagnostic baseline."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from proactive.networks.encoders.common import CleanEncoder, require_model_input


class CleanOnlyMLPEncoder(nn.Module):
    """Use no probe evidence; accepted shared fields are deliberately ignored."""

    encoder_name = "clean_mlp"

    def __init__(self, clean_dim: int = 4, state_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.clean_encoder = CleanEncoder(clean_dim, 64, dropout)
        self.output = nn.Sequential(
            nn.Linear(64, state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.state_dim = state_dim

    def forward(self, model_input: Mapping[str, torch.Tensor]) -> torch.Tensor:
        require_model_input(model_input)
        return self.output(self.clean_encoder(model_input["clean_features"]))

