"""Reviewer-facing zero-acquisition diagnostic controls."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from proactive.networks.diagnostic import DiagnosticHeads, DiagnosticOutput


class ScalarConfidenceDiagnostic(nn.Module):
    """Diagnostic baseline that receives only normalized clean answer probability."""

    def __init__(self, state_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
        )
        self.heads = DiagnosticHeads(state_dim=state_dim, dropout=dropout)

    def forward(self, model_input: Mapping[str, torch.Tensor]) -> DiagnosticOutput:
        clean = model_input["clean_features"]
        if clean.ndim != 2 or clean.shape[1] != 4:
            raise ValueError("Scalar baseline expects clean_features [B, 4]")
        return self.heads(self.encoder(clean[:, :1]))
