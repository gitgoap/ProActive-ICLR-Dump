"""Shared components for all mandatory diagnostic encoders."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from proactive.train.state_data import PROBE_NUMERIC_FEATURES, PROBE_ORDER


def require_model_input(model_input: Mapping[str, torch.Tensor]) -> None:
    expected = {
        "clean_features",
        "probe_numeric",
        "acquired_mask",
        "sequence_indices",
        "remaining_budget",
        "max_budget",
        "action_mask",
    }
    if set(model_input) != expected:
        raise ValueError(
            f"Model input schema mismatch: expected {sorted(expected)}, got {sorted(model_input)}"
        )
    clean = model_input["clean_features"]
    probe = model_input["probe_numeric"]
    mask = model_input["acquired_mask"]
    if clean.ndim != 2:
        raise ValueError("clean_features must have shape [batch, features]")
    if probe.ndim != 3 or probe.shape[1:] != (
        len(PROBE_ORDER),
        len(PROBE_NUMERIC_FEATURES),
    ):
        raise ValueError("probe_numeric has the wrong shape")
    if mask.shape != probe.shape[:2]:
        raise ValueError("acquired_mask has the wrong shape")


class CleanEncoder(nn.Module):
    def __init__(self, clean_dim: int = 4, output_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(clean_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, clean_features: torch.Tensor) -> torch.Tensor:
        return self.network(clean_features)


class ProbeTokenizer(nn.Module):
    def __init__(
        self,
        action_embedding_dim: int = 16,
        token_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(len(PROBE_ORDER), action_embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(len(PROBE_NUMERIC_FEATURES) + action_embedding_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        probe_numeric: torch.Tensor,
        acquired_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = probe_numeric.shape[0]
        indices = torch.arange(len(PROBE_ORDER), device=probe_numeric.device)
        identity = self.action_embedding(indices).unsqueeze(0).expand(batch_size, -1, -1)
        tokens = self.network(torch.cat([identity, probe_numeric], dim=-1))
        return tokens * acquired_mask.unsqueeze(-1).to(tokens.dtype)


class BudgetEmbedding(nn.Module):
    def __init__(self, output_dim: int = 16, max_budget: int = 7) -> None:
        super().__init__()
        if output_dim < 0:
            raise ValueError("Budget embedding dimension must be nonnegative")
        self.max_budget = max_budget
        self.output_dim = output_dim
        self.embedding = (
            nn.Embedding(max_budget + 1, output_dim) if output_dim > 0 else None
        )

    def forward(self, remaining_budget: torch.Tensor) -> torch.Tensor:
        if torch.any(remaining_budget < 0) or torch.any(remaining_budget > self.max_budget):
            raise ValueError("remaining_budget is outside the configured embedding range")
        if self.embedding is None:
            return torch.zeros(
                (*remaining_budget.shape, 0),
                dtype=torch.float32,
                device=remaining_budget.device,
            )
        return self.embedding(remaining_budget.long())
