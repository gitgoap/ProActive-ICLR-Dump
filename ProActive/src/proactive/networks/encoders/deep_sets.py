"""Plan §7.2 sum-pooled Deep Sets encoder."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from proactive.networks.encoders.common import (
    BudgetEmbedding,
    CleanEncoder,
    ProbeTokenizer,
    require_model_input,
)


class DeepSetsEncoder(nn.Module):
    encoder_name = "deep_sets"

    def __init__(
        self,
        clean_dim: int = 4,
        token_dim: int = 64,
        action_embedding_dim: int = 16,
        budget_embedding_dim: int = 16,
        phi_hidden_dim: int = 128,
        pooled_dim: int = 128,
        state_dim: int = 128,
        dropout: float = 0.1,
        max_budget: int = 7,
    ) -> None:
        super().__init__()
        self.clean_encoder = CleanEncoder(clean_dim, 64, dropout)
        self.probe_tokenizer = ProbeTokenizer(action_embedding_dim, token_dim, dropout)
        self.phi = nn.Sequential(
            nn.Linear(token_dim, phi_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phi_hidden_dim, pooled_dim),
            nn.ReLU(),
        )
        self.budget_embedding = BudgetEmbedding(budget_embedding_dim, max_budget)
        self.rho = nn.Sequential(
            nn.Linear(64 + pooled_dim + 1 + budget_embedding_dim, state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.state_dim = state_dim

    def forward(self, model_input: Mapping[str, torch.Tensor]) -> torch.Tensor:
        require_model_input(model_input)
        clean = self.clean_encoder(model_input["clean_features"])
        mask = model_input["acquired_mask"].bool()
        tokens = self.probe_tokenizer(model_input["probe_numeric"], mask)
        transformed = self.phi(tokens) * mask.unsqueeze(-1).to(tokens.dtype)
        pooled = transformed.sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).to(clean.dtype)
        budget = self.budget_embedding(model_input["remaining_budget"])
        return self.rho(torch.cat([clean, pooled, count, budget], dim=-1))

