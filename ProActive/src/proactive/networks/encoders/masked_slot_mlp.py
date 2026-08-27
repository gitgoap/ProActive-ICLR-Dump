"""Plan §7.3 exact invariant masked-slot MLP control."""

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
from proactive.train.state_data import PROBE_ORDER


class MaskedSlotMLPEncoder(nn.Module):
    encoder_name = "masked_slot_mlp"

    def __init__(
        self,
        clean_dim: int = 4,
        token_dim: int = 64,
        action_embedding_dim: int = 16,
        budget_embedding_dim: int = 16,
        state_dim: int = 128,
        dropout: float = 0.1,
        max_budget: int = 7,
    ) -> None:
        super().__init__()
        self.clean_encoder = CleanEncoder(clean_dim, 64, dropout)
        self.probe_tokenizer = ProbeTokenizer(action_embedding_dim, token_dim, dropout)
        self.budget_embedding = BudgetEmbedding(budget_embedding_dim, max_budget)
        input_dim = 64 + len(PROBE_ORDER) * token_dim + len(PROBE_ORDER) + budget_embedding_dim
        self.rho = nn.Sequential(
            nn.Linear(input_dim, state_dim),
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
        slots = self.probe_tokenizer(model_input["probe_numeric"], mask).flatten(start_dim=1)
        budget = self.budget_embedding(model_input["remaining_budget"])
        return self.rho(
            torch.cat([clean, slots, mask.to(clean.dtype), budget], dim=-1)
        )

