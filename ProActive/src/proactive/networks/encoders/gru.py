"""Mandatory order-sensitive GRU baseline from Plan §7.4."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from proactive.networks.encoders.common import (
    BudgetEmbedding,
    CleanEncoder,
    ProbeTokenizer,
    require_model_input,
)


class GRUEvidenceEncoder(nn.Module):
    encoder_name = "gru"

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
        self.clean_encoder = CleanEncoder(clean_dim, token_dim, dropout)
        self.probe_tokenizer = ProbeTokenizer(action_embedding_dim, token_dim, dropout)
        self.gru = nn.GRU(token_dim, state_dim, batch_first=True)
        self.budget_embedding = BudgetEmbedding(budget_embedding_dim, max_budget)
        self.rho = nn.Sequential(
            nn.Linear(state_dim + token_dim + 1 + budget_embedding_dim, state_dim),
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
        slots = self.probe_tokenizer(model_input["probe_numeric"], mask)
        order = model_input["sequence_indices"].long()
        if order.shape != mask.shape:
            raise ValueError("sequence_indices has the wrong shape")
        lengths = mask.sum(dim=1).long()
        safe_order = order.clamp_min(0)
        sequence = slots.gather(
            1, safe_order.unsqueeze(-1).expand(-1, -1, slots.shape[-1])
        )
        sequence = torch.cat([clean.unsqueeze(1), sequence], dim=1)
        packed = pack_padded_sequence(
            sequence,
            (lengths + 1).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        budget = self.budget_embedding(model_input["remaining_budget"])
        count = lengths.unsqueeze(1).to(clean.dtype)
        return self.rho(torch.cat([hidden[-1], clean, count, budget], dim=-1))

