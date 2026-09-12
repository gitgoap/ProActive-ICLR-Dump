"""Action-conditioned value-of-information head (Plan §9.3)."""

from __future__ import annotations

import torch
from torch import nn

from proactive.networks.diagnostic import DiagnosticModel
from proactive.train.state_data import PROBE_ORDER


class ActionConditionedVOIHead(nn.Module):
    """Predict one value for each non-STOP probe; STOP has fixed value zero."""

    def __init__(
        self,
        state_dim: int = 128,
        action_embedding_dim: int = 16,
        budget_embedding_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_budget: int = 7,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(len(PROBE_ORDER), action_embedding_dim)
        if budget_embedding_dim < 0:
            raise ValueError("budget_embedding_dim must be nonnegative")
        self.budget_embedding = (
            nn.Embedding(max_budget + 1, budget_embedding_dim)
            if budget_embedding_dim > 0
            else None
        )
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_embedding_dim + budget_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.max_budget = max_budget

    def forward(self, hidden: torch.Tensor, remaining_budget: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or remaining_budget.shape != (hidden.shape[0],):
            raise ValueError("VOI head expects hidden [B,D] and remaining_budget [B]")
        if torch.any(remaining_budget < 0) or torch.any(remaining_budget > self.max_budget):
            raise ValueError("remaining_budget is outside configured range")
        batch = hidden.shape[0]
        actions = self.action_embedding(
            torch.arange(len(PROBE_ORDER), device=hidden.device)
        ).unsqueeze(0).expand(batch, -1, -1)
        if self.budget_embedding is None:
            budget = hidden.new_zeros((batch, len(PROBE_ORDER), 0))
        else:
            budget = self.budget_embedding(remaining_budget.long()).unsqueeze(1).expand(
                -1, len(PROBE_ORDER), -1
            )
        state = hidden.unsqueeze(1).expand(-1, len(PROBE_ORDER), -1)
        non_stop = self.network(torch.cat([state, actions, budget], dim=-1)).squeeze(-1)
        stop = torch.zeros((batch, 1), dtype=non_stop.dtype, device=non_stop.device)
        return torch.cat([non_stop, stop], dim=1)


class FrozenDiagnosticPolicy(nn.Module):
    """A frozen diagnostic encoder plus trainable action-conditioned VOI head."""

    def __init__(self, diagnostic: DiagnosticModel, head: ActionConditionedVOIHead) -> None:
        super().__init__()
        self.diagnostic = diagnostic
        self.voi_head = head
        for parameter in self.diagnostic.parameters():
            parameter.requires_grad_(False)
        self.diagnostic.eval()

    def train(self, mode: bool = True) -> "FrozenDiagnosticPolicy":
        super().train(mode)
        # Dropout in the frozen encoder/heads must never perturb VOI inputs.
        self.diagnostic.eval()
        return self

    def encode(self, model_input: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self.diagnostic.encoder(model_input)

    def forward(self, model_input: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encode(model_input)
        return self.voi_head(hidden, model_input["remaining_budget"])
