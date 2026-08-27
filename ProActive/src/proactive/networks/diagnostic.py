"""Shared diagnostic heads and model factory for the Week 5 bake-off."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
from torch import nn

from proactive.networks.encoders import (
    CleanOnlyMLPEncoder,
    DeepSetsEncoder,
    GRUEvidenceEncoder,
    MaskedSlotMLPEncoder,
)


ENCODER_NAMES = ("clean_mlp", "gru", "masked_slot_mlp", "deep_sets")


@dataclass(frozen=True)
class DiagnosticOutput:
    hidden: torch.Tensor
    bit_logits: torch.Tensor
    six_way_logits: torch.Tensor
    signature: torch.Tensor


class DiagnosticHeads(nn.Module):
    def __init__(self, state_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()

        def head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(state_dim, output_dim),
            )

        self.source_bits = head(3)
        self.six_way = head(6)
        self.signature = head(3)

    def forward(self, hidden: torch.Tensor) -> DiagnosticOutput:
        return DiagnosticOutput(
            hidden=hidden,
            bit_logits=self.source_bits(hidden),
            six_way_logits=self.six_way(hidden),
            signature=self.signature(hidden),
        )


class DiagnosticModel(nn.Module):
    def __init__(self, encoder: nn.Module, state_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if getattr(encoder, "state_dim", None) != state_dim:
            raise ValueError("Encoder state_dim does not match diagnostic heads")
        self.encoder = encoder
        self.heads = DiagnosticHeads(state_dim, dropout)

    def forward(self, model_input: Mapping[str, torch.Tensor]) -> DiagnosticOutput:
        return self.heads(self.encoder(model_input))


def build_diagnostic_model(name: str, architecture: Mapping[str, Any]) -> DiagnosticModel:
    if name not in ENCODER_NAMES:
        raise ValueError(f"Unknown encoder {name!r}; expected one of {ENCODER_NAMES}")
    common = {
        "clean_dim": 4,
        "state_dim": int(architecture.get("state_dim", 128)),
        "dropout": float(architecture.get("dropout", 0.1)),
    }
    if name == "clean_mlp":
        encoder = CleanOnlyMLPEncoder(**common)
    else:
        evidence = {
            **common,
            "token_dim": int(architecture.get("probe_token_dim", 64)),
            "action_embedding_dim": int(architecture.get("action_embedding_dim", 16)),
            "budget_embedding_dim": int(architecture.get("budget_embedding_dim", 16)),
            "max_budget": int(architecture.get("max_budget", 7)),
        }
        if name == "deep_sets":
            encoder = DeepSetsEncoder(
                **evidence,
                phi_hidden_dim=int(architecture.get("phi_hidden_dim", 128)),
                pooled_dim=int(architecture.get("pooled_dim", 128)),
            )
        elif name == "masked_slot_mlp":
            encoder = MaskedSlotMLPEncoder(**evidence)
        else:
            encoder = GRUEvidenceEncoder(**evidence)
    return DiagnosticModel(
        encoder,
        state_dim=common["state_dim"],
        dropout=common["dropout"],
    )

