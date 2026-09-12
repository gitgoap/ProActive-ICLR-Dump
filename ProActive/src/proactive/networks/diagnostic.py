"""Shared diagnostic heads and model factory for the Week 5 bake-off."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

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


class IndependentSourceDiagnosticModel(nn.Module):
    """Ablation with a separate evidence encoder for each source bit.

    The main model lets visual, language-prior, and alignment supervision shape
    one shared representation.  This comparison removes that sharing: each
    binary source predictor owns a fresh encoder and head.  Six-way and
    signature supervision retain a fourth encoder so the output contract and
    downstream policy interface remain identical.
    """

    def __init__(
        self,
        encoder_factory: Callable[[], nn.Module],
        *,
        state_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        def head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(state_dim, output_dim),
            )

        self.source_encoders = nn.ModuleList([encoder_factory() for _ in range(3)])
        self.source_heads = nn.ModuleList([head(1) for _ in range(3)])
        # ``FrozenDiagnosticPolicy`` deliberately consumes this public encoder.
        # In the ablation it is trained by the six-way and signature objectives,
        # while source-bit gradients remain isolated in ``source_encoders``.
        self.encoder = encoder_factory()
        self.six_way = head(6)
        self.signature = head(3)
        if any(getattr(item, "state_dim", None) != state_dim for item in self.source_encoders):
            raise ValueError("Independent source encoder state_dim mismatch")
        if getattr(self.encoder, "state_dim", None) != state_dim:
            raise ValueError("Independent shared-task encoder state_dim mismatch")

    def forward(self, model_input: Mapping[str, torch.Tensor]) -> DiagnosticOutput:
        source_hidden = [encoder(model_input) for encoder in self.source_encoders]
        bit_logits = torch.cat(
            [head(hidden) for head, hidden in zip(self.source_heads, source_hidden)],
            dim=1,
        )
        hidden = self.encoder(model_input)
        return DiagnosticOutput(
            hidden=hidden,
            bit_logits=bit_logits,
            six_way_logits=self.six_way(hidden),
            signature=self.signature(hidden),
        )


def build_diagnostic_model(name: str, architecture: Mapping[str, Any]) -> nn.Module:
    if name not in ENCODER_NAMES:
        raise ValueError(f"Unknown encoder {name!r}; expected one of {ENCODER_NAMES}")
    common = {
        "clean_dim": 4,
        "state_dim": int(architecture.get("state_dim", 128)),
        "dropout": float(architecture.get("dropout", 0.1)),
    }
    def build_encoder() -> nn.Module:
        if name == "clean_mlp":
            return CleanOnlyMLPEncoder(**common)
        evidence = {
            **common,
            "token_dim": int(architecture.get("probe_token_dim", 64)),
            "action_embedding_dim": int(architecture.get("action_embedding_dim", 16)),
            "budget_embedding_dim": int(architecture.get("budget_embedding_dim", 16)),
            "max_budget": int(architecture.get("max_budget", 7)),
        }
        if name == "deep_sets":
            return DeepSetsEncoder(
                **evidence,
                phi_hidden_dim=int(architecture.get("phi_hidden_dim", 128)),
                pooled_dim=int(architecture.get("pooled_dim", 128)),
            )
        if name == "masked_slot_mlp":
            return MaskedSlotMLPEncoder(**evidence)
        return GRUEvidenceEncoder(**evidence)

    mode = str(architecture.get("multi_task_mode", "shared_encoder"))
    if mode == "independent_source_encoders":
        return IndependentSourceDiagnosticModel(
            build_encoder,
            state_dim=common["state_dim"],
            dropout=common["dropout"],
        )
    if mode != "shared_encoder":
        raise ValueError(
            "Unknown multi_task_mode; expected shared_encoder or "
            "independent_source_encoders"
        )
    return DiagnosticModel(
        build_encoder(),
        state_dim=common["state_dim"],
        dropout=common["dropout"],
    )
