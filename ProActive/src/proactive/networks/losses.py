"""Plan-exact diagnostic and policy objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from proactive.networks.diagnostic import DiagnosticOutput


@dataclass(frozen=True)
class DiagnosticLoss:
    total: torch.Tensor
    source_bits: torch.Tensor
    six_way: torch.Tensor
    signature: torch.Tensor


def diagnostic_loss(
    output: DiagnosticOutput,
    targets: Mapping[str, torch.Tensor],
    *,
    bit_pos_weight: torch.Tensor,
    class_weight: torch.Tensor,
    six_way_weight: float = 0.5,
    signature_weight: float = 0.1,
    reduction: str = "mean",
) -> DiagnosticLoss:
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")
    bit_raw = F.binary_cross_entropy_with_logits(
        output.bit_logits,
        targets["source_bits"].to(output.bit_logits.dtype),
        pos_weight=bit_pos_weight.to(output.bit_logits.device),
        reduction="none",
    ).mean(dim=-1)
    six_raw = F.cross_entropy(
        output.six_way_logits,
        targets["six_way"].long(),
        weight=class_weight.to(output.six_way_logits.device),
        reduction="none",
    )
    signature_raw = F.mse_loss(
        output.signature,
        targets["signature"].to(output.signature.dtype),
        reduction="none",
    ).mean(dim=-1)
    total_raw = bit_raw + six_way_weight * six_raw + signature_weight * signature_raw
    if reduction == "mean":
        return DiagnosticLoss(
            total=total_raw.mean(),
            source_bits=bit_raw.mean(),
            six_way=six_raw.mean(),
            signature=signature_raw.mean(),
        )
    return DiagnosticLoss(total_raw, bit_raw, six_raw, signature_raw)


def compute_training_weights(
    source_bits: torch.Tensor,
    six_way: torch.Tensor,
    num_classes: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_bits.ndim != 2 or source_bits.shape[1] != 3:
        raise ValueError("source_bits must have shape [N, 3]")
    if six_way.ndim != 1 or six_way.shape[0] != source_bits.shape[0]:
        raise ValueError("six_way must have shape [N]")
    positives = source_bits.float().sum(dim=0)
    negatives = source_bits.shape[0] - positives
    if torch.any(positives <= 0) or torch.any(negatives <= 0):
        raise ValueError("Every source bit must have positive and negative training examples")
    bit_pos_weight = negatives / positives
    counts = torch.bincount(six_way.long(), minlength=num_classes).float()
    if torch.any(counts <= 0):
        raise ValueError("Every six-way class must appear in the training split")
    class_weight = source_bits.shape[0] / (num_classes * counts)
    return bit_pos_weight, class_weight


@dataclass(frozen=True)
class PolicyLoss:
    total: torch.Tensor
    ranking: torch.Tensor
    mse: torch.Tensor
    action_ce: torch.Tensor


def policy_loss(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    margin: float = 0.05,
    mse_weight: float = 0.25,
    action_ce_weight: float = 0.5,
) -> PolicyLoss:
    """Plan §9.5 ranking + regression + best-action classification loss."""

    if predicted_values.shape != target_values.shape or legal_mask.shape != target_values.shape:
        raise ValueError("Policy tensors must have identical [batch, actions] shapes")
    legal_mask = legal_mask.bool()
    if torch.any(legal_mask.sum(dim=1) == 0):
        raise ValueError("Each policy row must have at least one legal action")
    masked_predictions = predicted_values.masked_fill(~legal_mask, float("-inf"))
    masked_targets = target_values.masked_fill(~legal_mask, float("-inf"))
    # STOP is the target when every legal acquisition has nonpositive value,
    # including exact zero ties. Otherwise choose the highest positive probe.
    non_stop_targets = masked_targets[:, :-1]
    best_non_stop_value, best_non_stop_action = non_stop_targets.max(dim=1)
    best_action = torch.where(
        best_non_stop_value > 0.0,
        best_non_stop_action,
        torch.full_like(best_non_stop_action, target_values.shape[1] - 1),
    )
    action_ce = F.cross_entropy(masked_predictions, best_action)
    mse = ((predicted_values - target_values).square() * legal_mask).sum() / legal_mask.sum()

    ranking_terms = []
    action_count = predicted_values.shape[1]
    for left in range(action_count):
        for right in range(action_count):
            ordered = legal_mask[:, left] & legal_mask[:, right] & (
                target_values[:, left] > target_values[:, right]
            )
            if ordered.any():
                ranking_terms.append(
                    F.relu(
                        margin
                        - (predicted_values[ordered, left] - predicted_values[ordered, right])
                    )
                )
    ranking = (
        torch.cat(ranking_terms).mean()
        if ranking_terms
        else predicted_values.sum() * 0.0
    )
    total = ranking + mse_weight * mse + action_ce_weight * action_ce
    return PolicyLoss(total=total, ranking=ranking, mse=mse, action_ce=action_ce)
