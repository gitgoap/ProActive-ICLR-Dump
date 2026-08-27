"""Small-network diagnostic training and validation helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from proactive.eval.diagnostic_metrics import diagnostic_metrics, within_group_metrics
from proactive.networks.diagnostic import DiagnosticModel
from proactive.networks.losses import diagnostic_loss
from proactive.train.state_data import FeatureNormalizer, permute_sequence_indices


def move_model_input(value: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in value.items()}


def move_targets(value: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in value.items()}


def train_epoch(
    model: DiagnosticModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    normalizer: FeatureNormalizer,
    bit_pos_weight: torch.Tensor,
    class_weight: torch.Tensor,
    six_way_weight: float,
    signature_weight: float,
    device: torch.device,
    gradient_clip_norm: float,
    randomize_gru_order: bool,
    permutation_generator: torch.Generator,
) -> Dict[str, float]:
    model.train()
    totals = {"total": 0.0, "source_bits": 0.0, "six_way": 0.0, "signature": 0.0}
    examples = 0
    for batch in loader:
        model_input = move_model_input(batch["model_input"], device)
        targets = move_targets(batch["targets"], device)
        if randomize_gru_order:
            model_input["sequence_indices"] = permute_sequence_indices(
                model_input["sequence_indices"],
                model_input["acquired_mask"],
                generator=permutation_generator,
            )
        model_input = normalizer.transform(model_input)
        optimizer.zero_grad(set_to_none=True)
        loss = diagnostic_loss(
            model(model_input),
            targets,
            bit_pos_weight=bit_pos_weight,
            class_weight=class_weight,
            six_way_weight=six_way_weight,
            signature_weight=signature_weight,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        count = int(targets["six_way"].shape[0])
        examples += count
        for name in totals:
            totals[name] += float(getattr(loss, name).detach().cpu()) * count
    if examples == 0:
        raise ValueError("Training loader produced no examples")
    return {name: total / examples for name, total in totals.items()}


@torch.no_grad()
def predict(
    model: DiagnosticModel,
    loader: DataLoader,
    *,
    normalizer: FeatureNormalizer,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    bit_probabilities = []
    six_way_probabilities = []
    signatures = []
    bit_targets = []
    six_way_targets = []
    signature_targets = []
    metadata = []
    state_ids = []
    max_budgets = []
    for batch in loader:
        model_input = move_model_input(batch["model_input"], device)
        targets = move_targets(batch["targets"], device)
        output = model(normalizer.transform(model_input))
        bit_probabilities.append(torch.sigmoid(output.bit_logits).cpu())
        six_way_probabilities.append(torch.softmax(output.six_way_logits, dim=-1).cpu())
        signatures.append(output.signature.cpu())
        bit_targets.append(targets["source_bits"].cpu())
        six_way_targets.append(targets["six_way"].cpu())
        signature_targets.append(targets["signature"].cpu())
        audit = batch["audit_metadata"]
        if not isinstance(audit, Mapping):
            raise ValueError("Collated audit_metadata must remain a mapping")
        audit_keys = tuple(audit)
        audit_count = len(audit[audit_keys[0]]) if audit_keys else 0
        metadata.extend(
            [{key: audit[key][index] for key in audit_keys} for index in range(audit_count)]
        )
        state_ids.extend(batch["state_id"])
        max_budgets.extend(model_input["max_budget"].detach().cpu().tolist())
    if not bit_probabilities:
        raise ValueError("Prediction loader produced no examples")
    return {
        "bit_probabilities": torch.cat(bit_probabilities).numpy(),
        "six_way_probabilities": torch.cat(six_way_probabilities).numpy(),
        "signature_predictions": torch.cat(signatures).numpy(),
        "bit_targets": torch.cat(bit_targets).numpy().astype(np.int64),
        "six_way_targets": torch.cat(six_way_targets).numpy().astype(np.int64),
        "signature_targets": torch.cat(signature_targets).numpy(),
        "metadata": metadata,
        "state_id": state_ids,
        "max_budget": max_budgets,
    }


def evaluate_predictions(predictions: Mapping[str, Any]) -> Dict[str, Any]:
    return diagnostic_metrics(
        bit_probabilities=predictions["bit_probabilities"],
        six_way_probabilities=predictions["six_way_probabilities"],
        bit_targets=predictions["bit_targets"],
        six_way_targets=predictions["six_way_targets"],
        signature_predictions=predictions["signature_predictions"],
        signature_targets=predictions["signature_targets"],
    )


def stratified_prediction_metrics(predictions: Mapping[str, Any]) -> Dict[str, Any]:
    inputs = {
        "bit_probabilities": predictions["bit_probabilities"],
        "six_way_probabilities": predictions["six_way_probabilities"],
        "bit_targets": predictions["bit_targets"],
        "six_way_targets": predictions["six_way_targets"],
        "signature_predictions": predictions["signature_predictions"],
        "signature_targets": predictions["signature_targets"],
    }
    metadata = predictions["metadata"]
    return {
        "pooled": evaluate_predictions(predictions),
        "within_dataset": within_group_metrics(
            group_values=[row["dataset"] for row in metadata],
            minimum_rows=25,
            metric_inputs=inputs,
        ),
        "within_model": within_group_metrics(
            group_values=[row["model_id"] for row in metadata],
            minimum_rows=25,
            metric_inputs=inputs,
        ),
        "within_budget": within_group_metrics(
            group_values=[str(value) for value in predictions["max_budget"]],
            minimum_rows=25,
            metric_inputs=inputs,
        ),
    }
