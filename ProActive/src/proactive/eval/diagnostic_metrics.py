"""Diagnostic metrics required by Plan §21.1."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    roc_auc_score,
)

from proactive.train.state_data import SIX_WAY_LABELS, SOURCE_BITS


def diagnostic_metrics(
    bit_probabilities: Sequence[Sequence[float]],
    six_way_probabilities: Sequence[Sequence[float]],
    bit_targets: Sequence[Sequence[int]],
    six_way_targets: Sequence[int],
    signature_predictions: Sequence[Sequence[float]] | None = None,
    signature_targets: Sequence[Sequence[float]] | None = None,
    *,
    allow_undefined_auroc: bool = False,
) -> Dict[str, Any]:
    bit_probability = np.asarray(bit_probabilities, dtype=np.float64)
    six_probability = np.asarray(six_way_probabilities, dtype=np.float64)
    bit_truth = np.asarray(bit_targets, dtype=np.int64)
    six_truth = np.asarray(six_way_targets, dtype=np.int64)
    if bit_probability.shape != bit_truth.shape or bit_probability.ndim != 2 or bit_probability.shape[1] != 3:
        raise ValueError("Bit probabilities/targets must have matching [N, 3] shape")
    if six_probability.shape != (bit_truth.shape[0], len(SIX_WAY_LABELS)):
        raise ValueError("Six-way probabilities have the wrong shape")
    if six_truth.shape != (bit_truth.shape[0],):
        raise ValueError("Six-way targets have the wrong shape")
    if not np.isfinite(bit_probability).all() or not np.isfinite(six_probability).all():
        raise ValueError("Metric probabilities must be finite")
    if (bit_probability < 0).any() or (bit_probability > 1).any():
        raise ValueError("Bit probabilities must lie in [0, 1]")
    if not np.allclose(six_probability.sum(axis=1), 1.0, atol=1.0e-6):
        raise ValueError("Six-way probability rows must sum to one")
    bit_prediction = (bit_probability >= 0.5).astype(np.int64)
    six_prediction = six_probability.argmax(axis=1)
    per_bit_auroc: Dict[str, float | None] = {}
    unavailable_auroc: Dict[str, Dict[str, Any]] = {}
    for index, name in enumerate(SOURCE_BITS):
        observed_classes = np.unique(bit_truth[:, index])
        if observed_classes.size != 2:
            if not allow_undefined_auroc:
                raise ValueError(f"Cannot compute AUROC: bit {name!r} has one target class")
            per_bit_auroc[name] = None
            unavailable_auroc[name] = {
                "reason": "AUROC requires both target classes",
                "observed_classes": observed_classes.astype(int).tolist(),
            }
            continue
        per_bit_auroc[name] = float(
            roc_auc_score(bit_truth[:, index], bit_probability[:, index])
        )
    result: Dict[str, Any] = {
        "record_count": int(bit_truth.shape[0]),
        "source_bit_micro_f1": float(f1_score(bit_truth, bit_prediction, average="micro", zero_division=0)),
        "source_bit_macro_f1": float(f1_score(bit_truth, bit_prediction, average="macro", zero_division=0)),
        "source_bit_hamming_loss": float(hamming_loss(bit_truth, bit_prediction)),
        "source_bit_auroc": per_bit_auroc,
        "six_way_macro_f1": float(f1_score(six_truth, six_prediction, average="macro", zero_division=0)),
        "six_way_balanced_accuracy": float(balanced_accuracy_score(six_truth, six_prediction)),
        "six_way_confusion_matrix": confusion_matrix(
            six_truth, six_prediction, labels=list(range(len(SIX_WAY_LABELS)))
        ).tolist(),
        "six_way_labels": list(SIX_WAY_LABELS),
    }
    if unavailable_auroc:
        result["metrics_scientifically_valid"] = False
        result["source_bit_auroc_unavailable"] = unavailable_auroc
    if (signature_predictions is None) != (signature_targets is None):
        raise ValueError("Signature predictions and targets must be provided together")
    if signature_predictions is not None:
        prediction = np.asarray(signature_predictions, dtype=np.float64)
        target = np.asarray(signature_targets, dtype=np.float64)
        if prediction.shape != target.shape or prediction.shape != (bit_truth.shape[0], 3):
            raise ValueError("Signature prediction/target shape mismatch")
        result["signature_mse"] = float(np.square(prediction - target).mean())
    return result


def within_group_metrics(
    *,
    group_values: Sequence[str],
    minimum_rows: int,
    metric_inputs: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    groups = np.asarray(group_values, dtype=object)
    if groups.shape != (len(metric_inputs["six_way_targets"]),):
        raise ValueError("Group vector length mismatch")
    output: Dict[str, Dict[str, Any]] = {}
    for group in sorted(set(str(value) for value in groups.tolist())):
        indices = np.flatnonzero(groups == group)
        if indices.size < minimum_rows:
            continue
        sliced = {
            key: np.asarray(value)[indices]
            for key, value in metric_inputs.items()
            if value is not None
        }
        try:
            output[group] = diagnostic_metrics(**sliced)
        except ValueError as exc:
            output[group] = {"record_count": int(indices.size), "is_valid": False, "error": str(exc)}
    return output
