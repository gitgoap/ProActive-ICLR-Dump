"""
Confound audits and Section 28 triggers for ProActive (Plan §3.3, §28).

Audits dataset and model proxy confounds:
1. Per-dataset and per-model relation applicability rates.
2. Per-dataset and per-model source bit activation rates (b_V, b_L, b_A).
3. Six-way failure label distributions across benchmarks.
4. Dataset/model-ID control baseline evaluation hooks.
5. Plan §28 trigger: If dataset/model ID control is within 2.0 performance
   points of the full system, triggers warning and narrows claim.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def audit_dataset_and_model_confounds(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Audit relation applicability and label distributions by dataset and model."""
    dataset_counts: Dict[str, int] = defaultdict(int)
    model_counts: Dict[str, int] = defaultdict(int)

    # Applicability counters
    dataset_rel_app: Dict[str, int] = defaultdict(int)
    model_rel_app: Dict[str, int] = defaultdict(int)

    # Source bit counters
    dataset_bits: Dict[str, Dict[str, int]] = defaultdict(lambda: {"V": 0, "L": 0, "A": 0})
    model_bits: Dict[str, Dict[str, int]] = defaultdict(lambda: {"V": 0, "L": 0, "A": 0})

    # Six-way label distributions
    dataset_labels: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    model_labels: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total = len(records)
    for r in records:
        ds = r.get("dataset", "unknown")
        mod = r.get("model_id", "unknown")
        dataset_counts[ds] += 1
        model_counts[mod] += 1

        rel_app = bool(r.get("relation_applicable", False))
        if rel_app:
            dataset_rel_app[ds] += 1
            model_rel_app[mod] += 1

        t_bits = r.get("teacher_bits", {})
        if t_bits.get("visual", 0):
            dataset_bits[ds]["V"] += 1
            model_bits[mod]["V"] += 1
        if t_bits.get("language", 0):
            dataset_bits[ds]["L"] += 1
            model_bits[mod]["L"] += 1
        if t_bits.get("alignment", 0):
            dataset_bits[ds]["A"] += 1
            model_bits[mod]["A"] += 1

        lbl6 = r.get("teacher_label6", "unknown")
        dataset_labels[ds][lbl6] += 1
        model_labels[mod][lbl6] += 1

    # Format rates
    by_dataset: Dict[str, Any] = {}
    for ds, count in dataset_counts.items():
        by_dataset[ds] = {
            "total_count": count,
            "relation_applicable_rate": dataset_rel_app[ds] / count if count > 0 else 0.0,
            "visual_bit_rate": dataset_bits[ds]["V"] / count if count > 0 else 0.0,
            "language_bit_rate": dataset_bits[ds]["L"] / count if count > 0 else 0.0,
            "alignment_bit_rate": dataset_bits[ds]["A"] / count if count > 0 else 0.0,
            "label6_distribution": {k: v / count for k, v in dataset_labels[ds].items()},
        }

    by_model: Dict[str, Any] = {}
    for mod, count in model_counts.items():
        by_model[mod] = {
            "total_count": count,
            "relation_applicable_rate": model_rel_app[mod] / count if count > 0 else 0.0,
            "visual_bit_rate": model_bits[mod]["V"] / count if count > 0 else 0.0,
            "language_bit_rate": model_bits[mod]["L"] / count if count > 0 else 0.0,
            "alignment_bit_rate": model_bits[mod]["A"] / count if count > 0 else 0.0,
            "label6_distribution": {k: v / count for k, v in model_labels[mod].items()},
        }

    return {
        "total_records": total,
        "by_dataset": by_dataset,
        "by_model": by_model,
    }


def check_section_28_trigger(
    full_system_metric: float,
    id_control_metric: float,
    threshold: float = 2.0,
    metric_name: str = "Macro-F1",
) -> Dict[str, Any]:
    """Check Section 28 confound trigger rule (Plan §28).

    Rule: If an ID-only control baseline achieves within `threshold` points
    of the full diagnostic system, the trigger fires:
    - Flags shortcut risk
    - Recommends stripping shortcut features from encoder
    - Narrows paper claim to within-dataset diagnostic sensitivity.
    """
    gap = full_system_metric - id_control_metric
    triggered = (gap <= threshold)

    result = {
        "metric_name": metric_name,
        "full_system_metric": full_system_metric,
        "id_control_metric": id_control_metric,
        "gap": gap,
        "threshold": threshold,
        "triggered": triggered,
        "action_required": "None" if not triggered else "REMOVE_SHORTCUT_FEATURES_AND_NARROW_CLAIM",
        "message": (
            f"ID Control ({id_control_metric:.2f}) is within {gap:.2f} points of Full System ({full_system_metric:.2f}). "
            f"Plan §28 trigger FIRED. Model may be exploiting dataset shortcuts."
            if triggered else
            f"Gap is {gap:.2f} points (> {threshold:.2f}). Plan §28 trigger PASS."
        ),
    }
    return result
