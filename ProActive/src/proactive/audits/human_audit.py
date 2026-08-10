"""Deterministic sampling utilities for the blinded behavioural audit."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, List, Mapping, Set, Tuple


LABELS = (
    "no-failure",
    "visual",
    "language-prior",
    "alignment",
    "mixed",
    "unclear",
)


def audit_rank(seed: int, namespace: str, key: Tuple[str, str]) -> str:
    return hashlib.sha256(
        f"{seed}|{namespace}|{key[0]}|{key[1]}".encode("utf-8")
    ).hexdigest()


def _borderline_score(label: Mapping[str, Any], thresholds: Mapping[str, Any]) -> float:
    signature = label["teacher_signature"]
    distances = (
        abs(float(signature["V"]) - float(thresholds["visual_conf_threshold"])),
        abs(float(signature["L"]) - float(thresholds["blank_conf_ratio_threshold"])),
        abs(float(signature["A"]) - float(thresholds["grounding_conf_threshold"])),
    )
    return min(distances)


def select_audit_keys(
    teachers: Mapping[Tuple[str, str], Mapping[str, Any]],
    labels: Mapping[Tuple[str, str], Mapping[str, Any]],
    total: int,
    natural_count: int,
    target_per_label: int,
    seed: int,
    thresholds: Mapping[str, Any],
    allowed_splits: Set[str],
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], str]]:
    """Select one deterministic natural subset plus a targeted complement."""
    if total <= 0 or not 0 <= natural_count <= total:
        raise ValueError("Audit total/natural_count are inconsistent")
    eligible = [
        key
        for key, label in labels.items()
        if key in teachers
        and label.get("split") in allowed_splits
        and label.get("teacher_label6") in LABELS
    ]
    if len(eligible) < total:
        raise ValueError(f"Audit needs {total} eligible rows, found {len(eligible)}")

    natural = sorted(eligible, key=lambda key: audit_rank(seed, "natural", key))[:natural_count]
    selected: List[Tuple[str, str]] = list(natural)
    selected_set = set(selected)
    provenance = {key: "natural" for key in natural}
    counts = Counter(labels[key]["teacher_label6"] for key in natural)
    slice_counts = Counter(
        (labels[key]["dataset"], labels[key]["model_id"]) for key in natural
    )

    candidates: Dict[str, List[Tuple[str, str]]] = {}
    for label_name in LABELS:
        candidates[label_name] = [
            key
            for key in eligible
            if key not in selected_set and labels[key]["teacher_label6"] == label_name
        ]

    while len(selected) < total:
        deficits = {
            label_name: max(0, target_per_label - counts[label_name])
            for label_name in LABELS
        }
        available_labels = [name for name in LABELS if candidates[name]]
        if not available_labels:
            raise ValueError("Audit candidates exhausted before reaching requested total")
        positive_deficits = [name for name in available_labels if deficits[name] > 0]
        pool = positive_deficits or available_labels
        chosen_label = min(pool, key=lambda name: (-deficits[name], LABELS.index(name)))

        def candidate_priority(key: Tuple[str, str]) -> Tuple[Any, ...]:
            label = labels[key]
            slice_key = (label["dataset"], label["model_id"])
            return (
                slice_counts[slice_key],
                _borderline_score(label, thresholds),
                audit_rank(seed, "targeted", key),
            )

        key = min(candidates[chosen_label], key=candidate_priority)
        candidates[chosen_label].remove(key)
        selected.append(key)
        selected_set.add(key)
        provenance[key] = "targeted_borderline"
        counts[chosen_label] += 1
        label = labels[key]
        slice_counts[(label["dataset"], label["model_id"])] += 1

    return selected, provenance

