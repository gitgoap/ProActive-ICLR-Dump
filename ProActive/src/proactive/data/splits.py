"""
Grouped split construction for the ProActive project.

All data is split at the GROUP level, not the individual record level.
A group is defined by (dataset, image_id, question_id). All model
outputs, prompt variants, and probe outcomes for the same base instance
stay in the same split.  (Plan §12.3)

Split ratios: 70/10/10/10 → train / val / cal / test
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SPLIT_NAMES = ("train", "val", "cal", "test")
DEFAULT_RATIOS = (0.70, 0.10, 0.10, 0.10)


def compute_group_id(
    dataset: str,
    image_id: str,
    question_id: str,
) -> str:
    """Compute a deterministic group ID for split assignment.

    All records sharing this group ID will land in the same split.
    (Plan §12.3)
    """
    key = f"{dataset}|{image_id}|{question_id}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"sha256:{h[:16]}"


def assign_split(
    group_id: str,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> str:
    """Deterministically assign a group_id to a split.

    Uses hashing so the assignment is reproducible and independent
    of the order records are processed. The seed is mixed into
    the hash to allow different split configurations.
    """
    assert len(ratios) == len(SPLIT_NAMES), (
        f"Expected {len(SPLIT_NAMES)} ratios, got {len(ratios)}"
    )
    assert abs(sum(ratios) - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {sum(ratios)}"
    )

    # Hash group_id with seed for deterministic assignment
    key = f"{seed}|{group_id}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Convert first 8 hex chars to a float in [0, 1)
    bucket = int(h[:8], 16) / (16**8)

    cumulative = 0.0
    for i, ratio in enumerate(ratios):
        cumulative += ratio
        if bucket < cumulative:
            return SPLIT_NAMES[i]
    return SPLIT_NAMES[-1]


def build_grouped_splits(
    records: List[Dict[str, Any]],
    group_id_field: str = "group_id",
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Assign splits to a list of records based on their group_id.

    Modifies records in-place by adding a 'split' field.
    Returns the same list for convenience.
    """
    for record in records:
        gid = record[group_id_field]
        record["split"] = assign_split(gid, ratios=ratios, seed=seed)
    return records


def validate_no_group_overlap(
    records: List[Dict[str, Any]],
    group_id_field: str = "group_id",
    split_field: str = "split",
) -> Tuple[bool, List[str]]:
    """Verify that no group_id appears in more than one split.

    Returns (is_valid, list_of_violations).
    """
    group_splits: Dict[str, set] = {}
    for record in records:
        gid = record[group_id_field]
        split = record[split_field]
        if gid not in group_splits:
            group_splits[gid] = set()
        group_splits[gid].add(split)

    violations = []
    for gid, splits in group_splits.items():
        if len(splits) > 1:
            violations.append(
                f"Group {gid} appears in multiple splits: {splits}"
            )

    return (len(violations) == 0, violations)


def get_split_stats(
    records: List[Dict[str, Any]],
    split_field: str = "split",
    group_id_field: str = "group_id",
) -> Dict[str, Dict[str, int]]:
    """Return record counts and group counts per split."""
    stats: Dict[str, Dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        split_records = [r for r in records if r[split_field] == split_name]
        split_groups = {r[group_id_field] for r in split_records}
        stats[split_name] = {
            "records": len(split_records),
            "groups": len(split_groups),
        }
    return stats
