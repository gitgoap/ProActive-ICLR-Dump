"""
Manifest construction and I/O.

A manifest is a JSONL file where each record represents one
(dataset, image, question) instance with its group_id, split
assignment, and metadata needed for teacher cache generation.
(Plan §14.1, §18)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from proactive.utils.hashing import compute_group_id, hash_manifest
from proactive.utils.io import read_jsonl, write_jsonl
from proactive.data.hallusion_contract import (
    HALLUSION_BINARY,
    HALLUSION_OPEN_ENDED,
)


# ---------------------------------------------------------------------------
# Manifest record schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "instance_id",
    "group_id",
    "dataset",
    "image_path",
    "question",
    "gold_answer",
    "relation_applicable",
}


def make_manifest_record(
    dataset: str,
    image_id: str,
    question_id: str,
    image_path: str,
    question: str,
    gold_answer: str,
    relation_applicable: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a single manifest record with a computed group_id.

    The instance_id is a human-readable unique key.
    The group_id is a hash for split assignment.
    """
    instance_id = f"{dataset}_{image_id}_{question_id}"
    group_id = compute_group_id(dataset, image_id, question_id)

    record = {
        "instance_id": instance_id,
        "group_id": group_id,
        "dataset": dataset,
        "image_id": image_id,
        "question_id": question_id,
        "image_path": image_path,
        "question": question,
        "gold_answer": gold_answer,
        "relation_applicable": relation_applicable,
    }
    if extra:
        record.update(extra)
    return record


def validate_manifest(records: List[Dict[str, Any]]) -> List[str]:
    """Validate that all required fields are present in every record.

    Returns list of error messages (empty if valid).
    """
    errors = []
    seen_ids = set()
    for i, record in enumerate(records):
        missing = REQUIRED_FIELDS - set(record.keys())
        if missing:
            errors.append(f"Record {i}: missing fields {missing}")
        iid = record.get("instance_id", f"__missing_{i}")
        if iid in seen_ids:
            errors.append(f"Record {i}: duplicate instance_id '{iid}'")
        seen_ids.add(iid)
        if str(record.get("dataset", "")).lower() == "hallusionbench":
            answer_type = record.get("answer_type")
            if answer_type not in {HALLUSION_BINARY, HALLUSION_OPEN_ENDED}:
                errors.append(
                    f"Record {i}: invalid HallusionBench answer_type {answer_type!r}"
                )
                continue
            if record.get("answer_contract_version") != 1:
                errors.append(
                    f"Record {i}: unsupported HallusionBench answer contract"
                )
            expected_match_mode = (
                "exact_alias"
                if answer_type == HALLUSION_OPEN_ENDED
                else "binary_exact"
            )
            if record.get("answer_match_mode") != expected_match_mode:
                errors.append(
                    f"Record {i}: invalid HallusionBench answer_match_mode"
                )
            if str(record.get("benchmark_gold_answer", "")) not in {"0", "1", "2"}:
                errors.append(
                    f"Record {i}: invalid HallusionBench benchmark_gold_answer"
                )
            references = record.get("reference_answers")
            if (
                not isinstance(references, list)
                or not references
                or any(not isinstance(value, str) or not value.strip() for value in references)
            ):
                errors.append(
                    f"Record {i}: invalid HallusionBench reference_answers"
                )
            if answer_type == HALLUSION_BINARY:
                if record.get("gold_answer") not in {"yes", "no", "uncertain"}:
                    errors.append(
                        f"Record {i}: binary HallusionBench gold is not normalized"
                    )
            elif not str(record.get("gt_answer_details", "")).strip():
                errors.append(
                    f"Record {i}: open-ended HallusionBench row lacks gt_answer_details"
                )
        if str(record.get("dataset", "")).lower() == "vizwiz":
            if record.get("answer_contract_version") != 1:
                errors.append(f"Record {i}: unsupported VizWiz answer contract")
            if record.get("answer_match_mode") != "normalized_exact":
                errors.append(f"Record {i}: invalid VizWiz answer_match_mode")
            if (
                record.get("vizwiz_gold_policy")
                != "normalized_majority_source_order_tiebreak_v1"
            ):
                errors.append(f"Record {i}: invalid VizWiz gold policy")
            counts = record.get("vizwiz_answer_counts")
            if (
                not isinstance(counts, dict)
                or not counts
                or any(
                    not isinstance(key, str)
                    or not key
                    or not isinstance(value, int)
                    or value <= 0
                    for key, value in counts.items()
                )
            ):
                errors.append(f"Record {i}: invalid VizWiz answer counts")
            else:
                gold = record.get("gold_answer")
                top_count = max(counts.values())
                expected_tie_size = sum(
                    value == top_count for value in counts.values()
                )
                if gold not in counts or counts.get(gold) != top_count:
                    errors.append(
                        f"Record {i}: VizWiz gold is not a majority candidate"
                    )
                if record.get("vizwiz_tied_top_answer_count") != expected_tie_size:
                    errors.append(f"Record {i}: invalid VizWiz top-tie count")
                if record.get("reference_answers") != [gold]:
                    errors.append(f"Record {i}: invalid VizWiz reference answers")
    return errors


def save_manifest(
    records: List[Dict[str, Any]],
    output_path: str | Path,
    overwrite: bool = False,
) -> str:
    """Save manifest to JSONL and return its content hash."""
    write_jsonl(records, output_path, overwrite=overwrite)
    return hash_manifest(records)


def load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    """Load a manifest JSONL file."""
    return read_jsonl(path)
