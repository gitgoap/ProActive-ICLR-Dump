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
