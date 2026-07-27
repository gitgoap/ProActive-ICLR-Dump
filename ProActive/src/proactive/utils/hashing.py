"""
Hashing utilities for reproducibility tracking.

Every experiment must record hashes of prompts, generation configs,
image transformations, and dataset manifests so that results can be
traced back to exact inputs.  (Plan §14.1, §27.5)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def _stable_json(obj: Any) -> str:
    """JSON-serialize with sorted keys for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def hash_string(s: str) -> str:
    """SHA-256 hash of a string, returned as hex."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_dict(d: Dict[str, Any]) -> str:
    """SHA-256 hash of a dictionary (sorted-key JSON serialization)."""
    return hash_string(_stable_json(d))


def hash_prompt(prompt_text: str) -> str:
    """Hash a prompt template or formatted prompt."""
    return hash_string(prompt_text)


def hash_generation_config(config: Dict[str, Any]) -> str:
    """Hash generation config (temperature, max_tokens, etc.)."""
    return hash_dict(config)


def hash_transform(
    transform_name: str,
    params: Dict[str, Any],
) -> str:
    """Hash an image transformation spec (probe type + parameters)."""
    spec = {"transform": transform_name, **params}
    return hash_dict(spec)


def hash_manifest(records: list) -> str:
    """Hash a full manifest (list of record dicts) for integrity checking."""
    h = hashlib.sha256()
    for record in records:
        h.update(_stable_json(record).encode("utf-8"))
    return h.hexdigest()


def compute_group_id(
    dataset: str,
    image_id: str,
    question_id: str,
) -> str:
    """Compute a group ID for split construction (Plan §12.3).

    All model outputs, prompt variants, severity variants, and probe
    outcomes for the same base instance must share this group ID
    to prevent train/test leakage.
    """
    key = f"{dataset}|{image_id}|{question_id}"
    return "sha256:" + hash_string(key)[:16]
