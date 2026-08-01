"""
Teacher cache JSONL schema validator (Plan §14.1).

Validates teacher cache records against the exact schema contract:
1. All top-level required fields present and non-empty.
2. Split must be non-empty and strictly in ('train', 'val').
3. score_method in ('generation_logits', 'teacher_forced').
4. Clean object with valid normalized answer and bounded finite scores:
   - answer_prob in [0, 1]
   - token_entropy_mean >= 0
   - token_margin_mean in [-1, 1]
   - correct in (0, 1)
5. Probes dict contains all mandatory probes (blank, blur, crop, brightness, noise, grounding).
6. Every mandatory probe observation must be valid for valid teacher instances.
7. Probe conf_shift in [-1, 1].
8. Relation probe present and applicable when relation_applicable=True.
9. Teacher signature V, L, A non-negative finite floats.
10. Teacher bits in {0, 1}.
11. teacher_label6 in valid SixWayState enum values.
12. prompt_hash and generation_config_hash non-empty hex strings.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List

from proactive.features.evidence_state import SixWayState

logger = logging.getLogger(__name__)

REQUIRED_TOP_LEVEL = [
    "instance_id",
    "group_id",
    "dataset",
    "split",
    "model_id",
    "model_revision",
    "image_path",
    "prompt_text",
    "gold_answer",
    "score_method",
    "clean",
    "probes",
    "teacher_signature",
    "teacher_bits",
    "teacher_label6",
    "prompt_hash",
    "generation_config_hash",
]

REQUIRED_CLEAN = [
    "raw_answer",
    "norm_answer",
    "correct",
    "answer_prob",
    "token_entropy_mean",
    "token_margin_mean",
]

MANDATORY_PROBES = ["blank", "blur", "crop", "brightness", "noise", "grounding"]
VALID_SPLITS = {"train", "val"}
VALID_SCORE_METHODS = {"generation_logits", "teacher_forced"}
VALID_SIX_WAY_STATES = {s.value for s in SixWayState} | {"unclear", "invalid"}


def is_finite(val: Any) -> bool:
    """Check if value is a finite float or int."""
    if isinstance(val, (int, float)):
        return math.isfinite(val)
    return False


def is_hex_string(val: Any) -> bool:
    """Check if string is a valid non-empty hex hash."""
    if not isinstance(val, str) or not val:
        return False
    return bool(re.match(r'^[0-9a-fA-F]+$', val))


def validate_record(record: Dict[str, Any], idx: int = 0) -> List[str]:
    """Validate a single teacher cache record against Plan §14.1 with strict checks."""
    errors = []

    # 1. Top-level keys
    for k in REQUIRED_TOP_LEVEL:
        if k not in record or record[k] is None:
            errors.append(f"Row {idx}: Missing or null top-level key '{k}'")

    # 2. Split safety (no leakage of cal/test into train/val teacher cache)
    split = record.get("split", "")
    if split not in VALID_SPLITS:
        errors.append(f"Row {idx}: Invalid split '{split}'. Must be one of {VALID_SPLITS}")

    # 3. Score method
    score_method = record.get("score_method", "")
    if score_method not in VALID_SCORE_METHODS:
        errors.append(f"Row {idx}: Invalid score_method '{score_method}'. Must be in {VALID_SCORE_METHODS}")

    # 4. Hash fields
    for hash_key in ("prompt_hash", "generation_config_hash"):
        h_val = record.get(hash_key, "")
        if not is_hex_string(h_val):
            errors.append(f"Row {idx}: Field '{hash_key}' is not a valid non-empty hex hash: {h_val}")

    is_record_valid = record.get("valid", True)

    # 5. Clean block
    clean = record.get("clean", {})
    if not isinstance(clean, dict):
        errors.append(f"Row {idx}: 'clean' field is not a dict")
    else:
        for ck in REQUIRED_CLEAN:
            if ck not in clean:
                errors.append(f"Row {idx}: Clean missing key '{ck}'")

        correct = clean.get("correct")
        if correct not in (0, 1, True, False):
            errors.append(f"Row {idx}: Clean 'correct' must be in {{0, 1}}, got: {correct}")

        prob = clean.get("answer_prob")
        if not is_finite(prob) or not (0.0 <= prob <= 1.0001):
            errors.append(f"Row {idx}: Clean 'answer_prob' out of [0, 1] bounds or non-finite: {prob}")

        entropy = clean.get("token_entropy_mean")
        if not is_finite(entropy) or entropy < -1e-6:
            errors.append(f"Row {idx}: Clean 'token_entropy_mean' must be non-negative, got: {entropy}")

        margin = clean.get("token_margin_mean")
        if not is_finite(margin) or not (-1.0001 <= margin <= 1.0001):
            errors.append(f"Row {idx}: Clean 'token_margin_mean' out of [-1, 1] bounds: {margin}")

    # 6. Probes block
    probes = record.get("probes", {})
    if not isinstance(probes, dict):
        errors.append(f"Row {idx}: 'probes' field is not a dict")
    else:
        for p in MANDATORY_PROBES:
            if p not in probes:
                errors.append(f"Row {idx}: Missing mandatory probe '{p}' in probes dict")
            else:
                p_obs = probes[p]
                conf_shift = p_obs.get("conf_shift")
                if not is_finite(conf_shift) or not (-1.0001 <= conf_shift <= 1.0001):
                    errors.append(f"Row {idx}: Probe '{p}' conf_shift out of [-1, 1] bounds: {conf_shift}")

                if is_record_valid and not p_obs.get("valid", True):
                    errors.append(
                        f"Row {idx}: Mandatory probe '{p}' is marked valid=False ({p_obs.get('invalid_reason')}) in a valid teacher instance"
                    )

        if record.get("relation_applicable", False):
            if "relation" not in probes:
                errors.append(f"Row {idx}: relation_applicable is True but 'relation' probe is missing")
            elif not probes["relation"].get("applicable", False):
                errors.append(f"Row {idx}: relation_applicable is True but 'relation' probe has applicable=False")

    # 7. Teacher bits, signature, and labels
    t_bits = record.get("teacher_bits", {})
    for b in ("visual", "language", "alignment"):
        b_val = t_bits.get(b)
        if b_val not in (0, 1):
            errors.append(f"Row {idx}: Teacher bit '{b}' must be 0 or 1, got: {b_val}")

    t_sig = record.get("teacher_signature", {})
    for s in ("V", "L", "A"):
        s_val = t_sig.get(s)
        if not is_finite(s_val) or s_val < -1e-6:
            errors.append(f"Row {idx}: Non-finite or negative teacher signature component '{s}': {s_val}")

    t_label6 = record.get("teacher_label6", "")
    if t_label6 not in VALID_SIX_WAY_STATES:
        errors.append(f"Row {idx}: Invalid teacher_label6 '{t_label6}'. Must be one of {VALID_SIX_WAY_STATES}")

    return errors


def validate_file(file_path: Path) -> Dict[str, Any]:
    """Validate all records in a JSONL teacher cache file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    total_rows = 0
    valid_rows = 0
    all_errors = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total_rows += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                errs = validate_record(rec, idx)
                if errs:
                    all_errors.extend(errs)
                else:
                    valid_rows += 1
            except Exception as e:
                all_errors.append(f"Row {idx}: JSON parse error: {e}")

    report = {
        "file_path": str(file_path),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": total_rows - valid_rows,
        "is_valid": len(all_errors) == 0 and total_rows > 0,
        "errors": all_errors[:100],  # cap at first 100
    }
    return report
