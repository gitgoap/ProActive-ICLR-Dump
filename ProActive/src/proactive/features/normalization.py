"""
Answer normalization per dataset type.

Each dataset has an authoritative normalization rule (Plan §14.4).
This module provides a single entry point that dispatches to the
correct normalizer based on dataset name.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Binary normalizers
# ---------------------------------------------------------------------------

_YES_VARIANTS = {
    "yes", "yeah", "yep", "yup", "y", "correct", "right", "true",
    "affirmative", "indeed", "absolutely", "sure", "positive",
}
_NO_VARIANTS = {
    "no", "nope", "nah", "n", "incorrect", "wrong", "false",
    "negative", "not",
}


def normalize_yes_no(raw: str) -> str:
    """Normalize to 'yes' or 'no'. Returns 'unknown' if ambiguous."""
    cleaned = raw.strip().lower().rstrip(".")
    # Take the first word for multi-word answers
    first_word = cleaned.split()[0] if cleaned.split() else ""
    if first_word in _YES_VARIANTS or cleaned in _YES_VARIANTS:
        return "yes"
    if first_word in _NO_VARIANTS or cleaned in _NO_VARIANTS:
        return "no"
    # Check if the full answer contains a clear indicator
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    return "unknown"


def normalize_true_false(raw: str) -> str:
    """Normalize to 'true' or 'false'. Returns 'unknown' if ambiguous."""
    cleaned = raw.strip().lower().rstrip(".")
    first_word = cleaned.split()[0] if cleaned.split() else ""
    if first_word in ("true", "correct", "yes", "right"):
        return "true"
    if first_word in ("false", "incorrect", "no", "wrong"):
        return "false"
    return "unknown"


# ---------------------------------------------------------------------------
# Free-form normalizer (VizWiz-VQA style)
# ---------------------------------------------------------------------------

# Common articles and filler words to strip
_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Unanswerable variants for VizWiz
_UNANSWERABLE = {
    "unanswerable", "not answerable", "cannot be answered",
    "can't be answered", "cannot answer", "not sure",
    "i don't know", "i do not know", "n/a", "na",
    "unsuitable", "unsuitable image",
}


def normalize_freeform(raw: str) -> str:
    """Normalize free-form answers (VizWiz style).

    Steps: lowercase → remove punctuation → remove articles →
    collapse whitespace → normalize unanswerable variants.
    """
    text = raw.strip().lower()

    # Check unanswerable variants first
    if text in _UNANSWERABLE:
        return "unanswerable"

    # Remove punctuation
    text = _PUNCT_RE.sub(" ", text)
    # Remove articles
    words = text.split()
    words = [w for w in words if w not in _ARTICLES]
    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", " ".join(words)).strip()

    # Final unanswerable check after normalization
    if text in _UNANSWERABLE or not text:
        return "unanswerable"

    return text


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Map dataset names to normalizer types
_DATASET_NORMALIZER = {
    "hallusionbench": "yes_no",
    "pope": "yes_no",
    "repope": "yes_no",
    "vizwiz": "freeform",
    "vizwiz_vqa": "freeform",
    "vsr": "true_false",
    "gqa_relation": "true_false",
    "prehal": "yes_no",
    "illusionbench": "yes_no",
}


def normalize_answer(
    raw_answer: str,
    dataset: str,
    normalizer_type: Optional[str] = None,
) -> str:
    """Normalize an answer according to dataset-specific rules.

    Args:
        raw_answer: The raw model output string.
        dataset: Dataset name (e.g., 'hallusionbench', 'vizwiz').
        normalizer_type: Override the normalizer type. If None, inferred
            from dataset name.

    Returns:
        Normalized answer string.
    """
    if normalizer_type is None:
        key = dataset.lower().replace("-", "").replace(" ", "_")
        normalizer_type = _DATASET_NORMALIZER.get(key)
        if normalizer_type is None:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Known: "
                f"{sorted(_DATASET_NORMALIZER.keys())}. "
                f"Provide normalizer_type explicitly."
            )

    if normalizer_type == "yes_no":
        return normalize_yes_no(raw_answer)
    elif normalizer_type == "true_false":
        return normalize_true_false(raw_answer)
    elif normalizer_type == "freeform":
        return normalize_freeform(raw_answer)
    else:
        raise ValueError(f"Unknown normalizer type: {normalizer_type}")
