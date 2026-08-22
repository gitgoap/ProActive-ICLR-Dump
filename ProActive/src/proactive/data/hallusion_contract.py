"""Deterministic answer contracts for the released HallusionBench data.

HallusionBench is predominantly binary, but the official JSON contains a
small collection of genuinely open-ended table questions.  ``gt_answer`` is
only a benchmark-level 0/1 indicator for those rows; the natural-language
reference lives in ``gt_answer_details``.  This module classifies the question
form from source text and provides fail-closed helpers shared by loading,
validation, and tests.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


HALLUSION_BINARY = "binary"
HALLUSION_OPEN_ENDED = "open_ended"
HALLUSION_ANSWER_TYPES = {HALLUSION_BINARY, HALLUSION_OPEN_ENDED}

_OPEN_INTERROGATIVE_RE = re.compile(r"^(?:which|what|who|where|when|how)\b", re.IGNORECASE)
_VISUAL_CONTEXT_PREFIX_RE = re.compile(
    r"^according\s+to\s+(?:the\s+)?(?:image|table|chart)\s*,\s*",
    re.IGNORECASE,
)

_BINARY_GOLD = {
    "0": "no",
    "1": "yes",
    "2": "uncertain",
    "no": "no",
    "yes": "yes",
    "uncertain": "uncertain",
}


def classify_hallusion_answer_type(question: str) -> str:
    """Classify from question grammar, never from a model prediction.

    The released anomalies either begin directly with a WH interrogative or
    use an explicit ``According to the image/table/chart, which ...`` prefix.
    Relative clauses such as ``According to the image which is about ...,
    does ...`` deliberately remain binary because there is no comma-delimited
    visual-context prefix followed by an interrogative.
    """

    text = str(question).strip()
    core = _VISUAL_CONTEXT_PREFIX_RE.sub("", text, count=1).strip()
    if _OPEN_INTERROGATIVE_RE.match(core):
        return HALLUSION_OPEN_ENDED
    return HALLUSION_BINARY


def normalize_hallusion_gold(raw_gold: Any) -> str:
    """Map the official binary indicator to a literal answer fail closed."""

    key = str(raw_gold).strip().lower()
    if key not in _BINARY_GOLD:
        raise ValueError(
            f"Unsupported HallusionBench binary gt_answer {raw_gold!r}; "
            "expected one of 0/1/2 or no/yes/uncertain"
        )
    return _BINARY_GOLD[key]


def hallusion_source_key(item: Mapping[str, Any]) -> str:
    """Return a stable official-annotation identity for curated references."""

    required = ("category", "subcategory", "set_id", "figure_id", "question_id")
    missing = [field for field in required if field not in item]
    if missing:
        raise ValueError(f"HallusionBench source row is missing identity fields: {missing}")
    return "|".join(str(item[field]).strip() for field in required)

