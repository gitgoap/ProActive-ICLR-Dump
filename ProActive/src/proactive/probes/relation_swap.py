"""
Relation swap probe for the ProActive diagnostic system.

Implements the reversible spatial-relation lexicon (Plan §13.3)
and the applicability checks required before running the relation probe.

A relation swap is legal ONLY when:
1. Exactly one reversible relation is detected in the statement.
2. If annotated relation is present, it matches the detected relation and is in the reversible lexicon.
3. Grammatical and semantic context confirms spatial relation usage (rejects non-spatial verbs/adjectives).
4. The statement has no ambiguous negation.
5. The expected gold answer logically reverses ("true" <-> "false", "yes" <-> "no").
6. Round-trip swapping restores the original text: swap(swap(q)) == q.

Represents relation outcomes with explicit statuses:
`invariant`, `changed_correctly`, `changed_incorrectly`, `invalid`, `not_applicable`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from proactive.features.normalization import normalize_answer


# ---------------------------------------------------------------------------
# Reversible relation lexicon  (Plan §13.3)
# ---------------------------------------------------------------------------

REVERSIBLE_PAIRS: List[Tuple[str, str]] = [
    ("left", "right"),
    ("above", "below"),
    ("in front of", "behind"),
    ("on top of", "under"),
]

_FORWARD: Dict[str, str] = {}
for a, b in REVERSIBLE_PAIRS:
    _FORWARD[a] = b
    _FORWARD[b] = a

ALL_RELATIONS: List[str] = sorted(
    list(_FORWARD.keys()), key=len, reverse=True
)

# Canonical mapping for dataset-annotated relations (e.g. VSR relations)
ANNOTATED_RELATION_MAP: Dict[str, str] = {
    # left <-> right
    "left": "left",
    "to the left of": "left",
    "at the left side of": "left",
    "on the left of": "left",
    "on the left side of": "left",
    "left of": "left",
    "right": "right",
    "to the right of": "right",
    "at the right side of": "right",
    "on the right of": "right",
    "on the right side of": "right",
    "right of": "right",
    # above <-> below
    "above": "above",
    "below": "below",
    # in front of <-> behind
    "in front of": "in front of",
    "behind": "behind",
    # on top of <-> under
    "on top of": "on top of",
    "under": "under",
}

INVERSE_GOLD_MAP: Dict[str, str] = {
    "true": "false",
    "false": "true",
    "yes": "no",
    "no": "yes",
    "1": "0",
    "0": "1",
}


class RelationSwapStatus(str, Enum):
    """Explicit outcome status for relation swap probe (Contract 2)."""
    INVARIANT = "invariant"                     # Model gave same answer despite swapped premise (bad, triggers b_A)
    CHANGED_CORRECTLY = "changed_correctly"     # Model correctly flipped to swapped gold
    CHANGED_INCORRECTLY = "changed_incorrectly" # Model flipped but to wrong state
    INVALID = "invalid"                         # Malformed, unknown, or clean baseline incorrect
    NOT_APPLICABLE = "not_applicable"           # Relation probe not legal for this instance


@dataclass
class RelationSwapResult:
    """Result of attempting a relation swap."""
    applicable: bool
    original_text: str
    swapped_text: Optional[str] = None
    relation_found: Optional[str] = None
    relation_swapped_to: Optional[str] = None
    swapped_gold_answer: Optional[str] = None
    round_trip_ok: bool = False
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Grammatical context checks
# ---------------------------------------------------------------------------

NON_SPATIAL_PATTERNS = [
    # "left" as verb meaning departed: "left the room", "left home", "left his coat"
    r'\bleft\s+(?:the|a|an|his|her|their|my|our|this|that|home|school|work|office|room|building|station|place|car|table)\b',
    # "right" as adjective meaning correct: "answer is right", "you are right", "is the answer right"
    r'\b(?:answer|response|statement|guess|choice)\s+(?:is|was)\s+right\b',
    r'\bis\s+(?:it|the\s+answer|that|this)\s+right\b',
    r'\b(?:all|that\'s)\s+right\b',
    r'\bright\s+(?:now|here|there|away|after|before)\b',
]


def _is_non_spatial_usage(text: str) -> bool:
    """Detect common non-spatial usage of words like 'left' and 'right'."""
    text_lower = text.lower()
    for pattern in NON_SPATIAL_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def _find_relations(text: str) -> List[Tuple[str, int, int]]:
    """Find all reversible relations in the text with boundary matching."""
    text_lower = text.lower()
    found: List[Tuple[str, int, int]] = []
    used_ranges: List[Tuple[int, int]] = []

    for relation in ALL_RELATIONS:
        pattern = r'\b' + re.escape(relation) + r'\b'
        for match in re.finditer(pattern, text_lower):
            start, end = match.start(), match.end()
            overlap = any(
                s < end and start < e for s, e in used_ranges
            )
            if not overlap:
                found.append((relation, start, end))
                used_ranges.append((start, end))

    return found


def _has_ambiguous_negation(text: str) -> bool:
    """Check if the text contains complex or ambiguous negation."""
    text_lower = text.lower()
    negation_words = ["not", "n't", "no", "never", "neither", "nor"]
    neg_count = sum(
        len(re.findall(r'\b' + re.escape(w) + r'\b', text_lower))
        for w in negation_words
    )
    return neg_count > 1


def invert_gold_answer(gold_answer: Optional[str], dataset: str = "vsr") -> Optional[str]:
    """Logically invert a binary gold answer."""
    if gold_answer is None:
        return None
    norm_gold = normalize_answer(str(gold_answer), dataset).lower()
    return INVERSE_GOLD_MAP.get(norm_gold, None)


# ---------------------------------------------------------------------------
# Core swap logic
# ---------------------------------------------------------------------------

def swap_relation(
    text: str,
    annotated_relation: Optional[str] = None,
    gold_answer: Optional[str] = None,
    _check_round_trip: bool = True,
) -> RelationSwapResult:
    """Attempt to swap a single reversible relation in the text.

    Args:
        text: Input statement/question to be swapped.
        annotated_relation: Optional dataset annotation of the relation.
        gold_answer: Optional clean gold answer to invert.
        _check_round_trip: Whether to verify round-trip restoration.
    """
    # 1. Non-spatial filter (rejects "left the room", "answer is right")
    if _is_non_spatial_usage(text):
        return RelationSwapResult(
            applicable=False,
            original_text=text,
            reason="non_spatial_usage_detected",
        )

    # 2. Check annotated relation if provided
    canonical_ann: Optional[str] = None
    if annotated_relation is not None:
        norm_ann = annotated_relation.strip().lower()
        if norm_ann not in ANNOTATED_RELATION_MAP:
            return RelationSwapResult(
                applicable=False,
                original_text=text,
                reason=f"annotated_relation_not_reversible: '{annotated_relation}'",
            )
        canonical_ann = ANNOTATED_RELATION_MAP[norm_ann]

    # 3. Find candidate relations in text
    relations = _find_relations(text)

    if len(relations) == 0:
        return RelationSwapResult(
            applicable=False,
            original_text=text,
            reason="no_reversible_relation_found",
        )

    if len(relations) > 1:
        return RelationSwapResult(
            applicable=False,
            original_text=text,
            reason="multiple_relations_found",
        )

    if _has_ambiguous_negation(text):
        return RelationSwapResult(
            applicable=False,
            original_text=text,
            reason="ambiguous_negation",
        )

    relation_found, start, end = relations[0]

    # 4. Check alignment with annotated relation
    if canonical_ann is not None:
        if canonical_ann != relation_found:
            return RelationSwapResult(
                applicable=False,
                original_text=text,
                reason=f"annotated_relation_mismatch: '{canonical_ann}' vs detected '{relation_found}'",
            )

    # 5. Check gold answer reversibility if provided
    swapped_gold: Optional[str] = None
    if gold_answer is not None:
        swapped_gold = invert_gold_answer(gold_answer)
        if swapped_gold is None:
            return RelationSwapResult(
                applicable=False,
                original_text=text,
                reason=f"gold_answer_not_invertible: '{gold_answer}'",
            )

    target = _FORWARD[relation_found]

    # 6. Case-preserving replacement
    original_in_text = text[start:end]
    if original_in_text[0].isupper():
        replacement = target[0].upper() + target[1:]
    else:
        replacement = target

    swapped_text = text[:start] + replacement + text[end:]

    # 7. Round-trip verification
    round_trip_ok = True
    if _check_round_trip:
        round_trip_result = swap_relation(
            swapped_text,
            annotated_relation=target,
            gold_answer=swapped_gold,
            _check_round_trip=False,
        )
        round_trip_ok = (
            round_trip_result.applicable
            and round_trip_result.swapped_text == text
        )
        if not round_trip_ok:
            return RelationSwapResult(
                applicable=False,
                original_text=text,
                swapped_text=swapped_text,
                relation_found=relation_found,
                relation_swapped_to=target,
                swapped_gold_answer=swapped_gold,
                round_trip_ok=False,
                reason="round_trip_failed",
            )

    return RelationSwapResult(
        applicable=True,
        original_text=text,
        swapped_text=swapped_text,
        relation_found=relation_found,
        relation_swapped_to=target,
        swapped_gold_answer=swapped_gold,
        round_trip_ok=round_trip_ok,
    )


def check_relation_applicable(
    text: str,
    annotated_relation: Optional[str] = None,
    gold_answer: Optional[str] = None,
) -> bool:
    """Check if relation swap is applicable for this text."""
    return swap_relation(text, annotated_relation, gold_answer).applicable


def evaluate_relation_swap(
    original_answer_correct: Optional[bool],
    original_norm_answer: str,
    swapped_norm_answer: str,
    original_gold: str,
    swapped_gold: str,
    dataset: str = "vsr",
) -> RelationSwapStatus:
    """Evaluate relation probe outcome with fail-closed status (Contract 2).

    Rules:
    1. If original clean answer is not correct (or not evaluated), return INVALID.
    2. If original gold == swapped gold (not an opposite-gold pair), return INVALID.
    3. If swapped answer is unknown, malformed, or empty, return INVALID.
    4. If swapped answer matches swapped gold, return CHANGED_CORRECTLY.
    5. If swapped answer matches original answer (invariant to swap), return INVARIANT.
    6. Otherwise, return CHANGED_INCORRECTLY.
    """
    if original_answer_correct is not True:
        return RelationSwapStatus.INVALID

    norm_orig_gold = normalize_answer(original_gold, dataset)
    norm_swap_gold = normalize_answer(swapped_gold, dataset)

    if norm_orig_gold == norm_swap_gold or norm_orig_gold in ("", "unknown"):
        return RelationSwapStatus.INVALID

    norm_swap_ans = normalize_answer(swapped_norm_answer, dataset)
    norm_orig_ans = normalize_answer(original_norm_answer, dataset)

    if norm_swap_ans in ("", "unknown", "invalid"):
        return RelationSwapStatus.INVALID

    if norm_swap_ans == norm_swap_gold:
        return RelationSwapStatus.CHANGED_CORRECTLY
    elif norm_swap_ans == norm_orig_ans:
        return RelationSwapStatus.INVARIANT
    else:
        return RelationSwapStatus.CHANGED_INCORRECTLY


def get_swap_invariance(
    original_answer_correct: Optional[bool],
    swapped_answer: str,
    original_gold: str,
    swapped_gold: str,
    original_norm_answer: str = "",
    dataset: str = "vsr",
) -> Optional[bool]:
    """Return boolean swap invariance for backward compatibility with signature.

    Returns:
        True: swap invariance detected (bad for model, triggers b_A).
        False: model changed correctly.
        None: invalid test (does not trigger b_A).
    """
    status = evaluate_relation_swap(
        original_answer_correct=original_answer_correct,
        original_norm_answer=original_norm_answer,
        swapped_norm_answer=swapped_answer,
        original_gold=original_gold,
        swapped_gold=swapped_gold,
        dataset=dataset,
    )
    if status == RelationSwapStatus.INVARIANT:
        return True
    elif status == RelationSwapStatus.CHANGED_CORRECTLY:
        return False
    else:
        # INVALID or CHANGED_INCORRECTLY -> None (does not trigger b_A)
        return None
