"""
Shared score extraction utilities used by all model adapters.

Computes confidence, entropy, margin, and other features from
token-level outputs regardless of model-specific API differences.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


def logprobs_to_confidence(token_logprobs: List[float]) -> float:
    """Length-normalized confidence: exp( mean(logprobs) )."""
    if not token_logprobs:
        return 0.0
    return math.exp(sum(token_logprobs) / len(token_logprobs))


def logprobs_to_entropy(
    token_distributions: List[Dict[str, float]],
) -> float:
    """Mean token entropy from per-position probability distributions."""
    if not token_distributions:
        return 0.0
    total = 0.0
    for dist in token_distributions:
        h = 0.0
        for p in dist.values():
            if p > 0:
                h -= p * math.log(p)
        total += h
    return total / len(token_distributions)


def logprobs_to_margin(
    token_distributions: List[Dict[str, float]],
) -> float:
    """Mean top-1 minus top-2 margin."""
    if not token_distributions:
        return 0.0
    total = 0.0
    for dist in token_distributions:
        probs = sorted(dist.values(), reverse=True)
        top1 = probs[0] if len(probs) >= 1 else 0.0
        top2 = probs[1] if len(probs) >= 2 else 0.0
        total += (top1 - top2)
    return total / len(token_distributions)


def validate_scores(
    answer_prob: float,
    entropy: float,
    margin: float,
) -> Tuple[bool, List[str]]:
    """Validate that extracted scores are finite and in expected ranges.

    Returns (is_valid, list_of_issues).
    """
    issues = []
    if not math.isfinite(answer_prob):
        issues.append(f"answer_prob is not finite: {answer_prob}")
    if not math.isfinite(entropy):
        issues.append(f"entropy is not finite: {entropy}")
    if not math.isfinite(margin):
        issues.append(f"margin is not finite: {margin}")
    if answer_prob < 0 or answer_prob > 1:
        issues.append(f"answer_prob out of [0,1]: {answer_prob}")
    if entropy < 0:
        issues.append(f"entropy is negative: {entropy}")
    return (len(issues) == 0, issues)
