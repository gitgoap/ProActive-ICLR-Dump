"""
Clean feature extraction from MLLM generation outputs.

Computes the clean feature vector g(x) as defined in Plan §5.3:
  [c, H_bar, m_bar, l, IQA, qtype, answerability, relation_available]

Reference: Plan §14.3 for the exact formulas.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from proactive.features.evidence_state import CleanFeatures


def compute_confidence(token_logprobs: List[float]) -> float:
    """Length-normalized confidence (Plan §14.3).

    c = exp( (1/T) * sum(log p(y_t | y_{<t}, x)) )
    """
    if not token_logprobs:
        return 0.0
    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(mean_logprob)


def compute_answer_logprob(token_logprobs: List[float]) -> float:
    """Total answer log-probability: sum of token log-probs."""
    return sum(token_logprobs) if token_logprobs else float("-inf")


def compute_entropy(
    token_distributions: List[Dict[str, float]],
    top_k: int = 50,
) -> float:
    """Mean token entropy (Plan §14.3).

    H_bar = (1/T) * sum( -sum(p(v) * log(p(v))) for each token )

    token_distributions: list of {token: prob} dicts per position.
    Uses top-k approximation when full vocab is unavailable.
    """
    if not token_distributions:
        return 0.0

    total_entropy = 0.0
    for dist in token_distributions:
        token_entropy = 0.0
        for prob in dist.values():
            if prob > 0:
                token_entropy -= prob * math.log(prob)
        total_entropy += token_entropy

    return total_entropy / len(token_distributions)


def compute_margin(
    token_distributions: List[Dict[str, float]],
) -> float:
    """Mean token margin: top-1 prob minus top-2 prob (Plan §14.3).

    m_bar = (1/T) * sum( p_t^(1) - p_t^(2) )
    """
    if not token_distributions:
        return 0.0

    total_margin = 0.0
    for dist in token_distributions:
        sorted_probs = sorted(dist.values(), reverse=True)
        top1 = sorted_probs[0] if len(sorted_probs) >= 1 else 0.0
        top2 = sorted_probs[1] if len(sorted_probs) >= 2 else 0.0
        total_margin += (top1 - top2)

    return total_margin / len(token_distributions)


def extract_clean_features(
    raw_answer: str,
    norm_answer: str,
    token_logprobs: List[float],
    token_distributions: Optional[List[Dict[str, float]]] = None,
    answer_len_tokens: Optional[int] = None,
    relation_available: bool = False,
    correct: Optional[bool] = None,
    latency_ms: Optional[float] = None,
    qtype: Optional[str] = None,
    answerability: Optional[float] = None,
    iqa_features: Optional[Dict[str, float]] = None,
) -> CleanFeatures:
    """Build CleanFeatures from raw generation outputs.

    Args:
        raw_answer: Raw model output.
        norm_answer: Normalized answer string.
        token_logprobs: Log-probabilities for each generated token.
        token_distributions: Optional top-k distributions per token.
        answer_len_tokens: Number of generated tokens. Inferred from
            token_logprobs if not provided.
        relation_available: Whether relation swap is legal.
        correct: Whether the answer matches the gold (training-time only).
        latency_ms: Generation latency in milliseconds.
        qtype: Coarse question type.
        answerability: Test-time answerability cue.
        iqa_features: Image quality features.
    """
    if answer_len_tokens is None:
        answer_len_tokens = len(token_logprobs)

    confidence = compute_confidence(token_logprobs)
    logprob = compute_answer_logprob(token_logprobs)

    # Entropy and margin require full distributions
    entropy = 0.0
    margin = 0.0
    if token_distributions:
        entropy = compute_entropy(token_distributions)
        margin = compute_margin(token_distributions)

    return CleanFeatures(
        raw_answer=raw_answer,
        norm_answer=norm_answer,
        answer_prob=confidence,
        token_entropy_mean=entropy,
        token_margin_mean=margin,
        answer_len_tokens=answer_len_tokens,
        relation_available=relation_available,
        iqa_features=iqa_features,
        qtype=qtype,
        answerability=answerability,
        answer_logprob=logprob,
        latency_ms=latency_ms,
        correct=correct,
    )
