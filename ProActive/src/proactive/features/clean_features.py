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


class MissingScoreError(ValueError):
    """Raised when token logprobs are missing or empty."""
    pass


class MissingDistributionError(ValueError):
    """Raised when token distributions are missing or empty."""
    pass


def compute_confidence(token_logprobs: List[float]) -> float:
    """Length-normalized confidence (Plan §14.3).

    c = exp( (1/T) * sum(log p(y_t | y_{<t}, x)) )
    """
    if not token_logprobs:
        raise MissingScoreError("Cannot compute confidence: token_logprobs is empty or None")
    for lp in token_logprobs:
        if not math.isfinite(lp):
            raise ValueError(f"Non-finite token log-probability encountered: {lp}")
    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    c = math.exp(mean_logprob)
    if not math.isfinite(c) or c < 0.0 or c > 1.0001:
        raise ValueError(f"Invalid confidence {c} computed from mean_logprob={mean_logprob}")
    return min(1.0, max(0.0, c))


def compute_answer_logprob(token_logprobs: List[float]) -> float:
    """Total answer log-probability: sum of token log-probs."""
    if not token_logprobs:
        raise MissingScoreError("Cannot compute total logprob: token_logprobs is empty or None")
    for lp in token_logprobs:
        if not math.isfinite(lp):
            raise ValueError(f"Non-finite token log-probability encountered: {lp}")
    return sum(token_logprobs)


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
        raise MissingDistributionError("Cannot compute entropy: token_distributions is empty or None")

    total_entropy = 0.0
    for dist in token_distributions:
        if not dist:
            raise MissingDistributionError("Encountered empty token distribution dict")
        token_entropy = 0.0
        for prob in dist.values():
            if not math.isfinite(prob) or prob < 0:
                raise ValueError(f"Invalid probability in token distribution: {prob}")
            if prob > 0:
                token_entropy -= prob * math.log(prob)
        if not math.isfinite(token_entropy):
            raise ValueError("Non-finite token entropy computed")
        total_entropy += token_entropy

    return total_entropy / len(token_distributions)


def compute_margin(
    token_distributions: List[Dict[str, float]],
) -> float:
    """Mean token margin: top-1 prob minus top-2 prob (Plan §14.3).

    m_bar = (1/T) * sum( p_t^(1) - p_t^(2) )
    """
    if not token_distributions:
        raise MissingDistributionError("Cannot compute margin: token_distributions is empty or None")

    total_margin = 0.0
    for dist in token_distributions:
        if not dist:
            raise MissingDistributionError("Encountered empty token distribution dict")
        sorted_probs = sorted(dist.values(), reverse=True)
        top1 = sorted_probs[0] if len(sorted_probs) >= 1 else 0.0
        top2 = sorted_probs[1] if len(sorted_probs) >= 2 else 0.0
        if not (math.isfinite(top1) and math.isfinite(top2)):
            raise ValueError(f"Non-finite top probabilities in margin calculation: top1={top1}, top2={top2}")
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
    score_method: str = "generation_logits",
    valid: bool = True,
    invalid_reason: Optional[str] = None,
) -> CleanFeatures:
    """Build CleanFeatures from raw generation outputs with strict score checks.

    Args:
        raw_answer: Raw model output.
        norm_answer: Normalized answer string.
        token_logprobs: Log-probabilities for each generated token.
        token_distributions: Optional top-k distributions per token.
        answer_len_tokens: Number of generated tokens.
        relation_available: Whether relation swap is legal.
        correct: Whether the answer matches the gold (training-time only).
        latency_ms: Generation latency in milliseconds.
        qtype: Coarse question type.
        answerability: Test-time answerability cue.
        iqa_features: Image quality features.
        score_method: "generation_logits" or "teacher_forced".
        valid: Whether features are valid.
        invalid_reason: Reason if marked invalid.
    """
    if not valid:
        return CleanFeatures(
            raw_answer=raw_answer,
            norm_answer=norm_answer,
            answer_prob=0.0,
            token_entropy_mean=0.0,
            token_margin_mean=0.0,
            answer_len_tokens=0,
            relation_available=relation_available,
            score_method=score_method,
            valid=False,
            invalid_reason=invalid_reason,
            iqa_features=iqa_features,
            qtype=qtype,
            answerability=answerability,
            answer_logprob=float("-inf"),
            latency_ms=latency_ms,
            correct=correct,
        )

    # Validate score inputs strictly
    if not token_logprobs or not token_distributions:
        return CleanFeatures(
            raw_answer=raw_answer,
            norm_answer=norm_answer,
            answer_prob=0.0,
            token_entropy_mean=0.0,
            token_margin_mean=0.0,
            answer_len_tokens=0,
            relation_available=relation_available,
            score_method=score_method,
            valid=False,
            invalid_reason="missing_scores_or_distributions",
            iqa_features=iqa_features,
            qtype=qtype,
            answerability=answerability,
            answer_logprob=float("-inf"),
            latency_ms=latency_ms,
            correct=correct,
        )

    try:
        confidence = compute_confidence(token_logprobs)
        logprob = compute_answer_logprob(token_logprobs)
        entropy = compute_entropy(token_distributions)
        margin = compute_margin(token_distributions)
        length = answer_len_tokens if answer_len_tokens is not None else len(token_logprobs)

        return CleanFeatures(
            raw_answer=raw_answer,
            norm_answer=norm_answer,
            answer_prob=confidence,
            token_entropy_mean=entropy,
            token_margin_mean=margin,
            answer_len_tokens=length,
            relation_available=relation_available,
            score_method=score_method,
            valid=True,
            invalid_reason=None,
            iqa_features=iqa_features,
            qtype=qtype,
            answerability=answerability,
            answer_logprob=logprob,
            latency_ms=latency_ms,
            correct=correct,
        )
    except Exception as e:
        return CleanFeatures(
            raw_answer=raw_answer,
            norm_answer=norm_answer,
            answer_prob=0.0,
            token_entropy_mean=0.0,
            token_margin_mean=0.0,
            answer_len_tokens=0,
            relation_available=relation_available,
            score_method=score_method,
            valid=False,
            invalid_reason=f"score_computation_error: {str(e)}",
            iqa_features=iqa_features,
            qtype=qtype,
            answerability=answerability,
            answer_logprob=float("-inf"),
            latency_ms=latency_ms,
            correct=correct,
        )
