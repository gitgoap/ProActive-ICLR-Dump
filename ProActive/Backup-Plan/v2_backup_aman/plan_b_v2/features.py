"""Leakage-safe prefix features for the Plan B V2 correction selector.

Only the clean answer and probes acquired at the requested budget are visible.
The confidence feature set is deliberately named ``cached_confidence_proxy``:
probe absolute scores are reconstructed from clean scores plus cached shifts and
are not treated as independently verified probabilities.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from plan_b.core import Candidate, Observation, PlanBError, SOURCE_ORDER


TRANSFORM_SOURCES = ("blur", "crop", "brightness", "noise", "blank")

STRUCTURAL_FEATURE_NAMES = (
    *(f"source_{source}" for source in SOURCE_ORDER),
    "candidate_vote_share",
    "candidate_support_fraction",
    "candidate_transform_support_fraction",
    "candidate_independent_grounding_support_fraction",
    "candidate_is_grounding",
    "candidate_has_grounding_support",
    "candidate_is_clean",
    "candidate_agrees_clean",
    "grounding_agrees_clean",
    "clean_vote_share",
    "top_vote_share",
    "top_two_vote_gap",
    "candidate_vote_gap",
    "pairwise_agreement_rate",
    "normalized_vote_entropy",
    "distinct_answer_fraction",
    "usable_observation_fraction",
    "acquired_probe_fraction",
)

CONFIDENCE_FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES + (
    "confidence_available_fraction",
    "clean_answer_probability",
    "candidate_mean_answer_probability",
    "candidate_max_answer_probability",
    "candidate_minus_clean_probability",
    "candidate_probability_gap_to_best_other",
    "clean_token_margin",
    "candidate_mean_token_margin",
    "candidate_minus_clean_token_margin",
    "clean_token_entropy",
    "candidate_mean_token_entropy",
    "clean_minus_candidate_token_entropy",
)


def _finite_number(value: Any, *, name: str, instance_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanBError(f"Missing/non-numeric {name} for {instance_id}")
    result = float(value)
    if not math.isfinite(result):
        raise PlanBError(f"Non-finite {name} for {instance_id}")
    return result


def _source_payload(row: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    instance_id = str(row.get("instance_id", "<unknown>"))
    if source == "clean":
        payload = row.get("clean")
    else:
        probes = row.get("probes")
        payload = probes.get(source) if isinstance(probes, Mapping) else None
    if not isinstance(payload, Mapping):
        raise PlanBError(f"Missing {source} payload for {instance_id}")
    return payload


def source_confidence(row: Mapping[str, Any], source: str) -> tuple[float, float, float]:
    """Return (answer probability, token margin, token entropy).

    Probe values are reconstructed from the clean value plus the stored shift.
    This is why these quantities are an explicitly isolated proxy ablation.
    """

    instance_id = str(row.get("instance_id", "<unknown>"))
    clean = _source_payload(row, "clean")
    clean_prob = _finite_number(clean.get("answer_prob"), name="clean.answer_prob", instance_id=instance_id)
    clean_margin = _finite_number(
        clean.get("token_margin_mean"), name="clean.token_margin_mean", instance_id=instance_id
    )
    clean_entropy = _finite_number(
        clean.get("token_entropy_mean"), name="clean.token_entropy_mean", instance_id=instance_id
    )
    if source == "clean":
        values = (clean_prob, clean_margin, clean_entropy)
    else:
        payload = _source_payload(row, source)
        values = (
            clean_prob + _finite_number(payload.get("conf_shift"), name=f"{source}.conf_shift", instance_id=instance_id),
            clean_margin + _finite_number(
                payload.get("margin_shift"), name=f"{source}.margin_shift", instance_id=instance_id
            ),
            clean_entropy + _finite_number(
                payload.get("entropy_shift"), name=f"{source}.entropy_shift", instance_id=instance_id
            ),
        )
    if not all(math.isfinite(value) for value in values):
        raise PlanBError(f"Invalid reconstructed confidence for {instance_id}/{source}")
    return values


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 1 or len(counts) <= 1:
        return 0.0
    raw = -sum((count / total) * math.log(count / total) for count in counts if count)
    return raw / math.log(len(counts))


def structural_features(
    candidate: Candidate,
    candidates: Sequence[Candidate],
    observations: Sequence[Observation],
    acquired_cost: int,
) -> tuple[float, ...]:
    usable = [observation for observation in observations if observation.usable]
    usable_votes = len(usable)
    if usable_votes <= 0:
        raise PlanBError("Cannot construct V2 features without usable observations")
    by_source = {observation.source: observation for observation in observations}
    clean = by_source.get("clean")
    grounding = by_source.get("grounding")
    counts = sorted((item.vote_count for item in candidates), reverse=True)
    top = counts[0] / usable_votes
    second = counts[1] / usable_votes if len(counts) > 1 else 0.0
    other_best = max((item.vote_count for item in candidates if item.norm_answer != candidate.norm_answer), default=0)
    agreeing_pairs = sum(count * (count - 1) // 2 for count in counts)
    all_pairs = usable_votes * (usable_votes - 1) // 2
    clean_votes = next(
        (item.vote_count for item in candidates if clean is not None and item.norm_answer == clean.norm_answer), 0
    )
    transform_support = sum(source in candidate.sources for source in TRANSFORM_SOURCES)
    grounding_support = sum(
        source in candidate.sources for source in TRANSFORM_SOURCES if source in by_source and by_source[source].usable
    ) if "grounding" in candidate.sources else 0
    available_transforms = sum(
        source in by_source and by_source[source].usable for source in TRANSFORM_SOURCES
    )
    values = [float(source in candidate.sources) for source in SOURCE_ORDER]
    values.extend(
        [
            candidate.vote_count / usable_votes,
            candidate.vote_count / len(SOURCE_ORDER),
            transform_support / max(1, available_transforms),
            grounding_support / max(1, available_transforms),
            float("grounding" in candidate.sources),
            float("grounding" in candidate.sources and grounding_support > 0),
            float("clean" in candidate.sources),
            float(clean is not None and clean.usable and candidate.norm_answer == clean.norm_answer),
            float(
                clean is not None
                and grounding is not None
                and clean.usable
                and grounding.usable
                and clean.norm_answer == grounding.norm_answer
            ),
            clean_votes / usable_votes,
            top,
            top - second,
            (candidate.vote_count - other_best) / usable_votes,
            agreeing_pairs / all_pairs if all_pairs else 1.0,
            _entropy(counts),
            len(candidates) / len(SOURCE_ORDER),
            usable_votes / len(SOURCE_ORDER),
            acquired_cost / 6.0,
        ]
    )
    if len(values) != len(STRUCTURAL_FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise PlanBError("Invalid V2 structural feature vector")
    return tuple(values)


def cached_confidence_features(
    row: Mapping[str, Any],
    candidate: Candidate,
    candidates: Sequence[Candidate],
    observations: Sequence[Observation],
    acquired_cost: int,
) -> tuple[float, ...]:
    base = list(structural_features(candidate, candidates, observations, acquired_cost))
    usable = [observation for observation in observations if observation.usable]
    source_values = {observation.source: source_confidence(row, observation.source) for observation in usable}
    candidate_values = [source_values[source] for source in candidate.sources]
    clean_prob, clean_margin, clean_entropy = source_confidence(row, "clean")
    candidate_prob = sum(value[0] for value in candidate_values) / len(candidate_values)
    candidate_margin = sum(value[1] for value in candidate_values) / len(candidate_values)
    candidate_entropy = sum(value[2] for value in candidate_values) / len(candidate_values)
    other_probabilities = []
    for other in candidates:
        if other.norm_answer == candidate.norm_answer:
            continue
        vals = [source_values[source][0] for source in other.sources]
        other_probabilities.append(sum(vals) / len(vals))
    best_other = max(other_probabilities, default=candidate_prob)
    base.extend(
        [
            len(source_values) / max(1, len(usable)),
            clean_prob,
            candidate_prob,
            max(value[0] for value in candidate_values),
            candidate_prob - clean_prob,
            candidate_prob - best_other,
            clean_margin,
            candidate_margin,
            candidate_margin - clean_margin,
            clean_entropy,
            candidate_entropy,
            clean_entropy - candidate_entropy,
        ]
    )
    if len(base) != len(CONFIDENCE_FEATURE_NAMES) or not all(math.isfinite(value) for value in base):
        raise PlanBError("Invalid V2 cached-confidence feature vector")
    return tuple(base)


def feature_vector(
    feature_set: str,
    row: Mapping[str, Any],
    candidate: Candidate,
    candidates: Sequence[Candidate],
    observations: Sequence[Observation],
    acquired_cost: int,
) -> tuple[float, ...]:
    if feature_set == "structural_v2":
        return structural_features(candidate, candidates, observations, acquired_cost)
    if feature_set == "structural_plus_cached_confidence_proxy_v2":
        return cached_confidence_features(row, candidate, candidates, observations, acquired_cost)
    raise PlanBError(f"Unknown V2 feature set: {feature_set}")


def feature_names(feature_set: str) -> tuple[str, ...]:
    if feature_set == "structural_v2":
        return STRUCTURAL_FEATURE_NAMES
    if feature_set == "structural_plus_cached_confidence_proxy_v2":
        return CONFIDENCE_FEATURE_NAMES
    raise PlanBError(f"Unknown V2 feature set: {feature_set}")
