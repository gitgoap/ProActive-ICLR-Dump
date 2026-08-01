"""
Probe runner: orchestrates running all applicable probes for one instance.

Given a manifest record, a loaded MLLM adapter, and the clean generation
output, the probe runner applies each legal probe independently to the
ORIGINAL image-question pair and returns ProbeObservation objects.

CRITICAL: Every probe is applied to the original input, never chained
on top of another probe's output. (Plan §2.5)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from proactive.features.clean_features import (
    compute_confidence,
    compute_entropy,
    compute_margin,
)
from proactive.features.evidence_state import (
    ProbeAction,
    ProbeObservation,
    NON_STOP_PROBES,
    VISUAL_PROBES,
)
from proactive.features.normalization import normalize_answer
from proactive.features.semantic import compute_semantic_match
from proactive.models.base_adapter import GenerationOutput, MLLMAdapter
from proactive.probes.image_transforms import (
    apply_image_transform,
    get_transform_hash,
    derive_transform_seed,
    CANONICAL_SEVERITIES,
)
from proactive.probes.relation_swap import swap_relation
from proactive.prompts.templates import (
    make_dataset_prompt,
    make_grounding_prompt,
    make_relation_prompt,
    parse_grounding_output,
)
from proactive.utils.hashing import hash_prompt, hash_generation_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probe observation builder
# ---------------------------------------------------------------------------

def _build_probe_observation(
    probe_id: ProbeAction,
    raw_answer: str,
    norm_answer: str,
    clean_norm_answer: str,
    clean_answer_prob: float,
    clean_entropy: float,
    clean_margin: float,
    dataset: str,
    applicable: bool,
    severity: Optional[float],
    prompt_text: str,
    gen_config: Dict[str, Any],
    token_logprobs: List[float],
    token_distributions: Optional[List[Dict[str, float]]] = None,
    transform_hash: Optional[str] = None,
    latency_ms: Optional[float] = None,
    valid: bool = True,
    invalid_reason: Optional[str] = None,
    parse_status: str = "ok",
    score_method: str = "generation_logits",
    semantic_threshold: float = 0.82,
    embedding_fn: Optional[Callable[[str, str], float]] = None,
) -> ProbeObservation:
    """Build a ProbeObservation from generation output and clean baselines."""
    # Fail-closed check: if invalid, do not record flips or conf shifts
    if not valid:
        return ProbeObservation(
            probe_id=probe_id,
            raw_answer=raw_answer,
            norm_answer=norm_answer,
            flip=False,
            conf_shift=0.0,
            entropy_shift=0.0,
            margin_shift=0.0,
            exact_match=0.0,
            semantic_match=0.0,
            applicable=applicable,
            severity=severity,
            latency_ms=latency_ms,
            prompt_hash=hash_prompt(prompt_text),
            image_transform_hash=transform_hash,
            generation_config_hash=hash_generation_config(gen_config),
            valid=False,
            invalid_reason=invalid_reason,
            parse_status=parse_status,
            score_method=score_method,
            answer_prob=0.0,
            token_entropy_mean=0.0,
            token_margin_mean=0.0,
        )

    # Compute probe-side features with finite validation
    try:
        probe_prob = compute_confidence(token_logprobs)
        probe_entropy = compute_entropy(token_distributions or [])
        probe_margin = compute_margin(token_distributions or [])
    except Exception as e:
        logger.debug(f"Probe {probe_id.value} score computation failed: {e}")
        return ProbeObservation(
            probe_id=probe_id,
            raw_answer=raw_answer,
            norm_answer=norm_answer,
            flip=False,
            conf_shift=0.0,
            entropy_shift=0.0,
            margin_shift=0.0,
            exact_match=0.0,
            semantic_match=0.0,
            applicable=applicable,
            severity=severity,
            latency_ms=latency_ms,
            prompt_hash=hash_prompt(prompt_text),
            image_transform_hash=transform_hash,
            generation_config_hash=hash_generation_config(gen_config),
            valid=False,
            invalid_reason=f"score_computation_failed: {str(e)}",
            parse_status=parse_status,
            score_method=score_method,
            answer_prob=0.0,
            token_entropy_mean=0.0,
            token_margin_mean=0.0,
        )

    conf_shift = probe_prob - clean_answer_prob
    entropy_shift = probe_entropy - clean_entropy
    margin_shift = probe_margin - clean_margin

    # Answer matching
    flip = (norm_answer != clean_norm_answer)
    exact_match = 1.0 if (norm_answer == clean_norm_answer) else 0.0

    semantic_match = 0.0
    try:
        semantic_match = compute_semantic_match(
            pred_answer=norm_answer,
            target_answer=clean_norm_answer,
            dataset=dataset,
            threshold=semantic_threshold,
            embedding_fn=embedding_fn,
        )
    except Exception as e:
        logger.debug(f"Semantic match computation failed for probe {probe_id.value}: {e}")
        semantic_match = exact_match

    return ProbeObservation(
        probe_id=probe_id,
        raw_answer=raw_answer,
        norm_answer=norm_answer,
        flip=flip,
        conf_shift=conf_shift,
        entropy_shift=entropy_shift,
        margin_shift=margin_shift,
        exact_match=exact_match,
        semantic_match=semantic_match,
        applicable=applicable,
        severity=severity,
        latency_ms=latency_ms,
        prompt_hash=hash_prompt(prompt_text),
        image_transform_hash=transform_hash,
        generation_config_hash=hash_generation_config(gen_config),
        valid=True,
        invalid_reason=None,
        parse_status=parse_status,
        score_method=score_method,
        answer_prob=probe_prob,
        token_entropy_mean=probe_entropy,
        token_margin_mean=probe_margin,
    )


# ---------------------------------------------------------------------------
# Individual probe runners
# ---------------------------------------------------------------------------

def _run_visual_probe(
    probe_id: ProbeAction,
    adapter: MLLMAdapter,
    image: Image.Image,
    prompt_text: str,
    dataset: str,
    clean_norm_answer: str,
    clean_prob: float,
    clean_entropy: float,
    clean_margin: float,
    severity: Optional[float] = None,
    instance_id: str = "",
    global_seed: int = 42,
    score_method: str = "generation_logits",
    semantic_threshold: float = 0.82,
    embedding_fn: Optional[Callable[[str, str], float]] = None,
) -> ProbeObservation:
    """Run a visual image-transform probe."""
    probe_name = probe_id.value
    actual_severity = severity if severity is not None else CANONICAL_SEVERITIES.get(probe_name, 0.0)
    noise_seed = derive_transform_seed(global_seed, instance_id, probe_name, actual_severity)

    transformed = apply_image_transform(
        image, probe_name, severity=actual_severity, noise_seed=noise_seed
    )
    transform_hash = get_transform_hash(
        probe_name, severity=actual_severity, noise_seed=noise_seed
    )

    gen_output = adapter.generate(transformed, prompt_text)
    norm_answer = normalize_answer(gen_output.raw_answer, dataset)

    # Score method consistency
    token_lps = gen_output.token_logprobs
    token_dists = gen_output.token_distributions

    if score_method == "teacher_forced" or not token_lps or not all(math.isfinite(lp) for lp in token_lps):
        score_out = adapter.score(transformed, prompt_text, norm_answer)
        token_lps = score_out.token_logprobs
        token_dists = score_out.token_distributions

    return _build_probe_observation(
        probe_id=probe_id,
        raw_answer=gen_output.raw_answer,
        norm_answer=norm_answer,
        clean_norm_answer=clean_norm_answer,
        clean_answer_prob=clean_prob,
        clean_entropy=clean_entropy,
        clean_margin=clean_margin,
        dataset=dataset,
        applicable=True,
        severity=actual_severity,
        prompt_text=prompt_text,
        gen_config=adapter.get_generation_config_dict(),
        token_logprobs=token_lps,
        token_distributions=token_dists,
        transform_hash=transform_hash,
        latency_ms=gen_output.latency_ms,
        score_method=score_method,
        semantic_threshold=semantic_threshold,
        embedding_fn=embedding_fn,
    )


def _run_grounding_probe(
    adapter: MLLMAdapter,
    image: Image.Image,
    question: str,
    dataset: str,
    clean_norm_answer: str,
    clean_prob: float,
    clean_entropy: float,
    clean_margin: float,
    score_method: str = "generation_logits",
    semantic_threshold: float = 0.82,
    embedding_fn: Optional[Callable[[str, str], float]] = None,
) -> ProbeObservation:
    """Run the grounding probe (describe-then-answer) with isolated final answer scoring."""
    prompt_text = make_grounding_prompt(question, dataset)
    gen_output = adapter.generate(image, prompt_text)

    # Parse machine-readable answer
    parsed = parse_grounding_output(gen_output.raw_answer, dataset)

    if not parsed.is_valid:
        return _build_probe_observation(
            probe_id=ProbeAction.GROUNDING,
            raw_answer=gen_output.raw_answer,
            norm_answer=parsed.norm_final_answer or "invalid",
            clean_norm_answer=clean_norm_answer,
            clean_answer_prob=clean_prob,
            clean_entropy=clean_entropy,
            clean_margin=clean_margin,
            dataset=dataset,
            applicable=True,
            severity=None,
            prompt_text=prompt_text,
            gen_config=adapter.get_generation_config_dict(),
            token_logprobs=[],
            valid=False,
            invalid_reason=parsed.invalid_reason,
            parse_status=parsed.parse_status,
            score_method=score_method,
            latency_ms=gen_output.latency_ms,
        )

    # Score ONLY the final answer tokens
    token_lps: List[float] = []
    token_dists: Optional[List[Dict[str, float]]] = None

    if score_method == "teacher_forced":
        score_prompt = f"{prompt_text}\n{parsed.description}\nFINAL_ANSWER:"
        score_out = adapter.score(image, score_prompt, f" {parsed.norm_final_answer}")
        token_lps = score_out.token_logprobs
        token_dists = score_out.token_distributions
    else:
        ans_len = max(1, len(parsed.raw_final_answer.split()))
        if gen_output.token_logprobs:
            token_lps = gen_output.token_logprobs[-ans_len:]
            if gen_output.token_distributions:
                token_dists = gen_output.token_distributions[-ans_len:]

    return _build_probe_observation(
        probe_id=ProbeAction.GROUNDING,
        raw_answer=gen_output.raw_answer,
        norm_answer=parsed.norm_final_answer,
        clean_norm_answer=clean_norm_answer,
        clean_answer_prob=clean_prob,
        clean_entropy=clean_entropy,
        clean_margin=clean_margin,
        dataset=dataset,
        applicable=True,
        severity=None,
        prompt_text=prompt_text,
        gen_config=adapter.get_generation_config_dict(),
        token_logprobs=token_lps,
        token_distributions=token_dists,
        latency_ms=gen_output.latency_ms,
        valid=True,
        parse_status=parsed.parse_status,
        score_method=score_method,
        semantic_threshold=semantic_threshold,
        embedding_fn=embedding_fn,
    )


def _run_relation_probe(
    adapter: MLLMAdapter,
    image: Image.Image,
    question: str,
    dataset: str,
    clean_norm_answer: str,
    clean_prob: float,
    clean_entropy: float,
    clean_margin: float,
    swapped_question: Optional[str] = None,
    swapped_gold_answer: Optional[str] = None,
    annotated_relation: Optional[str] = None,
    gold_answer: Optional[str] = None,
    score_method: str = "generation_logits",
    semantic_threshold: float = 0.82,
    embedding_fn: Optional[Callable[[str, str], float]] = None,
) -> ProbeObservation:
    """Run the relation swap probe."""
    if swapped_question is not None:
        swap_text = swapped_question
        applicable = True
        reason = None
    else:
        swap_result = swap_relation(
            question,
            annotated_relation=annotated_relation,
            gold_answer=gold_answer,
        )
        applicable = swap_result.applicable
        swap_text = swap_result.swapped_text
        reason = swap_result.reason

    if not applicable or not swap_text:
        return ProbeObservation(
            probe_id=ProbeAction.RELATION,
            raw_answer="",
            norm_answer="",
            flip=False,
            conf_shift=0.0,
            entropy_shift=0.0,
            margin_shift=0.0,
            exact_match=1.0,
            semantic_match=1.0,
            applicable=False,
            severity=None,
            latency_ms=0.0,
            valid=True,
            invalid_reason=reason,
            parse_status="not_applicable",
            score_method=score_method,
        )

    prompt_text = make_relation_prompt(swap_text, dataset)
    gen_output = adapter.generate(image, prompt_text)
    norm_answer = normalize_answer(gen_output.raw_answer, dataset)

    token_lps = gen_output.token_logprobs
    token_dists = gen_output.token_distributions

    if score_method == "teacher_forced" or not token_lps or not all(math.isfinite(lp) for lp in token_lps):
        score_out = adapter.score(image, prompt_text, norm_answer)
        token_lps = score_out.token_logprobs
        token_dists = score_out.token_distributions

    return _build_probe_observation(
        probe_id=ProbeAction.RELATION,
        raw_answer=gen_output.raw_answer,
        norm_answer=norm_answer,
        clean_norm_answer=clean_norm_answer,
        clean_answer_prob=clean_prob,
        clean_entropy=clean_entropy,
        clean_margin=clean_margin,
        dataset=dataset,
        applicable=True,
        severity=None,
        prompt_text=prompt_text,
        gen_config=adapter.get_generation_config_dict(),
        token_logprobs=token_lps,
        token_distributions=token_dists,
        latency_ms=gen_output.latency_ms,
        score_method=score_method,
        semantic_threshold=semantic_threshold,
        embedding_fn=embedding_fn,
    )


# ---------------------------------------------------------------------------
# Full probe runner
# ---------------------------------------------------------------------------

def run_all_probes(
    adapter: MLLMAdapter,
    image: Image.Image,
    question: str,
    dataset: str,
    clean_norm_answer: str,
    clean_answer_prob: float,
    clean_entropy: float,
    clean_margin: float,
    relation_applicable: bool = False,
    swapped_question: Optional[str] = None,
    swapped_gold_answer: Optional[str] = None,
    annotated_relation: Optional[str] = None,
    gold_answer: Optional[str] = None,
    instance_id: str = "",
    severities: Optional[Dict[str, float]] = None,
    global_seed: int = 42,
    score_method: str = "generation_logits",
    semantic_threshold: float = 0.82,
    embedding_fn: Optional[Callable[[str, str], float]] = None,
) -> Dict[ProbeAction, ProbeObservation]:
    """Run all applicable probes for one instance independently on the ORIGINAL input."""
    severities = severities or {}
    observations: Dict[ProbeAction, ProbeObservation] = {}

    standard_prompt = make_dataset_prompt(question, dataset)

    # 1. Blank image probe (Language prior bit b_L)
    obs_blank = _run_visual_probe(
        probe_id=ProbeAction.BLANK,
        adapter=adapter,
        image=image,
        prompt_text=standard_prompt,
        dataset=dataset,
        clean_norm_answer=clean_norm_answer,
        clean_prob=clean_answer_prob,
        clean_entropy=clean_entropy,
        clean_margin=clean_margin,
        severity=None,
        instance_id=instance_id,
        global_seed=global_seed,
        score_method=score_method,
        semantic_threshold=semantic_threshold,
        embedding_fn=embedding_fn,
    )
    observations[ProbeAction.BLANK] = obs_blank

    # 2. Visual perturbation probes (Visual bit b_V)
    for probe_id in VISUAL_PROBES:
        probe_name = probe_id.value
        severity = severities.get(probe_name)
        obs = _run_visual_probe(
            probe_id=probe_id,
            adapter=adapter,
            image=image,
            prompt_text=standard_prompt,
            dataset=dataset,
            clean_norm_answer=clean_norm_answer,
            clean_prob=clean_answer_prob,
            clean_entropy=clean_entropy,
            clean_margin=clean_margin,
            severity=severity,
            instance_id=instance_id,
            global_seed=global_seed,
            score_method=score_method,
            semantic_threshold=semantic_threshold,
            embedding_fn=embedding_fn,
        )
        observations[probe_id] = obs

    # 3. Grounding probe
    obs_ground = _run_grounding_probe(
        adapter=adapter,
        image=image,
        question=question,
        dataset=dataset,
        clean_norm_answer=clean_norm_answer,
        clean_prob=clean_answer_prob,
        clean_entropy=clean_entropy,
        clean_margin=clean_margin,
        score_method=score_method,
        semantic_threshold=semantic_threshold,
        embedding_fn=embedding_fn,
    )
    observations[ProbeAction.GROUNDING] = obs_ground

    # 4. Relation probe (if applicable)
    if relation_applicable:
        obs_rel = _run_relation_probe(
            adapter=adapter,
            image=image,
            question=question,
            dataset=dataset,
            clean_norm_answer=clean_norm_answer,
            clean_prob=clean_answer_prob,
            clean_entropy=clean_entropy,
            clean_margin=clean_margin,
            swapped_question=swapped_question,
            swapped_gold_answer=swapped_gold_answer,
            annotated_relation=annotated_relation,
            gold_answer=gold_answer,
            score_method=score_method,
            semantic_threshold=semantic_threshold,
            embedding_fn=embedding_fn,
        )
        observations[ProbeAction.RELATION] = obs_rel

    return observations
