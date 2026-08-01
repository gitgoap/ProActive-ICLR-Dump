"""
Teacher cache instance processing pipeline (Plan §14).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image

from proactive.features.clean_features import (
    compute_confidence,
    compute_entropy,
    compute_margin,
    compute_answer_logprob,
)
from proactive.features.evidence_state import ProbeAction
from proactive.features.normalization import normalize_answer
from proactive.features.semantic import SemanticMatcher, get_default_semantic_matcher
from proactive.models.base_adapter import MLLMAdapter
from proactive.probes.probe_runner import run_all_probes
from proactive.probes.relation_swap import (
    evaluate_relation_swap,
    RelationSwapStatus,
)
from proactive.prompts.templates import make_dataset_prompt
from proactive.teacher.label_computation import (
    compute_teacher_labels,
    InvalidMandatoryProbeError,
)
from proactive.utils.hashing import hash_generation_config, hash_prompt

def _load_image_safely(image_path: str | Path | Image.Image, dataset_name: str = "") -> Image.Image:
    """Load image from path with fallback to PROACTIVE_DATA_ROOT, cwd/data, and standard subdirectories."""
    if isinstance(image_path, Image.Image):
        return image_path

    p = Path(image_path)
    if p.exists():
        return Image.open(p).convert("RGB")

    # Collect possible data roots
    possible_roots = []
    env_root = os.environ.get("PROACTIVE_DATA_ROOT")
    if env_root:
        possible_roots.append(Path(env_root))

    possible_roots.extend([
        Path.cwd() / "data",
        Path.cwd() / "data" / "POPE",
        Path.cwd().parent / "data",
        Path.home() / "MMUQ" / "data",
        Path.home() / "ProActive" / "data",
        Path("/home/aman/MMUQ/data"),
        Path("/home/aman/ProActive/data"),
    ])

    # Check candidates under all known roots
    for root_p in possible_roots:
        if not root_p.exists():
            continue
        candidates = [
            root_p / p.name,
            root_p / "POPE" / "images" / p.name,
            root_p / "POPE" / p.name,
            root_p / "pope" / "images" / p.name,
            root_p / "coco" / "val2014" / p.name,
            root_p / "COCO" / "val2014" / p.name,
            root_p / "val2014" / p.name,
            root_p / "VSR" / "val2017" / p.name,
            root_p / "VSR" / "train2017" / p.name,
            root_p / "VSR" / "images" / p.name,
            root_p / "vsr" / "val2017" / p.name,
            root_p / "vsr" / "train2017" / p.name,
            root_p / "VizWiz" / "Images" / "val" / p.name,
            root_p / "VizWiz" / "Images" / "train" / p.name,
            root_p / "VizWiz" / "images" / "val" / p.name,
            root_p / "VizWiz" / "val" / p.name,
            root_p / "vizwiz" / "Images" / "val" / p.name,
            root_p / "vizwiz" / "val" / p.name,
            root_p / "HallusionBench" / "hallusion_bench" / "VD" / "illusion" / p.name,
            root_p / "HallusionBench" / "hallusion_bench" / "VS" / p.name,
            root_p / "HallusionBench" / "hallusion_bench" / p.name,
            root_p / "hallusion_bench" / "VD" / "illusion" / p.name,
            root_p / "hallusion_bench" / "VS" / p.name,
            root_p / "hallusion_bench" / p.name,
        ]
        for cand in candidates:
            if cand.exists():
                return Image.open(cand).convert("RGB")

    # Direct open if not resolved, allowing standard FileNotFoundError with clear path
    return Image.open(image_path).convert("RGB")


def process_instance(
    record: dict,
    adapter: MLLMAdapter,
    dataset_name: str,
    model_id: str,
    model_revision: str,
    severities: Optional[Dict[str, float]] = None,
    global_seed: int = 42,
    semantic_matcher: Optional[SemanticMatcher] = None,
    semantic_threshold: float = 0.82,
) -> dict:
    """Process one manifest record: clean inference + all probes + labels."""
    instance_id = record["instance_id"]
    image_path = record["image_path"]
    question = record["question"]
    gold_answer = record["gold_answer"]
    relation_applicable = record.get("relation_applicable", False)
    swapped_question = record.get("swapped_question")
    swapped_gold_answer = record.get("swapped_gold_answer")
    annotated_relation = record.get("annotated_relation", record.get("relation"))

    # Determine embedding function if semantic matcher provided or available
    embedding_fn: Optional[Callable[[str, str], float]] = None
    if semantic_matcher is not None and semantic_matcher.is_available:
        embedding_fn = semantic_matcher.similarity
    else:
        gm = get_default_semantic_matcher()
        if gm and gm.is_available:
            embedding_fn = gm.similarity

    # Load image safely
    img = _load_image_safely(image_path, dataset_name)

    # --- Clean inference ---
    prompt = make_dataset_prompt(question, dataset_name)
    clean_gen = adapter.generate(img, prompt)

    norm_answer = normalize_answer(clean_gen.raw_answer, dataset_name)
    norm_gold = normalize_answer(gold_answer, dataset_name)
    correct = (norm_answer == norm_gold)

    # Score method selection & validation
    score_method = "generation_logits"
    clean_token_lps = clean_gen.token_logprobs
    clean_token_dists = clean_gen.token_distributions

    # Check if generation logits are missing or non-finite
    if not clean_token_lps or not all(math.isfinite(lp) for lp in clean_token_lps):
        logger.debug(f"Missing/non-finite generation logits for {instance_id}, using teacher-forced scoring")
        score_method = "teacher_forced"
        score_out = adapter.score(img, prompt, norm_answer)
        clean_token_lps = score_out.token_logprobs
        clean_token_dists = score_out.token_distributions

    clean_valid = True
    clean_invalid_reason = None
    clean_prob = 0.0
    clean_logprob = float("-inf")
    clean_entropy = 0.0
    clean_margin = 0.0

    try:
        clean_prob = compute_confidence(clean_token_lps)
        clean_logprob = compute_answer_logprob(clean_token_lps)
        clean_entropy = compute_entropy(clean_token_dists or [])
        clean_margin = compute_margin(clean_token_dists or [])
    except Exception as e:
        clean_valid = False
        clean_invalid_reason = f"clean_score_error: {str(e)}"
        logger.debug(f"Clean feature score error for {instance_id}: {e}")

    # --- Run all probes ---
    probe_obs = run_all_probes(
        adapter=adapter,
        image=img,
        question=question,
        dataset=dataset_name,
        clean_norm_answer=norm_answer,
        clean_answer_prob=clean_prob,
        clean_entropy=clean_entropy,
        clean_margin=clean_margin,
        relation_applicable=relation_applicable,
        swapped_question=swapped_question,
        swapped_gold_answer=swapped_gold_answer,
        annotated_relation=annotated_relation,
        gold_answer=gold_answer,
        instance_id=instance_id,
        severities=severities,
        global_seed=global_seed,
        score_method=score_method,
        semantic_threshold=semantic_threshold,
        embedding_fn=embedding_fn,
    )

    # --- Evaluate relation swap outcome ---
    rel_status = RelationSwapStatus.NOT_APPLICABLE
    swap_inv = None
    if relation_applicable and ProbeAction.RELATION in probe_obs:
        rel_obs = probe_obs[ProbeAction.RELATION]
        if rel_obs.applicable and rel_obs.valid:
            if swapped_gold_answer is not None:
                exp_swap_gold = swapped_gold_answer
            else:
                exp_swap_gold = "false" if norm_gold in ("true", "yes", "1") else "true"

            rel_status = evaluate_relation_swap(
                original_answer_correct=correct,
                original_norm_answer=norm_answer,
                swapped_norm_answer=rel_obs.norm_answer,
                original_gold=norm_gold,
                swapped_gold=exp_swap_gold,
                dataset=dataset_name,
            )
            if rel_status == RelationSwapStatus.INVARIANT:
                swap_inv = True
            elif rel_status == RelationSwapStatus.CHANGED_CORRECTLY:
                swap_inv = False

    # --- Compute teacher labels (Fail-closed) ---
    valid_teacher = clean_valid
    teacher_invalid_reason = clean_invalid_reason
    labels = None

    if valid_teacher:
        try:
            labels = compute_teacher_labels(
                probe_observations=probe_obs,
                clean_answer_prob=clean_prob,
                clean_correct=correct,
                relation_applicable=relation_applicable,
                swap_invariance=swap_inv,
                benchmark_family=record.get("category", record.get("pope_split")),
                strict_validation=True,
            )
        except InvalidMandatoryProbeError as e:
            valid_teacher = False
            teacher_invalid_reason = str(e)
            logger.debug(f"Invalid mandatory probe for {instance_id}: {e}")

    # --- Serialize output ---
    probes_dict = {action.value: obs.to_dict() for action, obs in probe_obs.items()}

    output = {
        "instance_id": instance_id,
        "group_id": record.get("group_id", "default_group"),
        "dataset": dataset_name,
        "split": record.get("split", ""),
        "model_id": model_id,
        "model_revision": model_revision,
        "image_path": str(image_path),
        "prompt_text": prompt,
        "gold_answer": gold_answer,
        "relation_applicable": relation_applicable,
        "relation_outcome_status": rel_status.value,
        "score_method": score_method,
        "valid": valid_teacher,
        "invalid_reason": teacher_invalid_reason,
        "clean": {
            "raw_answer": clean_gen.raw_answer,
            "norm_answer": norm_answer,
            "correct": int(correct),
            "answer_logprob": clean_logprob,
            "answer_prob": clean_prob,
            "token_entropy_mean": clean_entropy,
            "token_margin_mean": clean_margin,
            "answer_len_tokens": clean_gen.answer_len_tokens,
            "latency_ms": clean_gen.latency_ms,
            "score_method": score_method,
            "valid": clean_valid,
        },
        "probes": probes_dict,
        "teacher_signature": {
            "V": labels.teacher_signature.V if labels else 0.0,
            "L": labels.teacher_signature.L if labels else 0.0,
            "A": labels.teacher_signature.A if labels else 0.0,
        },
        "teacher_bits": {
            "visual": int(labels.source_bits.visual) if labels else 0,
            "language": int(labels.source_bits.language) if labels else 0,
            "alignment": int(labels.source_bits.alignment) if labels else 0,
        },
        "teacher_label6": labels.six_way_state.value if labels else "unclear",
        "benchmark_family": labels.benchmark_family if (labels and labels.benchmark_family) else "",
        "swap_invariance": swap_inv,
        "prompt_hash": hash_prompt(prompt),
        "generation_config_hash": hash_generation_config(
            adapter.get_generation_config_dict()
        ),
    }
    return output
