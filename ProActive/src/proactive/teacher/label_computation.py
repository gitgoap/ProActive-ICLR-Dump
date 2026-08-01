"""
Teacher label computation from full-probe cache.

Computes the three source bits (b_V, b_L, b_A), the continuous
teacher signature, and the six-way reporting label from the
complete set of probe observations. (Plan §6.1–6.4)

Threshold defaults match Plan §6.2. They must be frozen by end
of Week 3 using pilot sanity checks on train/val data, never test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from proactive.features.evidence_state import (
    ProbeAction,
    ProbeObservation,
    SixWayState,
    SourceBits,
    TeacherLabels,
    TeacherSignature,
)
from proactive.probes.relation_swap import RelationSwapStatus


class InvalidMandatoryProbeError(ValueError):
    """Raised when a mandatory probe is missing or marked invalid."""
    pass


# Visual perturbation probes for b_V (Plan §6.2: excludes blank)
PERTURBATION_VISUAL_PROBES = [
    ProbeAction.BLUR,
    ProbeAction.CROP,
    ProbeAction.BRIGHTNESS,
    ProbeAction.NOISE,
]

# Mandatory baseline probes for any instance
MANDATORY_PROBES = [
    ProbeAction.BLANK,
    ProbeAction.BLUR,
    ProbeAction.CROP,
    ProbeAction.BRIGHTNESS,
    ProbeAction.NOISE,
    ProbeAction.GROUNDING,
]


# ---------------------------------------------------------------------------
# Default thresholds  (Plan §6.2)
# ---------------------------------------------------------------------------

@dataclass
class LabelThresholds:
    """Thresholds for deriving source bits from probe observations (Plan §6.2)."""
    # b_V: visual fragility
    visual_flip_threshold: float = 0.25   # mean flip rate across perturbation probes
    visual_conf_threshold: float = 0.15   # mean |delta_c| across perturbation probes

    # b_L: language-prior persistence
    blank_semantic_match_required: bool = True   # blank answer semantically matches clean
    blank_conf_ratio_threshold: float = 0.80     # c_blank / (c_clean + eps)

    # b_A: alignment / relation instability
    grounding_flip_triggers: bool = True         # valid flip on grounding -> b_A
    grounding_conf_threshold: float = 0.15       # |delta_c| on valid grounding -> b_A

    eps: float = 1e-8


DEFAULT_THRESHOLDS = LabelThresholds()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_mandatory_probes(
    probe_observations: Dict[ProbeAction, ProbeObservation],
    relation_applicable: bool = False,
) -> None:
    """Verify that all mandatory probe observations are present and valid (Fail-closed)."""
    for p in MANDATORY_PROBES:
        if p not in probe_observations:
            raise InvalidMandatoryProbeError(f"Missing mandatory probe observation: {p.value}")
        if not probe_observations[p].valid:
            reason = probe_observations[p].invalid_reason or "unknown_error"
            raise InvalidMandatoryProbeError(
                f"Mandatory probe observation '{p.value}' is invalid ({reason})"
            )

    if relation_applicable:
        if ProbeAction.RELATION not in probe_observations:
            raise InvalidMandatoryProbeError("Missing relation probe observation for relation-applicable instance")
        if not probe_observations[ProbeAction.RELATION].valid:
            reason = probe_observations[ProbeAction.RELATION].invalid_reason or "unknown_error"
            raise InvalidMandatoryProbeError(f"Relation probe is marked invalid ({reason})")


# ---------------------------------------------------------------------------
# Source bit computation  (Plan §6.2)
# ---------------------------------------------------------------------------

def compute_visual_bit(
    probe_observations: Dict[ProbeAction, ProbeObservation],
    thresholds: LabelThresholds = DEFAULT_THRESHOLDS,
) -> bool:
    """Compute b_V: visual fragility bit across perturbation probes {blur, crop, brightness, noise}."""
    visual_obs = [
        probe_observations[p]
        for p in PERTURBATION_VISUAL_PROBES
        if p in probe_observations and probe_observations[p].applicable and probe_observations[p].valid
    ]

    if not visual_obs:
        return False

    mean_flip = sum(float(o.flip) for o in visual_obs) / len(visual_obs)
    mean_conf_shift = sum(abs(o.conf_shift) for o in visual_obs) / len(visual_obs)

    return (
        mean_flip >= thresholds.visual_flip_threshold
        or mean_conf_shift >= thresholds.visual_conf_threshold
    )


def compute_language_bit(
    probe_observations: Dict[ProbeAction, ProbeObservation],
    clean_answer_prob: float,
    thresholds: LabelThresholds = DEFAULT_THRESHOLDS,
) -> bool:
    """Compute b_L: language-prior persistence bit from blank probe."""
    blank_obs = probe_observations.get(ProbeAction.BLANK)
    if blank_obs is None or not blank_obs.applicable or not blank_obs.valid:
        return False

    # Semantic match requirement
    if thresholds.blank_semantic_match_required and blank_obs.semantic_match != 1.0:
        return False

    blank_conf = clean_answer_prob + blank_obs.conf_shift
    conf_ratio = blank_conf / (clean_answer_prob + thresholds.eps)

    return conf_ratio >= thresholds.blank_conf_ratio_threshold


def compute_alignment_bit(
    probe_observations: Dict[ProbeAction, ProbeObservation],
    swap_invariance: Optional[bool] = None,
    relation_applicable: bool = False,
    thresholds: LabelThresholds = DEFAULT_THRESHOLDS,
) -> bool:
    """Compute b_A: alignment / relation instability bit."""
    grounding_obs = probe_observations.get(ProbeAction.GROUNDING)

    grounding_triggered = False
    if grounding_obs is not None and grounding_obs.applicable and grounding_obs.valid:
        if thresholds.grounding_flip_triggers and grounding_obs.flip:
            grounding_triggered = True
        if abs(grounding_obs.conf_shift) >= thresholds.grounding_conf_threshold:
            grounding_triggered = True

    # Relation triggers b_A strictly when swap_invariance is True
    relation_triggered = bool(
        relation_applicable
        and swap_invariance is True
    )

    return grounding_triggered or relation_triggered


# ---------------------------------------------------------------------------
# Continuous teacher signature  (Plan §6.4)
# ---------------------------------------------------------------------------

def compute_teacher_signature(
    probe_observations: Dict[ProbeAction, ProbeObservation],
    clean_answer_prob: float,
    relation_applicable: bool = False,
    swap_invariance: Optional[bool] = None,
) -> TeacherSignature:
    """Compute continuous teacher signature (V, L, A) in R^3."""
    visual_obs = [
        probe_observations[p]
        for p in PERTURBATION_VISUAL_PROBES
        if p in probe_observations and probe_observations[p].applicable and probe_observations[p].valid
    ]
    v_score = 0.0
    if visual_obs:
        v_score = sum(abs(o.conf_shift) for o in visual_obs) / len(visual_obs)

    l_score = 0.0
    blank_obs = probe_observations.get(ProbeAction.BLANK)
    if blank_obs is not None and blank_obs.applicable and blank_obs.valid:
        blank_conf = clean_answer_prob + blank_obs.conf_shift
        l_score = (blank_conf / (clean_answer_prob + 1e-8)) * blank_obs.semantic_match

    a_score = 0.0
    grounding_obs = probe_observations.get(ProbeAction.GROUNDING)
    if grounding_obs is not None and grounding_obs.applicable and grounding_obs.valid:
        a_score = max(a_score, abs(grounding_obs.conf_shift))
        if grounding_obs.flip:
            a_score = max(a_score, 0.5)

    if relation_applicable and swap_invariance is True:
        a_score = max(a_score, 1.0)

    return TeacherSignature(V=v_score, L=l_score, A=a_score)


# ---------------------------------------------------------------------------
# Full teacher label computation
# ---------------------------------------------------------------------------

def compute_teacher_labels(
    probe_observations: Dict[ProbeAction, ProbeObservation],
    clean_answer_prob: float,
    clean_correct: bool,
    relation_applicable: bool = False,
    swap_invariance: Optional[bool] = None,
    benchmark_family: Optional[str] = None,
    thresholds: LabelThresholds = DEFAULT_THRESHOLDS,
    strict_validation: bool = True,
) -> TeacherLabels:
    """Compute complete teacher labels from full probe observations."""
    if strict_validation:
        validate_mandatory_probes(probe_observations, relation_applicable)

    b_v = compute_visual_bit(probe_observations, thresholds)
    b_l = compute_language_bit(probe_observations, clean_answer_prob, thresholds)
    b_a = compute_alignment_bit(
        probe_observations, swap_invariance, relation_applicable, thresholds
    )

    source_bits = SourceBits(visual=b_v, language=b_l, alignment=b_a)
    six_way = TeacherLabels.compute_six_way(source_bits, clean_correct)
    signature = compute_teacher_signature(
        probe_observations, clean_answer_prob, relation_applicable, swap_invariance
    )

    return TeacherLabels(
        source_bits=source_bits,
        six_way_state=six_way,
        teacher_signature=signature,
        benchmark_family=benchmark_family,
    )
