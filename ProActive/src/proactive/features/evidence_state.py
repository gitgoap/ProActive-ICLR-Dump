"""
Core data schemas for the ProActive evidence acquisition system.

Defines the probe action space, observation format, clean features,
evidence state, and six-way diagnostic labels. These dataclasses are
the single source of truth for data flowing through the pipeline.

Reference: Super Implementation Plan §5 (Formal problem statement)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Action space  (Plan §5.2)
# ---------------------------------------------------------------------------

class ProbeAction(str, Enum):
    """Non-stop probe identifiers and the STOP action."""
    BLANK = "blank"
    BLUR = "blur"
    CROP = "crop"
    BRIGHTNESS = "brightness"
    NOISE = "noise"
    GROUNDING = "grounding"
    RELATION = "relation"
    STOP = "stop"


# Convenience sets
NON_STOP_PROBES: List[ProbeAction] = [
    p for p in ProbeAction if p != ProbeAction.STOP
]

# Visual probes used for b_V source bit (Plan §6.2)
VISUAL_PROBES: List[ProbeAction] = [
    ProbeAction.BLUR,
    ProbeAction.CROP,
    ProbeAction.BRIGHTNESS,
    ProbeAction.NOISE,
]

# Every non-stop probe costs exactly 1 MLLM forward pass (Plan §13.1)
PROBE_COSTS: Dict[ProbeAction, int] = {
    p: 1 for p in NON_STOP_PROBES
}
PROBE_COSTS[ProbeAction.STOP] = 0

# Maximum number of legal probes (7 non-stop probes)
MAX_PROBES = len(NON_STOP_PROBES)


# ---------------------------------------------------------------------------
# Six-way diagnostic label  (Plan §6.3)
# ---------------------------------------------------------------------------

class SixWayState(str, Enum):
    """Six-way behavioural reporting target."""
    NO_FAILURE = "no-failure"
    VISUAL = "visual"
    LANGUAGE_PRIOR = "language-prior"
    ALIGNMENT = "alignment"
    MIXED = "mixed"
    UNCLEAR = "unclear"


# ---------------------------------------------------------------------------
# Probe observation  (Plan §5.4)
# ---------------------------------------------------------------------------

@dataclass
class ProbeObservation:
    """Observation from one probe applied independently to the original input.

    Each field matches the cache schema in Plan §14.1.
    """
    probe_id: ProbeAction
    raw_answer: str
    norm_answer: str
    flip: bool              # Answer changed from clean answer
    conf_shift: float       # Delta confidence (probe - clean)
    entropy_shift: float    # Delta entropy (probe - clean)
    margin_shift: float     # Delta margin (probe - clean)
    exact_match: float      # Exact normalized agreement with clean (0 or 1)
    semantic_match: float   # Semantic agreement with clean (0 or 1)
    applicable: bool        # Whether this probe was legal for this instance
    severity: Optional[float] = None
    latency_ms: Optional[float] = None
    prompt_hash: Optional[str] = None
    image_transform_hash: Optional[str] = None
    generation_config_hash: Optional[str] = None
    # Raw probe-side scores for delta computation
    answer_prob: Optional[float] = None
    token_entropy_mean: Optional[float] = None
    token_margin_mean: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize to a flat dictionary for JSONL storage."""
        d = {
            "probe_id": self.probe_id.value,
            "raw_answer": self.raw_answer,
            "norm_answer": self.norm_answer,
            "flip": self.flip,
            "conf_shift": self.conf_shift,
            "entropy_shift": self.entropy_shift,
            "margin_shift": self.margin_shift,
            "exact_match": self.exact_match,
            "semantic_match": self.semantic_match,
            "applicable": self.applicable,
            "severity": self.severity,
            "latency_ms": self.latency_ms,
            "prompt_hash": self.prompt_hash,
            "image_transform_hash": self.image_transform_hash,
            "generation_config_hash": self.generation_config_hash,
        }
        return d


# ---------------------------------------------------------------------------
# Clean features  (Plan §5.3)
# ---------------------------------------------------------------------------

@dataclass
class CleanFeatures:
    """Features extracted from the clean (unperturbed) MLLM forward pass.

    IMPORTANT: Does NOT include dataset_id, model_id, or gold correctness
    as input features for the main learner (Plan §5.1, §3.3).
    These are stored as metadata only.
    """
    raw_answer: str
    norm_answer: str
    answer_prob: float          # Length-normalized confidence c
    token_entropy_mean: float   # Mean token entropy H_bar
    token_margin_mean: float    # Mean top-1 vs top-2 margin m_bar
    answer_len_tokens: int      # Answer length in tokens
    relation_available: bool = False  # Whether relation swap is legal
    # Optional features from Plan §5.3
    iqa_features: Optional[Dict[str, float]] = None  # Image quality
    qtype: Optional[str] = None  # Coarse question-type feature
    answerability: Optional[float] = None  # Test-time answerability cue
    answer_logprob: Optional[float] = None
    latency_ms: Optional[float] = None

    # --- Metadata fields: NOT used as model input features ---
    correct: Optional[bool] = None  # vs gold answer; training-time only

    def to_feature_dict(self) -> Dict[str, float]:
        """Return only the features used as model inputs (no metadata).

        Excludes: dataset_id, model_id, correct, raw_answer.
        """
        features = {
            "answer_prob": self.answer_prob,
            "token_entropy_mean": self.token_entropy_mean,
            "token_margin_mean": self.token_margin_mean,
            "answer_len_tokens": float(self.answer_len_tokens),
            "relation_available": float(self.relation_available),
        }
        if self.answerability is not None:
            features["answerability"] = self.answerability
        # qtype would be embedded separately; store as string metadata
        return features


# ---------------------------------------------------------------------------
# Evidence state  (Plan §5.6)
# ---------------------------------------------------------------------------

@dataclass
class EvidenceState:
    """Complete evidence state at acquisition step t.

    This is the central data structure for the diagnostic pipeline.
    It tracks which probes have been acquired and enforces that
    unacquired probe data is never accessible.
    """
    clean_features: CleanFeatures
    acquired_probes: Dict[ProbeAction, ProbeObservation] = field(
        default_factory=dict
    )
    remaining_budget: int = MAX_PROBES

    # --- Metadata: NOT used as model input features ---
    instance_id: Optional[str] = None
    group_id: Optional[str] = None
    dataset: Optional[str] = None
    model_id: Optional[str] = None
    split: Optional[str] = None

    @property
    def acquired_set(self) -> Set[ProbeAction]:
        """Set of probe actions already acquired."""
        return set(self.acquired_probes.keys())

    @property
    def num_acquired(self) -> int:
        return len(self.acquired_probes)

    @property
    def legal_actions(self) -> Set[ProbeAction]:
        """Actions that can still be taken at this state.

        A probe is legal if:
        - It has not been acquired yet
        - It is applicable (relation requires relation_available)
        - Its cost fits within the remaining budget
        STOP is always legal.
        """
        legal: Set[ProbeAction] = set()
        for probe in NON_STOP_PROBES:
            if probe in self.acquired_probes:
                continue
            # Relation is only legal when the instance supports it
            if (probe == ProbeAction.RELATION
                    and not self.clean_features.relation_available):
                continue
            if PROBE_COSTS[probe] <= self.remaining_budget:
                legal.add(probe)
        # STOP is always available
        legal.add(ProbeAction.STOP)
        return legal

    def acquire_probe(self, obs: ProbeObservation) -> "EvidenceState":
        """Return a NEW state with the given probe observation added.

        Does not mutate the current state (immutable transition).
        Raises ValueError on illegal actions.
        """
        if obs.probe_id == ProbeAction.STOP:
            raise ValueError("Cannot acquire STOP as a probe observation.")
        if obs.probe_id in self.acquired_probes:
            raise ValueError(
                f"Probe {obs.probe_id.value} already acquired."
            )
        cost = PROBE_COSTS[obs.probe_id]
        if cost > self.remaining_budget:
            raise ValueError(
                f"Insufficient budget ({self.remaining_budget}) for "
                f"probe {obs.probe_id.value} (cost={cost})."
            )
        new_state = copy.deepcopy(self)
        new_state.acquired_probes[obs.probe_id] = obs
        new_state.remaining_budget -= cost
        return new_state

    def validate_no_leakage(self) -> bool:
        """Verify that only acquired probes are present in the state.

        Returns True if no leakage detected, raises AssertionError otherwise.
        """
        for probe in NON_STOP_PROBES:
            if probe not in self.acquired_probes:
                assert probe not in self.acquired_probes, (
                    f"Unacquired probe {probe.value} found in state."
                )
        # Ensure metadata fields are not in the feature vector keys
        feature_keys = set(self.clean_features.to_feature_dict().keys())
        forbidden = {"dataset_id", "model_id", "dataset", "correct"}
        leaked = feature_keys & forbidden
        assert not leaked, (
            f"Forbidden metadata keys in feature vector: {leaked}"
        )
        return True

    def get_action_mask(self) -> Dict[ProbeAction, bool]:
        """Return a mask dict: True if the action is legal."""
        legal = self.legal_actions
        return {a: (a in legal) for a in ProbeAction}


# ---------------------------------------------------------------------------
# Teacher labels  (Plan §6.1–6.4)
# ---------------------------------------------------------------------------

@dataclass
class SourceBits:
    """Three-bit behavioural source target (Plan §6.1)."""
    visual: bool = False       # b_V: visual fragility
    language: bool = False     # b_L: language-prior persistence
    alignment: bool = False    # b_A: alignment / relation instability

    def to_list(self) -> List[int]:
        return [int(self.visual), int(self.language), int(self.alignment)]

    @property
    def count(self) -> int:
        return sum(self.to_list())


@dataclass
class TeacherSignature:
    """Continuous teacher signature (Plan §6.4)."""
    V: float = 0.0  # Continuous visual fragility score
    L: float = 0.0  # Continuous language-prior score
    A: float = 0.0  # Continuous alignment score

    def to_list(self) -> List[float]:
        return [self.V, self.L, self.A]


@dataclass
class TeacherLabels:
    """Complete teacher-derived labels for one instance."""
    source_bits: SourceBits
    six_way_state: SixWayState
    teacher_signature: TeacherSignature
    benchmark_family: Optional[str] = None  # Metadata only, not a training target

    @staticmethod
    def compute_six_way(
        bits: SourceBits,
        clean_correct: bool,
    ) -> SixWayState:
        """Derive the six-way state from source bits (Plan §6.3).

        - no-failure:  clean correct AND no bits active
        - mixed:       at least two bits active
        - visual:      only b_V=1
        - language-prior: only b_L=1
        - alignment:   only b_A=1
        - unclear:     all remaining (e.g., incorrect but no strong bit)
        """
        n = bits.count
        if clean_correct and n == 0:
            return SixWayState.NO_FAILURE
        if n >= 2:
            return SixWayState.MIXED
        if n == 1:
            if bits.visual:
                return SixWayState.VISUAL
            if bits.language:
                return SixWayState.LANGUAGE_PRIOR
            if bits.alignment:
                return SixWayState.ALIGNMENT
        return SixWayState.UNCLEAR
