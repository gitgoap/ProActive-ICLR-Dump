"""
Tests for evidence state masking and leakage prevention.

Verifies:
- EvidenceState does not expose unacquired probe data
- No dataset_id or model_id in the main feature vector
- Immutable state transitions
- Legal action masking
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.features.evidence_state import (
    CleanFeatures,
    EvidenceState,
    ProbeAction,
    ProbeObservation,
    SixWayState,
    SourceBits,
    TeacherLabels,
    NON_STOP_PROBES,
    PROBE_COSTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_clean_features(**kwargs) -> CleanFeatures:
    defaults = {
        "raw_answer": "yes",
        "norm_answer": "yes",
        "answer_prob": 0.85,
        "token_entropy_mean": 0.3,
        "token_margin_mean": 0.6,
        "answer_len_tokens": 1,
        "relation_available": False,
    }
    defaults.update(kwargs)
    return CleanFeatures(**defaults)


def _make_probe_obs(
    probe_id: ProbeAction = ProbeAction.BLUR,
    **kwargs,
) -> ProbeObservation:
    defaults = {
        "probe_id": probe_id,
        "raw_answer": "no",
        "norm_answer": "no",
        "flip": True,
        "conf_shift": -0.3,
        "entropy_shift": 0.2,
        "margin_shift": -0.1,
        "exact_match": 0.0,
        "semantic_match": 0.0,
        "applicable": True,
    }
    defaults.update(kwargs)
    return ProbeObservation(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvidenceStateMasking:

    def test_empty_state_has_no_probes(self):
        """A fresh state has no acquired probes."""
        state = EvidenceState(clean_features=_make_clean_features())
        assert len(state.acquired_probes) == 0
        assert state.num_acquired == 0

    def test_unacquired_probes_not_in_state(self):
        """Unacquired probes must not appear in acquired_probes."""
        state = EvidenceState(clean_features=_make_clean_features())
        obs = _make_probe_obs(ProbeAction.BLUR)
        state2 = state.acquire_probe(obs)

        for probe in NON_STOP_PROBES:
            if probe == ProbeAction.BLUR:
                assert probe in state2.acquired_probes
            else:
                assert probe not in state2.acquired_probes

    def test_validate_no_leakage_passes(self):
        """Validation passes on a properly constructed state."""
        state = EvidenceState(clean_features=_make_clean_features())
        assert state.validate_no_leakage() is True

    def test_no_forbidden_keys_in_features(self):
        """Feature dict must not contain dataset_id, model_id, etc."""
        cf = _make_clean_features()
        feature_dict = cf.to_feature_dict()
        forbidden = {"dataset_id", "model_id", "dataset", "correct"}
        leaked = set(feature_dict.keys()) & forbidden
        assert not leaked, f"Forbidden keys in features: {leaked}"


class TestImmutableTransitions:

    def test_acquire_returns_new_state(self):
        """acquire_probe returns a NEW state, not mutating the original."""
        state = EvidenceState(clean_features=_make_clean_features())
        obs = _make_probe_obs(ProbeAction.BLANK)
        state2 = state.acquire_probe(obs)

        assert ProbeAction.BLANK not in state.acquired_probes
        assert ProbeAction.BLANK in state2.acquired_probes
        assert state.remaining_budget == 7
        assert state2.remaining_budget == 6

    def test_cannot_acquire_same_probe_twice(self):
        """Acquiring the same probe twice raises ValueError."""
        state = EvidenceState(clean_features=_make_clean_features())
        obs = _make_probe_obs(ProbeAction.BLUR)
        state2 = state.acquire_probe(obs)

        with pytest.raises(ValueError, match="already acquired"):
            state2.acquire_probe(obs)

    def test_cannot_acquire_stop(self):
        """STOP cannot be acquired as a probe observation."""
        state = EvidenceState(clean_features=_make_clean_features())
        obs = _make_probe_obs(ProbeAction.STOP)

        with pytest.raises(ValueError, match="STOP"):
            state.acquire_probe(obs)


class TestLegalActions:

    def test_empty_state_all_probes_legal(self):
        """All applicable probes are legal from an empty state."""
        cf = _make_clean_features(relation_available=False)
        state = EvidenceState(clean_features=cf)
        legal = state.legal_actions
        # 6 non-stop probes (relation not available) + STOP = 7
        assert ProbeAction.STOP in legal
        assert ProbeAction.RELATION not in legal
        assert len(legal) == 7  # 6 probes + STOP

    def test_relation_legal_when_available(self):
        """Relation is legal when relation_available=True."""
        cf = _make_clean_features(relation_available=True)
        state = EvidenceState(clean_features=cf)
        assert ProbeAction.RELATION in state.legal_actions

    def test_acquired_probe_not_legal(self):
        """An acquired probe is removed from legal actions."""
        state = EvidenceState(clean_features=_make_clean_features())
        obs = _make_probe_obs(ProbeAction.BLUR)
        state2 = state.acquire_probe(obs)
        assert ProbeAction.BLUR not in state2.legal_actions

    def test_budget_exhaustion(self):
        """When budget is 0, only STOP is legal."""
        state = EvidenceState(
            clean_features=_make_clean_features(),
            remaining_budget=0,
        )
        assert state.legal_actions == {ProbeAction.STOP}

    def test_budget_limits_actions(self):
        """Cannot acquire a probe that costs more than remaining budget."""
        state = EvidenceState(
            clean_features=_make_clean_features(),
            remaining_budget=0,
        )
        obs = _make_probe_obs(ProbeAction.NOISE)
        with pytest.raises(ValueError, match="Insufficient budget"):
            state.acquire_probe(obs)


class TestSixWayStateComputation:

    def test_no_failure(self):
        bits = SourceBits(visual=False, language=False, alignment=False)
        assert TeacherLabels.compute_six_way(bits, True) == SixWayState.NO_FAILURE

    def test_visual_only(self):
        bits = SourceBits(visual=True, language=False, alignment=False)
        assert TeacherLabels.compute_six_way(bits, True) == SixWayState.VISUAL

    def test_language_only(self):
        bits = SourceBits(visual=False, language=True, alignment=False)
        assert TeacherLabels.compute_six_way(bits, True) == SixWayState.LANGUAGE_PRIOR

    def test_alignment_only(self):
        bits = SourceBits(visual=False, language=False, alignment=True)
        assert TeacherLabels.compute_six_way(bits, True) == SixWayState.ALIGNMENT

    def test_mixed(self):
        bits = SourceBits(visual=True, language=True, alignment=False)
        assert TeacherLabels.compute_six_way(bits, True) == SixWayState.MIXED

    def test_unclear_incorrect_no_bits(self):
        bits = SourceBits(visual=False, language=False, alignment=False)
        assert TeacherLabels.compute_six_way(bits, False) == SixWayState.UNCLEAR
