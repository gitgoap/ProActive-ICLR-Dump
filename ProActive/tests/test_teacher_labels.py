"""
Tests for teacher label computation.

Validates:
- Mandatory probe presence validation
- Source-bit threshold logic (b_V, b_L, b_A)
- Six-way state derivation
- Boundary/edge cases at threshold values
- Continuous teacher signature computation
- Integration with TeacherLabels
"""

import pytest

from proactive.features.evidence_state import (
    ProbeAction,
    ProbeObservation,
    SixWayState,
    SourceBits,
    TeacherLabels,
)
from proactive.teacher.label_computation import (
    compute_visual_bit,
    compute_language_bit,
    compute_alignment_bit,
    compute_teacher_labels,
    compute_teacher_signature,
    validate_mandatory_probes,
    InvalidMandatoryProbeError,
    LabelThresholds,
    DEFAULT_THRESHOLDS,
)


def _make_obs(
    probe_id: ProbeAction,
    flip: bool = False,
    conf_shift: float = 0.0,
    semantic_match: float = 0.0,
    applicable: bool = True,
    valid: bool = True,
) -> ProbeObservation:
    """Shorthand to create a probe observation for testing."""
    return ProbeObservation(
        probe_id=probe_id,
        raw_answer="test",
        norm_answer="test",
        flip=flip,
        conf_shift=conf_shift,
        entropy_shift=0.0,
        margin_shift=0.0,
        exact_match=1.0 if not flip else 0.0,
        semantic_match=semantic_match,
        applicable=applicable,
        valid=valid,
    )


class TestMandatoryValidation:
    def test_missing_probe_raises_error(self):
        obs = {
            ProbeAction.BLANK: _make_obs(ProbeAction.BLANK),
            ProbeAction.BLUR: _make_obs(ProbeAction.BLUR),
            # Missing CROP, BRIGHTNESS, NOISE, GROUNDING
        }
        with pytest.raises(InvalidMandatoryProbeError, match="Missing mandatory probe observation"):
            validate_mandatory_probes(obs)

    def test_missing_relation_when_applicable(self):
        obs = {
            ProbeAction.BLANK: _make_obs(ProbeAction.BLANK),
            ProbeAction.BLUR: _make_obs(ProbeAction.BLUR),
            ProbeAction.CROP: _make_obs(ProbeAction.CROP),
            ProbeAction.BRIGHTNESS: _make_obs(ProbeAction.BRIGHTNESS),
            ProbeAction.NOISE: _make_obs(ProbeAction.NOISE),
            ProbeAction.GROUNDING: _make_obs(ProbeAction.GROUNDING),
        }
        with pytest.raises(InvalidMandatoryProbeError, match="Missing relation probe observation"):
            validate_mandatory_probes(obs, relation_applicable=True)


class TestVisualBit:
    def test_no_visual_probes(self):
        assert compute_visual_bit({}) is False

    def test_all_stable_no_trigger(self):
        obs = {
            ProbeAction.BLUR: _make_obs(ProbeAction.BLUR, flip=False, conf_shift=0.05),
            ProbeAction.CROP: _make_obs(ProbeAction.CROP, flip=False, conf_shift=0.05),
            ProbeAction.BRIGHTNESS: _make_obs(ProbeAction.BRIGHTNESS, flip=False, conf_shift=0.05),
            ProbeAction.NOISE: _make_obs(ProbeAction.NOISE, flip=False, conf_shift=0.05),
        }
        assert compute_visual_bit(obs) is False

    def test_high_flip_rate_triggers(self):
        obs = {
            ProbeAction.BLUR: _make_obs(ProbeAction.BLUR, flip=True),
            ProbeAction.CROP: _make_obs(ProbeAction.CROP, flip=False),
            ProbeAction.BRIGHTNESS: _make_obs(ProbeAction.BRIGHTNESS, flip=False),
            ProbeAction.NOISE: _make_obs(ProbeAction.NOISE, flip=False),
        }
        assert compute_visual_bit(obs) is True

    def test_high_conf_shift_triggers(self):
        obs = {
            ProbeAction.BLUR: _make_obs(ProbeAction.BLUR, conf_shift=-0.30),
            ProbeAction.CROP: _make_obs(ProbeAction.CROP, conf_shift=0.10),
            ProbeAction.BRIGHTNESS: _make_obs(ProbeAction.BRIGHTNESS, conf_shift=0.10),
            ProbeAction.NOISE: _make_obs(ProbeAction.NOISE, conf_shift=-0.10),
        }
        assert compute_visual_bit(obs) is True


class TestLanguageBit:
    def test_blank_high_conf_ratio_triggers(self):
        obs = {
            ProbeAction.BLANK: _make_obs(ProbeAction.BLANK, conf_shift=-0.05, semantic_match=1.0)
        }
        # clean_prob = 0.8, blank_conf = 0.75 -> ratio = 0.75/0.8 = 0.9375 >= 0.80
        assert compute_language_bit(obs, clean_answer_prob=0.8) is True

    def test_blank_low_conf_ratio_no_trigger(self):
        obs = {
            ProbeAction.BLANK: _make_obs(ProbeAction.BLANK, conf_shift=-0.50, semantic_match=1.0)
        }
        # clean_prob = 0.8, blank_conf = 0.30 -> ratio = 0.30/0.8 = 0.375 < 0.80
        assert compute_language_bit(obs, clean_answer_prob=0.8) is False

    def test_blank_no_semantic_match_no_trigger(self):
        obs = {
            ProbeAction.BLANK: _make_obs(ProbeAction.BLANK, conf_shift=0.0, semantic_match=0.0)
        }
        assert compute_language_bit(obs, clean_answer_prob=0.8) is False


class TestAlignmentBit:
    def test_grounding_flip_triggers(self):
        obs = {ProbeAction.GROUNDING: _make_obs(ProbeAction.GROUNDING, flip=True)}
        assert compute_alignment_bit(obs) is True

    def test_grounding_invalid_does_not_trigger(self):
        obs = {ProbeAction.GROUNDING: _make_obs(ProbeAction.GROUNDING, flip=True, valid=False)}
        assert compute_alignment_bit(obs) is False

    def test_relation_swap_invariance_triggers(self):
        obs = {ProbeAction.GROUNDING: _make_obs(ProbeAction.GROUNDING, flip=False)}
        assert compute_alignment_bit(obs, swap_invariance=True, relation_applicable=True) is True

    def test_relation_swap_invalid_does_not_trigger(self):
        obs = {ProbeAction.GROUNDING: _make_obs(ProbeAction.GROUNDING, flip=False)}
        assert compute_alignment_bit(obs, swap_invariance=None, relation_applicable=True) is False


class TestFullLabelComputation:
    def test_stable_instance(self):
        obs = {
            ProbeAction.BLANK: _make_obs(ProbeAction.BLANK, flip=False, conf_shift=-0.02, semantic_match=0.0),
            ProbeAction.BLUR: _make_obs(ProbeAction.BLUR, flip=False, conf_shift=0.01),
            ProbeAction.CROP: _make_obs(ProbeAction.CROP, flip=False, conf_shift=0.01),
            ProbeAction.BRIGHTNESS: _make_obs(ProbeAction.BRIGHTNESS, flip=False, conf_shift=0.01),
            ProbeAction.NOISE: _make_obs(ProbeAction.NOISE, flip=False, conf_shift=0.01),
            ProbeAction.GROUNDING: _make_obs(ProbeAction.GROUNDING, flip=False, conf_shift=0.01),
        }
        labels = compute_teacher_labels(
            probe_observations=obs,
            clean_answer_prob=0.9,
            clean_correct=True,
        )
        assert labels.six_way_state == SixWayState.NO_FAILURE
        assert labels.source_bits.visual is False
        assert labels.source_bits.language is False
        assert labels.source_bits.alignment is False

    def test_teacher_signature_populated(self):
        obs = {
            ProbeAction.BLANK: _make_obs(ProbeAction.BLANK, conf_shift=-0.1, semantic_match=1.0),
            ProbeAction.BLUR: _make_obs(ProbeAction.BLUR, conf_shift=-0.2),
            ProbeAction.CROP: _make_obs(ProbeAction.CROP, conf_shift=-0.1),
            ProbeAction.BRIGHTNESS: _make_obs(ProbeAction.BRIGHTNESS, conf_shift=-0.1),
            ProbeAction.NOISE: _make_obs(ProbeAction.NOISE, conf_shift=-0.1),
            ProbeAction.GROUNDING: _make_obs(ProbeAction.GROUNDING, conf_shift=-0.1),
        }
        labels = compute_teacher_labels(
            probe_observations=obs,
            clean_answer_prob=0.8,
            clean_correct=False,
        )
        sig = labels.teacher_signature
        assert isinstance(sig.V, float) and sig.V >= 0
        assert isinstance(sig.L, float) and sig.L >= 0
        assert isinstance(sig.A, float) and sig.A >= 0
