"""
Tests for confound audits, Section 28 triggers, and teacher schema validator (Plan §3.3, §14.1, §28).
"""

import json
import tempfile
from pathlib import Path
import pytest

from proactive.audits.confound_audit import (
    audit_dataset_and_model_confounds,
    check_section_28_trigger,
)
from proactive.audits.schema_validator import validate_record, validate_file
from proactive.features.evidence_state import CleanFeatures


class TestCleanFeaturesActionMask:
    def test_default_excludes_relation_available(self):
        cf = CleanFeatures(
            raw_answer="yes",
            norm_answer="yes",
            answer_prob=0.95,
            token_entropy_mean=0.2,
            token_margin_mean=0.8,
            answer_len_tokens=1,
            relation_available=True,
        )
        f_dict = cf.to_feature_dict()
        assert "relation_available" not in f_dict

    def test_explicit_ablation_includes_relation_available(self):
        cf = CleanFeatures(
            raw_answer="yes",
            norm_answer="yes",
            answer_prob=0.95,
            token_entropy_mean=0.2,
            token_margin_mean=0.8,
            answer_len_tokens=1,
            relation_available=True,
        )
        f_dict = cf.to_feature_dict(include_relation_available=True)
        assert f_dict["relation_available"] == 1.0


class TestConfoundAudits:
    def test_audit_rates(self):
        records = [
            {
                "dataset": "pope",
                "model_id": "llava",
                "relation_applicable": False,
                "teacher_bits": {"visual": 1, "language": 0, "alignment": 0},
                "teacher_label6": "visual",
            },
            {
                "dataset": "vsr",
                "model_id": "llava",
                "relation_applicable": True,
                "teacher_bits": {"visual": 0, "language": 0, "alignment": 1},
                "teacher_label6": "alignment",
            },
        ]
        audit = audit_dataset_and_model_confounds(records)
        assert audit["total_records"] == 2
        assert audit["by_dataset"]["pope"]["relation_applicable_rate"] == 0.0
        assert audit["by_dataset"]["vsr"]["relation_applicable_rate"] == 1.0
        assert audit["by_model"]["llava"]["visual_bit_rate"] == 0.5

    def test_section_28_trigger(self):
        # Gap <= 2.0 -> trigger fires
        res_fired = check_section_28_trigger(full_system_metric=75.0, id_control_metric=73.5, threshold=2.0)
        assert res_fired["triggered"] is True
        assert res_fired["action_required"] == "REMOVE_SHORTCUT_FEATURES_AND_NARROW_CLAIM"

        # Gap > 2.0 -> trigger does not fire
        res_pass = check_section_28_trigger(full_system_metric=75.0, id_control_metric=70.0, threshold=2.0)
        assert res_pass["triggered"] is False


class TestSchemaValidation:
    def test_valid_record(self):
        rec = {
            "instance_id": "pope_001",
            "group_id": "group_1",
            "dataset": "pope",
            "split": "train",
            "model_id": "llava_1.5_7b",
            "model_revision": "main",
            "image_path": "data/sample.jpg",
            "prompt_text": "Is there a cat?",
            "gold_answer": "yes",
            "score_method": "generation_logits",
            "clean": {
                "raw_answer": "yes",
                "norm_answer": "yes",
                "correct": 1,
                "answer_prob": 0.95,
                "token_entropy_mean": 0.1,
                "token_margin_mean": 0.8,
            },
            "probes": {
                "blank": {"conf_shift": -0.5, "valid": True},
                "blur": {"conf_shift": -0.1, "valid": True},
                "crop": {"conf_shift": -0.1, "valid": True},
                "brightness": {"conf_shift": -0.1, "valid": True},
                "noise": {"conf_shift": -0.1, "valid": True},
                "grounding": {"conf_shift": -0.1, "valid": True},
            },
            "teacher_signature": {"V": 0.1, "L": 0.5, "A": 0.1},
            "teacher_bits": {"visual": 0, "language": 1, "alignment": 0},
            "teacher_label6": "language-prior",
            "prompt_hash": "abc123",
            "generation_config_hash": "def456",
        }
        errs = validate_record(rec, 0)
        assert len(errs) == 0, f"Errors found: {errs}"

    def test_missing_mandatory_probe_in_schema(self):
        rec = {
            "instance_id": "pope_001",
            "group_id": "group_1",
            "dataset": "pope",
            "split": "train",
            "model_id": "llava_1.5_7b",
            "model_revision": "main",
            "image_path": "data/sample.jpg",
            "prompt_text": "Is there a cat?",
            "gold_answer": "yes",
            "score_method": "generation_logits",
            "clean": {
                "raw_answer": "yes",
                "norm_answer": "yes",
                "correct": 1,
                "answer_prob": 0.95,
                "token_entropy_mean": 0.1,
                "token_margin_mean": 0.8,
            },
            "probes": {
                "blank": {"conf_shift": -0.5, "valid": True},
                # Missing blur, crop, etc.
            },
            "teacher_signature": {"V": 0.1, "L": 0.5, "A": 0.1},
            "teacher_bits": {"visual": 0, "language": 1, "alignment": 0},
            "teacher_label6": "language-prior",
            "prompt_hash": "abc123",
            "generation_config_hash": "def456",
        }
        errs = validate_record(rec, 0)
        assert any("Missing mandatory probe" in e for e in errs)

    def test_invalid_split_rejected(self):
        rec = {
            "instance_id": "pope_001",
            "group_id": "group_1",
            "dataset": "pope",
            "split": "test",  # Test split is rejected in teacher cache
            "model_id": "llava_1.5_7b",
            "model_revision": "main",
            "image_path": "data/sample.jpg",
            "prompt_text": "Is there a cat?",
            "gold_answer": "yes",
            "score_method": "generation_logits",
            "clean": {
                "raw_answer": "yes",
                "norm_answer": "yes",
                "correct": 1,
                "answer_prob": 0.95,
                "token_entropy_mean": 0.1,
                "token_margin_mean": 0.8,
            },
            "probes": {
                "blank": {"conf_shift": -0.5, "valid": True},
                "blur": {"conf_shift": -0.1, "valid": True},
                "crop": {"conf_shift": -0.1, "valid": True},
                "brightness": {"conf_shift": -0.1, "valid": True},
                "noise": {"conf_shift": -0.1, "valid": True},
                "grounding": {"conf_shift": -0.1, "valid": True},
            },
            "teacher_signature": {"V": 0.1, "L": 0.5, "A": 0.1},
            "teacher_bits": {"visual": 0, "language": 1, "alignment": 0},
            "teacher_label6": "language-prior",
            "prompt_hash": "abc123",
            "generation_config_hash": "def456",
        }
        errs = validate_record(rec, 0)
        assert any("Invalid split" in e for e in errs)
