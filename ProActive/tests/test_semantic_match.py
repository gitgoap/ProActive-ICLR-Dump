"""
Tests for semantic answer matching and threshold calibration (Plan §14.5).
"""

import pytest
from proactive.features.semantic import (
    compute_semantic_match,
    calibrate_semantic_threshold,
    SemanticProvenance,
    compute_reference_match,
)


class TestSemanticMatch:
    def test_binary_exact_match(self):
        assert compute_semantic_match("yes", "Yes.", "pope") == 1.0
        assert compute_semantic_match("no", "No", "pope") == 1.0
        assert compute_semantic_match("true", "True", "vsr") == 1.0
        assert compute_semantic_match("false", "False", "vsr") == 1.0

    def test_binary_strictness(self):
        # Binary datasets reject anything non-exact
        assert compute_semantic_match("yes", "no", "pope") == 0.0
        assert compute_semantic_match("true", "false", "vsr") == 0.0

    def test_freeform_exact_and_similarity(self):
        # Exact match in freeform
        assert compute_semantic_match("dog", "dog", "vizwiz") == 1.0

        # High similarity
        def mock_embed_high(a, b):
            return 0.95

        assert compute_semantic_match("brown dog", "dog brown", "vizwiz", threshold=0.82, embedding_fn=mock_embed_high) == 1.0

        # Low similarity
        def mock_embed_low(a, b):
            return 0.40

        assert compute_semantic_match("dog", "airplane", "vizwiz", threshold=0.82, embedding_fn=mock_embed_low) == 0.0

    def test_open_hallusion_reference_aliases(self):
        assert compute_reference_match(
            "Niger",
            ["Niger", "According to the table, Niger has the largest rate."],
            "hallusionbench",
            answer_type="open_ended",
        ) == 1.0
        assert compute_reference_match(
            "December 2022 and July 2023",
            ["December 2022 and July 2023", "Dec 22 and Jul 23"],
            "hallusionbench",
            answer_type="open_ended",
        ) == 1.0
        assert compute_reference_match(
            "unanswerable",
            ["unanswerable", "No, inconsistency in table"],
            "hallusionbench",
            answer_type="open_ended",
        ) == 1.0

    def test_open_hallusion_semantic_detail_fallback(self):
        def mock_embed(a, b):
            return 0.95 if "germany" in b else 0.1

        assert compute_reference_match(
            "Germany",
            ["Germany had the highest GDP in Europe in 2021."],
            "hallusionbench",
            threshold=0.82,
            embedding_fn=mock_embed,
            answer_type="open_ended",
        ) == 1.0

    def test_open_hallusion_exact_alias_mode_rejects_related_wrong_entity(self):
        def misleading_embed(a, b):
            return 0.99

        assert compute_reference_match(
            "France",
            ["Germany", "Germany had the highest GDP in Europe in 2021."],
            "hallusionbench",
            threshold=0.50,
            embedding_fn=misleading_embed,
            answer_type="open_ended",
            semantic_fallback=False,
        ) == 0.0

    def test_provenance_dataclass(self):
        prov = SemanticProvenance(
            dataset="vizwiz",
            matching_mode="embedding_similarity",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            model_revision="e4ce9877abf3edee10b0257f22713854020a4004",
            threshold=0.82,
            calibration_split="train_val",
            calibration_metrics={"precision": 0.92, "recall": 0.90, "f1": 0.91},
        )
        d = prov.to_dict()
        assert d["threshold"] == 0.82
        assert d["calibration_split"] == "train_val"
        assert d["calibration_metrics"]["f1"] == 0.91

    def test_calibration_function(self):
        pairs = [
            ("apple", "apple pie", True),
            ("red car", "automobile", True),
            ("cat", "dog", False),
            ("computer", "ocean", False),
        ]

        def sim_fn(s1, s2):
            if (s1, s2) in [("apple", "apple pie"), ("red car", "automobile")]:
                return 0.85
            return 0.20

        thresh, prov = calibrate_semantic_threshold(pairs, sim_fn, target_recall=0.90)
        assert thresh <= 0.85
        assert prov.calibration_metrics["recall"] >= 0.90
