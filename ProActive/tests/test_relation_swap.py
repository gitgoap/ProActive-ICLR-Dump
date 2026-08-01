"""
Tests for the relation swap probe.

Validates:
- Round-trip equality: swap(swap(q)) == q
- Applicability logic (single relation, no ambiguous negation)
- Edge cases (no relation, multiple relations, partial matches)
- Lexicon completeness
- Explicit RelationSwapStatus and fail-closed swap invariance
"""

import pytest

from proactive.probes.relation_swap import (
    swap_relation,
    check_relation_applicable,
    evaluate_relation_swap,
    get_swap_invariance,
    RelationSwapStatus,
    REVERSIBLE_PAIRS,
    ALL_RELATIONS,
    _find_relations,
    _has_ambiguous_negation,
)


def _make_spatial_sentence(relation: str) -> str:
    """Construct a grammatically sound spatial sentence."""
    if relation in ("left", "right"):
        return f"The cat is on the {relation} side of the dog"
    return f"The cat is {relation} the dog"


class TestBasicSwap:
    @pytest.mark.parametrize("a,b", REVERSIBLE_PAIRS)
    def test_forward_swap(self, a, b):
        text = _make_spatial_sentence(a)
        result = swap_relation(text)
        assert result.applicable, f"Failed for {a} -> {b}: {result.reason}"
        assert result.swapped_text == _make_spatial_sentence(b)

    @pytest.mark.parametrize("a,b", REVERSIBLE_PAIRS)
    def test_reverse_swap(self, a, b):
        text = _make_spatial_sentence(b)
        result = swap_relation(text)
        assert result.applicable, f"Failed for {b} -> {a}: {result.reason}"
        assert result.swapped_text == _make_spatial_sentence(a)


class TestRoundTrip:
    @pytest.mark.parametrize("a,b", REVERSIBLE_PAIRS)
    def test_round_trip_forward(self, a, b):
        original = _make_spatial_sentence(a)
        result = swap_relation(original)
        assert result.applicable, f"Failed: {result.reason}"
        assert result.round_trip_ok

        result2 = swap_relation(result.swapped_text)
        assert result2.applicable
        assert result2.swapped_text == original

    def test_round_trip_complex_sentence(self):
        text = "The person on the left is holding a cup"
        result = swap_relation(text)
        if result.applicable:
            assert result.round_trip_ok
            result2 = swap_relation(result.swapped_text)
            assert result2.swapped_text == text


class TestApplicability:
    def test_no_relation(self):
        result = swap_relation("What color is this object?")
        assert not result.applicable
        assert result.reason == "no_reversible_relation_found"

    def test_multiple_relations(self):
        result = swap_relation("The cat is above the table and left of the chair")
        assert not result.applicable
        assert result.reason == "multiple_relations_found"

    def test_ambiguous_negation(self):
        result = swap_relation("The cat is not not above the dog")
        assert not result.applicable
        assert result.reason == "ambiguous_negation"

    def test_single_negation_ok(self):
        result = swap_relation("The cat is not above the dog")
        assert result.applicable

    def test_annotated_relation_mismatch(self):
        result = swap_relation("The cat is above the dog", annotated_relation="below")
        assert not result.applicable
        assert "annotated_relation_mismatch" in result.reason

    def test_reject_verb_departed(self):
        result = swap_relation("The man left the room.")
        assert not result.applicable
        assert result.reason == "non_spatial_usage_detected"

    def test_reject_adjective_correct(self):
        result = swap_relation("Is the answer right?")
        assert not result.applicable
        assert result.reason == "non_spatial_usage_detected"


class TestEvaluateRelationSwap:
    def test_changed_correctly(self):
        status = evaluate_relation_swap(
            original_answer_correct=True,
            original_norm_answer="true",
            swapped_norm_answer="false",
            original_gold="true",
            swapped_gold="false",
        )
        assert status == RelationSwapStatus.CHANGED_CORRECTLY
        assert get_swap_invariance(True, "false", "true", "false", "true") is False

    def test_invariant_detected(self):
        status = evaluate_relation_swap(
            original_answer_correct=True,
            original_norm_answer="true",
            swapped_norm_answer="true",
            original_gold="true",
            swapped_gold="false",
        )
        assert status == RelationSwapStatus.INVARIANT
        assert get_swap_invariance(True, "true", "true", "false", "true") is True

    def test_invalid_when_clean_incorrect(self):
        status = evaluate_relation_swap(
            original_answer_correct=False,
            original_norm_answer="false",
            swapped_norm_answer="true",
            original_gold="true",
            swapped_gold="false",
        )
        assert status == RelationSwapStatus.INVALID
        assert get_swap_invariance(False, "true", "true", "false", "false") is None

    def test_invalid_when_swapped_malformed(self):
        status = evaluate_relation_swap(
            original_answer_correct=True,
            original_norm_answer="true",
            swapped_norm_answer="unknown",
            original_gold="true",
            swapped_gold="false",
        )
        assert status == RelationSwapStatus.INVALID
        assert get_swap_invariance(True, "unknown", "true", "false", "true") is None


class TestLexiconCompleteness:
    def test_all_pairs_bidirectional(self):
        for a, b in REVERSIBLE_PAIRS:
            assert a in ALL_RELATIONS
            assert b in ALL_RELATIONS

    def test_expected_pair_count(self):
        assert len(REVERSIBLE_PAIRS) == 4
