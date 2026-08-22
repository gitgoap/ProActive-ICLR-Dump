"""
Tests for grounding probe output parsing and validation (Plan §13.4, §14.5).
"""

import pytest
from proactive.prompts.templates import parse_grounding_output, make_grounding_prompt


class TestGroundingParsing:
    def test_clean_final_answer_tag_yes(self):
        text = "I see a brown dog sitting on a rug in the living room.\nFINAL_ANSWER: yes"
        res = parse_grounding_output(text, "pope")
        assert res.is_valid is True
        assert res.parse_status == "ok"
        assert res.norm_final_answer == "yes"
        assert "brown dog" in res.description

    def test_clean_final_answer_tag_false(self):
        text = "The cat is on the left side of the table.\nFINAL_ANSWER: false"
        res = parse_grounding_output(text, "vsr")
        assert res.is_valid is True
        assert res.parse_status == "ok"
        assert res.norm_final_answer == "false"
        assert "cat is on the left" in res.description

    def test_malformed_binary_answer(self):
        text = "I see an object.\nFINAL_ANSWER: It is possible, but I cannot be sure."
        res = parse_grounding_output(text, "pope")
        assert res.is_valid is False
        assert res.parse_status == "malformed"
        assert res.invalid_reason is not None

    def test_regex_fallback_last_line(self):
        text = "I see a bird in the tree.\nYes"
        res = parse_grounding_output(text, "pope")
        assert res.is_valid is True
        assert res.parse_status == "regex_fallback"
        assert res.norm_final_answer == "yes"

    def test_empty_output(self):
        res = parse_grounding_output("", "pope")
        assert res.is_valid is False
        assert res.parse_status == "empty"

    def test_freeform_final_answer(self):
        text = "The person is holding a white bottle with prescription text.\nFINAL_ANSWER: aspirin"
        res = parse_grounding_output(text, "vizwiz")
        assert res.is_valid is True
        assert res.parse_status == "ok"
        assert res.norm_final_answer == "aspirin"

    def test_terminal_answer_phrase_recovers_binary_without_tag(self):
        text = "The image contains a dog beside a bench.\nThe answer is yes."
        res = parse_grounding_output(text, "pope")
        assert res.is_valid is True
        assert res.parse_status == "terminal_answer_fallback"
        assert res.raw_final_answer == "yes."
        assert res.norm_final_answer == "yes"

    def test_terminal_answer_phrase_recovers_freeform_without_tag(self):
        text = "The label is legible on the bottle.\nThe answer is **aspirin**"
        res = parse_grounding_output(text, "vizwiz")
        assert res.is_valid is True
        assert res.parse_status == "terminal_answer_fallback"
        assert res.raw_final_answer == "aspirin"
        assert res.norm_final_answer == "aspirin"

    def test_bare_freeform_last_line_remains_invalid(self):
        text = "The label is legible on the bottle.\naspirin"
        res = parse_grounding_output(text, "vizwiz")
        assert res.is_valid is False
        assert res.parse_status == "malformed"

    def test_conflicting_explicit_binary_answer_remains_invalid(self):
        text = "The image is difficult to interpret.\nThe answer is yes or no."
        res = parse_grounding_output(text, "pope")
        assert res.is_valid is False
        assert res.parse_status == "malformed"

    def test_empty_tag_is_not_converted_to_unanswerable(self):
        res = parse_grounding_output("Description.\nFINAL_ANSWER:", "vizwiz")
        assert res.is_valid is False
        assert res.parse_status == "malformed"
        assert "no answer content" in res.invalid_reason

    def test_explicit_vizwiz_unanswerable_remains_valid(self):
        res = parse_grounding_output(
            "The image is too blurry to read.\nFINAL_ANSWER: unanswerable", "vizwiz"
        )
        assert res.is_valid is True
        assert res.norm_final_answer == "unanswerable"

    def test_explicit_vizwiz_unknown_is_structured_abstention(self):
        res = parse_grounding_output(
            "The label cannot be read.\nFINAL_ANSWER: Unknown", "vizwiz"
        )
        assert res.is_valid is True
        assert res.norm_final_answer == "unanswerable"

    def test_short_isolated_freeform_answer_is_recovered(self):
        res = parse_grounding_output(
            "The prize dialog shows three required letters.\n\nMVG", "vizwiz"
        )
        assert res.is_valid is True
        assert res.norm_final_answer == "mvg"
        assert res.parse_status == "terminal_answer_fallback"

    def test_embedded_terminal_binary_answer_is_recovered(self):
        res = parse_grounding_output(
            "The conditions are insufficient.\nTherefore, the answer is no — it cannot be concluded.",
            "hallusionbench",
        )
        assert res.is_valid is True
        assert res.norm_final_answer == "no"

    def test_freeform_prose_is_not_treated_as_isolated_answer(self):
        res = parse_grounding_output(
            "The label is blurred.\nThe contents of this container cannot be determined from the available image.",
            "vizwiz",
        )
        assert res.is_valid is False

    def test_open_hallusion_answer_uses_freeform_contract(self):
        res = parse_grounding_output(
            "The table lists five countries.\nFINAL_ANSWER: Niger",
            "hallusionbench",
            answer_type="open_ended",
        )
        assert res.is_valid is True
        assert res.norm_final_answer == "niger"

    def test_binary_hallusion_numeric_indicator(self):
        res = parse_grounding_output(
            "The object is visible.\nFINAL_ANSWER: 1", "hallusionbench"
        )
        assert res.is_valid is True
        assert res.norm_final_answer == "yes"
