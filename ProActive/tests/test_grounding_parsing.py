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
