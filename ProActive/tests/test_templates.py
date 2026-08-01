"""
Tests for prompt templates and grounding formatting (Plan §13.1, §13.4).
"""

import pytest

from proactive.prompts.templates import (
    make_binary_prompt,
    make_freeform_prompt,
    make_dataset_prompt,
    make_grounding_prompt,
    make_relation_prompt,
    parse_grounding_output,
)


class TestPromptTemplates:
    def test_binary_prompt_vsr(self):
        p = make_binary_prompt("The cat is left of the dog.", "vsr")
        assert "true or false" in p
        assert "Statement: The cat is left of the dog." in p

    def test_binary_prompt_pope(self):
        p = make_binary_prompt("Is there a cat in the image?", "pope")
        assert "yes or no" in p

    def test_freeform_prompt_vizwiz(self):
        p = make_freeform_prompt("What is this item?")
        assert "Answer the question concisely." in p

    def test_dataset_prompt_dispatch(self):
        assert "yes or no" in make_dataset_prompt("Is there a dog?", "pope")
        assert "true or false" in make_dataset_prompt("The ball is under the bed.", "vsr")
        assert "Answer the question concisely." in make_dataset_prompt("What is this?", "vizwiz")

    def test_grounding_prompt_contains_final_answer_tag(self):
        p_pope = make_grounding_prompt("Is there a dog?", "pope")
        assert "FINAL_ANSWER: <yes or no>" in p_pope
        assert "describe what you see" in p_pope.lower()

        p_vsr = make_grounding_prompt("The cat is on the mat.", "vsr")
        assert "FINAL_ANSWER: <true or false>" in p_vsr

        p_viz = make_grounding_prompt("What is written on the can?", "vizwiz")
        assert "FINAL_ANSWER: <answer>" in p_viz

    def test_relation_prompt(self):
        p = make_relation_prompt("The cat is right of the dog.", "vsr")
        assert "The cat is right of the dog." in p
        assert "true or false" in p
