"""
Tests for answer normalization.

Covers binary (yes/no, true/false) and free-form (VizWiz) normalizers
with various edge cases.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.features.normalization import (
    normalize_answer,
    normalize_yes_no,
    normalize_true_false,
    normalize_freeform,
    normalize_hallusion_binary,
)


class TestYesNoNormalizer:

    @pytest.mark.parametrize("raw,expected", [
        ("Yes", "yes"),
        ("yes", "yes"),
        ("YES", "yes"),
        ("Yes.", "yes"),
        ("Yeah", "yes"),
        ("yep", "yes"),
        ("No", "no"),
        ("no", "no"),
        ("NO", "no"),
        ("No.", "no"),
        ("nope", "no"),
        ("Yes, the object is there.", "yes"),
        ("No, I don't see it.", "no"),
    ])
    def test_basic_cases(self, raw, expected):
        assert normalize_yes_no(raw) == expected

    def test_ambiguous_returns_unknown(self):
        assert normalize_yes_no("maybe") == "unknown"
        assert normalize_yes_no("I'm not sure") == "unknown"


class TestTrueFalseNormalizer:

    @pytest.mark.parametrize("raw,expected", [
        ("True", "true"),
        ("true", "true"),
        ("TRUE", "true"),
        ("True.", "true"),
        ("False", "false"),
        ("false", "false"),
        ("FALSE", "false"),
        ("False.", "false"),
        ("Correct", "true"),
        ("Incorrect", "false"),
    ])
    def test_basic_cases(self, raw, expected):
        assert normalize_true_false(raw) == expected

    def test_ambiguous(self):
        assert normalize_true_false("uncertain") == "unknown"


class TestFreeformNormalizer:

    def test_lowercase_and_strip(self):
        assert normalize_freeform("  A Cat  ") == "cat"

    def test_remove_punctuation(self):
        assert normalize_freeform("it's a dog!") == "it s dog"

    def test_remove_articles(self):
        assert normalize_freeform("the red car") == "red car"
        assert normalize_freeform("A big house") == "big house"

    def test_unanswerable(self):
        assert normalize_freeform("unanswerable") == "unanswerable"
        assert normalize_freeform("not answerable") == "unanswerable"
        assert normalize_freeform("I don't know") == "unanswerable"
        assert normalize_freeform("no answer") == "unanswerable"
        assert normalize_freeform("cannot be determined") == "unanswerable"

    def test_collapse_whitespace(self):
        assert normalize_freeform("red    car") == "red car"


class TestNormalizeAnswerDispatch:

    def test_hallusionbench(self):
        assert normalize_answer("Yes", "hallusionbench") == "yes"
        assert normalize_answer("1", "hallusionbench") == "yes"
        assert normalize_answer("0", "hallusionbench") == "no"
        assert normalize_answer("2", "hallusionbench") == "uncertain"

    def test_hallusion_binary_does_not_reinterpret_open_answer(self):
        assert normalize_hallusion_binary("Niger") == "unknown"

    def test_pope(self):
        assert normalize_answer("No", "pope") == "no"

    def test_vsr(self):
        assert normalize_answer("True", "vsr") == "true"

    def test_vizwiz(self):
        result = normalize_answer("A red car", "vizwiz")
        assert result == "red car"

    def test_gqa_relation(self):
        assert normalize_answer("False", "gqa_relation") == "false"

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            normalize_answer("yes", "nonexistent_dataset")

    def test_explicit_normalizer_type(self):
        """Override normalizer type regardless of dataset name."""
        assert normalize_answer("True", "anything", normalizer_type="true_false") == "true"
