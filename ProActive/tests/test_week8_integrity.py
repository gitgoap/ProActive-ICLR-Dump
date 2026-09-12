"""CPU-only regression checks for closed answers and report integrity."""

import json
import tempfile
import unittest
from pathlib import Path

from proactive.features.semantic import compute_semantic_match
from proactive.utils.hashing import hash_dict
from scripts.validate_week8 import _read_signed


class ClosedAnswerTests(unittest.TestCase):
    @staticmethod
    def forbidden_embedding(left, right):
        raise AssertionError("Closed answers must never use embeddings")

    def test_heldout_closed_answers(self):
        for dataset, normalizer, left, right in (
            ("prehal", None, "A", "B"),
            ("illusionbench", "multiple_choice", "A", "B"),
            ("illusionbench", "true_false", "true", "false"),
        ):
            with self.subTest(dataset=dataset, normalizer=normalizer):
                self.assertEqual(compute_semantic_match(
                    left, right, dataset, normalizer_type=normalizer,
                    embedding_fn=self.forbidden_embedding,
                ), 0.0)
                self.assertEqual(compute_semantic_match(
                    left, left, dataset, normalizer_type=normalizer,
                    embedding_fn=self.forbidden_embedding,
                ), 1.0)

    def test_explicit_closed_contract_does_not_require_registered_dataset(self):
        for normalizer, left, right in (
            ("multiple_choice", "A", "B"),
            ("yes_no", "yes", "no"),
            ("true_false", "true", "false"),
        ):
            with self.subTest(normalizer=normalizer):
                self.assertEqual(compute_semantic_match(
                    left, right, "new_closed_dataset", normalizer_type=normalizer,
                    embedding_fn=self.forbidden_embedding,
                ), 0.0)

    def test_freeform_and_open_hallusion_still_use_embeddings(self):
        for dataset, answer_type in (("vizwiz", None), ("hallusionbench", "open_ended")):
            with self.subTest(dataset=dataset):
                self.assertEqual(compute_semantic_match(
                    "dog", "canine", dataset, answer_type=answer_type,
                    embedding_fn=lambda left, right: 0.99,
                ), 1.0)


class ReportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".week8_integrity_", dir=Path(__file__).resolve().parents[1]
        )
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "report.json"

    def read_report(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")
        errors = []
        result = _read_signed(self.path, errors)
        return result, errors

    def test_valid_report_round_trip(self):
        report = {"is_valid": True, "row_count": 1200}
        report["report_sha256"] = hash_dict(report)
        result, errors = self.read_report(report)
        self.assertEqual(result, report)
        self.assertEqual(errors, [])

    def test_missing_empty_null_and_nonstring_hash_are_rejected(self):
        for hash_field in ({}, {"report_sha256": None}, {"report_sha256": ""}, {"report_sha256": 1}):
            with self.subTest(hash_field=hash_field):
                result, errors = self.read_report({"is_valid": True, **hash_field})
                self.assertIsNone(result)
                self.assertTrue(errors)

    def test_tampered_report_is_rejected(self):
        report = {"is_valid": True, "row_count": 1200}
        report["report_sha256"] = hash_dict(report)
        report["row_count"] = 1199
        result, errors = self.read_report(report)
        self.assertIsNone(result)
        self.assertIn("mismatch", errors[0])

    def test_non_object_json_is_rejected(self):
        for value in ([], None, "text", 42):
            with self.subTest(value=value):
                result, errors = self.read_report(value)
                self.assertIsNone(result)
                self.assertIn("JSON object", errors[0])

    def test_malformed_and_missing_reports_are_rejected(self):
        self.path.write_text("{invalid", encoding="utf-8")
        errors = []
        self.assertIsNone(_read_signed(self.path, errors))
        self.assertTrue(errors)
        errors = []
        self.assertIsNone(_read_signed(self.path.parent / "missing.json", errors))
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
