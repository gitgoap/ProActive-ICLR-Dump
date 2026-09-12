"""
End-to-end integration tests for Week 3 pilot pipeline.

Simulates pilot caching using MockMLLMAdapter, verifies JSONL schema,
and tests the pilot analysis and candidate config generation workflow.
"""

import json
import tempfile
from pathlib import Path
from PIL import Image
import pytest

from proactive.audits.pilot_analysis import (
    compute_probe_statistics,
    compute_severity_grid_statistics,
    select_canonical_severities,
    generate_candidate_week3_config,
    compute_full_run_estimates,
)
from proactive.audits.schema_validator import validate_file
from proactive.models.base_adapter import GenerationOutput, MLLMAdapter, ScoringOutput
from proactive.probes.probe_runner import run_all_probes
from proactive.teacher.cache_builder import process_instance


class MockAdapter(MLLMAdapter):
    """Deterministic mock adapter for pipeline testing."""
    def __init__(self, answer: str = "yes"):
        super().__init__(model_path="mock/path")
        self.answer = answer
        self._is_loaded = True

    def load_model(self):
        self._is_loaded = True

    def get_model_revision(self) -> str:
        return "mock_revision"

    def generate(self, image, prompt, max_new_tokens=32, **kwargs):
        if "FINAL_ANSWER:" in prompt:
            text = f"I see a room with objects.\nFINAL_ANSWER: {self.answer}"
        else:
            text = self.answer
        return GenerationOutput(
            raw_answer=text,
            token_logprobs=[-0.1, -0.05],
            token_distributions=[
                {self.answer: 0.90, "other": 0.10},
                {self.answer: 0.95, "other": 0.05},
            ],
            answer_len_tokens=2,
            latency_ms=45.0,
        )

    def score(self, image, prompt, target_text, **kwargs):
        return ScoringOutput(
            token_logprobs=[-0.1, -0.05],
            token_distributions=[
                {self.answer: 0.90, "other": 0.10},
                {self.answer: 0.95, "other": 0.05},
            ],
            total_logprob=-0.15,
            latency_ms=20.0,
        )


class TestWeek3PipelineIntegration:
    def test_full_pipeline_mock_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_path = tmp_path / "test.jpg"
            Image.new("RGB", (64, 64), (200, 100, 50)).save(img_path)

            record = {
                "instance_id": "test_inst_001",
                "group_id": "grp_01",
                "dataset": "pope",
                "split": "train",
                "image_path": str(img_path),
                "question": "Is there a dog?",
                "gold_answer": "yes",
                "relation_applicable": False,
            }

            adapter = MockAdapter()
            res = process_instance(
                record=record,
                adapter=adapter,
                dataset_name="pope",
                model_id="mock_model",
                model_revision="main",
            )

            # Write JSONL
            jsonl_file = tmp_path / "pilot.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")

            # Run schema validator
            report = validate_file(jsonl_file)
            assert report["is_valid"] is True, f"Schema errors: {report['errors']}"
            assert report["valid_rows"] == 1
            assert report["invalid_rows"] == 0

            # Run analysis components
            records = [res]
            grid_stats = compute_severity_grid_statistics(records)
            selected_sev = select_canonical_severities(grid_stats)
            stats = compute_probe_statistics(records)
            assert "blank" in stats
            assert "noise" in stats

            cand_config_path = tmp_path / "candidate_config.yaml"
            generate_candidate_week3_config(stats, {}, cand_config_path, selected_severities=selected_sev)
            assert cand_config_path.exists()

            estimates = compute_full_run_estimates(records, total_manifest_examples=100, total_models=1, gpus=1)
            assert estimates["projected_total_hours"] > 0

    def test_heldout_row_normalizer_reaches_clean_and_probe_paths(self):
        """Mixed-format datasets must use the normalizer declared by each row."""

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "illusion.png"
            Image.new("RGB", (64, 64), (100, 100, 100)).save(img_path)
            record = {
                "instance_id": "illusionbench_example_1",
                "group_id": "illusion_group_1",
                "dataset": "illusionbench",
                "split": "shift",
                "image_path": str(img_path),
                "question": "Which option is correct?\nA. First\nB. Second",
                "gold_answer": "B",
                "answer_type": "multiple_choice",
                "normalizer_type": "multiple_choice",
                "answer_match_mode": "choice_exact",
                "relation_applicable": False,
            }

            result = process_instance(
                record=record,
                adapter=MockAdapter(answer="B"),
                dataset_name="illusionbench",
                model_id="mock_model",
                model_revision="main",
            )

            assert result["valid"] is True
            assert result["normalizer_type"] == "multiple_choice"
            assert result["clean"]["norm_answer"] == "B"
            assert all(
                probe["norm_answer"] == "B"
                for probe in result["probes"].values()
                if probe["applicable"]
            )
