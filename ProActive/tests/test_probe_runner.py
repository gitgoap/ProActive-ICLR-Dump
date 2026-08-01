"""
Tests for probe runner orchestration (Plan §2.5, §13).
"""

import pytest
from PIL import Image

from proactive.features.evidence_state import ProbeAction
from proactive.models.base_adapter import GenerationOutput, MLLMAdapter, ScoringOutput
from proactive.probes.probe_runner import run_all_probes


class MockRunnerAdapter(MLLMAdapter):
    def __init__(self, answer="yes"):
        super().__init__(model_path="mock/path")
        self._answer = answer
        self._is_loaded = True

    def load_model(self):
        self._is_loaded = True

    def get_model_revision(self) -> str:
        return "main"

    def generate(self, image, prompt, max_new_tokens=32, **kwargs):
        if "FINAL_ANSWER:" in prompt:
            text = f"Visual description.\nFINAL_ANSWER: {self._answer}"
        else:
            text = self._answer
        return GenerationOutput(
            raw_answer=text,
            token_logprobs=[-0.1],
            token_distributions=[{"yes": 0.90, "no": 0.10}],
            answer_len_tokens=1,
            latency_ms=20.0,
        )

    def score(self, image, prompt, target_text, **kwargs):
        return ScoringOutput(
            token_logprobs=[-0.1],
            token_distributions=[{"yes": 0.90, "no": 0.10}],
            total_logprob=-0.1,
            latency_ms=10.0,
        )


class TestProbeRunner:
    def test_run_all_probes_standard(self):
        adapter = MockRunnerAdapter(answer="yes")
        img = Image.new("RGB", (32, 32), (100, 100, 100))

        obs = run_all_probes(
            adapter=adapter,
            image=img,
            question="Is there a dog?",
            dataset="pope",
            clean_norm_answer="yes",
            clean_answer_prob=0.90,
            clean_entropy=0.10,
            clean_margin=0.80,
            relation_applicable=False,
            instance_id="pope_01",
        )

        assert ProbeAction.BLANK in obs
        assert ProbeAction.BLUR in obs
        assert ProbeAction.CROP in obs
        assert ProbeAction.BRIGHTNESS in obs
        assert ProbeAction.NOISE in obs
        assert ProbeAction.GROUNDING in obs
        assert ProbeAction.RELATION not in obs

        assert obs[ProbeAction.BLANK].valid is True
        assert obs[ProbeAction.GROUNDING].valid is True
        assert obs[ProbeAction.GROUNDING].parse_status in ("ok", "regex_fallback")

    def test_run_all_probes_with_relation(self):
        adapter = MockRunnerAdapter(answer="true")
        img = Image.new("RGB", (32, 32), (100, 100, 100))

        obs = run_all_probes(
            adapter=adapter,
            image=img,
            question="The cat is left of the dog.",
            dataset="vsr",
            clean_norm_answer="true",
            clean_answer_prob=0.90,
            clean_entropy=0.10,
            clean_margin=0.80,
            relation_applicable=True,
            instance_id="vsr_01",
        )

        assert ProbeAction.RELATION in obs
        assert obs[ProbeAction.RELATION].applicable is True
