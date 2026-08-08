"""Adversarial safety tests for the Week 3 pilot execution path."""

import json
import importlib.util
import csv
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from proactive.audits.schema_validator import validate_file
from proactive.data.loaders import load_hallusionbench
from proactive.features.semantic import calibrate_semantic_threshold_from_scores
from proactive.models.base_adapter import GenerationOutput, MLLMAdapter, ScoringOutput
from proactive.teacher.cache_builder import process_severity_grid_instance

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_pilot_cache.py"
_SPEC = importlib.util.spec_from_file_location("run_pilot_cache", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
_RUN_PILOT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_PILOT)
pilot_record_key = _RUN_PILOT.pilot_record_key
read_existing_pilot_keys = _RUN_PILOT.read_existing_pilot_keys
stratified_sample = _RUN_PILOT.stratified_sample

_CALIBRATE_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_semantic_threshold.py"
_CAL_SPEC = importlib.util.spec_from_file_location("calibrate_semantic_threshold", _CALIBRATE_SCRIPT)
assert _CAL_SPEC and _CAL_SPEC.loader
_CALIBRATE = importlib.util.module_from_spec(_CAL_SPEC)
_CAL_SPEC.loader.exec_module(_CALIBRATE)


class CountingAdapter(MLLMAdapter):
    def __init__(self):
        super().__init__(model_path="mock/path")
        self._is_loaded = True
        self.generate_calls = 0

    def load_model(self):
        self._is_loaded = True

    def get_model_revision(self) -> str:
        return "mock_revision"

    def generate(self, image, prompt, max_new_tokens=32, **kwargs):
        self.generate_calls += 1
        answer = "scene description\nFINAL_ANSWER: yes" if "FINAL_ANSWER:" in prompt else "yes"
        return GenerationOutput(
            raw_answer=answer,
            token_logprobs=[-0.1],
            token_distributions=[{"yes": 0.9, "no": 0.1}],
            answer_len_tokens=1,
            latency_ms=10.0,
        )

    def score(self, image, prompt, target_text, **kwargs):
        return ScoringOutput(
            token_logprobs=[-0.1],
            token_distributions=[{"yes": 0.9, "no": 0.1}],
            total_logprob=-0.1,
            latency_ms=5.0,
        )


def test_stratified_sample_fills_exact_limit_and_is_deterministic():
    records = [
        {
            "instance_id": f"id_{index}",
            "split": "train" if index % 3 else "val",
            "category": f"category_{index % 7}",
        }
        for index in range(137)
    ]
    first = stratified_sample(records, 100, seed=42)
    second = stratified_sample(records, 100, seed=42)
    assert len(first) == 100
    assert [row["instance_id"] for row in first] == [row["instance_id"] for row in second]
    assert len({row["instance_id"] for row in first}) == 100


def test_existing_duplicate_cache_fails_closed(tmp_path: Path):
    output = tmp_path / "cache.jsonl"
    row = {"instance_id": "same", "model_id": "m", "dataset": "pope"}
    output.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate pilot key"):
        read_existing_pilot_keys(output, "canonical", "m", "pope")


def test_severity_key_includes_probe_and_value():
    row = {
        "instance_id": "i",
        "pilot_severity_probe": "blur",
        "pilot_severity_value": 8.0,
    }
    assert pilot_record_key(row, "severity_grid") == ("i", "blur", 8.0)


def test_compact_severity_grid_uses_15_generations_and_validates(tmp_path: Path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    record = {
        "instance_id": "pilot_1",
        "group_id": "group_1",
        "dataset": "pope",
        "split": "train",
        "image_path": str(image_path),
        "question": "Is there an object?",
        "gold_answer": "yes",
        "relation_applicable": False,
    }
    adapter = CountingAdapter()
    rows = process_severity_grid_instance(
        record,
        adapter,
        dataset_name="pope",
        model_id="mock",
        model_revision="mock_revision",
    )
    assert len(rows) == 12
    assert adapter.generate_calls == 15  # 7 canonical + 8 non-canonical severities
    assert all(row["record_type"] == "severity_pilot" for row in rows)
    assert all(len(row["probes"]) == 1 for row in rows)

    output = tmp_path / "severity.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = validate_file(output)
    assert report["is_valid"], report["errors"]
    assert report["unique_records"] == 12


def test_hallusionbench_filters_text_only_before_limit(tmp_path: Path):
    annotations = [
        {"id": "text", "question": "Text only?", "gt_answer": "no"},
        {"id": "image_1", "filename": "one.png", "question": "One?", "gt_answer": "yes"},
        {"id": "image_2", "filename": "two.png", "question": "Two?", "gt_answer": "no"},
    ]
    (tmp_path / "HallusionBench.json").write_text(json.dumps(annotations), encoding="utf-8")
    records = load_hallusionbench(
        {
            "dataset_name": "hallusionbench",
            "data_path": str(tmp_path),
            "annotation_file": "HallusionBench.json",
        },
        limit=2,
    )
    assert len(records) == 2
    assert all(record["image_path"] for record in records)
    assert {record["image_id"] for record in records} == {"image_1", "image_2"}


def test_semantic_audit_spans_score_range_and_calibrates():
    rows = [
        {
            "cosine_similarity": score,
            "model_id": "m",
            "instance_id": f"i{index}",
            "probe": "blur",
        }
        for index, score in enumerate([index / 100 for index in range(100)])
    ]
    selected = _CALIBRATE.select_evenly_across_scores(rows, 5)
    assert [row["cosine_similarity"] for row in selected] == [0.0, 0.25, 0.5, 0.74, 0.99]

    threshold, metrics = calibrate_semantic_threshold_from_scores(
        [(0.95, True), (0.90, True), (0.70, False), (0.60, False)],
        target_recall=0.90,
    )
    assert 0.72 <= threshold <= 0.90
    assert metrics["recall"] >= 0.90


def test_semantic_calibration_rejects_single_class():
    with pytest.raises(ValueError, match="positive and negative"):
        calibrate_semantic_threshold_from_scores([(0.9, True), (0.8, True)])


def test_semantic_calibration_report_records_annotator(tmp_path: Path):
    labeled_csv = tmp_path / "audit.csv"
    with labeled_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "cosine_similarity", "human_match"],
        )
        writer.writeheader()
        writer.writerow({"split": "train", "cosine_similarity": "0.95", "human_match": "1"})
        writer.writerow({"split": "val", "cosine_similarity": "0.60", "human_match": "0"})

    output_report = tmp_path / "report.json"
    args = SimpleNamespace(
        annotator="Test Annotator",
        labeled_csv=str(labeled_csv),
        min_labels=2,
        target_recall=0.90,
        output_report=str(output_report),
        overwrite=True,
    )
    _CALIBRATE.calibrate_audit(args)
    report = json.loads(output_report.read_text(encoding="utf-8"))
    assert report["annotator"] == "Test Annotator"
    assert report["calibrated_at_utc"].endswith("+00:00")
    assert report["source_sha256"]
