import csv
import json
from pathlib import Path

import pytest

from scripts.index_state_manifest import _state_identity
from proactive.data.heldout import (
    deterministic_stratified_sample,
    load_illusionbench_release,
    load_prehal_release,
    parse_illusion_multiple_choice,
)
from proactive.data.manifests import validate_manifest
from proactive.features.normalization import normalize_answer
from proactive.prompts.templates import parse_grounding_output


def test_prehal_release_loader_is_strict_and_grouped(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "mmbench"
    image_dir.mkdir(parents=True)
    (image_dir / "one.png").write_bytes(b"image")
    (image_dir / "two.png").write_bytes(b"image")
    rows = [
        {"index": "1", "question": "What is shown?", "A": "cat", "B": "dog", "C": "car", "D": "bus", "answer": "B", "image": "/mmbench/one.png", "hallucination_type": "existence "},
        {"index": "2", "question": "Choose one", "A": "red", "B": "blue", "C": "green", "D": "black", "answer": "A", "image": "/mmbench/two.png", "hallucination_type": "attribute"},
    ]
    with open(tmp_path / "dataset.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {"annotation_file": "dataset.csv", "expected_source_rows": 2, "sample_cap": 2, "split_seed": 42, "official_url": "official", "revision": "pinned", "license": "cc-by-4.0"}
    result = load_prehal_release(tmp_path, config)
    assert len(result.records) == 2
    assert result.exclusions == []
    assert result.records[0]["split"] == "shift"
    assert result.records[0]["normalizer_type"] == "multiple_choice"
    assert result.audit["normalized_hallucination_type_counts"]["existence"] == 1
    assert validate_manifest(result.records) == []


def test_illusion_parser_converts_roman_options_fail_closed() -> None:
    prompt, options, answer = parse_illusion_multiple_choice(
        "Which animal? i) cat ii) dog iii) horse", "ii) dog"
    )
    assert prompt.splitlines()[-1] == "C. horse"
    assert options == {"A": "cat", "B": "dog", "C": "horse"}
    assert answer == "B"
    with pytest.raises(ValueError, match="answer_text_option_mismatch"):
        parse_illusion_multiple_choice("Which animal? i) cat ii) dog", "ii) cat")


def test_illusion_release_uniformly_excludes_missing_and_malformed(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "one.png").write_bytes(b"image")
    source = [
        {"image_property": {"image_name": "one.png", "Difficult Level": 1, "Category": "A", "Description": "description"}, "qa_data": [
            {"Question": "Is this true?", "Question Type": "TF", "Correct Answer": "True"},
            {"Question": "Choose. i) one ii) two", "Question Type": "Select", "Correct Answer": "ii) two"},
            {"Question": "No option markers", "Question Type": "Select", "Correct Answer": "one"},
        ]},
        {"image_property": {"image_name": "missing.png", "Difficult Level": 0, "Category": "B", "Description": "missing"}, "qa_data": [
            {"Question": "Is this true?", "Question Type": "TF", "Correct Answer": "False"}
        ]},
    ]
    (tmp_path / "Image_properties.json").write_text(json.dumps(source), encoding="utf-8")
    config = {"annotation_file": "Image_properties.json", "expected_annotation_images": 2, "expected_archive_images": 1, "expected_missing_annotation_images": 1, "expected_unreferenced_archive_images": 0, "expected_source_qa_rows": 4, "sample_cap": 2, "split_seed": 42, "official_url": "official", "revision": "pinned", "license": None, "license_status": "not_declared"}
    result = load_illusionbench_release(tmp_path, config)
    assert len(result.records) == 2
    assert result.audit["excluded_rows"] == 2
    assert result.audit["exclusion_counts"] == {"malformed_option_sequence": 1, "missing_image": 1}
    assert validate_manifest(result.records) == []


def test_deterministic_stratified_sampling_is_order_independent() -> None:
    rows = [{"instance_id": f"x{index}", "heldout_stratum": "a" if index < 7 else "b"} for index in range(10)]
    first, counts = deterministic_stratified_sample(rows, cap=5, seed=42)
    second, _ = deterministic_stratified_sample(list(reversed(rows)), cap=5, seed=42)
    assert [row["instance_id"] for row in first] == [row["instance_id"] for row in second]
    assert sum(counts.values()) == 5


def test_prehal_same_basename_in_different_folders_has_distinct_identity(
    tmp_path: Path,
) -> None:
    for folder in ("source_a", "source_b"):
        image_dir = tmp_path / "images" / folder
        image_dir.mkdir(parents=True)
        (image_dir / "1.png").write_bytes(b"image")
    rows = [
        {
            "index": str(index),
            "question": "Choose one",
            "A": "cat",
            "B": "dog",
            "C": "car",
            "D": "bus",
            "answer": "A",
            "image": f"/{folder}/1.png",
            "hallucination_type": "existence",
        }
        for index, folder in enumerate(("source_a", "source_b"), start=1)
    ]
    with open(tmp_path / "dataset.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = load_prehal_release(
        tmp_path,
        {
            "annotation_file": "dataset.csv",
            "expected_source_rows": 2,
            "expected_unique_images": 2,
            "sample_cap": 2,
            "split_seed": 42,
            "official_url": "official",
            "revision": "pinned",
            "license": "cc-by-4.0",
        },
    )
    assert result.audit["source_unique_images"] == 2
    assert len({row["image_id"] for row in result.records}) == 2
    assert len({row["group_id"] for row in result.records}) == 2


def test_multiple_choice_normalization_and_grounding_contract() -> None:
    assert normalize_answer("(b).", "prehal") == "B"
    assert normalize_answer("option C", "illusionbench", normalizer_type="multiple_choice") == "C"
    assert normalize_answer("maybe", "prehal") == "unknown"
    parsed = parse_grounding_output(
        "The second option matches the image.\nFINAL_ANSWER: B",
        "prehal",
        answer_type="multiple_choice",
        normalizer_type="multiple_choice",
    )
    assert parsed.is_valid is True
    assert parsed.norm_final_answer == "B"


def test_week8_state_index_reads_shift_identity_from_metadata() -> None:
    row = {
        "record_type": "partial_state_v1",
        "metadata": {
            "split": "shift",
            "model_id": "model-a",
            "instance_id": "example-a",
        },
    }
    assert _state_identity(row, Path("states.jsonl"), 1) == (
        "shift",
        "model-a",
        "example-a",
    )


@pytest.mark.parametrize(
    "row, message",
    [
        ({"record_type": "partial_state_v1"}, "Missing state metadata"),
        (
            {
                "record_type": "partial_state_v1",
                "metadata": {
                    "split": "shift",
                    "model_id": "model-a",
                    "instance_id": "",
                },
            },
            "Missing metadata.instance_id",
        ),
        (
            {
                "record_type": "partial_state_v1",
                "metadata": {
                    "split": "unknown",
                    "model_id": "model-a",
                    "instance_id": "example-a",
                },
            },
            "Unknown metadata.split",
        ),
    ],
)
def test_week8_state_index_rejects_malformed_metadata(
    row: dict[str, object], message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        _state_identity(row, Path("states.jsonl"), 1)
