import json
from pathlib import Path

import pytest

from proactive.data.hallusion_contract import (
    HALLUSION_BINARY,
    HALLUSION_OPEN_ENDED,
    classify_hallusion_answer_type,
)
from proactive.data.loaders import load_hallusionbench
from proactive.data.manifests import validate_manifest


@pytest.mark.parametrize(
    "question",
    [
        "which country has the largest population growth rate in 2023",
        "According to the table, which country in Europe has the higher GDP?",
        "According to the image, which month has the highest percentage change?",
    ],
)
def test_open_question_classification_is_content_based(question: str) -> None:
    assert classify_hallusion_answer_type(question) == HALLUSION_OPEN_ENDED


def test_relative_which_clause_remains_binary() -> None:
    question = (
        "According to the image which is about Euler's Number, does the value "
        "range from 2.70 to 2.71?"
    )
    assert classify_hallusion_answer_type(question) == HALLUSION_BINARY


def test_loader_preserves_mixed_answer_contract(tmp_path: Path) -> None:
    annotations = [
        {
            "category": "VS",
            "subcategory": "table",
            "visual_input": "1",
            "set_id": "0",
            "figure_id": "1",
            "question_id": "0",
            "question": "Is France first?",
            "gt_answer": "0",
            "gt_answer_details": "No, Germany is first.",
            "filename": "binary.png",
        },
        {
            "category": "VS",
            "subcategory": "table",
            "visual_input": "1",
            "set_id": "1",
            "figure_id": "1",
            "question_id": "0",
            "question": "According to the table, which country is first?",
            "gt_answer": "1",
            "gt_answer_details": "Germany is first.",
            "filename": "open.png",
        },
    ]
    annotation_path = tmp_path / "HallusionBench.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    overlay_path = tmp_path / "open_references.json"
    overlay_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "matching_mode": "exact_alias",
                "records": {
                    "VS|table|1|1|0": {"canonical_answers": ["Germany"]}
                },
            }
        ),
        encoding="utf-8",
    )
    records = load_hallusionbench(
        {
            "dataset_name": "hallusionbench",
            "data_path": str(tmp_path),
            "annotation_file": annotation_path.name,
            "open_ended_reference_file": str(overlay_path),
            "expected_open_ended_image_records": 1,
        }
    )
    assert validate_manifest(records) == []
    binary, opened = records
    assert binary["answer_type"] == HALLUSION_BINARY
    assert binary["gold_answer"] == "no"
    assert binary["benchmark_gold_answer"] == "0"
    assert opened["answer_type"] == HALLUSION_OPEN_ENDED
    assert opened["answer_match_mode"] == "exact_alias"
    assert opened["gold_answer"] == "Germany is first."
    assert opened["reference_answers"] == ["Germany", "Germany is first."]


def test_open_question_without_overlay_fails_closed(tmp_path: Path) -> None:
    annotation = {
        "category": "VS",
        "subcategory": "table",
        "visual_input": "1",
        "set_id": "9",
        "figure_id": "1",
        "question_id": "0",
        "question": "Which country is first?",
        "gt_answer": "1",
        "gt_answer_details": "Niger",
        "filename": "x.png",
    }
    (tmp_path / "HallusionBench.json").write_text(
        json.dumps([annotation]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no author-verified reference"):
        load_hallusionbench(
            {
                "dataset_name": "hallusionbench",
                "data_path": str(tmp_path),
                "annotation_file": "HallusionBench.json",
            }
        )


def test_repository_overlay_has_exactly_fourteen_source_rows() -> None:
    overlay = json.loads(
        Path("configs/data/hallusionbench_open_ended_references.json").read_text(
            encoding="utf-8"
        )
    )
    assert overlay["schema_version"] == 1
    assert overlay["matching_mode"] == "exact_alias"
    assert len(overlay["records"]) == 14
