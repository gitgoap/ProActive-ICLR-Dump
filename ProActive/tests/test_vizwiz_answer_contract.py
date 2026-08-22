from __future__ import annotations

import copy

import pytest

from proactive.data.loaders import VIZWIZ_GOLD_POLICY, select_vizwiz_gold
from scripts.migrate_hallusion_answer_contract import (
    _update_vizwiz_answer_contract,
    validate_manifest_transition,
)


def test_vizwiz_majority_aggregates_after_normalization() -> None:
    winner, counts, tie_size = select_vizwiz_gold(
        [
            {"answer": "The cat"},
            {"answer": "cat."},
            {"answer": "dog"},
        ]
    )
    assert winner == "cat"
    assert counts == {"cat": 2, "dog": 1}
    assert tie_size == 1


def test_vizwiz_tie_uses_released_source_order() -> None:
    winner, counts, tie_size = select_vizwiz_gold(
        [{"answer": "loading"}, {"answer": "unanswerable"}]
    )
    assert winner == "loading"
    assert counts == {"loading": 1, "unanswerable": 1}
    assert tie_size == 2


def _vizwiz_old() -> dict:
    return {
        "instance_id": "vizwiz_0",
        "group_id": "group:vizwiz_0",
        "dataset": "vizwiz",
        "image_id": "0",
        "question_id": "0",
        "image_path": "/data/0.jpg",
        "question": "What is shown?",
        "gold_answer": "unanswerable",
        "relation_applicable": False,
        "answerable": 1,
        "split": "train",
    }


def _vizwiz_new() -> dict:
    row = copy.deepcopy(_vizwiz_old())
    row.update(
        {
            "gold_answer": "loading",
            "answer_contract_version": 1,
            "answer_match_mode": "normalized_exact",
            "reference_answers": ["loading"],
            "vizwiz_gold_policy": VIZWIZ_GOLD_POLICY,
            "vizwiz_answer_counts": {"loading": 1, "unanswerable": 1},
            "vizwiz_tied_top_answer_count": 2,
        }
    )
    return row


def test_manifest_transition_allows_only_declared_vizwiz_contract_change() -> None:
    old = [_vizwiz_old()]
    new = [_vizwiz_new()]
    # The transition contract also requires exactly 14 Hallusion open rows.
    for index in range(14):
        old_row = {
            "instance_id": f"hallusionbench_{index}",
            "group_id": f"group:hallusionbench_{index}",
            "dataset": "hallusionbench",
            "image_id": str(index),
            "question_id": str(index),
            "image_path": f"/data/{index}.png",
            "question": "Which country is first?",
            "gold_answer": "1",
            "relation_applicable": False,
            "category": "VS",
            "subcategory": "table",
            "split": "train",
        }
        new_row = copy.deepcopy(old_row)
        new_row.update(
            {
                "gold_answer": "Niger",
                "benchmark_gold_answer": "1",
                "answer_contract_version": 1,
                "answer_type": "open_ended",
                "answer_match_mode": "exact_alias",
                "gt_answer_details": "Niger",
                "reference_answers": ["Niger"],
            }
        )
        old.append(old_row)
        new.append(new_row)

    validate_manifest_transition(old, new)

    changed = copy.deepcopy(new)
    changed[0]["question"] = "A changed question"
    with pytest.raises(ValueError, match="Unexpected VizWiz manifest drift"):
        validate_manifest_transition(old, changed)


def test_vizwiz_migration_recomputes_clean_correctness_without_inference() -> None:
    teacher = {
        "instance_id": "vizwiz_0",
        "clean": {"raw_answer": "Loading.", "norm_answer": "loading", "correct": 0},
        "probes": {},
    }
    _update_vizwiz_answer_contract(teacher, _vizwiz_new())
    assert teacher["gold_answer"] == "loading"
    assert teacher["clean"]["norm_answer"] == "loading"
    assert teacher["clean"]["correct"] == 1
    assert teacher["vizwiz_gold_policy"] == VIZWIZ_GOLD_POLICY


def test_vizwiz_migration_refreshes_parser_dependent_probe_features() -> None:
    teacher = {
        "instance_id": "vizwiz_0",
        "clean": {
            "raw_answer": "the label is unreadable",
            "norm_answer": "label is unreadable",
            "correct": 0,
        },
        "probes": {
            "grounding": {
                "raw_answer": (
                    "The image is too blurry.\n"
                    "FINAL_ANSWER: cannot be determined"
                ),
                "norm_answer": "cannot be determined",
                "flip": True,
                "exact_match": 0.0,
                "semantic_match": 0.0,
                "valid": True,
                "parse_status": "ok",
            }
        },
    }
    calls = []

    def similarity(left: str, right: str) -> float:
        calls.append((left, right))
        return 0.9

    _update_vizwiz_answer_contract(
        teacher,
        _vizwiz_new(),
        semantic_threshold=0.5,
        embedding_fn=similarity,
    )

    grounding = teacher["probes"]["grounding"]
    assert grounding["norm_answer"] == "unanswerable"
    assert grounding["flip"] is True
    assert grounding["exact_match"] == 0.0
    assert grounding["semantic_match"] == 1.0
    assert calls == [("unanswerable", "label is unreadable")]
