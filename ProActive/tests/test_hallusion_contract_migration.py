from __future__ import annotations

import copy

import pytest

from scripts.migrate_hallusion_answer_contract import (
    _migrate_shard,
    _update_answer_features,
    validate_manifest_transition,
)
from proactive.teacher.label_computation import LabelThresholds
from proactive.utils.io import file_sha256, write_jsonl


def _manifest_row(instance_id: str, *, dataset: str = "hallusionbench") -> dict:
    return {
        "instance_id": instance_id,
        "group_id": f"group:{instance_id}",
        "dataset": dataset,
        "image_id": instance_id,
        "question_id": "0",
        "image_path": f"/data/{instance_id}.png",
        "question": "Is the statement correct?",
        "gold_answer": "1" if dataset == "hallusionbench" else "yes",
        "relation_applicable": False,
        "category": "VS" if dataset == "hallusionbench" else "",
        "subcategory": "table" if dataset == "hallusionbench" else "",
        "split": "train",
    }


def _contract_row(old: dict, *, open_ended: bool) -> dict:
    row = copy.deepcopy(old)
    row["benchmark_gold_answer"] = str(old["gold_answer"])
    row["answer_contract_version"] = 1
    if open_ended:
        row["answer_type"] = "open_ended"
        row["answer_match_mode"] = "exact_alias"
        row["gold_answer"] = "Niger has the largest rate."
        row["gt_answer_details"] = "Niger has the largest rate."
        row["reference_answers"] = ["Niger", "Niger has the largest rate."]
    else:
        row["answer_type"] = "binary"
        row["answer_match_mode"] = "binary_exact"
        row["gold_answer"] = "yes"
        row["gt_answer_details"] = "Yes."
        row["reference_answers"] = ["yes"]
    return row


def test_manifest_transition_retains_all_rows_and_identifies_exactly_14_open() -> None:
    old = [_manifest_row(f"hallusionbench_{index}") for index in range(15)]
    old.append(_manifest_row("pope_0", dataset="pope"))
    new = [
        _contract_row(row, open_ended=index < 14)
        for index, row in enumerate(old[:15])
    ]
    new.append(copy.deepcopy(old[-1]))

    old_by_id, new_by_id, open_ids = validate_manifest_transition(old, new)

    assert len(old_by_id) == len(new_by_id) == 16
    assert len(open_ids) == 14
    assert "hallusionbench_14" not in open_ids


def test_manifest_transition_rejects_unrelated_dataset_drift() -> None:
    old = [_manifest_row(f"hallusionbench_{index}") for index in range(14)]
    old.append(_manifest_row("pope_0", dataset="pope"))
    new = [_contract_row(row, open_ended=True) for row in old[:14]]
    changed = copy.deepcopy(old[-1])
    changed["question"] = "Changed after inference"
    new.append(changed)

    with pytest.raises(ValueError, match="Non-HallusionBench manifest row changed"):
        validate_manifest_transition(old, new)


def test_binary_migration_recomputes_features_from_raw_answers() -> None:
    row = {
        "instance_id": "hallusionbench_binary",
        "clean": {"raw_answer": "Yes", "norm_answer": "yes", "correct": 0},
        "probes": {
            "blank": {
                "raw_answer": "0",
                "norm_answer": "unknown",
                "valid": True,
                "flip": False,
                "exact_match": 1.0,
                "semantic_match": 1.0,
            },
            "grounding": {
                "raw_answer": "Description.\nFINAL_ANSWER: 1",
                "norm_answer": "yes",
                "valid": True,
                "flip": False,
                "exact_match": 1.0,
                "semantic_match": 1.0,
            },
        },
    }
    manifest = {
        "gold_answer": "no",
        "answer_type": "binary",
        "answer_contract_version": 1,
        "answer_match_mode": "binary_exact",
        "benchmark_gold_answer": "0",
        "gt_answer_details": "No.",
        "reference_answers": ["no"],
    }

    _update_answer_features(row, manifest)

    assert row["clean"]["norm_answer"] == "yes"
    assert row["clean"]["correct"] == 0
    assert row["probes"]["blank"]["norm_answer"] == "no"
    assert row["probes"]["blank"]["flip"] is True
    assert row["probes"]["blank"]["semantic_match"] == 0.0
    assert row["probes"]["grounding"]["norm_answer"] == "yes"
    assert row["probes"]["grounding"]["flip"] is False


def _probe(name: str, raw_answer: str = "Yes") -> dict:
    if name == "grounding":
        raw_answer = "The image supports it.\nFINAL_ANSWER: yes"
    return {
        "raw_answer": raw_answer,
        "norm_answer": "yes",
        "flip": False,
        "conf_shift": 0.0,
        "entropy_shift": 0.0,
        "margin_shift": 0.0,
        "exact_match": 1.0,
        "semantic_match": 1.0,
        "applicable": True,
        "valid": True,
        "parse_status": "ok",
        "score_method": "generation_logits",
    }


def test_shard_migration_reuses_binary_and_invalidates_all_open_rows(tmp_path) -> None:
    old = [_manifest_row(f"hallusionbench_{index}") for index in range(15)]
    new = [
        _contract_row(row, open_ended=index < 14)
        for index, row in enumerate(old)
    ]
    new_by_id = {row["instance_id"]: row for row in new}
    open_ids = {row["instance_id"] for row in new[:14]}
    old_manifest_path = tmp_path / "old.jsonl"
    write_jsonl(old, old_manifest_path)
    old_sha = file_sha256(old_manifest_path)
    teacher_path = tmp_path / "input" / (
        "teacher_qwen3_vl_8b_all_all_shard00-of-01.jsonl"
    )
    rows = []
    for index, manifest_row in enumerate(old):
        row = {
            "record_type": "teacher_cache",
            "instance_id": manifest_row["instance_id"],
            "group_id": manifest_row["group_id"],
            "dataset": manifest_row["dataset"],
            "split": manifest_row["split"],
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "model_revision": "a" * 40,
            "image_path": manifest_row["image_path"],
            "question": manifest_row["question"],
            "gold_answer": manifest_row["gold_answer"],
            "relation_applicable": False,
            "valid": True,
            "source_manifest_sha256": old_sha,
            "frozen_probe_config_sha256": "frozen-sha",
            "shard_id": 0,
            "num_shards": 1,
        }
        if index == 14:
            row.update(
                {
                    "clean": {
                        "raw_answer": "Yes",
                        "norm_answer": "yes",
                        "correct": 0,
                        "answer_prob": 0.9,
                        "valid": True,
                    },
                    "probes": {
                        name: _probe(name)
                        for name in (
                            "blank",
                            "blur",
                            "crop",
                            "brightness",
                            "noise",
                            "grounding",
                        )
                    },
                    "teacher_signature": {"V": 0.0, "L": 0.0, "A": 0.0},
                    "teacher_bits": {"visual": 0, "language": 0, "alignment": 0},
                    "teacher_label6": "unclear",
                    "benchmark_family": "VS",
                    "swap_invariance": None,
                }
            )
        rows.append(row)
    write_jsonl(rows, teacher_path)

    report = _migrate_shard(
        teacher_path=teacher_path,
        output_dir=tmp_path / "output",
        new_by_id=new_by_id,
        open_ids=open_ids,
        old_manifest_sha256=old_sha,
        new_manifest_sha256="new-manifest-sha",
        frozen_sha256="frozen-sha",
        thresholds=LabelThresholds(),
        overwrite=False,
    )

    assert report["selected_count"] == 15
    assert report["valid_count"] == 1
    assert report["failure_count"] == 0
    assert report["dropped_open_ended_count"] == 14
    assert report["pending_count"] == 14
