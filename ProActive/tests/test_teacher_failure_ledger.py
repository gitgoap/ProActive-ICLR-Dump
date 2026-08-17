"""Adversarial and integration tests for Week 4 failed-row provenance."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


SCRIPT_GLOBALS = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_teacher.py")
)
failure_ledger_path = SCRIPT_GLOBALS["_failure_ledger_path"]
load_failure_ledger = SCRIPT_GLOBALS["_load_failure_ledger"]
make_failure_record = SCRIPT_GLOBALS["_make_failure_record"]
write_failure_ledger = SCRIPT_GLOBALS["_write_failure_ledger"]

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "a" * 40
MANIFEST_HASH = "b" * 64
FROZEN_HASH = "c" * 64


def _failure(previous_attempts: int = 0) -> dict:
    return make_failure_record(
        manifest_record={
            "instance_id": "vizwiz_001",
            "dataset": "vizwiz",
            "split": "train",
        },
        model_id=MODEL_ID,
        model_revision=REVISION,
        error=ValueError("Mandatory probe grounding invalid"),
        invalid_teacher_record={
            "instance_id": "vizwiz_001",
            "valid": False,
            "probes": {
                "grounding": {
                    "raw_answer": "Description without a final answer.",
                    "valid": False,
                    "invalid_reason": "Missing FINAL_ANSWER tag",
                }
            },
        },
        previous_attempts=previous_attempts,
        manifest_sha256=MANIFEST_HASH,
        frozen_config_sha256=FROZEN_HASH,
        config_sha256="d" * 64,
        model_config_sha256="e" * 64,
        seed=42,
        shard_id=0,
        num_shards=4,
    )


def _load(path: Path) -> dict:
    return load_failure_ledger(
        path,
        model_id=MODEL_ID,
        selected_ids={"vizwiz_001"},
        manifest_sha256=MANIFEST_HASH,
        frozen_config_sha256=FROZEN_HASH,
        shard_id=0,
        num_shards=4,
    )


def test_failure_ledger_round_trip_preserves_raw_invalid_output(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher.jsonl"
    path = failure_ledger_path(teacher_path)
    row = _failure()
    key = (MODEL_ID, "vizwiz_001")
    write_failure_ledger(path, {key: row})

    loaded = _load(path)
    assert list(loaded) == [key]
    assert loaded[key]["attempt_count"] == 1
    grounding = loaded[key]["invalid_teacher_record"]["probes"]["grounding"]
    assert grounding["raw_answer"] == "Description without a final answer."


def test_failure_ledger_upsert_is_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "teacher.failures.jsonl"
    key = (MODEL_ID, "vizwiz_001")
    write_failure_ledger(path, {key: _failure()})
    write_failure_ledger(path, {key: _failure(previous_attempts=1)})

    loaded = _load(path)
    assert len(loaded) == 1
    assert loaded[key]["attempt_count"] == 2


def test_failure_ledger_rejects_provenance_drift(tmp_path: Path) -> None:
    path = tmp_path / "teacher.failures.jsonl"
    key = (MODEL_ID, "vizwiz_001")
    row = _failure()
    row["source_manifest_sha256"] = "f" * 64
    write_failure_ledger(path, {key: row})

    with pytest.raises(ValueError, match="provenance drift"):
        _load(path)
