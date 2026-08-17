"""CPU tests for uniform Week 4 grounding-cache reconstruction."""

from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

from proactive.teacher.label_computation import DEFAULT_THRESHOLDS
from proactive.utils.io import write_jsonl


ROOT = Path(__file__).parents[1]
REFRESH = runpy.run_path(str(ROOT / "scripts" / "refresh_grounding_cache.py"))
PIPELINE = runpy.run_path(str(ROOT / "tests" / "test_week4_pipeline.py"))
rebuild_teacher_row = REFRESH["_rebuild_teacher_row"]
validate_refresh_rows = REFRESH["_validate_refresh_rows"]
load_refresh_failures = REFRESH["_load_refresh_failures"]
write_refresh_failures = REFRESH["_write_refresh_failures"]
teacher_record = PIPELINE["_teacher_record"]


def _provenance() -> dict:
    return {
        "source_kind": "valid_teacher",
        "source_path": "teacher.jsonl",
        "source_file_sha256": "d" * 64,
        "source_record_sha256": "e" * 64,
    }


def test_rebuild_replaces_only_grounding_and_recomputes_labels() -> None:
    base = teacher_record()
    original_blur = copy.deepcopy(base["probes"]["blur"])
    grounding = copy.deepcopy(base["probes"]["grounding"])
    grounding.update(
        {
            "raw_answer": "Description.\nFINAL_ANSWER: no",
            "norm_answer": "no",
            "flip": True,
            "exact_match": 0.0,
            "semantic_match": 0.0,
            "conf_shift": -0.2,
            "generation_config_hash": "f" * 64,
        }
    )
    rebuilt = rebuild_teacher_row(
        base, grounding, _provenance(), DEFAULT_THRESHOLDS, 512
    )
    assert rebuilt["valid"] is True
    assert rebuilt["probes"]["blur"] == original_blur
    assert rebuilt["probes"]["grounding"]["norm_answer"] == "no"
    assert rebuilt["teacher_bits"]["alignment"] == 1
    assert rebuilt["grounding_refresh"]["uniform_max_new_tokens"] == 512
    assert rebuilt["grounding_refresh"]["effective_generation_config_sha256"] == "f" * 64


def test_refresh_resume_rejects_token_cap_and_source_drift(tmp_path: Path) -> None:
    row = teacher_record()
    row.update(
        {
            "record_type": "teacher_cache",
            "source_manifest_sha256": "a" * 64,
            "frozen_probe_config_sha256": "b" * 64,
            "shard_id": 0,
            "num_shards": 1,
            "grounding_refresh": {
                "schema_version": 1,
                "uniform_max_new_tokens": 512,
                **_provenance(),
            },
        }
    )
    path = tmp_path / "teacher.jsonl"
    write_jsonl([row], path)
    kwargs = {
        "model_id": row["model_id"],
        "selected_ids": {row["instance_id"]},
        "manifest_sha": "a" * 64,
        "frozen_sha": "b" * 64,
        "shard_id": 0,
        "num_shards": 1,
        "max_new_tokens": 512,
        "provenance": {(row["model_id"], row["instance_id"]): _provenance()},
    }
    assert validate_refresh_rows(path, **kwargs) == {
        (row["model_id"], row["instance_id"])
    }
    with pytest.raises(ValueError, match="token-cap drift"):
        validate_refresh_rows(path, **{**kwargs, "max_new_tokens": 768})
    bad_provenance = copy.deepcopy(kwargs["provenance"])
    bad_provenance[(row["model_id"], row["instance_id"])][
        "source_record_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="source provenance drift"):
        validate_refresh_rows(path, **{**kwargs, "provenance": bad_provenance})


def test_refresh_failure_ledger_roundtrip_is_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "teacher.failures.jsonl"
    key = ("test/model", "instance-1")
    row = {
        "record_type": "grounding_refresh_failure",
        "schema_version": 1,
        "model_id": key[0],
        "instance_id": key[1],
        "uniform_max_new_tokens": 512,
        "attempt_count": 2,
    }
    write_refresh_failures(path, {key: row})
    loaded = load_refresh_failures(
        path,
        model_id=key[0],
        selected_ids={key[1]},
        max_new_tokens=512,
    )
    assert list(loaded) == [key]
    assert loaded[key]["attempt_count"] == 2
