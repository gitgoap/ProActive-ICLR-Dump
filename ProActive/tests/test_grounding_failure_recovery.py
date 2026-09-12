"""Tests for deterministic, failure-triggered grounding recovery."""

from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

from proactive.prompts.templates import make_concise_grounding_retry_prompt
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_jsonl


ROOT = Path(__file__).parents[1]
RECOVERY = runpy.run_path(str(ROOT / "scripts" / "recover_grounding_failures.py"))
PIPELINE = runpy.run_path(str(ROOT / "tests" / "test_week4_pipeline.py"))

load_source_failures = RECOVERY["_load_source_failures"]
find_failure_base = RECOVERY["_find_failure_base"]
source_paths = RECOVERY["_source_paths"]
annotate_copy = RECOVERY["_annotate_copy"]
annotate_recovery = RECOVERY["_annotate_recovery"]
teacher_record = PIPELINE["_teacher_record"]


def _source_failure(
    *, path: Path, base: dict, source_kind: str = "valid_teacher"
) -> dict:
    return {
        "record_type": "grounding_refresh_failure",
        "schema_version": 1,
        "instance_id": base["instance_id"],
        "dataset": base["dataset"],
        "model_id": base["model_id"],
        "model_revision": base["model_revision"],
        "uniform_max_new_tokens": 1024,
        "attempt_count": 1,
        "error_type": "ValueError",
        "error_message": (
            "Missing FINAL_ANSWER tag and unable to resolve valid final answer"
        ),
        "grounding_observation": {
            "valid": False,
            "parse_status": "malformed",
            "raw_answer": "truncated reasoning",
        },
        "source_kind": source_kind,
        "source_path": str(path),
        "source_file_sha256": file_sha256(path),
        "source_record_sha256": hash_dict(base),
    }


def test_concise_retry_prompt_keeps_description_and_strict_answer_domain() -> None:
    binary = make_concise_grounding_retry_prompt("Is it red?", "hallusionbench")
    assert "visible evidence" in binary
    assert "Do not explain your reasoning" in binary
    assert "FINAL_ANSWER: <yes or no>" in binary
    assert binary.endswith("Write nothing after the FINAL_ANSWER line.")

    freeform = make_concise_grounding_retry_prompt("What is shown?", "vizwiz")
    assert "shortest sufficient answer" in freeform
    assert "FINAL_ANSWER: <answer>" in freeform

    relation = make_concise_grounding_retry_prompt("A is left of B", "vsr")
    assert "FINAL_ANSWER: <true or false>" in relation

    multiple_choice = make_concise_grounding_retry_prompt(
        "Which option?\nA. first\nB. second",
        "illusionbench",
        answer_type="multiple_choice",
    )
    assert "FINAL_ANSWER: <one option letter>" in multiple_choice


def test_shift_source_paths_use_the_heldout_filename_contract(tmp_path: Path) -> None:
    teacher, failures = source_paths(
        tmp_path,
        "qwen3_vl_8b",
        shard_id=0,
        num_shards=1,
        split="shift",
    )
    assert teacher.name == "teacher_qwen3_vl_8b_all_shift_shard00-of-01.jsonl"
    assert failures.name == (
        "teacher_qwen3_vl_8b_all_shift_shard00-of-01.failures.jsonl"
    )


def test_source_failure_loader_accepts_only_uniform_parse_failures(
    tmp_path: Path,
) -> None:
    base = teacher_record()
    source_path = tmp_path / "teacher.jsonl"
    write_jsonl([base], source_path)
    failure = _source_failure(path=source_path, base=base)
    failure_path = tmp_path / "teacher.failures.jsonl"
    write_jsonl([failure], failure_path)

    key = (base["model_id"], base["instance_id"])
    loaded = load_source_failures(
        failure_path,
        model_id=base["model_id"],
        selected_ids={base["instance_id"]},
        source_max_new_tokens=1024,
    )
    assert list(loaded) == [key]

    bad = copy.deepcopy(failure)
    bad["error_message"] = "CUDA out of memory"
    write_jsonl([bad], failure_path, overwrite=True)
    with pytest.raises(ValueError, match="Non-format failure"):
        load_source_failures(
            failure_path,
            model_id=base["model_id"],
            selected_ids={base["instance_id"]},
            source_max_new_tokens=1024,
        )


def test_failure_base_resolution_checks_file_and_record_hashes(
    tmp_path: Path,
) -> None:
    base = teacher_record()
    source_path = tmp_path / "teacher.jsonl"
    write_jsonl([base], source_path)
    failure = _source_failure(path=source_path, base=base)

    resolved = find_failure_base(
        failure, model_id=base["model_id"], instance_id=base["instance_id"]
    )
    assert resolved == base

    wrong_record = copy.deepcopy(failure)
    wrong_record["source_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="record hash drift"):
        find_failure_base(
            wrong_record,
            model_id=base["model_id"],
            instance_id=base["instance_id"],
        )

    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash drift"):
        find_failure_base(
            failure, model_id=base["model_id"], instance_id=base["instance_id"]
        )


def test_failure_ledger_base_resolution_is_fail_closed(tmp_path: Path) -> None:
    base = teacher_record()
    base["valid"] = False
    base["invalid_reason"] = "grounding invalid"
    wrapper = {
        "model_id": base["model_id"],
        "instance_id": base["instance_id"],
        "invalid_teacher_record": base,
    }
    source_path = tmp_path / "teacher.failures.jsonl"
    write_jsonl([wrapper], source_path)
    failure = _source_failure(
        path=source_path, base=base, source_kind="failure_ledger"
    )
    assert find_failure_base(
        failure, model_id=base["model_id"], instance_id=base["instance_id"]
    ) == base

    wrapper["invalid_teacher_record"] = None
    write_jsonl([wrapper], source_path, overwrite=True)
    failure["source_file_sha256"] = file_sha256(source_path)
    with pytest.raises(ValueError, match="found 0"):
        find_failure_base(
            failure, model_id=base["model_id"], instance_id=base["instance_id"]
        )


def test_teacher_failure_ledger_is_accepted_only_for_grounding_parse_failures(
    tmp_path: Path,
) -> None:
    base = teacher_record()
    base["valid"] = False
    base["invalid_reason"] = "Mandatory probe observation 'grounding' is invalid"
    base["normalizer_type"] = "multiple_choice"
    base["probes"]["grounding"].update(
        {
            "valid": False,
            "parse_status": "malformed",
            "invalid_reason": (
                "Multiple-choice answer is not one option letter: "
                "'None of the above' -> 'unknown'"
            ),
        }
    )
    failure = {
        "record_type": "teacher_failure",
        "schema_version": 1,
        "instance_id": base["instance_id"],
        "dataset": base["dataset"],
        "model_id": base["model_id"],
        "model_revision": base["model_revision"],
        "error_type": "ValueError",
        "error_message": base["invalid_reason"],
        "invalid_teacher_record": base,
    }
    path = tmp_path / "teacher.failures.jsonl"
    write_jsonl([failure], path)

    key = (base["model_id"], base["instance_id"])
    loaded = load_source_failures(
        path,
        model_id=base["model_id"],
        selected_ids={base["instance_id"]},
        source_max_new_tokens=1024,
    )
    assert list(loaded) == [key]
    assert find_failure_base(
        loaded[key], model_id=base["model_id"], instance_id=base["instance_id"]
    ) == base

    failure["invalid_teacher_record"]["probes"]["grounding"][
        "invalid_reason"
    ] = "CUDA out of memory"
    write_jsonl([failure], path, overwrite=True)
    with pytest.raises(ValueError, match="Non-format failure"):
        load_source_failures(
            path,
            model_id=base["model_id"],
            selected_ids={base["instance_id"]},
            source_max_new_tokens=1024,
        )


def test_recovery_metadata_distinguishes_copied_and_recovered_rows() -> None:
    base = teacher_record()
    provenance = {
        "source_kind": "grounding1024_valid_teacher",
        "source_path": "source.jsonl",
        "source_file_sha256": "a" * 64,
        "source_record_sha256": "b" * 64,
    }
    copied = annotate_copy(base, provenance)
    assert copied["grounding_recovery"]["status"] == "not_required"
    assert copied["grounding_recovery"]["policy_id"].endswith("_v1")

    source_failure = {
        "error_message": (
            "Missing FINAL_ANSWER tag and unable to resolve valid final answer"
        )
    }
    recovered = annotate_recovery(
        base,
        provenance=provenance,
        source_failure=source_failure,
        max_new_tokens=256,
    )
    metadata = recovered["grounding_recovery"]
    assert metadata["status"] == "recovered"
    assert metadata["retry_max_new_tokens"] == 256
    assert metadata["trigger_error"] == source_failure["error_message"]
