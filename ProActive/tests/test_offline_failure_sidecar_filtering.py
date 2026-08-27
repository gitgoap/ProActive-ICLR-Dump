"""Regression tests for excluding failure ledgers from offline artifacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_name", "prefix"),
    (
        ("build_labels", None),
        ("sample_states", "teacher_"),
        ("export_human_audit", "teacher_"),
    ),
)
def test_offline_discovery_excludes_failure_ledgers(
    tmp_path: Path, script_name: str, prefix: str | None
) -> None:
    teacher = tmp_path / "teacher_model_shard00-of-01.jsonl"
    failure = tmp_path / "teacher_model_shard00-of-01.failures.jsonl"
    teacher.write_text("{}\n", encoding="utf-8")
    failure.write_text("", encoding="utf-8")

    module = _load_script(script_name)
    files = (
        module._jsonl_files(tmp_path)
        if prefix is None
        else module._jsonl_files(tmp_path, prefix)
    )

    assert files == [teacher]


@pytest.mark.parametrize(
    ("script_name", "prefix"),
    (
        ("build_labels", None),
        ("sample_states", "teacher_"),
        ("export_human_audit", "teacher_"),
    ),
)
def test_explicit_failure_ledger_input_is_rejected(
    tmp_path: Path, script_name: str, prefix: str | None
) -> None:
    failure = tmp_path / "teacher_model_shard00-of-01.failures.jsonl"
    failure.write_text("", encoding="utf-8")
    module = _load_script(script_name)

    with pytest.raises(ValueError, match="Failure ledger"):
        if prefix is None:
            module._jsonl_files(failure)
        else:
            module._jsonl_files(failure, prefix)
