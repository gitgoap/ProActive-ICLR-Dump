"""Regression tests for immutable local model revision inspection."""

import runpy
from pathlib import Path


SCRIPT_GLOBALS = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "inspect_model_revisions.py")
)
huggingface_metadata_revision = SCRIPT_GLOBALS["huggingface_metadata_revision"]


def test_huggingface_metadata_uses_commit_not_git_blob_etag() -> None:
    commit = "a" * 40
    file_etag = "b" * 40

    assert huggingface_metadata_revision(f"{commit}\n{file_etag}\n1234.5\n") == commit


def test_huggingface_metadata_accepts_lfs_etag_but_ignores_it() -> None:
    commit = "c" * 40
    lfs_etag = "d" * 64

    assert huggingface_metadata_revision(f"{commit}\n{lfs_etag}\n1234.5\n") == commit


def test_huggingface_metadata_fails_closed_without_commit_first_line() -> None:
    assert huggingface_metadata_revision("not-a-commit\n" + "e" * 40) is None
