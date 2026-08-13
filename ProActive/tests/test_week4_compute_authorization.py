"""Fail-closed tests for staged versus full Week 4 compute approval."""

import runpy
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPT_GLOBALS = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_teacher.py")
)
enforce_compute_authorization = SCRIPT_GLOBALS["_enforce_compute_authorization"]


def _experiment(full_core_approved: bool = False) -> dict:
    return {
        "compute_authorization": {
            "staged_checks_approved": True,
            "staged_max_examples": 100,
            "staged_full_dataset_allowlist": ["vsr"],
            "full_core_approved": full_core_approved,
        }
    }


def _args(*, limit=None, dataset="all", dry_run=False) -> Namespace:
    return Namespace(limit=limit, dataset=dataset, dry_run=dry_run)


def test_approved_staged_limits_and_vsr_are_allowed() -> None:
    enforce_compute_authorization(_experiment(), _args(limit=100))
    enforce_compute_authorization(_experiment(), _args(dataset="vsr"))


def test_unapproved_full_core_run_fails_closed() -> None:
    with pytest.raises(SystemExit, match="Full Week 4 core generation"):
        enforce_compute_authorization(_experiment(), _args())


def test_separate_full_core_approval_unlocks_scope() -> None:
    enforce_compute_authorization(_experiment(full_core_approved=True), _args())


def test_missing_authorization_fails_closed() -> None:
    with pytest.raises(SystemExit, match="Missing compute_authorization"):
        enforce_compute_authorization({}, _args(limit=1))
