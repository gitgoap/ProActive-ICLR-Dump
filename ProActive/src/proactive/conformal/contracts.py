"""Freeze-bound calibration and locked-test authorization contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from proactive.train.checkpoints import validate_freeze_manifest
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256


def validate_final_aps_report(
    report: Mapping[str, Any], *, freeze_manifest_path: str | Path
) -> Dict[str, Any]:
    freeze_path = Path(freeze_manifest_path)
    validate_freeze_manifest(freeze_path, require_policy=True)
    if report.get("format_version") != "week7_final_aps_v1" or report.get("status") != "FINAL_FROZEN":
        raise ValueError("APS report is not a final Week 7 artifact")
    if report.get("fit_split") != "cal" or report.get("test_used") is not False:
        raise ValueError("APS report violates calibration/test split provenance")
    if report.get("freeze_manifest_sha256") != file_sha256(freeze_path):
        raise ValueError("APS report is bound to a different frozen stack")
    expected = report.get("thresholds_sha256")
    unsigned = {key: value for key, value in report.items() if key != "thresholds_sha256"}
    if expected != hash_dict(unsigned):
        raise ValueError("APS report self-hash mismatch")
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("APS report contains no thresholds")
    return dict(report)


def authorize_locked_test(
    *, freeze_manifest_path: str | Path, aps_report: Mapping[str, Any]
) -> None:
    """Raise unless both immutable gates exist and reference each other."""

    validate_final_aps_report(aps_report, freeze_manifest_path=freeze_manifest_path)
