#!/usr/bin/env python3
"""Validate a deterministic ablation rebuild without relying on torch ZIP bytes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuilt_history", required=True)
    parser.add_argument("--archived_history", required=True)
    parser.add_argument("--reference_aps", required=True)
    parser.add_argument("--reference_voi", required=True)
    parser.add_argument("--rebuilt_aps", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stopping_prefix(history: Sequence[Mapping[str, Any]], patience: int) -> list[Mapping[str, Any]]:
    if patience <= 0 or not history:
        raise ValueError("History and positive patience are required")
    best: tuple[float, float] | None = None
    stale = 0
    for index, row in enumerate(history):
        if int(row.get("epoch", -1)) != index:
            raise ValueError("History epochs are not contiguous from zero")
        validation = row.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError("History validation block is missing")
        score = (
            float(validation["source_bit_macro_f1"]),
            float(validation["six_way_macro_f1"]),
        )
        if any(not math.isfinite(value) for value in score):
            raise ValueError("History contains a non-finite selection score")
        if best is None or score > best:
            best = score
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            return list(history[: index + 1])
    raise ValueError("Archived history does not contain an early-stopping boundary")


def aps_scientific_payload(report: Mapping[str, Any]) -> Dict[str, Any]:
    ignored = {"checkpoint_path", "checkpoint_sha256", "report_sha256"}
    return {key: value for key, value in report.items() if key not in ignored}


def validate_equivalence(
    *,
    rebuilt_history: Sequence[Mapping[str, Any]],
    archived_history: Sequence[Mapping[str, Any]],
    reference_aps: Mapping[str, Any],
    rebuilt_aps: Mapping[str, Any],
    patience: int,
) -> Dict[str, Any]:
    expected_history = stopping_prefix(archived_history, patience)
    history_exact = list(rebuilt_history) == expected_history
    aps_exact = aps_scientific_payload(rebuilt_aps) == aps_scientific_payload(reference_aps)
    if not history_exact:
        raise ValueError("Rebuilt history differs before the original stopping boundary")
    if not aps_exact:
        raise ValueError("Rebuilt APS thresholds or metrics differ from the reference")
    return {
        "is_valid": True,
        "history_exact_through_stopping_boundary": True,
        "aps_thresholds_and_metrics_exact": True,
        "stopping_epoch": int(expected_history[-1]["epoch"]),
        "epoch_count": len(expected_history),
        "ignored_packaging_fields": sorted(
            ["checkpoint_path", "checkpoint_sha256", "report_sha256"]
        ),
    }


def main() -> None:
    args = parse_args()
    paths = {
        "rebuilt_history": Path(args.rebuilt_history),
        "archived_history": Path(args.archived_history),
        "reference_aps": Path(args.reference_aps),
        "reference_voi": Path(args.reference_voi),
        "rebuilt_aps": Path(args.rebuilt_aps),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {name}: {path}")
    reference_aps = _read_json(paths["reference_aps"])
    rebuilt_aps = _read_json(paths["rebuilt_aps"])
    reference_voi = _read_json(paths["reference_voi"])
    for name, report in (
        ("reference APS", reference_aps),
        ("rebuilt APS", rebuilt_aps),
    ):
        if not isinstance(report, Mapping):
            raise SystemExit(f"{name} is not a mapping")
        unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
        if report.get("report_sha256") != hash_dict(unsigned):
            raise SystemExit(f"{name} self-hash mismatch")
    if reference_voi.get("temporary_aps_sha256") != file_sha256(paths["reference_aps"]):
        raise SystemExit("Reference VOI does not bind the reference APS file")
    if reference_voi.get("checkpoint_sha256") != reference_aps.get("checkpoint_sha256"):
        raise SystemExit("Reference VOI/APS checkpoint binding mismatch")
    try:
        result = validate_equivalence(
            rebuilt_history=_read_json(paths["rebuilt_history"]),
            archived_history=_read_json(paths["archived_history"]),
            reference_aps=reference_aps,
            rebuilt_aps=rebuilt_aps,
            patience=args.patience,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Ablation rebuild equivalence failed: {exc}") from exc
    report: Dict[str, Any] = {
        "format_version": "week8_ablation_rebuild_equivalence_v1",
        **result,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }
    report["report_sha256"] = hash_dict(report)
    output = Path(args.output_report)
    write_json(report, output, overwrite=args.overwrite and output.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
