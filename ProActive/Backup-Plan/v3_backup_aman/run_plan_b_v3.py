#!/usr/bin/env python3
"""CLI for the frozen post-hoc Plan B V3 selective-correction study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
V1_ROOT = HERE.parent / "v1_backup_sourish_sir_plan"
V2_ROOT = HERE.parent / "v2_backup_aman"
for location in (V1_ROOT, V2_ROOT, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from plan_b.core import PlanBError, load_and_verify_inputs  # noqa: E402
from plan_b_v3.pipeline import (  # noqa: E402
    dependency_report,
    ensemble_specs,
    load_v1_config,
    load_v3_config,
    run_evaluate,
    run_prepare,
    validate_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Plan B V3 from frozen cached probes. Prepare uses train, then validation, "
            "and opens calibration only if validation passes. Test/shift require a separately "
            "confirmed command after both scientific gates pass."
        )
    )
    parser.add_argument("mode", choices=("preflight", "prepare", "evaluate", "validate"))
    parser.add_argument("--config", type=Path, default=HERE / "config" / "plan_b_v3_config.json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true")
    action.add_argument("--overwrite", action="store_true")
    parser.add_argument("--confirm-locked-evaluation", action="store_true")
    parser.add_argument("--require-evaluation", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    config = load_v3_config(config_path.resolve(), repo_root)
    if args.mode == "preflight":
        v1_config = load_v1_config(repo_root, config)
        records, inventory = load_and_verify_inputs(repo_root, v1_config)
        dependencies = dependency_report(ensemble_specs(config))
        payload = {
            "mode": "preflight",
            "is_valid": not dependencies["missing"],
            "device": args.device,
            "new_mllm_calls": 0,
            "post_hoc_exploratory": True,
            "loaded_records": len(records),
            "input_inventory": inventory,
            "dependencies": dependencies,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["is_valid"] else 1
    if args.mode == "prepare":
        payload = run_prepare(
            repo_root=repo_root,
            config=config,
            output_dir=output_dir.resolve(),
            resume=args.resume,
            overwrite=args.overwrite,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("status") == "FROZEN_CONFIRMATION_GATE_PASSED" else 2
    if args.mode == "evaluate":
        payload = run_evaluate(
            repo_root=repo_root,
            config=config,
            output_dir=output_dir.resolve(),
            resume=args.resume,
            overwrite=args.overwrite,
            confirm_locked_evaluation=args.confirm_locked_evaluation,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload = validate_outputs(output_dir.resolve(), args.require_evaluation, repo_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["is_valid"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlanBError as exc:
        print(f"PLAN_B_V3_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
