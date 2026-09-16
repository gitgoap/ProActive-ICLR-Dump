#!/usr/bin/env python3
"""Command-line entry point for the self-contained ProActive Plan B study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plan_b.core import PlanBError, load_and_verify_inputs  # noqa: E402
from plan_b.pipeline import (  # noqa: E402
    load_config,
    run_audit,
    run_evaluate,
    run_prepare,
    validate_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the cached-answer Plan B selector without any new MLLM calls. "
            "Preparation uses train/validation only; locked evaluation is separate."
        )
    )
    parser.add_argument(
        "mode",
        choices=("preflight", "audit", "prepare", "evaluate", "validate"),
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config" / "plan_b_config.json",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true")
    action.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit-only non-scientific row limit; forbidden for preparation/evaluation.",
    )
    parser.add_argument(
        "--confirm-locked-evaluation",
        action="store_true",
        help="Required to open test and held-out shift after the validation freeze passes.",
    )
    parser.add_argument(
        "--require-evaluation",
        action="store_true",
        help="In validate mode, also require and hash-check locked evaluation outputs.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise PlanBError("--limit must be positive")
    if args.limit is not None and args.mode not in {"preflight", "audit"}:
        raise PlanBError("--limit is allowed only for preflight or audit")

    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else (repo_root / args.config)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir)
    config = load_config(config_path.resolve())

    if args.mode == "preflight":
        records, inventory = load_and_verify_inputs(repo_root, config, limit=args.limit)
        _print(
            {
                "mode": "preflight",
                "is_valid": True,
                "device": args.device,
                "new_mllm_calls": 0,
                "loaded_records": len(records),
                "scientific_run": args.limit is None,
                "input_inventory": inventory,
            }
        )
        return 0

    if args.mode == "audit":
        report = run_audit(
            repo_root=repo_root,
            config=config,
            output_dir=output_dir.resolve(),
            resume=args.resume,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        _print(report)
        return 0 if report.get("is_valid") or args.limit is not None else 1

    if args.mode == "prepare":
        freeze = run_prepare(
            repo_root=repo_root,
            config=config,
            output_dir=output_dir.resolve(),
            resume=args.resume,
            overwrite=args.overwrite,
        )
        _print(freeze)
        return 0 if freeze.get("status") == "FROZEN_VALIDATION_GATE_PASSED" else 2

    if args.mode == "evaluate":
        report = run_evaluate(
            repo_root=repo_root,
            config=config,
            output_dir=output_dir.resolve(),
            resume=args.resume,
            overwrite=args.overwrite,
            confirm_locked_evaluation=args.confirm_locked_evaluation,
        )
        _print(report)
        # A scientifically negative result is still a successfully completed run.
        return 0

    report = validate_outputs(output_dir.resolve(), args.require_evaluation)
    _print(report)
    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlanBError as exc:
        print(f"PLAN_B_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
