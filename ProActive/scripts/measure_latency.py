#!/usr/bin/env python3
"""Measure controller overhead and cached end-to-end latency on fixed hardware."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import yaml

from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.voi import ActionConditionedVOIHead, FrozenDiagnosticPolicy
from proactive.train.checkpoints import POLICY_CHECKPOINT_VERSION, load_checkpoint, validate_freeze_manifest
from proactive.train.state_data import FeatureNormalizer
from proactive.train.vectorized import load_vectorized_split
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/week8_evaluation.yaml")
    parser.add_argument("--trajectory_path", required=True)
    parser.add_argument("--teacher_path", required=True)
    parser.add_argument("--freeze_manifest", default="outputs/week7_frozen/main_stack_freeze.json")
    parser.add_argument("--vector_manifest", default="outputs/week5_data/vectorized_manifest.json")
    parser.add_argument("--output_dir", default="outputs/week8_reports")
    parser.add_argument("--condition", default="proactive")
    parser.add_argument(
        "--split",
        choices=("train", "val", "cal", "test", "shift"),
        default="test",
        help="Vector split used for controller timing (Week 8 ablations use val)",
    )
    parser.add_argument("--budget", type=int, default=7)
    parser.add_argument("--target_coverage", type=float, default=0.90)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _teacher_index(path: Path) -> Dict[tuple[str, str], Mapping[str, Any]]:
    result = {}
    files = [path] if path.is_file() else sorted(path.glob("teacher_*.jsonl"))
    for source in files:
        if source.name.endswith(".failures.jsonl"):
            continue
        for row in iter_jsonl(source):
            key = (str(row["model_id"]), str(row["instance_id"]))
            if key in result:
                raise SystemExit(f"Duplicate teacher identity: {key}")
            result[key] = row
    if not result:
        raise SystemExit("No teacher records found")
    return result


def _percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    latency = config["latency"]
    warmup = int(latency["warmup_examples"])
    measured = int(latency["measured_examples"])
    if warmup < 1 or measured < 1 or latency.get("synchronize_cuda") is not True:
        raise SystemExit("Invalid Week 8 latency protocol")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("Fixed-hardware Week 8 latency requires an available CUDA device")

    freeze_path = Path(args.freeze_manifest)
    freeze = validate_freeze_manifest(freeze_path, require_policy=True)
    diagnostic_path = Path(freeze["artifacts"]["diagnostic_checkpoint"]["path"])
    policy_path = Path(freeze["artifacts"]["policy_checkpoint"]["path"])
    diagnostic = load_checkpoint(diagnostic_path, map_location="cpu")
    policy = load_checkpoint(policy_path, expected_version=POLICY_CHECKPOINT_VERSION, map_location="cpu")
    if policy.get("diagnostic_checkpoint_sha256") != file_sha256(diagnostic_path):
        raise SystemExit("Frozen policy/diagnostic mismatch")

    trajectory_path = Path(args.trajectory_path)
    selected = []
    for row in iter_jsonl(trajectory_path):
        if (
            row.get("condition") == args.condition
            and int(row.get("maximum_budget")) == args.budget
            and float(row.get("target_coverage")) == args.target_coverage
        ):
            selected.append(row)
    selected.sort(key=lambda row: hash_dict({"seed": config["seed"], "state_id": row["state_id"]}))
    if len(selected) < measured:
        raise SystemExit(f"Latency protocol needs {measured} rows, found {len(selected)}")
    selected = selected[:measured]
    teacher = _teacher_index(Path(args.teacher_path))

    generation_rows = []
    for row in selected:
        metadata = row["metadata"]
        key = (str(metadata["model_id"]), str(metadata["instance_id"]))
        source = teacher.get(key)
        if source is None:
            raise SystemExit(f"Missing teacher row for latency: {key}")
        clean_ms = source.get("clean", {}).get("latency_ms")
        if not isinstance(clean_ms, (int, float)) or clean_ms <= 0:
            raise SystemExit(f"Invalid clean latency for {key}")
        acquired_ms = 0.0
        for action in row["actions"]:
            if action == "stop":
                continue
            value = source.get("probes", {}).get(action, {}).get("latency_ms")
            if not isinstance(value, (int, float)) or value <= 0:
                raise SystemExit(f"Invalid probe latency for {key}/{action}")
            acquired_ms += float(value)
        total_ms = float(clean_ms) + acquired_ms
        generation_rows.append(
            {
                "model_id": key[0],
                "instance_id": key[1],
                "clean_latency_ms": float(clean_ms),
                "acquired_probe_latency_ms": acquired_ms,
                "total_generation_latency_ms": total_ms,
                "normalized_latency": total_ms / float(clean_ms),
                "additional_passes": int(row["acquisition_cost"]),
            }
        )

    if args.dry_run:
        print(json.dumps({"is_valid": True, "measured_rows": len(generation_rows)}, indent=2))
        return

    device = torch.device(args.device)
    diagnostic_model = build_diagnostic_model(
        diagnostic["encoder_name"], diagnostic["architecture"]
    )
    diagnostic_model.load_state_dict(diagnostic["model_state_dict"])
    policy_architecture = policy["policy_architecture"]
    voi_head = ActionConditionedVOIHead(
        state_dim=int(diagnostic["architecture"]["state_dim"]),
        action_embedding_dim=int(policy_architecture["action_embedding_dim"]),
        budget_embedding_dim=int(policy_architecture["budget_embedding_dim"]),
        hidden_dim=int(policy_architecture["hidden_dim"]),
        dropout=float(policy_architecture["dropout"]),
        max_budget=int(diagnostic["architecture"]["max_budget"]),
    )
    voi_head.load_state_dict(policy["model_state_dict"])
    controller = FrozenDiagnosticPolicy(diagnostic_model, voi_head).to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(diagnostic["normalizer"])
    vector = load_vectorized_split(Path(args.vector_manifest), args.split)
    if len(vector) < warmup + measured:
        raise SystemExit(f"Vectorized {args.split} split is too small for latency protocol")

    def infer(index: int) -> None:
        row = vector[index]
        batch = {key: value.unsqueeze(0).to(device) for key, value in row["model_input"].items()}
        with torch.no_grad():
            # Time the complete learned controller: diagnostic state encoder
            # plus the action-conditioned VOI head.  Timing only the
            # diagnostic encoder would understate policy overhead.
            controller(normalizer.transform(batch))

    for index in range(warmup):
        infer(index)
    torch.cuda.synchronize(device)
    controller_ms = []
    for index in range(warmup, warmup + measured):
        start = time.perf_counter_ns()
        infer(index)
        torch.cuda.synchronize(device)
        controller_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    properties = torch.cuda.get_device_properties(device)
    hardware = {
        "device": str(device),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "latency.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(generation_rows[0]))
        writer.writeheader()
        writer.writerows(generation_rows)
    report: Dict[str, Any] = {
        "format_version": "week8_latency_v1",
        "is_valid": True,
        "condition": args.condition,
        "evaluation_split": args.split,
        "budget": args.budget,
        "target_coverage": args.target_coverage,
        "warmup_examples": warmup,
        "measured_examples": measured,
        "cuda_synchronized": True,
        "hardware": hardware,
        "controller_latency_ms": {
            "mean": statistics.fmean(controller_ms),
            "median": statistics.median(controller_ms),
            "p95": _percentile(controller_ms, 0.95),
        },
        "controller_timed_components": [
            "diagnostic_state_encoder",
            "action_conditioned_voi_head",
        ],
        "generation_latency_source": "cached_per_forward_pass_measurements",
        "generation_latency_note": (
            "Probe/clean times were recorded during the cached MLLM generations; "
            "the controller overhead was remeasured on the hardware above."
        ),
        "normalized_latency_mean": statistics.fmean(row["normalized_latency"] for row in generation_rows),
        "total_generation_latency_ms_mean": statistics.fmean(row["total_generation_latency_ms"] for row in generation_rows),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "trajectory_sha256": file_sha256(trajectory_path),
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
    }
    report["report_sha256"] = hash_dict(report)
    report_path = output_dir / "latency_report.json"
    write_json(report, report_path, overwrite=args.overwrite and report_path.exists())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
