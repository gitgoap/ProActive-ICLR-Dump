#!/usr/bin/env python3
"""Build exact cached counterfactual VOI targets without touching cal/test."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import yaml

from proactive.conformal.aps import prediction_sets
from proactive.networks.diagnostic import build_diagnostic_model
from proactive.networks.losses import diagnostic_loss
from proactive.train.checkpoints import (
    freeze_includes_file,
    load_checkpoint,
    validate_freeze_manifest,
)
from proactive.train.ablations import FEATURE_ABLATIONS, apply_state_record_ablation
from proactive.train.state_data import FeatureNormalizer, collate_vectorized_states, vectorize_state
from proactive.train.voi import (
    add_cached_observation,
    build_multi_cost_voi_record,
    legal_non_stop_actions,
)
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl, write_json, write_jsonl


LOGGER = logging.getLogger("build_voi_targets")
ALLOWED_SPLITS = {"train", "val"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/policy_train.yaml")
    parser.add_argument("--manifest_path", required=True, help="Week 5 vectorized manifest")
    parser.add_argument("--state_manifest", default="outputs/week4_reports/final/state_manifest.json")
    parser.add_argument(
        "--state_dir",
        default=None,
        help="Override the diagnostic config's state directory (used by leakage-safe LOMO folds)",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--freeze_manifest", default=None)
    parser.add_argument("--temporary_aps", required=True)
    parser.add_argument("--teacher_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inference_batch_size", type=int, default=4096)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_unapproved_pilot", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def _thresholds(report: Mapping[str, Any], coverage: float, budgets: Sequence[int]) -> Dict[int, float]:
    if report.get("format_version") != "temporary_validation_aps_v1":
        raise SystemExit("VOI construction requires Week 5 temporary APS")
    if report.get("fit_split") != "val" or report.get("calibration_split_used") is not False or report.get("test_split_used") is not False:
        raise SystemExit("Temporary APS split provenance is unsafe")
    key = format(float(coverage), ".12g")
    result: Dict[int, float] = {}
    for budget in budgets:
        budget_block = report.get("thresholds", {}).get(str(budget))
        item = budget_block.get(key) if isinstance(budget_block, Mapping) else None
        if not isinstance(item, Mapping) or float(item.get("target_coverage", -1)) != float(coverage):
            raise SystemExit(f"Temporary APS has no {coverage} threshold for budget {budget}")
        result[budget] = float(item["threshold"])
    return result


@torch.no_grad()
def _evaluate_vectors(
    vectors: Sequence[Any],
    *,
    model: Any,
    normalizer: FeatureNormalizer,
    checkpoint: Mapping[str, Any],
    diagnostic_config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    six_probabilities: List[torch.Tensor] = []
    losses: List[torch.Tensor] = []
    for start in range(0, len(vectors), batch_size):
        batch = collate_vectorized_states(vectors[start : start + batch_size])
        model_input = {key: value.to(device) for key, value in batch["model_input"].items()}
        targets = {key: value.to(device) for key, value in batch["targets"].items()}
        output = model(normalizer.transform(model_input))
        per_row = diagnostic_loss(
            output,
            targets,
            bit_pos_weight=checkpoint["bit_pos_weight"],
            class_weight=checkpoint["class_weight"],
            six_way_weight=float(diagnostic_config["loss"]["six_way_weight"]),
            signature_weight=float(diagnostic_config["loss"]["signature_weight"]),
            reduction="none",
        )
        six_probabilities.append(torch.softmax(output.six_way_logits, dim=-1).cpu())
        losses.append(per_row.total.cpu())
    return {
        "six_way_probabilities": torch.cat(six_probabilities).numpy(),
        "diagnostic_loss": torch.cat(losses).numpy(),
    }


def _teacher_index(path: Path) -> Dict[tuple[str, str], Mapping[str, Any]]:
    result: Dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (row.get("model_id"), row.get("instance_id"))
        if not all(isinstance(value, str) and value for value in key):
            raise SystemExit(f"Malformed teacher identity in {path}")
        if key in result:
            raise SystemExit(f"Duplicate teacher identity in {path}: {key}")
        if row.get("valid") is not True:
            raise SystemExit(f"Invalid teacher row in recovered source: {key}")
        result[key] = row
    if not result:
        raise SystemExit(f"Empty teacher source: {path}")
    return result


def _jobs_for_records(
    records: Sequence[Mapping[str, Any]],
    *,
    teacher: Mapping[tuple[str, str], Mapping[str, Any]],
    budgets: Sequence[int],
    ablation_id: str | None = None,
    max_jobs: int | None = None,
) -> tuple[List[Any], List[Dict[str, Any]]]:
    vectors: List[Any] = []
    jobs: List[Dict[str, Any]] = []
    for source_state in records:
        state = (
            apply_state_record_ablation(source_state, ablation_id)
            if ablation_id is not None
            else dict(source_state)
        )
        if state is None:
            continue
        split = state.get("metadata", {}).get("split")
        if split not in ALLOWED_SPLITS:
            continue
        names = state.get("learner_input", {}).get("acquired_probe_names")
        if not isinstance(names, list):
            raise SystemExit("Malformed acquired probe names in state source")
        key = (state["metadata"]["model_id"], state["metadata"]["instance_id"])
        source_teacher = teacher.get(key)
        if source_teacher is None:
            raise SystemExit(f"Missing teacher row for state identity {key}")
        for budget in budgets:
            if len(names) > budget:
                continue
            current = vectorize_state(state, max_budget=budget)
            current_index = len(vectors)
            vectors.append(current)
            action_indices: Dict[str, int] = {}
            for action in legal_non_stop_actions(current):
                counterfactual = add_cached_observation(state, source_teacher, action)
                if ablation_id is not None:
                    counterfactual = apply_state_record_ablation(counterfactual, ablation_id)
                    if counterfactual is None:
                        raise SystemExit(
                            "A legal ablation counterfactual became unsupported"
                        )
                action_indices[action] = len(vectors)
                vectors.append(vectorize_state(counterfactual, max_budget=budget))
            jobs.append(
                {
                    "state": state,
                    "budget": budget,
                    "current_index": current_index,
                    "action_indices": action_indices,
                }
            )
            if max_jobs is not None and len(jobs) >= max_jobs:
                return vectors, jobs
    return vectors, jobs


def _records_for_source(
    state_path: Path,
    teacher_path: Path,
    *,
    budgets: Sequence[int],
    thresholds: Mapping[int, float],
    cost_multipliers: Sequence[float],
    eta_loss: float,
    model: Any,
    normalizer: FeatureNormalizer,
    checkpoint: Mapping[str, Any],
    diagnostic_config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    remaining_limit: int | None,
    provenance: Mapping[str, str],
    ablation_id: str | None = None,
) -> tuple[Dict[str, List[Dict[str, Any]]], int]:
    teacher = _teacher_index(teacher_path)
    state_records = list(iter_jsonl(state_path))
    vectors, jobs = _jobs_for_records(
        state_records,
        teacher=teacher,
        budgets=budgets,
        ablation_id=ablation_id,
        max_jobs=remaining_limit,
    )
    predictions = _evaluate_vectors(
        vectors,
        model=model,
        normalizer=normalizer,
        checkpoint=checkpoint,
        diagnostic_config=diagnostic_config,
        device=device,
        batch_size=batch_size,
    )
    output: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    produced = 0
    for job in jobs:
        if remaining_limit is not None and produced >= remaining_limit:
            break
        current_index = job["current_index"]
        budget = int(job["budget"])
        threshold = thresholds[budget]
        current_probability = predictions["six_way_probabilities"][current_index]
        current_entropy = float(
            -np.sum(current_probability * np.log(np.clip(current_probability, 1.0e-12, 1.0)))
        )
        current_set_size = len(prediction_sets([current_probability], threshold)[0])
        action_results: Dict[str, Dict[str, float]] = {}
        for action, index in job["action_indices"].items():
            probability = predictions["six_way_probabilities"][index]
            action_results[action] = {
                "set_size": len(prediction_sets([probability], threshold)[0]),
                "diagnostic_loss": float(predictions["diagnostic_loss"][index]),
                "entropy": float(
                    -np.sum(probability * np.log(np.clip(probability, 1.0e-12, 1.0)))
                ),
            }
        record = build_multi_cost_voi_record(
            state_record=job["state"],
            max_budget=budget,
            current_loss=float(predictions["diagnostic_loss"][current_index]),
            current_set_size=current_set_size,
            current_entropy=current_entropy,
            action_results=action_results,
            eta_loss=eta_loss,
            cost_multipliers=cost_multipliers,
            provenance=provenance,
        )
        output[record["metadata"]["split"]].append(record)
        produced += 1
    return dict(output), produced


def _validate_resume(path: Path, expected: Mapping[str, Any]) -> None:
    manifest = _read_json(path)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SystemExit(f"VOI resume refused: {key} drift")
    for entry in manifest.get("files", []):
        artifact = path.parent / entry["path"]
        if not artifact.exists() or file_sha256(artifact) != entry.get("sha256"):
            raise SystemExit(f"VOI resume refused: artifact drift at {artifact}")
        if sum(1 for _ in iter_jsonl(artifact)) != int(entry["row_count"]):
            raise SystemExit(f"VOI resume refused: row-count drift at {artifact}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.inference_batch_size <= 0:
        raise SystemExit("--inference_batch_size must be positive")
    config_path = Path(args.config)
    vector_manifest_path = Path(args.manifest_path)
    state_manifest_path = Path(args.state_manifest)
    checkpoint_path = Path(args.checkpoint)
    aps_path = Path(args.temporary_aps)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    approval = config.get("metadata", {}).get("approval_status")
    pilot = args.limit is not None and args.limit <= 100
    if approval != "APPROVED" and not args.dry_run and not (pilot and args.allow_unapproved_pilot):
        raise SystemExit("Week 6 settings are not APPROVED; only dry-run or explicit <=100-row pilot is allowed")
    if config["splits"] != {"train": "train", "selection": "val", "forbidden_during_week6": ["cal", "test"]}:
        raise SystemExit("Week 6 split firewall config drift")
    freeze_path = Path(args.freeze_manifest or config["week5_freeze_manifest"])
    freeze = validate_freeze_manifest(freeze_path, require_policy=False)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if not freeze_includes_file(
        freeze,
        checkpoint_path,
        artifact_name_prefix="diagnostic_checkpoint",
    ):
        raise SystemExit("VOI checkpoint is not included in the Week 5 diagnostic freeze")
    aps = _read_json(aps_path)
    if aps.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise SystemExit("Temporary APS/checkpoint hash mismatch")
    vector_manifest = _read_json(vector_manifest_path)
    ablation_id = vector_manifest.get("ablation_id")
    if ablation_id is not None and ablation_id not in FEATURE_ABLATIONS:
        raise SystemExit(f"Unsupported vector-manifest ablation_id: {ablation_id}")
    if vector_manifest.get("status") != "COMPLETE" and not pilot:
        raise SystemExit("Full VOI construction requires COMPLETE Week 5 vectors")
    if checkpoint.get("source_manifest_sha256") != file_sha256(vector_manifest_path):
        raise SystemExit("Diagnostic checkpoint/vectorized-manifest drift")
    diagnostic_config_path = Path(config["diagnostic_config"])
    with open(diagnostic_config_path, "r", encoding="utf-8") as handle:
        diagnostic_config = yaml.safe_load(handle)
    budgets = [int(value) for value in config["budgets"]]
    cost_multipliers = [float(value) for value in config["cost_multiplier_grid"]]
    coverage = float(config["temporary_aps_target_coverage"])
    thresholds = _thresholds(aps, coverage, budgets)
    teacher_dir = Path(args.teacher_path or config["teacher_dir"])
    output_dir = Path(args.output_dir or config["outputs"]["voi_dir"])
    state_manifest = _read_json(state_manifest_path)
    state_entries = state_manifest.get("files")
    if not isinstance(state_entries, list) or not state_entries:
        raise SystemExit("State manifest contains no files")
    source_state_dir = Path(args.state_dir or diagnostic_config["source_state_dir"])
    source_specs = []
    for entry in state_entries:
        state_path = source_state_dir / Path(entry["path"]).name
        teacher_name = state_path.name.replace("states_", "teacher_", 1)
        teacher_path = teacher_dir / teacher_name
        if not state_path.exists() or file_sha256(state_path) != entry.get("sha256"):
            raise SystemExit(f"State source hash mismatch: {state_path}")
        if not teacher_path.exists():
            raise SystemExit(f"Recovered teacher source missing: {teacher_path}")
        source_specs.append((state_path, teacher_path))
    final_manifest_path = output_dir / "voi_manifest.json"
    expected = {
        "format_version": "week6_voi_manifest_v1",
        "config_sha256": file_sha256(config_path),
        "vector_manifest_sha256": file_sha256(vector_manifest_path),
        "state_manifest_sha256": file_sha256(state_manifest_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "temporary_aps_sha256": file_sha256(aps_path),
        "ablation_id": ablation_id,
        "limit": args.limit,
    }
    if final_manifest_path.exists() and args.resume:
        _validate_resume(final_manifest_path, expected)
        LOGGER.info("Validated complete VOI output; skipping")
        return
    if final_manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {final_manifest_path}; use --resume or --overwrite")
    if args.dry_run:
        print(json.dumps({**expected, "source_files": len(source_specs), "budgets": budgets, "cost_multipliers": cost_multipliers, "output": str(final_manifest_path)}, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA requested but unavailable: {args.device}")
    model = build_diagnostic_model(checkpoint["encoder_name"], checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    provenance = {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "temporary_aps_sha256": file_sha256(aps_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "temporary_aps_fit_split": "val",
        "target_coverage": format(coverage, ".12g"),
        "ablation_id": str(ablation_id or "none"),
    }
    output_entries: List[Dict[str, Any]] = []
    sign_counts: Dict[str, Counter[str]] = {format(value, ".12g"): Counter() for value in cost_multipliers}
    best_counts: Dict[str, Counter[str]] = {format(value, ".12g"): Counter() for value in cost_multipliers}
    produced_total = 0
    progress_path = output_dir / "voi_progress.json"
    processed_sources: set[str] = set()

    def account_records(records: Iterable[Mapping[str, Any]]) -> None:
        for record in records:
            for multiplier, target in record["targets_by_cost_multiplier"].items():
                best_counts[multiplier][target["best_action"]] += 1
                for value in target["realized_voi"].values():
                    sign_counts[multiplier]["positive" if value > 0 else "negative" if value < 0 else "zero"] += 1

    if args.resume and progress_path.exists():
        progress = _read_json(progress_path)
        for key, value in expected.items():
            if progress.get(key) != value:
                raise SystemExit(f"VOI progress resume refused: {key} drift")
        output_entries = list(progress.get("files", []))
        processed_sources = set(progress.get("processed_sources", []))
        produced_total = int(progress.get("row_count", 0))
        for entry in output_entries:
            artifact = output_dir / entry["path"]
            if not artifact.exists() or file_sha256(artifact) != entry.get("sha256"):
                raise SystemExit(f"VOI progress artifact drift: {artifact}")
            account_records(iter_jsonl(artifact))
        LOGGER.info("Resuming after %s completed state sources", len(processed_sources))

    remaining = None if args.limit is None else args.limit - produced_total
    for state_path, teacher_path in source_specs:
        if remaining is not None and remaining <= 0:
            break
        if str(state_path) in processed_sources:
            continue
        split_records, produced = _records_for_source(
            state_path,
            teacher_path,
            budgets=budgets,
            thresholds=thresholds,
            cost_multipliers=cost_multipliers,
            eta_loss=float(config["eta_loss"]),
            model=model,
            normalizer=normalizer,
            checkpoint=checkpoint,
            diagnostic_config=diagnostic_config,
            device=device,
            batch_size=args.inference_batch_size,
            remaining_limit=remaining,
            provenance=provenance,
            ablation_id=ablation_id,
        )
        for split, records in sorted(split_records.items()):
            output_name = f"{state_path.stem}__{split}.jsonl"
            output_path = output_dir / output_name
            write_jsonl(records, output_path, overwrite=(args.overwrite or args.resume) and output_path.exists())
            output_entries.append(
                {
                    "path": output_name,
                    "split": split,
                    "row_count": len(records),
                    "sha256": file_sha256(output_path),
                    "source_state_path": str(state_path),
                    "source_state_sha256": file_sha256(state_path),
                    "source_teacher_path": str(teacher_path),
                    "source_teacher_sha256": file_sha256(teacher_path),
                }
            )
            account_records(records)
        produced_total += produced
        processed_sources.add(str(state_path))
        progress = {
            **expected,
            "status": "IN_PROGRESS",
            "processed_sources": sorted(processed_sources),
            "row_count": produced_total,
            "files": output_entries,
        }
        progress["progress_sha256"] = hash_dict(progress)
        write_json(progress, progress_path, overwrite=progress_path.exists())
        if remaining is not None:
            remaining -= produced
        LOGGER.info("Processed %s target rows from %s", produced, state_path)
    result: Dict[str, Any] = {
        **expected,
        "status": "PILOT" if args.limit is not None else "COMPLETE",
        "fit_splits": ["train", "val"],
        "calibration_used": False,
        "test_used": False,
        "budgets": budgets,
        "eta_loss": float(config["eta_loss"]),
        "cost_multipliers": cost_multipliers,
        "target_coverage": coverage,
        "row_count": produced_total,
        "split_counts": {
            split: sum(entry["row_count"] for entry in output_entries if entry["split"] == split)
            for split in sorted(ALLOWED_SPLITS)
        },
        "sign_counts": {key: dict(value) for key, value in sign_counts.items()},
        "best_action_counts": {key: dict(value) for key, value in best_counts.items()},
        "files": output_entries,
    }
    result["manifest_sha256"] = hash_dict(result)
    write_json(result, final_manifest_path, overwrite=args.overwrite and final_manifest_path.exists())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
