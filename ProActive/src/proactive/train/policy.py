"""Hash-bound Week 6 policy data, training, and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from proactive.networks.losses import policy_loss
from proactive.train.state_data import ACTION_ORDER
from proactive.train.vectorized import TensorStateDataset, project_model_input
from proactive.utils.hashing import hash_dict
from proactive.utils.io import file_sha256, iter_jsonl


VOI_MANIFEST_VERSION = "week6_voi_manifest_v1"


class VOITargetDataset(Dataset):
    def __init__(
        self,
        base: TensorStateDataset,
        records: Sequence[Mapping[str, Any]],
        *,
        split: str,
        cost_multiplier: float,
        target_kind: str = "voi",
        limit: int | None = None,
    ) -> None:
        if base.split != split or split not in {"train", "val"}:
            raise ValueError("VOI policy data are restricted to train/val")
        multiplier_key = format(float(cost_multiplier), ".12g")
        if target_kind not in {"voi", "entropy_reduction", "loss_only_voi"}:
            raise ValueError(
                "target_kind must be 'voi', 'entropy_reduction', or 'loss_only_voi'"
            )
        by_state = {state_id: index for index, state_id in enumerate(base.state_ids)}
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            if record.get("record_type") != "voi_target_multi_cost_v1":
                raise ValueError("VOI record version mismatch")
            expected_hash = record.get("record_sha256")
            unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
            if expected_hash != hash_dict(unsigned):
                raise ValueError("VOI record self-hash mismatch")
            if record.get("metadata", {}).get("split") != split:
                raise ValueError("VOI file mixes or mislabels splits")
            state_id = record.get("state_id")
            if state_id not in by_state:
                raise ValueError(f"VOI state is absent from vectorized {split}: {state_id}")
            if target_kind == "voi":
                target = record.get("targets_by_cost_multiplier", {}).get(multiplier_key)
                if not isinstance(target, Mapping):
                    raise ValueError(f"VOI record has no target for multiplier {multiplier_key}")
            elif target_kind == "entropy_reduction":
                components = record.get("counterfactual_components")
                if not isinstance(components, Mapping):
                    raise ValueError("VOI record lacks entropy-reduction components")
                entropy_values = {
                    action: float(component["entropy_reduction"])
                    for action, component in components.items()
                }
                positives = [action for action, value in entropy_values.items() if value > 0]
                target = {
                    "realized_voi": {"stop": 0.0, **entropy_values},
                    "best_action": (
                        max(positives, key=lambda action: (entropy_values[action], -ACTION_ORDER.index(action)))
                        if positives
                        else "stop"
                    ),
                }
            else:
                components = record.get("counterfactual_components")
                if not isinstance(components, Mapping):
                    raise ValueError("VOI record lacks diagnostic-loss components")
                loss_values = {
                    action: float(component["diagnostic_loss_reduction"])
                    for action, component in components.items()
                }
                positives = [action for action, value in loss_values.items() if value > 0]
                target = {
                    "realized_voi": {"stop": 0.0, **loss_values},
                    "best_action": (
                        max(
                            positives,
                            key=lambda action: (
                                loss_values[action],
                                -ACTION_ORDER.index(action),
                            ),
                        )
                        if positives
                        else "stop"
                    ),
                }
            identity = str(record.get("voi_id"))
            if identity in seen:
                raise ValueError(f"Duplicate VOI target: {identity}")
            seen.add(identity)
            values = target.get("realized_voi")
            legal = record.get("legal_actions")
            if not isinstance(values, Mapping) or not isinstance(legal, list):
                raise ValueError("Malformed VOI target values/legal actions")
            if set(values) != set(legal) or "stop" not in values or float(values["stop"]) != 0.0:
                raise ValueError("VOI target action coverage or STOP value is invalid")
            vector = torch.zeros(len(ACTION_ORDER), dtype=torch.float32)
            mask = torch.zeros(len(ACTION_ORDER), dtype=torch.bool)
            for action in legal:
                if action not in ACTION_ORDER:
                    raise ValueError(f"Unknown VOI action: {action}")
                index = ACTION_ORDER.index(action)
                value = float(values[action])
                if not np.isfinite(value):
                    raise ValueError("VOI target must be finite")
                vector[index] = value
                mask[index] = True
            base_index = by_state[state_id]
            budget = int(record["max_budget"])
            projected = project_model_input(base[base_index]["model_input"], budget)
            if not torch.equal(projected["action_mask"].bool(), mask):
                raise ValueError(f"VOI legal mask disagrees with state tensor: {identity}")
            rows.append(
                {
                    "base_index": base_index,
                    "budget": budget,
                    "target_values": vector,
                    "legal_mask": mask,
                    "best_action": ACTION_ORDER.index(str(target["best_action"])),
                    "voi_id": identity,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
        if not rows:
            raise ValueError(f"No VOI targets loaded for split {split}")
        self.base = base
        self.rows = rows
        self.split = split
        self.cost_multiplier = float(cost_multiplier)
        self.target_kind = target_kind

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.rows[index]
        base = self.base[item["base_index"]]
        return {
            "model_input": project_model_input(base["model_input"], item["budget"]),
            "target_values": item["target_values"],
            "legal_mask": item["legal_mask"],
            "best_action": item["best_action"],
            "voi_id": item["voi_id"],
            "audit_metadata": base["audit_metadata"],
        }


def load_voi_records(
    manifest_path: str | Path,
    split: str,
    *,
    require_complete: bool = True,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != VOI_MANIFEST_VERSION:
        raise ValueError("VOI manifest version mismatch")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != hash_dict(unsigned):
        raise ValueError("VOI manifest self-hash mismatch")
    if require_complete and manifest.get("status") != "COMPLETE":
        raise ValueError("Full policy training requires a COMPLETE VOI manifest")
    if split not in {"train", "val"}:
        raise ValueError("VOI loader refuses calibration/test splits")
    records: List[Dict[str, Any]] = []
    for entry in manifest.get("files", []):
        if entry.get("split") != split:
            continue
        path = manifest_path.parent / entry["path"]
        if not path.exists() or file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"VOI artifact hash mismatch: {path}")
        rows = list(iter_jsonl(path))
        if len(rows) != int(entry["row_count"]):
            raise ValueError(f"VOI artifact row-count mismatch: {path}")
        records.extend(rows)
    if not records:
        raise ValueError(f"VOI manifest has no {split} records")
    return manifest, records


def policy_epoch(
    model: Any,
    loader: Any,
    *,
    optimizer: torch.optim.Optimizer | None,
    normalizer: Any,
    device: torch.device,
    margin: float,
    mse_weight: float,
    action_ce_weight: float,
    gradient_clip_norm: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "ranking": 0.0, "mse": 0.0, "action_ce": 0.0}
    correct = 0
    stop_predictions = 0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            model_input = {key: value.to(device) for key, value in batch["model_input"].items()}
            model_input = normalizer.transform(model_input)
            targets = batch["target_values"].to(device)
            legal = batch["legal_mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predicted = model(model_input)
            losses = policy_loss(
                predicted,
                targets,
                legal,
                margin=margin,
                mse_weight=mse_weight,
                action_ce_weight=action_ce_weight,
            )
            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.voi_head.parameters(), gradient_clip_norm)
                optimizer.step()
            masked = predicted.masked_fill(~legal, float("-inf"))
            best_non_stop, best_index = masked[:, :-1].max(dim=1)
            chosen = torch.where(
                best_non_stop > 0,
                best_index,
                torch.full_like(best_index, len(ACTION_ORDER) - 1),
            )
            target_masked = targets.masked_fill(~legal, float("-inf"))
            target_non_stop, target_index = target_masked[:, :-1].max(dim=1)
            target_best = torch.where(
                target_non_stop > 0,
                target_index,
                torch.full_like(target_index, len(ACTION_ORDER) - 1),
            )
            batch_count = int(targets.shape[0])
            count += batch_count
            correct += int((chosen == target_best).sum().cpu())
            stop_predictions += int((chosen == len(ACTION_ORDER) - 1).sum().cpu())
            for name in totals:
                totals[name] += float(getattr(losses, name).detach().cpu()) * batch_count
    if count == 0:
        raise ValueError("Policy loader produced no rows")
    return {
        **{key: value / count for key, value in totals.items()},
        "best_action_accuracy": correct / count,
        "stop_prediction_rate": stop_predictions / count,
        "row_count": count,
    }
