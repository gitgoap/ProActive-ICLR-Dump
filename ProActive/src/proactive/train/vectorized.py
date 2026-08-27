"""Resume-safe vectorized state shards and budget projection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from proactive.train.state_data import ACTION_ORDER
from proactive.utils.io import file_sha256


TENSOR_SHARD_VERSION = "week5_tensor_shard_v1"
VECTOR_MANIFEST_VERSION = "week5_vector_manifest_v1"


def atomic_torch_save(value: Any, path: str | Path, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.stem + "_", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return path


def stack_vectorized_rows(rows: Sequence[Any], split: str) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot build an empty tensor shard")
    if any(row.metadata["split"] != split for row in rows):
        raise ValueError("Tensor shard mixes data splits")
    input_keys = tuple(rows[0].model_input)
    target_keys = tuple(rows[0].targets)
    return {
        "format_version": TENSOR_SHARD_VERSION,
        "split": split,
        "row_count": len(rows),
        "model_input": {
            key: torch.stack([row.model_input[key] for row in rows]) for key in input_keys
        },
        "targets": {
            key: torch.stack([row.targets[key] for row in rows]) for key in target_keys
        },
        "audit_metadata": {
            key: [row.metadata[key] for row in rows]
            for key in ("instance_id", "group_id", "dataset", "split", "model_id", "model_revision")
        },
        "state_id": [row.state_id for row in rows],
        "sampling_sources": [list(row.sampling_sources) for row in rows],
    }


def validate_tensor_shard(shard: Mapping[str, Any], expected_split: str | None = None) -> None:
    expected = {
        "format_version",
        "split",
        "row_count",
        "model_input",
        "targets",
        "audit_metadata",
        "state_id",
        "sampling_sources",
    }
    if set(shard) != expected or shard.get("format_version") != TENSOR_SHARD_VERSION:
        raise ValueError("Tensor-shard schema/version mismatch")
    split = shard.get("split")
    if expected_split is not None and split != expected_split:
        raise ValueError(f"Expected split {expected_split!r}, found {split!r}")
    count = shard.get("row_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Tensor shard row_count must be positive")
    model_input = shard.get("model_input")
    targets = shard.get("targets")
    if not isinstance(model_input, Mapping) or not isinstance(targets, Mapping):
        raise ValueError("Tensor shard input/target blocks must be mappings")
    for name, value in {**model_input, **targets}.items():
        if not isinstance(value, torch.Tensor) or value.shape[0] != count:
            raise ValueError(f"Tensor-shard field {name} has an invalid leading dimension")
    metadata = shard.get("audit_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("audit_metadata must be a mapping")
    if any(not isinstance(value, list) or len(value) != count for value in metadata.values()):
        raise ValueError("Tensor-shard audit metadata length mismatch")
    state_ids = shard.get("state_id")
    if not isinstance(state_ids, list) or len(state_ids) != count or len(state_ids) != len(set(state_ids)):
        raise ValueError("Tensor-shard state IDs are missing or duplicated")


class TensorStateDataset(Dataset):
    """Concatenate validated tensor shards without exposing metadata as input."""

    def __init__(self, shards: Sequence[Mapping[str, Any]], expected_split: str) -> None:
        if not shards:
            raise ValueError(f"No vectorized shards found for split {expected_split!r}")
        for shard in shards:
            validate_tensor_shard(shard, expected_split)
        self.split = expected_split
        self.model_input = {
            key: torch.cat([shard["model_input"][key] for shard in shards], dim=0)
            for key in shards[0]["model_input"]
        }
        self.targets = {
            key: torch.cat([shard["targets"][key] for shard in shards], dim=0)
            for key in shards[0]["targets"]
        }
        self.audit_metadata = {
            key: [item for shard in shards for item in shard["audit_metadata"][key]]
            for key in shards[0]["audit_metadata"]
        }
        self.state_ids = [item for shard in shards for item in shard["state_id"]]
        if len(self.state_ids) != len(set(self.state_ids)):
            raise ValueError(f"Duplicate state_id across {expected_split} tensor shards")

    def __len__(self) -> int:
        return len(self.state_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "model_input": {key: value[index] for key, value in self.model_input.items()},
            "targets": {key: value[index] for key, value in self.targets.items()},
            "audit_metadata": {key: value[index] for key, value in self.audit_metadata.items()},
            "state_id": self.state_ids[index],
        }


class IndexedStateDataset(Dataset):
    """A transparent subset view used for scientifically defined controls.

    The view retains the same learner/target separation as ``TensorStateDataset``
    and never exposes the selection predicate to the model.
    """

    def __init__(self, base: Dataset, indices: Sequence[int]) -> None:
        normalized = [int(value) for value in indices]
        if not normalized:
            raise ValueError("Indexed state dataset cannot be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Indexed state dataset contains duplicate indices")
        if min(normalized) < 0 or max(normalized) >= len(base):
            raise ValueError("Indexed state dataset contains an out-of-range index")
        self.base = base
        self.indices = normalized

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.base[self.indices[index]]


def empty_state_view(base: TensorStateDataset) -> IndexedStateDataset:
    """Return exactly one clean-only state per model-instance.

    Week 4 creates many partial states for every model-instance. Reusing all of
    them for the clean-only baseline would duplicate identical learner inputs
    and overweight instances with more legal subsets (notably relation rows).
    """

    acquired = base.model_input["acquired_mask"].sum(dim=1)
    indices = torch.nonzero(acquired == 0, as_tuple=False).flatten().tolist()
    identities = [
        (base.audit_metadata["model_id"][index], base.audit_metadata["instance_id"][index])
        for index in indices
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Clean-only view has duplicate model-instance identities")
    return IndexedStateDataset(base, indices)


class BudgetProjectedDataset(Dataset):
    """Logical state×budget expansion without duplicating stored tensors."""

    def __init__(
        self,
        base: TensorStateDataset,
        budgets: Sequence[int],
        limit: int | None = None,
    ) -> None:
        normalized = sorted(set(int(value) for value in budgets))
        if not normalized or normalized[0] < 0 or normalized[-1] > 7:
            raise ValueError("budgets must be unique integers in [0, 7]")
        if hasattr(base, "model_input"):
            acquired_counts = base.model_input["acquired_mask"].sum(dim=1).long()
        else:
            # Controls such as ``empty_state_view`` intentionally expose only a
            # Dataset interface.  Do not reach through that view and accidentally
            # restore rows that its scientific selection removed.
            acquired_counts = torch.tensor(
                [
                    int(base[index]["model_input"]["acquired_mask"].sum())
                    for index in range(len(base))
                ],
                dtype=torch.long,
            )
        base_indices: List[int] = []
        projected_budgets: List[int] = []
        for base_index, count in enumerate(acquired_counts.tolist()):
            for budget in normalized:
                if count <= budget:
                    base_indices.append(base_index)
                    projected_budgets.append(budget)
                    if limit is not None and len(base_indices) >= limit:
                        break
            if limit is not None and len(base_indices) >= limit:
                break
        if not base_indices:
            raise ValueError("No base state is compatible with the requested budgets")
        self.base = base
        self.base_indices = torch.tensor(base_indices, dtype=torch.long)
        self.budgets = torch.tensor(projected_budgets, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.base_indices.numel())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base_index = int(self.base_indices[index])
        budget = int(self.budgets[index])
        row = self.base[base_index]
        model_input = project_model_input(row["model_input"], budget)
        row["model_input"] = model_input
        row["max_budget"] = budget
        return row


def project_model_input(model_input: Mapping[str, torch.Tensor], budget: int) -> Dict[str, torch.Tensor]:
    if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 7:
        raise ValueError("budget must be an integer in [0, 7]")
    output = {key: value.clone() for key, value in model_input.items()}
    acquired = int(output["acquired_mask"].sum())
    if acquired > budget:
        raise ValueError("Projected state exceeds budget")
    remaining = budget - acquired
    output["remaining_budget"] = torch.tensor(remaining, dtype=torch.long)
    output["max_budget"] = torch.tensor(budget, dtype=torch.long)
    full_mask = output["action_mask"].bool()
    if remaining == 0:
        full_mask[:-1] = False
    full_mask[-1] = True
    output["action_mask"] = full_mask
    return output


def load_vectorized_split(
    manifest_path: str | Path,
    split: str,
    *,
    map_location: str | torch.device = "cpu",
) -> TensorStateDataset:
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != VECTOR_MANIFEST_VERSION:
        raise ValueError("Vectorized manifest version mismatch")
    entries = [entry for entry in manifest.get("files", []) if entry.get("split") == split]
    if not entries:
        raise ValueError(f"Vectorized manifest contains no {split!r} files")
    shards = []
    for entry in entries:
        path = manifest_path.parent / entry["path"]
        if not path.exists() or file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"Vectorized shard hash mismatch: {path}")
        shard = torch.load(path, map_location=map_location, weights_only=False)
        validate_tensor_shard(shard, split)
        if shard["row_count"] != entry.get("row_count"):
            raise ValueError(f"Vectorized shard row-count mismatch: {path}")
        shards.append(shard)
    return TensorStateDataset(shards, split)
