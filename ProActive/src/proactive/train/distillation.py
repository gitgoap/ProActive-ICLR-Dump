"""One-pass full-teacher distillation dataset and objective."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from proactive.train.vectorized import TensorStateDataset, project_model_input


class FullEvidencePairDataset(Dataset):
    """Pair one empty and one maximal legal state for every model-instance."""

    def __init__(self, base: TensorStateDataset) -> None:
        by_identity: Dict[tuple[str, str], Dict[str, Any]] = {}
        counts = base.model_input["acquired_mask"].sum(dim=1).tolist()
        for index, acquired in enumerate(counts):
            identity = (
                base.audit_metadata["model_id"][index],
                base.audit_metadata["instance_id"][index],
            )
            item = by_identity.setdefault(identity, {"empty": None, "full": None, "full_count": -1})
            if int(acquired) == 0:
                if item["empty"] is not None:
                    raise ValueError(f"Duplicate empty state for {identity}")
                item["empty"] = index
            if int(acquired) > int(item["full_count"]):
                item["full"] = index
                item["full_count"] = int(acquired)
        pairs = []
        for identity, item in sorted(by_identity.items()):
            if item["empty"] is None or item["full"] is None or item["full_count"] not in {6, 7}:
                raise ValueError(f"Missing empty/maximal state for {identity}")
            pairs.append((int(item["empty"]), int(item["full"])))
        if not pairs:
            raise ValueError("Full-evidence pair dataset is empty")
        self.base = base
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        empty_index, full_index = self.pairs[index]
        empty = self.base[empty_index]
        full = self.base[full_index]
        if empty["audit_metadata"] != full["audit_metadata"]:
            raise ValueError("Empty/full state metadata mismatch")
        return {
            "empty_model_input": project_model_input(empty["model_input"], 7),
            "full_model_input": project_model_input(full["model_input"], 7),
            "targets": empty["targets"],
            "audit_metadata": empty["audit_metadata"],
            "state_id": empty["state_id"],
        }


def distillation_loss(student: Any, teacher: Any, six_way_weight: float = 0.5, signature_weight: float = 0.1) -> torch.Tensor:
    bit = F.binary_cross_entropy_with_logits(
        student.bit_logits, torch.sigmoid(teacher.bit_logits.detach())
    )
    six = F.kl_div(
        F.log_softmax(student.six_way_logits, dim=-1),
        F.softmax(teacher.six_way_logits.detach(), dim=-1),
        reduction="batchmean",
    )
    signature = F.mse_loss(student.signature, teacher.signature.detach())
    return bit + six_way_weight * six + signature_weight * signature
