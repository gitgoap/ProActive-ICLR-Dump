"""Dataset/model identity controls kept outside the deployable learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class IdentityVocabulary:
    datasets: Tuple[str, ...]
    models: Tuple[str, ...]

    @classmethod
    def fit(cls, metadata: Mapping[str, Sequence[str]]) -> "IdentityVocabulary":
        datasets = tuple(sorted(set(metadata["dataset"])))
        models = tuple(sorted(set(metadata["model_id"])))
        if not datasets or not models:
            raise ValueError("Identity vocabulary requires dataset and model values")
        return cls(datasets=datasets, models=models)

    def transform(self, metadata: Mapping[str, Sequence[str]]) -> np.ndarray:
        datasets = list(metadata["dataset"])
        models = list(metadata["model_id"])
        if len(datasets) != len(models):
            raise ValueError("Dataset/model metadata length mismatch")
        result = np.zeros((len(datasets), len(self.datasets) + len(self.models)), dtype=np.float32)
        dataset_index = {value: index for index, value in enumerate(self.datasets)}
        model_index = {value: index for index, value in enumerate(self.models)}
        for row, (dataset, model) in enumerate(zip(datasets, models)):
            if dataset in dataset_index:
                result[row, dataset_index[dataset]] = 1.0
            if model in model_index:
                result[row, len(self.datasets) + model_index[model]] = 1.0
        return result

    def to_dict(self) -> Dict[str, list[str]]:
        return {"datasets": list(self.datasets), "models": list(self.models)}


def control_features(
    *,
    clean_features: np.ndarray,
    metadata: Mapping[str, Sequence[str]],
    vocabulary: IdentityVocabulary,
    probe_numeric: np.ndarray | None = None,
    acquired_mask: np.ndarray | None = None,
) -> np.ndarray:
    clean = np.asarray(clean_features, dtype=np.float32)
    if clean.ndim != 2 or clean.shape[1] != 4 or not np.isfinite(clean).all():
        raise ValueError("clean_features must be a finite [N, 4] matrix")
    identity = vocabulary.transform(metadata)
    if identity.shape[0] != clean.shape[0]:
        raise ValueError("Control metadata/features row mismatch")
    blocks = [clean, identity]
    if (probe_numeric is None) != (acquired_mask is None):
        raise ValueError("Probe numeric values and mask must be provided together")
    if probe_numeric is not None:
        probes = np.asarray(probe_numeric, dtype=np.float32)
        mask = np.asarray(acquired_mask, dtype=np.float32)
        if probes.shape[:2] != mask.shape or probes.ndim != 3 or probes.shape[0] != clean.shape[0]:
            raise ValueError("Malformed probe evidence for shortcut control")
        if not np.isfinite(probes).all() or not np.isfinite(mask).all():
            raise ValueError("Shortcut evidence must be finite")
        blocks.extend([probes.reshape(probes.shape[0], -1), mask])
    return np.concatenate(blocks, axis=1)
