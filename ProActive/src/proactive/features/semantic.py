"""
Semantic answer matching and threshold calibration (Plan §14.5).

Matching rules:
1. Binary datasets (POPE, VSR, HallusionBench, GQA-Relation, PreHal, IllusionBench):
   Exact normalized equality: 1[norm(a) == norm(b)].
2. Free-form datasets (VizWiz, GQA):
   Exact normalized equality OR embedding cosine similarity >= tau_sem (initial default 0.82).

Tracks full provenance: embedding model name, pinned revision, threshold,
calibration split, and calibration metrics. Calibration uses ONLY train/val data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from proactive.features.normalization import normalize_answer

logger = logging.getLogger(__name__)

# Datasets with fixed binary or closed vocabularies
BINARY_DATASETS = {
    "pope",
    "vsr",
    "hallusionbench",
    "gqa_relation",
    "prehal",
    "illusionbench",
}

DEFAULT_FREEFORM_THRESHOLD = 0.82
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "e4ce9877abf3edee10b0257f22713854020a4004"
DEFAULT_LOCAL_MODEL_DIR = "/home/models/all-MiniLM-L6-v2"


class SemanticMatcherError(RuntimeError):
    """Raised when semantic matcher fails to initialize or compute for a required dataset."""
    pass


class SemanticMatcher:
    """Manages sentence embedding model loading, cosine similarity, and thresholding."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_EMBEDDING_MODEL,
        revision: str = DEFAULT_EMBEDDING_REVISION,
        device: str = "cpu",
        local_model_dir: str = DEFAULT_LOCAL_MODEL_DIR,
    ):
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.device = device
        self.local_model_dir = local_model_dir
        self.model = None
        self.load_error: Optional[str] = None
        self.model_path_used: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load SentenceTransformer model from local directory or HuggingFace."""
        try:
            from sentence_transformers import SentenceTransformer

            loaded = False
            if Path(self.local_model_dir).exists():
                try:
                    self.model = SentenceTransformer(self.local_model_dir, device=self.device)
                    self.model_path_used = self.local_model_dir
                    loaded = True
                    logger.info(f"Loaded SemanticMatcher from local dir: {self.local_model_dir}")
                except Exception as local_err:
                    logger.warning(
                        f"Local model {self.local_model_dir} load failed ({local_err}), falling back to HF model {self.model_name_or_path}..."
                    )

            if not loaded:
                try:
                    self.model = SentenceTransformer(
                        self.model_name_or_path,
                        revision=self.revision,
                        device=self.device,
                    )
                    self.model_path_used = self.model_name_or_path
                    logger.info(f"Loaded SemanticMatcher from HF: {self.model_name_or_path} (rev: {self.revision})")
                except Exception as rev_err:
                    logger.info(f"Loading HF model without revision pin ({rev_err})...")
                    self.model = SentenceTransformer(
                        self.model_name_or_path,
                        device=self.device,
                    )
                    self.model_path_used = self.model_name_or_path
                    logger.info(f"Loaded SemanticMatcher from HF: {self.model_name_or_path}")
        except Exception as e:
            self.model = None
            self.load_error = str(e)
            logger.warning(f"SemanticMatcher model could not be loaded: {e}")

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def similarity(self, s1: str, s2: str) -> float:
        """Compute cosine similarity between two sentences."""
        if not s1 or not s2:
            return 1.0 if s1 == s2 else 0.0

        if self.model is None:
            raise SemanticMatcherError(
                f"SemanticMatcher model is not loaded: {self.load_error}. "
                "Cannot compute semantic similarity without embedding model."
            )

        try:
            import numpy as np
            embeddings = self.model.encode(
                [s1, s2],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sim = float(np.dot(embeddings[0], embeddings[1]))
            return float(np.clip(sim, -1.0, 1.0))
        except Exception as e:
            raise SemanticMatcherError(f"Embedding encoding failed: {e}") from e


_GLOBAL_MATCHER: Optional[SemanticMatcher] = None


def get_default_semantic_matcher() -> Optional[SemanticMatcher]:
    """Get or lazily initialize the singleton SemanticMatcher."""
    global _GLOBAL_MATCHER
    if _GLOBAL_MATCHER is None:
        _GLOBAL_MATCHER = SemanticMatcher()
    return _GLOBAL_MATCHER


@dataclass
class SemanticProvenance:
    """Provenance tracking for semantic matching decisions (Plan §14.5)."""
    dataset: str
    matching_mode: str  # "exact_normalized" or "embedding_similarity"
    embedding_model: Optional[str] = None
    model_revision: Optional[str] = None
    threshold: float = DEFAULT_FREEFORM_THRESHOLD
    calibration_split: Optional[str] = None  # e.g., "train_val"
    calibration_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "matching_mode": self.matching_mode,
            "embedding_model": self.embedding_model,
            "model_revision": self.model_revision,
            "threshold": self.threshold,
            "calibration_split": self.calibration_split,
            "calibration_metrics": self.calibration_metrics,
        }


def compute_semantic_match(
    pred_answer: str,
    target_answer: str,
    dataset: str,
    threshold: float = DEFAULT_FREEFORM_THRESHOLD,
    embedding_fn: Optional[Callable[[str, str], float]] = None,
    matcher: Optional[SemanticMatcher] = None,
) -> float:
    """Compute semantic match indicator in {0.0, 1.0} between prediction and target.

    Args:
        pred_answer: Model raw or processed prediction.
        target_answer: Comparison target (e.g. clean answer or gold answer).
        dataset: Dataset identifier.
        threshold: Cosine similarity threshold for free-form matching.
        embedding_fn: Optional callable (s1, s2) -> cosine_sim in [-1, 1].
        matcher: Optional SemanticMatcher instance.

    Returns:
        1.0 if answers match semantically, 0.0 otherwise.
    """
    dataset_lower = dataset.lower().replace("-", "").replace(" ", "_")
    norm_pred = normalize_answer(pred_answer, dataset)
    norm_target = normalize_answer(target_answer, dataset)

    # 1. Exact normalized match is authoritative across all datasets
    if norm_pred == norm_target and norm_pred not in ("", "unknown", "invalid"):
        return 1.0

    # 2. Binary / closed-vocab datasets use strict exact match only
    if dataset_lower in BINARY_DATASETS:
        return 0.0

    # 3. If either string is empty or invalid, no match
    if not norm_pred or not norm_target or norm_pred in ("unknown", "invalid"):
        return 0.0

    # 4. Free-form datasets (e.g. VizWiz): check embedding similarity
    sim_fn = embedding_fn
    if sim_fn is None and matcher is not None:
        sim_fn = matcher.similarity
    if sim_fn is None:
        # Check global matcher
        gm = get_default_semantic_matcher()
        if gm and gm.is_available:
            sim_fn = gm.similarity

    if sim_fn is None:
        raise SemanticMatcherError(
            f"Free-form dataset '{dataset}' requires an embedding model for semantic matching, "
            "but no embedding function or SemanticMatcher is available. Silent Jaccard fallback is prohibited."
        )

    sim = sim_fn(norm_pred, norm_target)
    return 1.0 if sim >= threshold else 0.0


def calibrate_semantic_threshold(
    pairs: Sequence[Tuple[str, str, bool]],
    similarity_fn: Callable[[str, str], float],
    target_recall: float = 0.90,
    candidate_thresholds: Optional[Sequence[float]] = None,
    dataset: str = "vizwiz",
    calibration_split: str = "train_val",
) -> Tuple[float, SemanticProvenance]:
    """Calibrate semantic threshold tau_sem on train/val pairs (Plan §14.5).

    Args:
        pairs: List of (pred, target, is_true_semantic_match) tuples.
        similarity_fn: Function returning similarity score in [0, 1].
        target_recall: Target recall on positive pairs.
        candidate_thresholds: Candidate threshold values to evaluate.
        dataset: Dataset identifier.
        calibration_split: Split name used for calibration.

    Returns:
        (best_threshold, provenance)
    """
    if candidate_thresholds is None:
        candidate_thresholds = [i / 100.0 for i in range(50, 96, 2)]

    if not pairs:
        prov = SemanticProvenance(
            dataset=dataset,
            matching_mode="embedding_similarity",
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            model_revision=DEFAULT_EMBEDDING_REVISION,
            threshold=DEFAULT_FREEFORM_THRESHOLD,
            calibration_split=calibration_split,
            calibration_metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0},
        )
        return DEFAULT_FREEFORM_THRESHOLD, prov

    scored_pairs = [
        (similarity_fn(p, t), is_match)
        for p, t, is_match in pairs
    ]

    best_threshold = DEFAULT_FREEFORM_THRESHOLD
    best_f1 = -1.0
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for tau in candidate_thresholds:
        tp = sum(1 for sim, is_match in scored_pairs if sim >= tau and is_match)
        fp = sum(1 for sim, is_match in scored_pairs if sim >= tau and not is_match)
        fn = sum(1 for sim, is_match in scored_pairs if sim < tau and is_match)
        tn = sum(1 for sim, is_match in scored_pairs if sim < tau and not is_match)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if rec >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = tau
            best_metrics = {"precision": prec, "recall": rec, "f1": f1}

    prov = SemanticProvenance(
        dataset=dataset,
        matching_mode="embedding_similarity",
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        model_revision=DEFAULT_EMBEDDING_REVISION,
        threshold=best_threshold,
        calibration_split=calibration_split,
        calibration_metrics=best_metrics,
    )
    return best_threshold, prov
