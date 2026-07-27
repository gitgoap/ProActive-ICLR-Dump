"""
Abstract base class for MLLM adapters.

Every model adapter must implement this interface so that the
teacher cache generator, clean inference, and probe system can
use any MLLM interchangeably.  (Plan §18)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GenerationOutput:
    """Output from a single MLLM generation call."""
    raw_answer: str
    token_logprobs: List[float] = field(default_factory=list)
    # Top-k token distributions per position (for entropy/margin)
    token_distributions: Optional[List[Dict[str, float]]] = None
    answer_len_tokens: int = 0
    latency_ms: float = 0.0
    # Generation metadata
    finish_reason: Optional[str] = None


@dataclass
class ScoringOutput:
    """Output from a teacher-forced scoring pass."""
    token_logprobs: List[float] = field(default_factory=list)
    token_distributions: Optional[List[Dict[str, float]]] = None
    total_logprob: float = 0.0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class MLLMAdapter(ABC):
    """Abstract adapter for multimodal large language models.

    Subclasses implement model-specific loading, generation, and scoring.
    The shared interface ensures all models can be used by the same
    teacher cache, clean inference, and probe pipelines.
    """

    def __init__(
        self,
        model_path: str,
        model_revision: str = "main",
        generation_config: Optional[Dict[str, Any]] = None,
        dtype: str = "auto",
        device: str = "cuda:0",
    ):
        self.model_path = model_path
        self.model_revision = model_revision
        self.generation_config = generation_config or self._default_gen_config()
        self.dtype = dtype
        self.device = device
        self.model = None
        self.processor = None

    @staticmethod
    def _default_gen_config() -> Dict[str, Any]:
        """Default deterministic generation config (Plan §14.3)."""
        return {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": 256,
            "num_beams": 1,
        }

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor onto the configured device."""
        ...

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> GenerationOutput:
        """Generate a response with token-level scores.

        Must use deterministic decoding. Must return token log-probs.
        """
        ...

    @abstractmethod
    def score(
        self,
        image: Image.Image,
        prompt: str,
        answer: str,
    ) -> ScoringOutput:
        """Teacher-forced scoring of a given answer.

        Used when generation APIs do not expose token logits.
        """
        ...

    @abstractmethod
    def get_model_revision(self) -> str:
        """Return the exact model revision (commit hash or tag)."""
        ...

    def unload_model(self) -> None:
        """Free GPU memory."""
        self.model = None
        self.processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get_generation_config_dict(self) -> Dict[str, Any]:
        """Return generation config as a serializable dict for hashing."""
        return dict(self.generation_config)
