"""InternVL3 adapter using the model's native image-token interface.

The preprocessing and conversation construction follow the MIT-licensed
OpenGVLab InternVL3 reference implementation. ProActive keeps the logic here
instead of relying on ``AutoTokenizer(..., images=...)`` because InternVL3's
custom checkpoint exposes a tokenizer plus ``InternVLChatModel``, not a
generic Hugging Face multimodal processor.
"""

from __future__ import annotations

import copy
import time
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from proactive.models.base_adapter import (
    GenerationOutput,
    MLLMAdapter,
    ScoringOutput,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_IMAGE_SIZE = 448
INTERNVL_MAX_PATCHES = 12
INTERNVL_TRANSFORMERS_REQUIRED = (4, 37, 2)


def _numeric_version_prefix(value: str) -> Tuple[int, int, int]:
    """Return a comparable three-part version prefix."""
    parts: List[int] = []
    for component in value.split("."):
        digits = "".join(ch for ch in component if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    if len(parts) != 3:
        raise RuntimeError(f"Unable to parse Transformers version: {value!r}")
    return tuple(parts)  # type: ignore[return-value]


def validate_internvl_transformers_version(value: str) -> None:
    """Fail before loading weights when the custom model API is incompatible."""
    parsed = _numeric_version_prefix(value)
    if parsed != INTERNVL_TRANSFORMERS_REQUIRED:
        raise RuntimeError(
            "The validated InternVL3 custom-code runtime requires Transformers 4.37.2; "
            f"found {value}. Use the isolated proactive-internvl environment; "
            "do not downgrade the Qwen/Gemma base environment."
        )


def _candidate_ratios(min_patches: int, max_patches: int) -> List[Tuple[int, int]]:
    ratios = {
        (width, height)
        for total in range(min_patches, max_patches + 1)
        for width, height in product(range(1, total + 1), repeat=2)
        if min_patches <= width * height <= max_patches
    }
    return sorted(
        ratios, key=lambda ratio: (ratio[0] * ratio[1], ratio[0], ratio[1])
    )


def _select_target_ratio(
    width: int,
    height: int,
    ratios: Iterable[Tuple[int, int]],
    image_size: int,
) -> Tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {(width, height)}")
    aspect = width / height
    area = width * height
    best: Optional[Tuple[int, int]] = None
    best_difference = float("inf")
    for ratio in ratios:
        difference = abs(aspect - ratio[0] / ratio[1])
        if difference < best_difference:
            best = ratio
            best_difference = difference
        elif difference == best_difference and best is not None:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best = ratio
    if best is None:
        raise ValueError("InternVL image tiling produced no candidate ratio")
    return best


def dynamic_preprocess(
    image: Image.Image,
    *,
    image_size: int = INTERNVL_IMAGE_SIZE,
    min_patches: int = 1,
    max_patches: int = INTERNVL_MAX_PATCHES,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """Deterministically tile one PIL image at InternVL's dynamic resolution."""
    if min_patches < 1 or max_patches < min_patches:
        raise ValueError("Require 1 <= min_patches <= max_patches")
    image = image.convert("RGB")
    grid_width, grid_height = _select_target_ratio(
        image.width,
        image.height,
        _candidate_ratios(min_patches, max_patches),
        image_size,
    )
    resized = image.resize(
        (image_size * grid_width, image_size * grid_height),
        resample=Image.Resampling.BICUBIC,
    )
    patches: List[Image.Image] = []
    for row in range(grid_height):
        for column in range(grid_width):
            left = column * image_size
            upper = row * image_size
            patches.append(
                resized.crop((left, upper, left + image_size, upper + image_size))
            )
    if len(patches) != grid_width * grid_height:
        raise RuntimeError("InternVL dynamic tiling count mismatch")
    if use_thumbnail and len(patches) > 1:
        patches.append(
            image.resize(
                (image_size, image_size), resample=Image.Resampling.BICUBIC
            )
        )
    return patches


def _topk_distribution(
    logits: torch.Tensor,
    tokenizer: Any,
    *,
    k: int = 50,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logprobs = F.log_softmax(logits, dim=-1)
    probabilities = torch.exp(logprobs)
    top_probs, top_indices = torch.topk(
        probabilities, k=min(k, probabilities.shape[-1])
    )
    distribution: Dict[str, float] = {}
    for probability, index in zip(top_probs.tolist(), top_indices.tolist()):
        token = tokenizer.decode([index], skip_special_tokens=False)
        distribution[token] = float(probability)
    return logprobs, distribution


class InternVLAdapter(MLLMAdapter):
    """Adapter for the custom-code OpenGVLab InternVL3-9B checkpoint."""

    def __init__(
        self,
        model_path: str = "/home/models/InternVL3-9B",
        model_revision: str = "main",
        generation_config: Optional[Dict[str, Any]] = None,
        dtype: str = "auto",
        device: str = "cuda:0",
    ):
        super().__init__(
            model_path=model_path,
            model_revision=model_revision,
            generation_config=generation_config,
            dtype=dtype,
            device=device,
        )
        self._model_dtype = torch.bfloat16
        self._image_transform = T.Compose(
            [
                T.Lambda(lambda value: value.convert("RGB")),
                T.Resize(
                    (INTERNVL_IMAGE_SIZE, INTERNVL_IMAGE_SIZE),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _resolve_dtype(self) -> torch.dtype:
        if self.dtype in {"auto", "bfloat16", "bf16"}:
            return torch.bfloat16
        if self.dtype in {"float16", "fp16"}:
            return torch.float16
        if self.dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported InternVL dtype: {self.dtype!r}")

    def load_model(self) -> None:
        """Load InternVL under its compatible, isolated Transformers runtime."""
        import transformers
        from transformers import AutoModel, AutoTokenizer

        validate_internvl_transformers_version(transformers.__version__)
        self._model_dtype = self._resolve_dtype()
        self.processor = AutoTokenizer.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            torch_dtype=self._model_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(self.device)

    def _pixel_values(self, image: Image.Image) -> torch.Tensor:
        patches = dynamic_preprocess(image)
        values = torch.stack([self._image_transform(patch) for patch in patches])
        return values.to(device=self.device, dtype=self._model_dtype)

    def _prompt_query(self, prompt: str, patch_count: int) -> Tuple[str, int]:
        if self.model is None or self.processor is None:
            raise RuntimeError("InternVL model is not loaded")
        question = prompt if "<image>" in prompt else f"<image>\n{prompt}"
        template = copy.deepcopy(self.model.conv_template)
        template.messages = []
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        context_token = "<IMG_CONTEXT>"
        image_tokens = (
            "<img>"
            + context_token * int(self.model.num_image_token) * patch_count
            + "</img>"
        )
        if query.count("<image>") != 1:
            raise RuntimeError("InternVL prompt must contain exactly one image placeholder")
        query = query.replace("<image>", image_tokens, 1)
        context_id = self.processor.convert_tokens_to_ids(context_token)
        if context_id is None or context_id == self.processor.unk_token_id:
            raise RuntimeError("InternVL tokenizer lacks <IMG_CONTEXT>")
        self.model.img_context_token_id = context_id
        eos_token_id = self.processor.convert_tokens_to_ids(template.sep.strip())
        if eos_token_id is None or eos_token_id == self.processor.unk_token_id:
            raise RuntimeError("InternVL tokenizer cannot resolve conversation separator")
        return query, int(eos_token_id)

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.processor(text, return_tensors="pt")
        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )

    def _multimodal_embeddings(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.model.extract_feature(pixel_values)
        embeddings = self.model.language_model.get_input_embeddings()(input_ids)
        batch, length, width = embeddings.shape
        flat_embeddings = embeddings.reshape(batch * length, width)
        flat_ids = input_ids.reshape(batch * length)
        selected = flat_ids == self.model.img_context_token_id
        flat_visual = visual.reshape(-1, width).to(flat_embeddings.device)
        if int(selected.sum().item()) != flat_visual.shape[0]:
            raise RuntimeError(
                "InternVL visual/token alignment mismatch: "
                f"tokens={int(selected.sum().item())}, visual={flat_visual.shape[0]}"
            )
        flat_embeddings[selected] = flat_visual
        return flat_embeddings.reshape(batch, length, width)

    @staticmethod
    def _generation_kwargs(
        config: Dict[str, Any], eos_token_id: int
    ) -> Dict[str, Any]:
        if config.get("do_sample", False):
            raise ValueError("InternVL teacher generation must be deterministic")
        if int(config.get("num_beams", 1)) != 1:
            raise ValueError("InternVL score extraction requires num_beams=1")
        values = {key: value for key, value in config.items() if value is not None}
        values["eos_token_id"] = eos_token_id
        values["return_dict_in_generate"] = True
        values["output_scores"] = True
        values.setdefault("use_cache", True)
        return values

    def generate(self, image: Image.Image, prompt: str) -> GenerationOutput:
        """Generate with native dynamic tiling and return per-token scores."""
        start = time.time()
        pixel_values = self._pixel_values(image)
        query, eos_token_id = self._prompt_query(prompt, pixel_values.shape[0])
        input_ids, attention_mask = self._tokenize(query)
        input_embeddings = self._multimodal_embeddings(pixel_values, input_ids)
        kwargs = self._generation_kwargs(self.generation_config, eos_token_id)

        with torch.no_grad():
            outputs = self.model.language_model.generate(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )

        scores: Sequence[torch.Tensor] = tuple(getattr(outputs, "scores", ()))
        sequence = outputs.sequences[0]
        generated_ids = sequence[-len(scores) :] if scores else sequence[:0]
        raw_answer = self.processor.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        token_logprobs: List[float] = []
        token_distributions: List[Dict[str, float]] = []
        for token_id, step_scores in zip(generated_ids.tolist(), scores):
            logprobs, distribution = _topk_distribution(
                step_scores[0], self.processor
            )
            token_logprobs.append(float(logprobs[token_id].item()))
            token_distributions.append(distribution)

        finish_reason = (
            "eos"
            if len(generated_ids) and generated_ids[-1].item() == eos_token_id
            else "length"
        )
        return GenerationOutput(
            raw_answer=raw_answer,
            token_logprobs=token_logprobs,
            token_distributions=token_distributions,
            answer_len_tokens=len(generated_ids),
            latency_ms=(time.time() - start) * 1000,
            finish_reason=finish_reason,
        )

    def score(self, image: Image.Image, prompt: str, answer: str) -> ScoringOutput:
        """Teacher-force exactly ``answer`` after the native assistant prefix."""
        start = time.time()
        pixel_values = self._pixel_values(image)
        query, _ = self._prompt_query(prompt, pixel_values.shape[0])
        prefix_ids, prefix_mask = self._tokenize(query)
        answer_encoded = self.processor(
            answer, add_special_tokens=False, return_tensors="pt"
        )
        answer_ids = answer_encoded["input_ids"].to(self.device)
        if answer_ids.shape[1] == 0:
            raise ValueError("Cannot teacher-force an empty InternVL answer")
        input_ids = torch.cat([prefix_ids, answer_ids], dim=1)
        answer_mask = torch.ones_like(answer_ids, device=self.device)
        attention_mask = torch.cat([prefix_mask, answer_mask], dim=1)
        input_embeddings = self._multimodal_embeddings(pixel_values, input_ids)

        with torch.no_grad():
            outputs = self.model.language_model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                return_dict=True,
            )

        prefix_length = prefix_ids.shape[1]
        answer_logits = outputs.logits[0, prefix_length - 1 : -1]
        targets = answer_ids[0]
        if answer_logits.shape[0] != targets.shape[0]:
            raise RuntimeError("InternVL teacher-forced score alignment mismatch")
        token_logprobs: List[float] = []
        token_distributions: List[Dict[str, float]] = []
        for target, logits in zip(targets.tolist(), answer_logits):
            logprobs, distribution = _topk_distribution(logits, self.processor)
            token_logprobs.append(float(logprobs[target].item()))
            token_distributions.append(distribution)
        return ScoringOutput(
            token_logprobs=token_logprobs,
            token_distributions=token_distributions,
            total_logprob=float(sum(token_logprobs)),
            latency_ms=(time.time() - start) * 1000,
        )

    def get_model_revision(self) -> str:
        return self.model_revision
