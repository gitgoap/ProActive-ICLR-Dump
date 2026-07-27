"""
Adapter for Gemma-4-E4B-it.

Server path: /home/models/paligemma-3b-mix-224
HuggingFace: google/paligemma-3b-mix-224

This is a STUB. Full implementation will be completed in Week 2.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from PIL import Image
import torch
import torch.nn.functional as F

from proactive.models.base_adapter import (
    GenerationOutput,
    MLLMAdapter,
    ScoringOutput,
)


class GemmaAdapter(MLLMAdapter):
    """Adapter for Gemma-4-E4B-it."""

    def __init__(
        self,
        model_path: str = "/home/models/gemma-4-E4B-it",
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

    def load_model(self) -> None:
        """Load Gemma-4 model and processor."""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            torch_dtype=self.dtype if self.dtype != "auto" else "auto",
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> GenerationOutput:
        """Generate with Gemma-4. Returns token log-probs."""
        start_time = time.time()
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        gen_kwargs = dict(self.generation_config)
        gen_kwargs["output_logits"] = True
        gen_kwargs["return_dict_in_generate"] = True
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        raw_answer = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        token_logprobs = []
        token_distributions = []
        
        if hasattr(outputs, "logits"):
            for i, logits_at_t in enumerate(outputs.logits):
                logits_at_t = logits_at_t[0]
                logprobs_at_t = F.log_softmax(logits_at_t, dim=-1)
                
                generated_token_id = generated_ids[i].item()
                token_logprob = logprobs_at_t[generated_token_id].item()
                token_logprobs.append(token_logprob)
                
                probs_at_t = torch.exp(logprobs_at_t)
                topk_probs, topk_indices = torch.topk(probs_at_t, k=min(50, probs_at_t.shape[-1]))
                dist = {}
                for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                    token_str = self.processor.decode([idx])
                    dist[token_str] = prob
                token_distributions.append(dist)
                
        latency_ms = (time.time() - start_time) * 1000
        
        return GenerationOutput(
            raw_answer=raw_answer,
            token_logprobs=token_logprobs,
            token_distributions=token_distributions,
            answer_len_tokens=len(generated_ids),
            latency_ms=latency_ms,
            finish_reason="stop",
        )

    def score(
        self,
        image: Image.Image,
        prompt: str,
        answer: str,
    ) -> ScoringOutput:
        """Teacher-forced scoring for Gemma-4."""
        start_time = time.time()
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        prompt_messages = [messages[0]]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(text=prompt_text, images=image, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits[0]
        input_ids = inputs.input_ids[0]
        
        answer_logits = logits[prompt_len-1 : len(input_ids)-1]
        answer_ids = input_ids[prompt_len:]
        
        token_logprobs = []
        token_distributions = []
        total_logprob = 0.0
        
        for i in range(len(answer_ids)):
            logprobs = F.log_softmax(answer_logits[i], dim=-1)
            target_id = answer_ids[i].item()
            token_lp = logprobs[target_id].item()
            token_logprobs.append(token_lp)
            total_logprob += token_lp
            
            probs = torch.exp(logprobs)
            topk_probs, topk_indices = torch.topk(probs, k=min(50, probs.shape[-1]))
            dist = {}
            for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                token_str = self.processor.decode([idx])
                dist[token_str] = prob
            token_distributions.append(dist)
            
        latency_ms = (time.time() - start_time) * 1000
        
        return ScoringOutput(
            token_logprobs=token_logprobs,
            token_distributions=token_distributions,
            total_logprob=total_logprob,
            latency_ms=latency_ms,
        )

    def get_model_revision(self) -> str:
        return self.model_revision
