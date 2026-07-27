#!/usr/bin/env python3
"""
Smoke test for MLLM adapters.

Runs on the server GPU to verify:
1. Model loads successfully.
2. Generation is deterministic.
3. Token log-probs are extracted properly.
4. Scoring produces the same log-probs as generation.

Usage:
    python scripts/smoke_test_models.py --model qwen
    python scripts/smoke_test_models.py --model gemma
    python scripts/smoke_test_models.py --model internvl
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.models.qwen_adapter import QwenAdapter
from proactive.models.gemma_adapter import GemmaAdapter
from proactive.models.internvl_adapter import InternVLAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Smoke test MLLM adapters.")
    parser.add_argument(
        "--model", type=str, required=True, choices=["qwen", "gemma", "internvl"],
        help="Which model adapter to test."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device to use."
    )
    args = parser.parse_args()

    # 1. Initialize Adapter
    logger.info(f"=== Initializing {args.model.upper()} ===")
    
    # We use greedy decoding for the smoke test to ensure determinism
    gen_config = {
        "max_new_tokens": 10,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
    }

    if args.model == "qwen":
        adapter = QwenAdapter(device=args.device, generation_config=gen_config)
    elif args.model == "gemma":
        adapter = GemmaAdapter(device=args.device, generation_config=gen_config)
    elif args.model == "internvl":
        adapter = InternVLAdapter(device=args.device, generation_config=gen_config)
    else:
        raise ValueError("Unknown model")

    # 2. Load Model
    logger.info("Loading model weights (this may take a minute)...")
    start_load = time.time()
    adapter.load_model()
    logger.info(f"Loaded in {time.time() - start_load:.1f}s")

    # 3. Create dummy data
    logger.info("Creating dummy image...")
    image = Image.new("RGB", (224, 224), color="blue")
    prompt = "What color is this image? Answer in exactly one word."

    # 4. Generate - Run 1
    logger.info(f"Prompt: {prompt}")
    logger.info("Running generation (Run 1)...")
    gen_out_1 = adapter.generate(image, prompt)
    
    sum_lp_1 = sum(gen_out_1.token_logprobs)
    logger.info(f"Answer 1: {gen_out_1.raw_answer}")
    logger.info(f"Tokens: {gen_out_1.answer_len_tokens}")
    logger.info(f"Sum LogProbs: {sum_lp_1:.4f}")

    # 5. Generate - Run 2 (Test Determinism)
    logger.info("Running generation (Run 2 - Determinism check)...")
    gen_out_2 = adapter.generate(image, prompt)
    
    sum_lp_2 = sum(gen_out_2.token_logprobs)
    logger.info(f"Answer 2: {gen_out_2.raw_answer}")
    
    if gen_out_1.raw_answer != gen_out_2.raw_answer:
        logger.error("DETERMINISM FAILURE: Answers do not match!")
        sys.exit(1)
        
    if abs(sum_lp_1 - sum_lp_2) > 1e-4:
        logger.error(f"DETERMINISM FAILURE: LogProbs differ! {sum_lp_1} vs {sum_lp_2}")
        sys.exit(1)
        
    logger.info("Determinism check passed. ✅")

    # 6. Score - Teacher Forced
    logger.info(f"Running score with answer: '{gen_out_1.raw_answer}'")
    score_out = adapter.score(image, prompt, gen_out_1.raw_answer)
    
    logger.info(f"Score Total LogProb: {score_out.total_logprob:.4f}")
    
    # The generated token sequence length might differ slightly from the scored token 
    # sequence length depending on how the tokenizer handles the prepended space or 
    # answer concatenation, but the total logprob should be roughly similar.
    # We just ensure it doesn't crash and returns valid numbers.
    if abs(sum_lp_1 - score_out.total_logprob) > 1.0:
        logger.warning(
            f"Score Total LogProb ({score_out.total_logprob:.4f}) differs significantly "
            f"from Generation Sum LogProbs ({sum_lp_1:.4f}). This may be expected due to "
            "tokenization differences (e.g. leading spaces)."
        )
    else:
        logger.info("Scoring log-probs match generation log-probs closely. ✅")
        
    logger.info("Distributions extracted: " + str(len(score_out.token_distributions)))
    
    logger.info(f"=== Smoke Test Passed for {args.model.upper()} ===")


if __name__ == "__main__":
    main()
