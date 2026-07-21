# Specifications and Definitions

## 1. Loading & Normalization Specifications

### 1.1 Model Loading Specs
- **OpenGVLab/InternVL3-8B**: Load with `transformers` using `AutoModel` and `AutoProcessor`. Requires `trust_remote_code=True` and `torch_dtype=torch.bfloat16`.
- **Qwen/Qwen3-VL-8B-Instruct**: Load with `Qwen3VLForConditionalGeneration` (or AutoModelForCausalLM) and its processor. Use `torch_dtype=torch.bfloat16`.
- **allenai/Molmo-7B-D-0924**: Load via `AutoModelForCausalLM` with `trust_remote_code=True`.
- **google/gemma-4-E4B-it**: Use standard Gemma 4 vision loading via `AutoModelForImageTextToText` or `AutoModelForCausalLM` and standard processor. Needs `bfloat16`.

### 1.2 Dataset Loading Specs
- **POPE**: Format is Yes/No questions about object existence.
- **HallusionBench**: Focuses on visual illusions and language hallucination (mixed). Requires standardizing visual/language splits.
- **VizWiz-VQA**: Open-ended real-world VQA (includes unanswerable questions).
- **Visual Spatial Reasoning (VSR)**: True/False statements about spatial relations between objects.
- **PRE-HAL**: Evaluates perception and reasoning hallucinations.
- **IllusionBench**: Focuses on visual illusions (held-out stress test).

### 1.3 Answer Normalization Specs
- Convert all generated text to lowercase.
- Strip leading/trailing whitespaces and common punctuation (`.`, `,`, `!`, `?`).
- For Yes/No/True/False datasets (POPE, VSR, HallusionBench):
  - Map keywords ("yes", "true", "correct") -> `1` or `"yes"`
  - Map keywords ("no", "false", "incorrect") -> `0` or `"no"`
- For open-ended (VizWiz): Remove standard stop words, extract core noun phrases or rely on exact match / VQA accuracy metrics if evaluating.

## 2. Baseline Definitions
- **Scalar confidence only**: One clean pass outputting the max logit probability (or sequence probability) of the generated answer.
- **Clean-only learned predictor**: A small classifier trained on clean features (confidence, question type, answer length) without any expensive diagnostic probes.
- **Fixed probe policies**: Always run probes in a fixed order (e.g., Visual-first, Blank-first, Grounding-first, Relation-first) until a budget is hit.
- **Random acquisition**: Randomly selects a probe to run under the same budget constraints.
- **Uncertainty-greedy**: Dynamically chooses the next probe based on maximum expected confidence shift or entropy.
- **Full-probe teacher**: Runs all probes for every instance (highest cost). Serves as the upper-bound reference.

## 3. Compute Budget Estimate
- **Models**: 4 (8B/7B class)
- **Datasets**: 6
- **Instances per dataset (scaling phase)**: ~10,000 total instances (across all splits).
- **Base inferences**: 4 models * 10,000 instances = 40,000 inferences.
- **Probed inferences** (assuming 4 probes per instance): 4 probes * 40,000 = 160,000 inferences.
- **Total inferences**: ~200,000.
- **Time estimate**: At ~0.5s to 1s per inference on an A100/H100, this will take roughly **30-55 GPU hours**.
- **Cost**: At ~$2-3/hour for an A100 GPU on typical cloud platforms, the full teacher probing will cost approximately **$100 - $165**.
- **Smoke test budget (50 samples)**: 50 * 6 datasets * 4 models * 5 passes (1 clean + 4 probes) = 6,000 inferences. (~1-2 GPU hours, highly feasible for local or quick cloud runs).
