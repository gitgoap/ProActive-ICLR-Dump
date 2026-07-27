# Project Status

**Current Phase:** Week 2 — Clean Inference and Model Adapters ✅ COMPLETE (Awaiting final dataset/model downloads)

**Completed Work:**
- Scaffolded repository structure and core data schemas.
- Completed Answer normalization and Clean feature extraction.
- Grouped split builder and dataset loaders.
- **Completed MLLM Adapters** (`qwen_adapter.py`, `gemma_adapter.py`, `internvl_adapter.py`) with full generation and scoring capabilities, including PyTorch tensor manipulation to extract precise token log-probs and top-k distributions.
- **Created and executed GPU smoke tests**.
  - `Qwen3-VL-8B-Instruct`: ✅ Passed determinism checks and successfully extracted distributions.
  - `gemma-4-E4B-it`: ✅ Passed determinism checks and successfully extracted distributions.

**Current Blocker:**
- None.

**Running or Awaiting Server Jobs:**
- `InternVL3-9B` download (Pending User Action)
- Dataset downloads for GQA, PRE-HAL, IllusionBench (Pending User Action)

**Next Tasks (Week 3):**
1. Implement the visual probes (blur, blank, brightness, noise, crop).
2. Implement the language probes (blank language, swap relation).
3. Cache the full teacher observations.
4. Finalize the exact thresholds for the three source bits (visual fragility, language prior, alignment).

**Deviations from the Plan:**
- None. We remain strictly aligned with the `v3.5_ProActive_Complete_Super_Implementation_Plan.md`.
