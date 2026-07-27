# Weekly Progress Tracker

This document records the **actual** work completed on a week-by-week basis during the execution of the ProActive project, serving as an empirical log distinct from the aspirational master plan.

---

## Week 1: Skeleton, Data Schemas, and Environment Setup
**Status:** Completed
**Dates:** ~July 24, 2026

**What we actually did:**
1. **Server Validation:** Ran `server_preflight.sh` to confirm GPU constraints (4x A6000), available RAM, and environment variables on `bumblebee.lcs2`.
2. **Architecture Scaffolding:** Created the 14 empty Python packages (`src/proactive/*`).
3. **Data Schemas:** Implemented core Pydantic data models for the project, including `EvidenceState`, `ProbeObservation`, `CleanFeatures`, `SourceBits`, `SixWayState`, and `TeacherLabels` in `src/proactive/data/`.
4. **Answer Normalization:** Built rule-based string normalizers for yes/no, true/false, and free-form outputs (`src/proactive/features/normalization.py`), and verified them with 65 passing unit tests.
5. **Data Loaders:** Built a grouped dataset splitter (`src/proactive/data/splits.py`) that strictly separates data based on image hashes to prevent test set leakage, along with dataset manifest builders.
6. **Configuration:** Created YAML configurations for 7 datasets (POPE, HallusionBench, VSR, VizWiz, PRE-HAL, IllusionBench, GQA) and 3 models (Qwen, Gemma, InternVL).

---

## Week 2: Model Adapters and Clean Inference
**Status:** Completed (Pending final InternVL/Dataset downloads)
**Dates:** ~July 24, 2026

**What we actually did:**
1. **Model Adapter Architecture:** Implemented the `MLLMAdapter` base class that mandates `generate` and `score` methods.
2. **Qwen Implementation:** Built `QwenAdapter` capable of processing dynamic vision info and extracting top-50 token distributions and chosen log-probabilities using PyTorch `F.log_softmax(logits)`.
3. **Gemma Implementation:** Built `GemmaAdapter` for `gemma-4-E4B-it` matching the exact generation and scoring signatures.
4. **InternVL Implementation:** Drafted `InternVLAdapter` for `InternVL3-9B` using its specific `<image>` interleaved chat templates.
5. **Smoke Testing:** Wrote `scripts/smoke_test_models.py` and executed it on the GPU server. Successfully generated deterministic results and extracted correct log distributions for both Qwen and Gemma natively.

---

## Week 3: Probes and Teacher Generation
**Status:** Not Started
**What we actually did:**
*(To be filled as work progresses)*
