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
4. **Answer Normalization:** Built rule-based string normalizers for yes/no, true/false, and free-form outputs (`src/proactive/features/normalization.py`), and verified them with passing unit tests.
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

## Week 3: Probes and Teacher Generation (Audit & Hardening)
**Status:** Implementation & Verification Complete; Ready for Server Pilot
**Dates:** ~August 1, 2026

**What we actually did:**
1. **Image Transform Probes:** Implemented 5 visual probes (blank, blur, crop, brightness, noise) in `src/proactive/probes/image_transforms.py` with deterministic SHA-256 seeding (`global_seed | instance_id | probe_name | severity`), pilot grid parameters, and visual sample export helper.
2. **Grounding Logic & Tagging:** Standardized `FINAL_ANSWER:` prompt formatting in `src/proactive/prompts/templates.py`. Parsed answers with fail-closed validation, marking malformed outputs explicitly as invalid rather than spurious flips.
3. **Relation Swap Logic:** Hardened relation swap probe with explicit `RelationSwapStatus` enum (`changed_correctly`, `invariant`, `invalid`, `not_applicable`), case preservation, round-trip verification, and fail-closed swap invariance computation.
4. **Feature & Confound Isolation:** Set `include_relation_available=False` as default for `CleanFeatures` to prevent dataset shortcut exploitation. Implemented Section 28 confound trigger audits (`src/proactive/audits/confound_audit.py`).
5. **Semantic Matching & Calibration:** Built Plan §14.5 semantic matching module in `src/proactive/features/semantic.py` supporting binary exact-normalized and free-form embedding similarity, provenance tracking, and threshold calibration.
6. **Probe Orchestrator:** Hardened `src/proactive/probes/probe_runner.py` to enforce independent application on original inputs, uniform scoring methods per instance, and mandatory probe completeness.
7. **Teacher Label Computation:** Refactored `src/proactive/teacher/label_computation.py` to validate mandatory probe observations, compute continuous teacher signatures $(V, L, A)$, source bits $(b_V, b_L, b_A)$, and six-way diagnostic states.
8. **Tooling & Validation:**
   - `scripts/run_pilot_cache.py`: Enforced mandatory `--manifest_path`, train/val-only filtering, deterministic stratified sampling, and severity grid support.
   - `scripts/validate_teacher_schema.py` & `src/proactive/audits/schema_validator.py`: Strict schema validator.
   - `scripts/analyze_pilot.py` & `src/proactive/audits/pilot_analysis.py`: Summary statistics, candidate config generation, and wall-clock estimation.
   - `scripts/check_week_completion.py`: Gate checker for all Week 3 requirements.
9. **Testing:** Added comprehensive unit and integration tests across 11 test modules. All 147 test cases pass (100% passing rate).

**Next Step (Server Execution):**
1. Run 1-example smoke test on server.
2. Run 10-example pilot.
3. Run 100-example pilot per model/dataset on train/val splits.
4. Validate schema with `scripts/validate_teacher_schema.py`.
5. Generate diagnostic plots, inspect 50 images per probe, and freeze configuration with `scripts/analyze_pilot.py --freeze`.
