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
**Status:** Completed (InternVL subsequently downloaded; GPU validation pending in Week 4)
**Dates:** ~July 24, 2026

**What we actually did:**
1. **Model Adapter Architecture:** Implemented the `MLLMAdapter` base class that mandates `generate` and `score` methods.
2. **Qwen Implementation:** Built `QwenAdapter` capable of processing dynamic vision info and extracting top-50 token distributions and chosen log-probabilities using PyTorch `F.log_softmax(logits)`.
3. **Gemma Implementation:** Built `GemmaAdapter` for `gemma-4-E4B-it` matching the exact generation and scoring signatures.
4. **InternVL Implementation:** Drafted `InternVLAdapter` for `InternVL3-9B` using its specific `<image>` interleaved chat templates.
5. **Smoke Testing:** Wrote `scripts/smoke_test_models.py` and executed it on the GPU server. Successfully generated deterministic results and extracted correct log distributions for both Qwen and Gemma natively.

**Later availability update (August 10, 2026):** InternVL3-9B is now downloaded
at `/home/models/InternVL3-9B` and pinned to immutable revision
`5f618513e35a9b85922341b8057feddfc8880e50`. This resolves the download blocker,
but does not count as the required InternVL GPU smoke/catch-up validation.

---

## Week 3: Probes and Teacher Generation (Audit & Hardening)
**Status:** COMPLETE
**Dates:** August 1–9, 2026

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
9. **Pilot safety hardening (August 7):** Prevented implicit append to existing caches, added duplicate-aware canonical/composite resume keys, retained backward-compatible pilot subsets while filling exactly 100 examples, and made append records durable with flush/fsync.
10. **Severity cost correction:** Replaced 12 repetitions of the full teacher suite with one canonical suite plus eight additional visual transformations (15 generations instead of 84 per severity-pilot instance). Severity rows now use a compact, explicitly validated non-training schema.
11. **Dataset correction:** HallusionBench now excludes its 178 text-only records before applying limits, preserving the intended 951 image-paired examples.
12. **Audit and freeze hardening:** Directory-wide schema/duplicate validation blocks analysis; severity safety constraints are hard gates; semantic threshold freezing requires a 50-pair human-labelled train/val audit; the completion gate requires two complete model/dataset pilot matrices and a human-reviewed frozen configuration.
13. **Testing:** Added adversarial tests for exact sampling, duplicate rejection, compact severity cost, HallusionBench filtering, and semantic calibration. Final local CPU suite: 158 passed.
14. **Server pilot matrix:** Completed Qwen and Gemma over POPE, HallusionBench, VizWiz, and VSR in canonical and compact severity modes: 16 files, 800 canonical records, 9,600 severity records, and 10,400 records total.
15. **Artifact validation:** Directory-wide validation reported zero duplicate rows and zero schema failures. Generated five required plots and 250 transformed-image inspection files.
16. **Semantic calibration:** Human-labelled 50 train/validation VizWiz answer pairs (17 positive, 33 negative). Frozen threshold `0.50` achieved precision `0.5926`, recall `0.9412`, and F1 `0.7273`.
17. **Configuration freeze:** Frozen severities are blur `8`, crop `0.65`, brightness `0.15`, and noise `25`. `configs/probes/frozen_week3_config.yaml` has status `FROZEN`.
18. **Completion gate:** `scripts/check_week_completion.py --mode full_week` passed against the synced server artifacts on August 9, 2026.

**Week 4 handoff:**
1. Populate the currently empty Week 4 requirement matrix from Plan §25.6.
2. Pin immutable model revisions and prepare the full teacher-cache run approval card.
3. Generate at least the Qwen and Gemma full canonical teacher caches after explicit approval.
4. Build labels, grouped partial states, audit packet, leakage report, and checksum manifests.

---

## Week 4: Full teacher cache, labels, and partial states
**Status:** IMPLEMENTED, NOT VALIDATED
**Dates:** August 9, 2026–present

**What has been implemented locally:**
1. Added `scripts/run_teacher.py`: deterministic model-instance sharding, every legal canonical probe, frozen configuration/manifest provenance, immutable revision enforcement, and duplicate-safe resume.
2. Added `scripts/build_labels.py` and `src/proactive/teacher/offline.py`: independent signature, source-bit, and six-way label reconstruction. Embedded GPU labels must match exactly.
3. Added `scripts/sample_states.py`: empty, all legal singleton, every fixed-baseline prefix, and 16 deterministic random subsets. Learner input contains only clean numeric features and acquired probe observations.
4. Added `scripts/export_human_audit.py`: blinded 180-row packet, three annotator sections, renamed lossless internal images, private key, instructions, and checksums.
5. Added `scripts/validate_week4.py`: readiness, daily teacher progress, and full completion modes; reproducible train/validation-only label/bit tables; teacher/label/state manifests.
6. Removed user-specific image-root fallbacks and prohibited silent unpinned semantic-model fallback.
7. Added read-only local model provenance inspection in `scripts/inspect_model_revisions.py`.
8. Populated the Week 4 requirement matrix, staged server plan, and InternVL catch-up record.
9. Full Qwen generation produced 7,238 valid teacher rows across four shards;
   53 rows failed closed at the grounding probe rather than entering the cache
   with fabricated labels.
10. Added conservative recovery for explicit terminal `The answer is ...`
    outputs, rejection of empty tags/conflicting binary answers, a parser-drift
    check over every existing resume row, and an atomic deduplicated failure
    ledger retaining invalid raw outputs and provenance.
11. After the parser-only retry preserved all 53 failures, added a separate
    uniform grounding-refresh cache: every Qwen and Gemma row receives exactly
    one 512-token grounding pass, labels are recomputed, source hashes are
    checked on resume, and the original cache is never modified.

**Local validation:**
- Initial Week 4 implementation added 10 focused tests and passed the then-complete 168-test CPU suite.
- The corrected model-revision inspector and compute-authorization guard pass seven focused regression tests; the expanded complete CPU suite passes `175` tests.
- Readiness dry-run reports 7,291 manifest rows, 14,582 expected Qwen/Gemma teacher rows, and 102,294 clean/probe passes.
- Deterministic four-way shard sizes are 1,801–1,847 rows/model; each model requires exactly 51,147 passes.
- Server evidence resolved and pinned immutable revisions for Qwen, Gemma, and InternVL on August 10, 2026.
- The owner approved the `0.80` collapse gate, 5/5 bit balance, 60+120 audit composition, interim two-model/final three-model audit policy, and staged GPU checks. Local readiness passes; full-core compute remains separately locked.
- The revised grounding parser was checked against all 7,238 existing Qwen
  rows with zero normalized-answer drift. Nine new recovery/ledger tests pass;
  the complete local CPU suite is `191 passed in 1.94s` after the uniform
  refresh implementation and failure-sidecar filtering.

**Still required for completion:**
1. Recover Qwen's 53 fail-closed rows, finish the Gemma four-shard cache, and
   pass daily checksum validation (full compute approved 2026-08-13).
2. InternVL GPU catch-up cache for the required final three-model audit.
3. Offline label/state artifacts and balance/leakage reports.
4. Final 180-example audit packet and a passing full Week 4 gate.
