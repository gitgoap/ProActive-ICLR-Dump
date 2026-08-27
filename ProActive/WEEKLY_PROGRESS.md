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
**Status:** COMPLETE
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
12. Audited the official HallusionBench source and found 14 open-ended
    image-table questions inside the 951-row image subset. Implemented a mixed
    per-record answer contract without excluding any examples: 937 binary rows
    use normalized benchmark indicators and 14 open rows use
    `gt_answer_details` plus a frozen author-audited alias overlay.
13. Added a conservative cache migration. It verifies that no non-Hallusion
    manifest row changed, recomputes binary correctness/features/labels from
    existing raw generations, invalidates all 14 prompt-changed rows for every
    model, and records source file/record plus old/new manifest hashes in a
    separate output directory.
14. Completed migration-v2 recovery and a uniform 1,024-token grounding
    refresh. Every Qwen/Gemma model-instance slot remains accounted for; Qwen
    has 7,285 valid plus 6 explicit failures, and Gemma has 7,288 valid plus 3
    explicit failures.
15. Rebuilt InternVL around its native image-token interface and isolated its
    official Transformers-4.37.2 runtime from the accepted Qwen/Gemma base
    environment. The focused 11-test adapter suite passed on the server; after
    adding grounding recovery coverage, all 26 focused recovery/parser/refresh
    tests and all 230 repository tests pass.
16. Completed InternVL 1/10/100-row and complete-VSR GPU stages with zero model
    failures. The 100-row stage covers all four active datasets; complete VSR
    contains 340 rows, 2,150 probes, and all 110 relation-applicable examples.
17. Validated the final grounding-recovery implementation with 26 focused
    tests, all 230 repository tests, and eight real-source dry-runs. The dry
    runs identify exactly six Qwen and three Gemma format failures under one
    uniform retry policy; no row is excluded.
18. Completed the approved nine-row GPU recovery. The final core cache contains
    7,291 valid Qwen rows and 7,291 valid Gemma rows, all failure ledgers are
    empty, and teacher-progress validation passes with 87,712 unique probes and
    zero errors.
19. Hardened offline artifact discovery so failure-ledger sidecars cannot enter
    label, state, or audit construction. The focused suite passes 17 tests and
    the complete server suite passes 236 tests.
20. Generated 14,582 offline labels and 388,623 leakage-safe partial states.
    Independent inspection found no duplicate IDs, forbidden learner metadata,
    observation/name mismatch, or missing mandatory sampling sources. The
    dominant train/validation class is 41.42%, below the approved 80% gate,
    and every dataset/model/source-bit slice clears the 5/5 balance minimum.
21. Exported and server-validated the 180-row human-audit packet with 30 rows
    per label, 180 images, all three models, and all four datasets. The first
    full report passes every scientific artifact check and is false only
    because the server copy of the InternVL catch-up document was stale.
22. Synced the final catch-up document and reran the full validator on
    2026-08-26. Every readiness, teacher, label/state, audit, and catch-up term
    passes with zero errors and warnings. Week 4 is COMPLETE.

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
- HallusionBench contract and migration tests include unit, adversarial, and
  end-to-end shard coverage. The complete local CPU suite passes `213 passed`
  on 2026-08-20. GPU behavior remains unclaimed until server logs
  validate the rebuilt manifest and migrated cache.
- The strict transition guard discovered 207 VizWiz gold-answer changes caused
  by hash-randomized set tie breaking. Replaced it with normalized majority and
  released-source-order tie resolution; manifests now preserve normalized
  counts/tie size, and migration recomputes correctness/labels without rerunning
  VizWiz inference.
- InternVL server validation passed 1, 10, 100, and 340-row stages with zero
  failures. A generic Week 3 schema check falsely rejected only cal/test rows;
  independent Week 4 checks confirmed every row, probe set, manifest field, and
  provenance hash is valid.

**Completion evidence:**
1. The uniform recovery resolved all nine remaining format failures with no
   exclusions; Qwen and Gemma each have 7,291 valid rows.
2. Offline generation produced 14,582 labels and 388,623 leakage-safe states.
3. The final audit packet contains 180 rows/images over three models and four
   datasets.
4. `validate_week4.py --mode full` passed on 2026-08-26 with zero errors and
   zero warnings. Human agreement scoring continues as a later parallel study.

---

## Week 5: Shared-state encoder bake-off
**Status:** COMPLETE
**Dates:** August 26, 2026–present

**Implemented locally:**
1. Strict metadata-free tensor contract and resume-safe vectorization of the
   frozen Week 4 states.
2. Clean-only MLP, canonical/random-order GRU, exact masked-slot MLP, and
   sum-pooled Deep Sets with shared source-bit, six-way, and signature heads.
3. Train-only normalization/class weighting, validation-only selection,
   temporary APS, identity shortcut controls, and permutation-drift metrics.
4. Three-seed selection gate with Deep Sets as the declared primary encoder;
   Set Transformer remains disabled unless the predeclared validation gate
   fires.
5. Unit, adversarial, resume, leakage, and synthetic end-to-end tests. Local
   syntax compilation passes; PyTorch execution awaits the server suite.
6. Server validation on 2026-08-26 passed all `279` tests, vectorized all
   388,623 states in 75.06 seconds, and passed isolated 1/10/100-row Deep Sets
   and canonical-GRU stages with exit code zero.
7. The owner approved the proposed Weeks 5–7 scientific settings on
   2026-08-26. Full Week 5 training is now authorized after regenerating the
   vectorized artifacts under the approved configuration hash.
8. Approved-hash vectorization and Week 5 readiness passed. The first two
   mandatory full checkpoints also passed: Deep Sets seed 42 completed in
   6:09.83 with source-bit/six-way Macro-F1 `0.8641/0.5025`; canonical GRU
   seed 42 completed in 2:31.51 with `0.8621/0.4675`. Thirteen checkpoints
   remain before the validation-only selection gate.
9. The latest synchronized snapshot contains `11/15` complete validation
   reports. Seed-42 temporary APS and permutation evaluation passed for Deep
   Sets and canonical GRU. Deep Sets produced exactly zero drift, while the
   canonical GRU produced nonzero order drift (mean JS `0.000586`, hidden
   relative drift `0.07684`, bit-probability L1 drift `0.02585`, and set
   disagreement `0.08522`), confirming the intended contrast.
10. The subsequent synchronized snapshot contains all `15/15` checkpoint
    validation reports, all `12/12` non-clean permutation reports, all `12/12`
    temporary APS reports, and the shortcut-control report. Aggregated
    validation evidence predicts that the Deep Sets gate will pass: its
    source-bit Macro-F1 is `0.864217`, only `0.000116` behind the best GRU and
    exactly invariant; its six-way Macro-F1 is the best at `0.508031`; and its
    advantage over the identity-only control is `0.097711`. The Set Transformer
    trigger is not reached. CPU-only signed selection and owner review remain.
11. The signed CPU-only selection report passed and selected Deep Sets seed 42
    with completion gate true. It used train/validation only, did not access
    calibration or test, passed the shortcut gate, and did not trigger the Set
    Transformer. Its report SHA-256 is `03a2e49f...36916`. The owner approved
    Deep Sets seed 42 and authorized the signed freeze on 2026-08-26; freeze
    creation and full validation remain.
12. The owner-approved freeze was written with Deep Sets seed 42 and validated
    successfully. Freeze SHA-256 is `0512ca85...a7929`; the full Week 5 report
    is valid with zero errors/warnings and SHA-256 `026ef289...207e`.

## Week 6: VOI targets, policy, and baseline frontier
**Status:** IMPLEMENTED, NOT VALIDATED; Week 5 prerequisite satisfied
**Dates:** August 26, 2026–present

**Implemented locally:**
1. Exact cached realized-VOI construction over train/validation only, with one
   counterfactual evaluation reused over the declared cost grid.
2. Action-conditioned VOI head, ranking/MSE/action-CE objective, legal masks,
   exact cost accounting, and STOP for nonpositive predicted value.
3. Main/GRU/masked learned-policy support plus random, four fixed,
   validation-selected dataset-specific fixed, non-leaking uncertainty,
   scalar, clean-only, distilled, full-teacher, oracle-next, and batched
   oracle-subset controls.
4. Three-seed validation selection, matched-cost frontiers, paper CSV/plots,
   action frequencies, oracle gaps, provenance gates, and Week 6 validator.
5. Server readiness passed on 2026-08-27 with zero errors. The VOI preflight
   verified eight hash-bound state/teacher sources, budgets `1/2/4`, all five
   approved cost multipliers, the frozen Deep Sets checkpoint, and no
   calibration/test access.
6. The 100-row CPU VOI construction audit passed in 4.36 seconds with 100
   train rows, no calibration/test access, mixed positive/negative targets at
   every cost, and monotonic STOP growth from 39% at cost 0 to 54% at cost
   0.4. Complete architecture-specific VOI construction is authorized.

## Week 7: Frozen stack, APS, and locked test
**Status:** IMPLEMENTED, NOT VALIDATED; blocked on Week 6 selection
**Dates:** August 26, 2026–present

**Implemented locally:**
1. Explicit owner-approved freeze of the selected diagnostic, policy,
   controls, dataset schedule, configs, and hashes before calibration.
2. Calibration-only frozen-policy trajectories and finite-sample APS at
   budgets 1/2/3/4/full for both 90% and 95% targets, including independently
   calibrated zero-acquisition controls.
3. Freeze-bound locked-test frontier and full permutation protocol with the
   GRU canonical/random-augmentation comparison.
4. Fail-closed coverage/comparator gates and an automatically generated Week 7
   go/no-go memo. No test execution is possible before freeze plus calibration.
