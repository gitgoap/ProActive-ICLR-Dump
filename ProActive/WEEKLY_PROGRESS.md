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
**Status:** COMPLETE
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
7. Complete Deep Sets VOI construction passed in 10:41.57 with 496,912
   train/validation rows and no forbidden-split access. The GRU comparison
   exposed a fail-closed freeze-contract inconsistency before inference; the
   implementation and regression test now accept any explicitly frozen
   diagnostic comparison while continuing to reject unfrozen checkpoints.
8. The freeze-contract correction passed 14 focused tests and the complete
   280-test server suite. Complete random-order GRU and masked-slot MLP VOI
   construction then passed in 10:46.19 and 10:43.50 respectively. All three
   architecture-specific corpora are `COMPLETE`, contain exactly 496,912
   train/validation targets (`436,429/60,483`), contain no calibration/test
   records, and preserve mixed positive/negative targets across the approved
   cost grid. The next gate is bounded learned-policy and train-only
   uncertainty-policy training before the full validation grid.
9. The bounded Deep Sets learned-VOI and train-only entropy-reduction policy
   pilots both exited zero. Each used exactly 100 train and 100 validation
   rows with no calibration/test access. The VOI policy selected epoch 19
   with validation loss `0.616179`, next-action agreement `0.72`, and STOP
   rate `0.40`; the uncertainty baseline selected epoch 13 with loss
   `0.541400`, agreement `0.66`, and STOP rate `0.46`. Complete training for
   these two paths is now the final measured step before scheduling the full
   Deep Sets cost/seed grid.
10. Complete Deep Sets policy training passed for cost multiplier `0.1`, seed
    42 in 25:59.92. It used all `436,429/60,483` train/validation targets,
    selected epoch 16, and achieved validation loss `0.580301`, next-action
    agreement `0.797365`, and STOP rate `0.534960`. The complete train-only
    entropy-reduction baseline passed in 19:26.90, selected epoch 10, and
    achieved loss `0.416474`, agreement `0.821586`, and STOP rate `0.406313`.
    Both reports are scientifically valid and explicitly exclude calibration
    and test. At measured throughput, the remaining 14-run Deep Sets grid is
    about six GPU-hours or roughly three wall-clock hours on two GPUs.
11. The owner-approved remaining grid completed on physical GPU 0 in 7.5586
    GPU-hours. The final matrix contains exactly 15/15 validation reports and
    60/60 expected policy files over all five cost multipliers and seeds
    42/43/44. Every report is scientifically valid with 436,429 training and
    60,483 validation targets; calibration/test flags are false, all checkpoint
    SHA-256 values match, and there are no missing, extra, or duplicate
    seed/cost combinations. Baseline staging is the next gate.
12. Scalar-confidence and one-pass-distilled baseline staging passed at limits
    1, 10, and 100. All six runs exited zero in 5.53--6.51 seconds, produced
    hash-matching best/last checkpoints and reports, and excluded calibration
    and test. The one-row reports correctly record undefined AUROCs as `null`
    and `metrics_scientifically_valid=false`; limits 10 and 100 have finite
    scientific metrics. Complete baseline training now awaits compute approval.
13. Complete scalar and distilled controls passed with scientifically valid
    full reports over all 1,404 validation model-instances, no calibration/test
    use, and matching checkpoint hashes. Scalar selected epoch 1 (Macro-F1
    `0.739826`, 15.75 seconds); distilled selected epoch 7 (Macro-F1 `0.757429`,
    24.53 seconds). Both happened sequentially on physical GPU 2 and remained
    far below the approved combined one-GPU-hour ceiling.
14. Temporary validation-only APS passed for clean-only, scalar-confidence,
    and one-pass-distilled controls. Each report fits all 1,404 validation
    rows at budgets 1/2/4 and coverages 0.90/0.95, excludes calibration/test,
    matches its checkpoint, and passes its signed self-hash. All three CPU jobs
    exited zero in under three seconds each.
15. The 100-row end-to-end validation frontier passed in 1:08.45 on physical
    GPU 2. All 15 required conditions, both oracles, budgets 1/2/4, full
    teacher at budget 7, signed dataset-specific schedule, and bound JSON/CSV/
    PNG hashes are present. Pilot indicators show ProActive beating random and
    fixed at one or more budgets; full 15-policy validation remains required.
16. The owner-approved complete Deep Sets validation-frontier matrix is now
    synchronized and audited. All 15/15 seed/cost reports are full validation
    runs over 1,404 model-instances, all 60 referenced report/CSV/PNG/schedule
    artifacts have matching hashes, and all 15 logs exit zero. The matrix used
    0.930472 GPU-hours total (214.87--238.12 seconds per run), far below the
    approved 7.5 GPU-hour ceiling. Read-only aggregation predicts lambda 0.0
    as the selection winner (mean source-bit Macro-F1 0.942567, mean cost
    2.284742, mean set size 2.167221), but the signed CPU-only selection and
    owner review remain mandatory. The GRU and masked-slot learned-policy
    comparison frontiers remain the other blocker to full Week 6 validation.
17. The formal CPU-only selection review produced a valid, self-hash-bound
    report and selected lambda 0.0 with primary seed 42 under the predeclared
    rule. No calibration or test data were accessed. ProActive beats random
    and fixed, budget progress passes, and the full-teacher-over-clean relative
    gain is 30.0581% against the 10% active-signal threshold. The command's
    review-stage exit code is intentionally nonzero pending explicit owner
    approval; Week 6 is not frozen or complete. GRU/masked comparison
    frontiers and the full validator remain required afterward.
18. The owner approved the measured lambda-0.0/seed-42 operating point on
    2026-09-04. This freezes the selection decision once the identical
    CPU-only command is rerun with `--approve_selection`; it does not authorize
    calibration/test access and does not replace the mandatory GRU/masked
    comparison frontiers or full Week 6 validation.
19. The CPU acceptance rerun reproduced the selected report without the
    review-stage stop. GRU and masked-slot VOI/entropy-reduction 100-row pilots
    then passed on GPU 3: 4/4 logs exit zero, 16/16 artifacts are present,
    checkpoint/report hashes match, each report uses 100 train and 100
    validation rows, and no calibration/test data were accessed. The four
    finite pilot losses are 0.755266, 0.533853, 0.548091, and 0.451863; total
    measured runtime is 129.85 seconds. `is_valid=false` correctly distinguishes
    bounded pilot checkpoints from complete scientific checkpoints. Full
    comparison training/frontiers and the full Week 6 validator remain.
20. The owner approved the complete random-permutation GRU and masked-slot MLP
    comparison bundle on 2026-09-05. It must run sequentially on a verified-free
    physical GPU 3, with a combined `3` GPU-hour ceiling, a `45`-minute timeout
    per training job, and a `15`-minute timeout per frontier. This authorization
    includes the architecture-bound scalar/distilled controls and temporary
    validation APS needed by each frontier; it does not unlock calibration or
    test data.
21. The complete comparison bundle was synchronized on 2026-09-06 after
    execution on verified-free physical GPU 1. All 14 logs exit zero in a
    combined 6,815.44 seconds (`1.8932` GPU-hours). Four complete policies,
    four architecture-bound one-pass controls, four temporary validation APS
    reports, and two complete frontiers are present and hash-consistent. Each
    frontier covers all 1,404 validation model-instances, 43 rows, 15
    conditions, budgets `1/2/4`, and both oracles. Both comparison policies
    beat random and fixed at one or more budgets. No calibration/test records
    were used; the CPU-only full Week 6 validator is now the final gate.
22. The CPU-only full Week 6 validator passed on 2026-09-06 with
    `is_valid=true`, `errors=[]`, approved configuration, and a complete
    496,912-row VOI corpus. Its self-hash independently reproduces as
    `fade7dc83ec2b94e3bbf3110457c18a800647026e7cc5de4ed05a8dedf29ff64`.
    Weeks 1--6 are therefore complete. Calibration/test remain locked while
    Week 7 first expands the validation-only dataset schedule to final budgets
    `1/2/3/4/7` and signs the main-stack freeze.
9. The bounded Deep Sets learned-VOI and train-only entropy-reduction policy
   pilots both exited zero. Each used exactly 100 train and 100 validation
   rows with no calibration/test access. The VOI policy selected epoch 19
   with validation loss `0.616179`, next-action agreement `0.72`, and STOP
   rate `0.40`; the uncertainty baseline selected epoch 13 with loss
   `0.541400`, agreement `0.66`, and STOP rate `0.46`. Complete training for
   these two paths is now the final measured step before scheduling the full
   Deep Sets cost/seed grid.

## Week 7: Frozen stack, APS, and locked test
**Status:** COMPLETE
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
5. The three validation-only static controls were expanded to the approved
   final budget grid `1/2/3/4/7` on 2026-09-06. Clean-only, scalar-confidence,
   and one-pass-distilled APS reports all contain both `0.90` and `0.95`
   targets, use the complete validation split, and record no calibration/test
   access. Their report hashes are `d4e62406...a292f`,
   `c2dc1bbc...ca732`, and `b8d81cab...7cd90`, respectively.
6. The selected Deep Sets lambda-0.0/seed-42 final-budget validation frontier
   completed on physical GPU 1 in `7:38.52` (`0.1274` GPU-hours), below the
   approved `0.25` GPU-hour and 15-minute limits. It is valid over 1,404
   validation model-instances, 71 frontier rows, all 15 conditions, both
   oracles, and budgets `1/2/3/4/7`. ProActive beats random and fixed at least
   once, and the CSV, figure, and dataset-schedule hashes match. Calibration
   and test were not used.

**Completion gate:** Passed. The immutable stack, final APS, complete locked
frontier, and all three permutation reports produced a signed `GO` decision.

7. The CPU-only freeze preview passed with exit code zero on 2026-09-06. The
   reviewed bundle contains the selected Deep Sets seed-42 diagnostic,
   lambda-0.0/seed-42 policy, signed Week 6 selection report, uncertainty,
   scalar and distilled controls, and the five-budget validation schedule.
   No freeze was written and calibration/test remain locked until explicit
   owner approval.
8. The owner approved the exact previewed main-stack boundary on 2026-09-06
   and authorized writing `outputs/week7_frozen/main_stack_freeze.json`.
   Calibration/test remain locked until the resulting manifest passes the
   synchronized hash audit.
9. The synchronized main-stack manifest passed that audit: status `FROZEN`,
   `owner_approved=true`, internal freeze hash `ec886b26...b21ef`, and all
   11 bound artifact hashes match. W7-01 is COMPLETE and the model-selection
   boundary is closed before any calibration or test access.
10. Week 7 readiness passed on 2026-09-06 with `is_valid=true`, approved
    configuration, zero errors, zero warnings, and self-hash
    `12e679f3...f136b`. `go_no_go=NO-GO` is the intentional readiness-mode
    state; the final locked validator alone can issue `GO`.
11. The owner approved complete calibration-only execution on 2026-09-06:
    frozen-policy trajectories on physical GPU 1 under `0.25` GPU-hours and a
    15-minute timeout, plus the three static APS calibration jobs on CPU in
    parallel. Test access remains prohibited.
12. Calibration completed and passed artifact audit. Five trajectory files
    contain exactly 1,500 rows each (7,500 total), every recorded file hash
    matches, the trajectory manifest is `COMPLETE`, and all provenance is
    calibration-only. The GPU job took 49.71 seconds (`0.0138` GPU-hours).
    Main APS plus clean/scalar/distilled APS are `FINAL_FROZEN`, cover budgets
    `1/2/3/4/7` at targets `0.90/0.95`, and bind to the signed freeze. All ten
    main APS cells pass the 0.03 undercoverage gate. W7-02 and W7-03 are
    COMPLETE; test has not yet been accessed.
13. The owner authorized the one-time complete locked-test bundle on
    2026-09-06: frontier on physical GPU 1 and primary Deep Sets plus two GRU
    permutation studies on physical GPU 2, only after free-device and
    contract-only checks. Combined compute is capped at one GPU-hour and no
    post-test tuning is permitted.
14. After GPU 1 failed its original empty-device preflight without reading test,
    the owner remapped the lightweight frontier/permutation jobs to shared
    physical GPUs 3/0 with a 1,536-MiB free-memory floor. All four contract
    checks and executions then exited zero. The frontier is complete and valid
    over 1,560 test model-instances, 142 rows, five budgets, both coverages,
    all controls, and both oracles; ProActive beats random and fixed. Every
    0.90/0.95 coverage cell passes the frozen 0.03 tolerance. Deep Sets records
    exactly zero permutation drift across 2,000 states, while canonical and
    random-order GRU provide the two required 2,000-state comparisons. Every
    self-hash and artifact binding matches. Aggregate GPU time was 548.41
    seconds (`0.1523` GPU-hours), below the approved one-hour ceiling. Only the
    CPU full validator remains; post-test tuning is prohibited.
15. The synchronized CPU-only full validator passed on 2026-09-07 with
    `is_valid=true`, `errors=[]`, and `go_no_go=GO`. Report self-hash
    `47fe3b99...9d0d7` reproduces, the emitted memo has no blocking findings,
    and its sole warning is the predeclared optional RAPS appendix trigger.
    APS remains the unchanged locked main method. Week 7 is COMPLETE and the
    project moves to Week 8 without any post-test tuning.

## Week 8: Generalization, held-out shift, ablations, latency, and audit
**Status:** IMPLEMENTED, NOT VALIDATED
**Dates:** September 8, 2026–present

**Implemented locally:**
1. Pinned official PRE-HAL and IllusionBench release contracts, strict loaders,
   safe download/extraction, deterministic 600-row-per-dataset sampling, and
   signed release/exclusion/manifests. PRE-HAL model weights are excluded.
2. Added mixed binary/multiple-choice prompting and normalization with
   fail-closed parsing. IllusionBench release defects are excluded uniformly
   before model inference and recorded by reason; no outcome-based filtering
   is permitted.
3. Extended the existing teacher, label, state, vectorization, and frontier
   path to the `shift` split. Final Week 7 APS is reused unchanged and all
   shift coverage is explicitly empirical only.
4. Added two-fold LOMO construction/freezing/evaluation, grouped bootstrap and
   paired intervals, within-dataset/model slices, deterministic qualitative
   cases, and fixed-hardware controller/MLLM latency accounting.
5. Added real component-ablation support: feature removal, no budget embedding,
   forced no-STOP rollout, loss-only VOI, no signature loss, independent source
   encoders, global APS, standardized evidence, and a fail-closed 15-item
   aggregation gate.
6. Added three independent blinded annotation packets, strict merge,
   adjudication, Fleiss-kappa/rule-match analysis, concise human instructions,
   and a Week 8 implementation/readiness/full validator.

**Pending evidence:** only the completed three-person human-audit artifacts and
the subsequent CPU-only full validator. Frozen shift, latency, two-fold LOMO,
the mandatory ablation aggregate, grouped statistics, slices, and qualitative
evidence are complete. The recovered held-out cache contains 2,400/2,400
accepted rows with zero unresolved failures and no exclusions. No post-shift
tuning is allowed.

**Server evidence and current pilot issue:**

7. The server suite passed 303 tests plus 16 subtests. PRE-HAL and
   IllusionBench were downloaded at their pinned revisions and verified; the
   signed shift manifest contains 600 rows from each dataset and was selected
   without model outputs or target-domain calibration.
8. The first Gemma one-row held-out stage on 2026-09-10 loaded the model but
   failed closed on its IllusionBench row because the special blank-image call
   omitted the row's `multiple_choice` normalizer. No valid teacher row was
   written. The blank call is corrected and an end-to-end regression now
   covers clean plus all applicable probes. After synchronization, 10 focused
   server tests passed. Resumed Gemma and Qwen stages each produced 1/1 and
   10/10 valid rows with zero unresolved failures, completing the parser gate.
9. The complete held-out traversal accounted for all requested rows but did not
   yet yield complete accepted caches: Qwen has 1,181 valid and 19 fail-closed
   rows; Gemma has 1,189 valid and 11 fail-closed rows. All 30 failures are
   mandatory-grounding format errors. A separate, provenance-preserving retry
   path was implemented and owner-approved for exactly this ledger-defined
   19-Qwen/11-Gemma scope; no row was dropped. Server tests and dry-run coverage
   formed the execution gate.
10. The approved recovery completed on 2026-09-11. Both model jobs exited zero,
    all 30 ledger-defined rows recovered, the separate recovered cache contains
    exactly 1,200 Qwen and 1,200 Gemma rows, and both failure ledgers are empty.
    The next gate is CPU construction of labels/states/vectorized shift data,
    followed by immutable Week 7 shift evaluation.
11. The CPU shift substrate is complete. Label construction produced 2,400
    rows, state sampling produced 63,687 leakage-safe states, the corrected
    metadata-aware index covers all 2,400 teachers, and two vectorized `shift`
    shards contain all 63,687 rows. The focused server suite passed 10 tests,
    and both tensor hashes match their signed manifest.
12. The frozen held-out shift frontier completed without target-domain
    calibration or post-test tuning. It evaluates 2,400 model instances across
    budgets `1/2/3/4/7`, both frozen coverage targets, all controls, and both
    oracles. ProActive exceeds the random schedule at budgets 1--4 for the
    primary 0.90 macro-F1 comparison; the signed report and bound hashes pass.
13. Fixed-hardware latency measurement is complete on an RTX A6000 with 10
    warm-ups and 100 synchronized measurements. Mean controller overhead is
    1.355 ms versus 9,163.951 ms mean cached generation latency.
14. The first Qwen LOMO-fold attempt stopped before training because the server
    executed a stale three-test builder that looked for state identity at the
    top level. Canonical `partial_state_v1` rows store `split` and `model_id`
    inside `metadata`. The corrected builder now prints revision
    `metadata_identity_v2`, reports observed identities on failure, and has an
    end-to-end filtering regression. No LOMO evidence was claimed.
15. The corrected server run passed all seven focused tests and materialized
    both LOMO folds with revision `metadata_identity_v2`. Holding out Qwen
    yields 194,249 states; holding out Gemma yields 194,374. Both folds contain
    source-model train/validation/calibration evidence and held-out-model test
    evidence only. Every referenced vector manifest, teacher file, label file,
    and state file exists and matches its recorded SHA-256. Fold construction
    is complete; diagnostic/policy training and source-only APS calibration
    remain pending.
16. The owner-approved two-fold LOMO bundle completed on 2026-09-11. It ran
    all 28 preparation, source-only calibration, freeze, and held-out
    evaluation stages successfully in 2,944 seconds (`0.818` aggregate
    GPU-hours), below its five-hour ceiling. Both 780-example held-out reports
    are valid; every report, CSV, trajectory, checkpoint, fold, freeze, and APS
    hash matches. At budget 7, ProActive source-bit Macro-F1 is `0.9692` on
    held-out Qwen versus `0.6826` clean-only and `0.6964` scalar, and `0.9948`
    on held-out Gemma versus `0.7956` and `0.7804`. Its corresponding six-way
    Macro-F1 values are `0.7927` and `0.8063`. Diagnostic transfer is
    therefore above both required controls. Coverage transfer is asymmetric:
    the 0.90-target coverage is `0.9910` for Qwen but `0.7115` for Gemma at
    budget 7. The paper claim is narrowed to useful diagnostic transfer and
    will report the Gemma calibration degradation explicitly; no held-out
    tuning or rerun is permitted.
17. The owner approved the complete 15-item Week 8 ablation bundle on
    2026-09-11: seed 42, validation-only evidence, at most two free GPUs, eight
    combined GPU-hours, 45-minute training caps, and 20-minute frontier caps.
    The implementation now applies feature removal consistently to diagnostic
    tensors, VOI counterfactuals, and acquired rollout evidence. GPU execution
    and the signed aggregate are pending.
18. The first full ablation attempt consumed 18,048 conservative GPU-seconds
    and completed `no_budget_embedding`, `no_confidence_shift`, and
    `no_relation_probe`; `no_answer_flip` and `shared_vs_independent_heads`
    reached partial policy checkpoints before their bounded timeouts. A resume
    then exposed a real idempotence defect: diagnostic `--resume` continued a
    run that had already stopped at epoch 10 and overwrote the selected
    `no_budget_embedding` checkpoint. The existing APS correctly rejected the
    new hash. Completed diagnostic and policy reports are now validated as
    immutable completion markers, histories continuing beyond the first
    stopping boundary are rejected. The epoch-zero recovery reproduced every
    epoch-0--10 metric exactly, but the raw `.pt` SHA changed because atomic
    PyTorch saves use a randomly named temporary ZIP root. Recovery therefore
    requires exact stopping-history and regenerated-APS scientific equality,
    archives the stale descendants, and rebuilds their hash chain rather than
    editing provenance. The ledger now includes the 228-second repair run.
19. The exact-history/APS-equivalence gate passed and the complete ablation
    continuation finished every remaining training, freeze, calibration, VOI,
    frontier, reference, and latency stage. The final ledger contains 28,275
    conservative GPU-seconds (`7.8542` GPU-hours), below the approved eight-hour
    ceiling. All 15 required evidence JSONs exist; their self-hashes and 29
    bound input hashes independently match. Final aggregation alone failed
    because repeated `--evidence` CLI occurrences replaced earlier values.
    The parser now accumulates repeated bindings and has a regression test.
    No GPU rerun is needed; the CPU-only signed aggregate remains pending.
20. The aggregation regression passed (`1 passed, 10 deselected`) and the
    CPU-only rerun produced the signed 15-item bundle. Independent checks
    reproduce report SHA-256 `ee0a3eb...e14b`, CSV SHA-256
    `7dfcbbb7...2613`, all 15 evidence hashes, and 121 aggregate data rows.
    `post_test_tuning_used=false`; W8-04 is COMPLETE.
21. The complete grouped Week 8 statistical analysis finished on 2026-09-13
    with exit code zero after 1:13:54. It analyzed 340,800 frozen shift
    trajectories using 2,000 grouped bootstrap resamples and produced 710
    confidence-interval rows, 120 paired-comparison rows, 160 slice rows, and
    20 deterministic qualitative cases. The analysis report self-hash and all
    bound hashes match. ProActive beats random at budgets 1--4, with paired
    source-bit Macro-F1 gains of `0.0458`, `0.1189`, `0.1474`, and `0.1178`.
    At the predeclared budget-7 primary comparison, random is slightly higher
    (`-0.0063` ProActive-minus-random; Holm-adjusted `p=0.003`), while ProActive
    uses about `0.42` fewer probes. This saturation result narrows the paper
    claim to constrained-budget value rather than universal dominance. W8-07,
    W8-08, and W8-09 are COMPLETE.
