# Project Status

**Current Phase:** Week 7 — stack freeze, final-budget validation schedule, calibration, and locked test

**Status:** Weeks 1–6 COMPLETE; Week 7 IMPLEMENTED, NOT VALIDATED

**Completed Work:**
- Weeks 1–2: repository/data scaffolding, grouped splits, normalization, clean features, and Qwen/Gemma/InternVL adapters.
- Qwen3-VL-8B-Instruct and Gemma-4-E4B-it passed real GPU smoke tests.
- Week 3 probe implementations and fail-closed teacher-label pipeline completed.
- Completed the Qwen/Gemma pilot matrix over POPE, HallusionBench, VizWiz, and VSR:
  - 16 cache files;
  - 800 canonical records;
  - 9,600 compact severity records;
  - 10,400 total valid records with zero duplicates and zero schema failures.
- Generated five required diagnostic plots and 50 inspection images for each of five visual probes (250 total).
- Completed the 50-pair train/validation VizWiz semantic audit: 17 positive and 33 negative labels.
- Froze semantic threshold `0.50` (precision `0.5926`, recall `0.9412`, F1 `0.7273`).
- Froze canonical visual severities: blur `8`, crop `0.65`, brightness `0.15`, noise `25`.
- `scripts/check_week_completion.py --mode full_week` passed against the synced server artifacts on 2026-08-09.
- Local CPU suite passed on 2026-08-09: `158 passed`.
- Implemented the Week 4 deterministic four-shard teacher runner with strict resume validation, frozen-config provenance, and immutable-revision enforcement.
- Implemented independent label recomputation, leakage-safe pre-policy partial states, a blinded 180-example audit exporter, class/bit reports, checksum manifests, and readiness/progress/full gates.
- Added Week 4 unit/adversarial/integration, revision-parser, compute-authorization,
  grounding-recovery, failure-ledger, HallusionBench answer-contract, and
  InternVL native-adapter regression tests. After adding the final grounding
  recovery and offline-sidecar tests, the isolated server CPU suite passes:
  `236 passed` on 2026-08-24. The focused recovery/parser/refresh suite passes
  `26` tests, and the focused offline-sidecar/Week-4 suite passes `17` tests.
- Audited the combined manifest: 7,291 rows (951 HallusionBench, 3,000 POPE, 3,000 VizWiz, 340 VSR), including 110 relation-applicable rows. Qwen plus Gemma require 14,582 teacher rows and 102,294 clean/probe passes.
- Accepted consistent server revision evidence and pinned Qwen
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, Gemma
  `292a7e278a400932df35f9fd4b1501edd04133a5`, and InternVL
  `5f618513e35a9b85922341b8057feddfc8880e50`.
- InternVL3-9B is downloaded at `/home/models/InternVL3-9B`; model availability
  and revision provenance are no longer blockers. Its first two one-row smoke
  attempts exposed a missing dependency and then Transformers-5/custom-code API
  incompatibility. The isolated
  Python-3.11/Transformers-4.37.2 environment and corrected native adapter now
  pass `11` focused adapter tests; the expanded repository suite now passes
  all `236` tests on the server. The
  corrected one-row GPU smoke also passed with one valid teacher row, six
  applicable probes, and zero failures in 101.6 seconds. The 10-row stage then
  passed with 10 unique valid rows, 60 probes, and zero failures in 170.0
  seconds. The 100-row and complete-VSR stages also passed with 100 and 340
  valid rows respectively and zero model failures.

**Current Blocker:**
- The complete Week 5 experiment matrix is synchronized: `15/15` checkpoint
  validation reports, `12/12` non-clean permutation studies, `12/12` temporary
  APS reports, and the shortcut-control report. Pre-gate inspection predicts a
  passed Deep Sets selection, no Set Transformer trigger, and an appendix-only
  RAPS trigger.
- The CPU-only signed Week 5 selection report passed and selected Deep Sets
  seed 42 without calibration/test access, and the owner approved the measured
  selection on 2026-08-26. The owner-approved freeze and full Week 5 validator
  subsequently passed with zero errors and warnings. Week 5 is COMPLETE.
- There is no remaining Week 4 data, GPU, or validation blocker. Weeks 6 and 7
  are implemented in advance but their execution remains scientifically gated
  by the Week 5 and Week 6 validation selections respectively.
- Week 6 readiness passed with zero errors on 2026-08-27. VOI preflight
  verified all eight source files and frozen hashes. The bounded audit and
  complete Deep Sets, random-order GRU, and masked-slot MLP VOI builds all
  passed. Each complete corpus contains 496,912 train/validation targets and
  no calibration/test records. Bounded learned-policy and non-leaking
  entropy-reduction policy training also passed with finite metrics, 100/100
  train/validation rows, and no calibration/test access. The immediate gate
  advanced through one complete run of each path. Both are scientifically
  valid; the main cost-0.1/seed-42 policy took 25:59.92 and the uncertainty
  baseline took 19:26.90. Server logs now show all 14 remaining validation-only
  policies completed successfully in 7.5586 GPU-hours. The complete resync now
  contains all 60 policy files: 15 best checkpoints, 15 last checkpoints, 15
  histories, and 15 validation reports. Every checkpoint hash, seed/cost
  combination, row count, and train/validation-only provenance check passes.
- The complete 15-run Deep Sets validation-frontier matrix is synchronized and
  independently audited: all 15 full reports and all 60 referenced artifacts
  are present and hash-valid, all reports cover 1,404 validation
  model-instances with both oracles, and every log exits zero. Total measured
  compute was 0.930472 GPU-hours. The subsequent signed CPU-only selection
  chose cost multiplier 0.0 and seed 42 and has been owner-approved. The
  required W6-13 GRU and masked-slot comparison frontiers are also present. The
  CPU-only full Week 6 validator subsequently passed with 496,912 complete VOI
  rows and zero errors, so Week 6 is COMPLETE.
- The formal CPU-only selection report is now synchronized and hash-valid. It
  selects cost multiplier 0.0 and primary seed 42, excludes calibration/test,
  and passes random, fixed, budget-progress, and active-signal gates. The full
  evidence gain over clean-only is 30.0581%, above the predeclared 10% minimum.
  The owner approved cost multiplier 0.0 and seed 42 on 2026-09-04, and the
  identical CPU-only acceptance rerun reproduced the signed report. The
  selection boundary is operationally recorded; calibration/test remain
  locked.
- GRU and masked-slot VOI plus entropy-reduction policy pilots passed on GPU 3
  at 100 train/100 validation rows. All four logs exit zero, all 16 artifacts
  are present and hash-valid, metrics are finite, and no calibration/test data
  were used. The intentional `is_valid=false` marks them as bounded pilots.
- The complete GRU-random and masked-slot comparison bundles are now
  synchronized and independently audited. All 14 jobs exited zero in `1.8932`
  GPU-hours on physical GPU 1. Each architecture has complete VOI and
  entropy-reduction policies, scalar/distilled controls, validation-only APS,
  and a 1,404-instance frontier with all 15 conditions and both oracles. All
  checked artifact hashes match, and neither calibration nor test was used.
- Week 7 final-budget static APS validation is synchronized and valid for the
  clean-only, scalar-confidence, and one-pass-distilled controls. All three
  reports use validation only, cover budgets `1/2/3/4/7` and targets
  `0.90/0.95`, have `limit=null`, and record no calibration/test access. The
  selected Deep Sets validation frontier subsequently passed on physical GPU
  1 in 7:38.52. It covers 1,404 validation model-instances, 71 frontier rows,
  all 15 conditions, both oracles, and budgets `1/2/3/4/7`; all referenced
  CSV, figure, and dataset-schedule hashes match. The schedule is validation
  only and both learned-policy comparison gates pass. The next task is the
  CPU-only main-stack freeze review.
- The owner-approved main-stack freeze is synchronized and valid. It has
  status `FROZEN`, internal freeze hash `ec886b26...b21ef`, and 11/11 bound
  artifact hashes match. The selection boundary is now closed. The next task
  CPU Week 7 readiness gate subsequently passed with `is_valid=true`, zero
  errors/warnings, and report self-hash `12e679f3...f136b`. Its `NO-GO` value
  is expected for readiness mode; only the final validator can emit `GO`. The
  next task is frozen-policy calibration-only trajectories; test remains
  locked.
- Final calibration is now synchronized and audited. The frozen policy produced
  7,500 calibration-only trajectories: 1,500 complete rows at each budget
  `1/2/3/4/7`, with all file hashes matching and no test access. Main APS and
  all three static-control APS reports are `FINAL_FROZEN`, use both `0.90` and
  `0.95`, bind to the main freeze, and have `limit=null`. All ten main APS
  calibration cells satisfy the approved 0.03 undercoverage tolerance. The
  trajectory GPU job took 49.71 seconds (`0.0138` GPU-hours).

**Resolved Week 4 history:**
- The Qwen/Gemma teacher cache is no longer blocked. The approved final
  grounding recovery completed on 2026-08-24: all six Qwen and three Gemma
  format failures recovered, all eight failure ledgers are empty, and official
  teacher-progress validation passes with 14,582/14,582 rows, 87,712 unique
  probe records, and zero errors. The active blocker is now generation and
  validation of offline labels, leakage-safe partial states, and the final
  three-model human-audit packet. Label/state generation has now completed
  with 14,582 labels and 388,623 unique leakage-safe states; independent checks
  pass. The server full report now validates the 180-row, three-model,
  four-dataset audit packet as well. Its only failing term is
  `catch_up.documented=false`, caused by the stale server copy of the InternVL
  catch-up document. After synchronization, the repeated full report passed
  every term with zero errors and warnings.
- The released HallusionBench JSON contains 14 genuinely open-ended image-table
  questions among the 951 image-paired records. The old manifest treated the
  benchmark-level `gt_answer` indicator as a literal answer for every row,
  making HallusionBench clean correctness stale. No rows will be excluded.
- Answer contract v1 is implemented locally: 937 binary rows use normalized
  `0/1/2 -> no/yes/uncertain`; the 14 open rows use official
  `gt_answer_details` plus an author-audited canonical-alias overlay. A
  fail-closed migration reuses unaffected inference, invalidates all 14 open
  rows independently of model outcomes, and writes a separate cache with
  old/new manifest and source-record hashes. Migration schema v2 is now server
  validated, including all signed artifact hashes and all eight resume dry-runs.
- The paper-facing rationale, complete 14-row mapping, deterministic scoring
  rule, migration policy, draft methods paragraph, and required evidence are
  maintained in `doc/docs/HALLUSIONBENCH_14_OPEN_ENDED_RECORDS.md`.
- The first strict migration attempt exposed a separate VizWiz reproducibility
  defect: unordered-set tie breaking changed 207 gold answers across identical
  rebuilds. Deterministic normalized-majority/source-order selection and
  CPU-only cache relabeling are implemented and documented in
  `doc/docs/VIZWIZ_DETERMINISTIC_GOLD_SELECTION.md`. Two server rebuilds under
  different Python hash seeds produced identical 7,291-row manifests, and the
  audited migration relabelled the cached outputs without VizWiz inference.
- Scientific settings and staged 1/10/100/full-VSR checks were owner-approved
  on 2026-08-10. Both local and server readiness validators now pass without
  errors.
- Full Qwen+Gemma generation was separately approved on 2026-08-13 after all
  staged gates passed. At approval time only physical GPU 1 was free; GPUs 0,
  2, and 3 were occupied by unrelated processes and must not be used until
  rechecked as free.
- Interim two-model work is allowed, but the final Week 4 human audit still
  requires InternVL.
- The first full Qwen pass produced 7,238/7,291 valid rows. All 53 missing rows
  failed closed at the grounding probe (30 HallusionBench, 23 VizWiz); they
  were not silently converted to labels. A conservative parser recovery and
  deduplicated raw-failure ledger are implemented locally and await server
  synchronization/retry.
- The first parser-only retry correctly preserved all 53 failures. Inspection
  showed that 36 outputs were truncated at the uniform 256-token ceiling and
  17 were explicit structured/bare answers. A separate uniform 512-token
  grounding refresh is now implemented locally for every Qwen and Gemma row;
  server validation is pending.

**Week 4 server evidence (complete):**
- Migration-v2 recovery is complete. Qwen has 7,255 valid plus 36
  grounding-only failures; Gemma has 7,217 valid plus 74 grounding-only
  failures. All 14 open Hallusion rows are valid for both models, every one of
  the 14,582 model-instance slots is accounted for, and no row was excluded.
- Server readiness plus Qwen and Gemma one-row stages passed independent
  teacher-progress validation with 2 rows, 12 probe records, and zero errors.
- Qwen 10-row and 100-row stages are pilot validated. The 100-row cache has 100
  teacher rows, 602 legal probe records, zero errors, and completed in 586.2
  seconds.
- Complete Qwen VSR is pilot validated and locally archived: 340 rows, 2,150
  probes, zero errors, 1,325.9 seconds, and matching decompressed/server SHA-256.
- InternVL3-9B is downloaded and pinned on the documented Week 4 catch-up path.
  Its isolated runtime plus corrected 1/10/100-row and complete-VSR GPU stages
  are server-validated. The catch-up now covers all four datasets and complete
  VSR with zero failures. GQA-Relation remains scheduled for Week 7–8.

**Next Tasks (Week 7):**
1. Obtain explicit authorization for the one-time locked-test bundle, then run
   the complete test frontier and three required permutation studies without
   tuning or reruns based on outcomes.
3. Continue the three-person human annotations in parallel; they remain a
   Week 8 agreement study and do not block Week 6 execution.

**Deviations from the Plan:**
- Week 3 core validation used two models over four active datasets; downloaded
  InternVL still requires its GPU catch-up run, and GQA-Relation remains
  scheduled for the catch-up window.
- Total Week 3 GPU-hours cannot be reconstructed exactly because several synced logs contain only resume/no-op or final fill segments. The available logs account for at least `2.9133` GPU-hours; future runs must retain complete start-to-finish logs.
- Week 4 policy-rollout and oracle-next partial subsets are explicitly deferred until those Week 5/6 artifacts exist; every current state records that dependency rather than fabricating unavailable trajectories.
