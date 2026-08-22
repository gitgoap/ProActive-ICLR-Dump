# Project Status

**Current Phase:** Week 4 — Full teacher cache, labels, and partial states

**Status:** IMPLEMENTED, NOT VALIDATED

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
  grounding-recovery, failure-ledger, and HallusionBench answer-contract
  regression tests. The complete local CPU suite passes: `213 passed` on
  2026-08-20.
- Audited the combined manifest: 7,291 rows (951 HallusionBench, 3,000 POPE, 3,000 VizWiz, 340 VSR), including 110 relation-applicable rows. Qwen plus Gemma require 14,582 teacher rows and 102,294 clean/probe passes.
- Accepted consistent server revision evidence and pinned Qwen
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, Gemma
  `292a7e278a400932df35f9fd4b1501edd04133a5`, and InternVL
  `5f618513e35a9b85922341b8057feddfc8880e50`.
- InternVL3-9B is downloaded at `/home/models/InternVL3-9B`; model availability
  and revision provenance are no longer blockers. Its first two one-row smoke
  attempts exposed a missing dependency and then Transformers-5/custom-code API
  incompatibility; no InternVL teacher row has yet been produced. An isolated
  Transformers-4.37.2 runtime and corrected native adapter are implemented
  locally and await server validation.

**Current Blocker:**
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

**Running or Awaiting Server Jobs:**
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
- InternVL3-9B is downloaded and pinned on the documented Week 4 catch-up path;
  GPU validation/cache generation have not started. GQA-Relation remains
  scheduled for Week 7–8.

**Next Tasks (Week 4):**
1. Run a uniform grounding-only refresh from the corrected base cache and
   validate exactly 7,291 rows/model with zero refresh failures. Do not build
   labels against the stale manifest/cache.
2. Build labels/states from the corrected cache, complete the InternVL
   catch-up/final audit, and run the full Week 4 gate.

**Deviations from the Plan:**
- Week 3 core validation used two models over four active datasets; downloaded
  InternVL still requires its GPU catch-up run, and GQA-Relation remains
  scheduled for the catch-up window.
- Total Week 3 GPU-hours cannot be reconstructed exactly because several synced logs contain only resume/no-op or final fill segments. The available logs account for at least `2.9133` GPU-hours; future runs must retain complete start-to-finish logs.
- Week 4 policy-rollout and oracle-next partial subsets are explicitly deferred until those Week 5/6 artifacts exist; every current state records that dependency rather than fabricating unavailable trajectories.
