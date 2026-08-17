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
  grounding-recovery, and failure-ledger regression tests. The complete local
  CPU suite passes: `192 passed in 1.70s` on 2026-08-17.
- Audited the combined manifest: 7,291 rows (951 HallusionBench, 3,000 POPE, 3,000 VizWiz, 340 VSR), including 110 relation-applicable rows. Qwen plus Gemma require 14,582 teacher rows and 102,294 clean/probe passes.
- Accepted consistent server revision evidence and pinned Qwen
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, Gemma
  `292a7e278a400932df35f9fd4b1501edd04133a5`, and InternVL
  `5f618513e35a9b85922341b8057feddfc8880e50`.
- InternVL3-9B is downloaded at `/home/models/InternVL3-9B`; model availability
  and revision provenance are no longer blockers. Its Week 4 GPU validation and
  teacher catch-up cache remain pending.

**Current Blocker:**
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
- Gemma full-core generation is running/awaiting final server synchronization.
- Qwen shards contain 7,238 valid rows: shard valid counts are 1,813, 1,805,
  1,785, and 1,835. The 53-row resume must be run after syncing the recovery
  code and after a suitable GPU is free.
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
1. Finish and validate the running Gemma shards; do not interrupt them for the
   Qwen recovery.
2. Sync the grounding-refresh script and updated parser, then run one Qwen and
   one Gemma grounding pass per manifest row in parallel on separate GPUs.
3. Validate the separate refreshed cache. Do not build labels until Qwen and
   Gemma both reach 7,291/7,291 with zero refresh failures.
4. Build labels/states, complete the InternVL catch-up/final audit, and run the
   full Week 4 gate.

**Deviations from the Plan:**
- Week 3 core validation used two models over four active datasets; downloaded
  InternVL still requires its GPU catch-up run, and GQA-Relation remains
  scheduled for the catch-up window.
- Total Week 3 GPU-hours cannot be reconstructed exactly because several synced logs contain only resume/no-op or final fill segments. The available logs account for at least `2.9133` GPU-hours; future runs must retain complete start-to-finish logs.
- Week 4 policy-rollout and oracle-next partial subsets are explicitly deferred until those Week 5/6 artifacts exist; every current state records that dependency rather than fabricating unavailable trajectories.
