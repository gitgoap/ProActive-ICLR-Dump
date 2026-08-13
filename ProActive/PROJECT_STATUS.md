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
- Added Week 4 unit/adversarial/integration, revision-parser, and compute-authorization regression tests. The complete local CPU suite passes: `175 passed in 2.73s` after staged approval.
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
- Full Qwen+Gemma generation is projected at about `33.23 GPU-hours` and remains
  unapproved. The runner fails closed on that scope until the staged logs pass
  review and the owner gives a separate high-cost approval.
- Interim two-model work is allowed, but the final Week 4 human audit still
  requires InternVL.

**Running or Awaiting Server Jobs:**
- None.
- Server readiness plus Qwen and Gemma one-row stages passed independent
  teacher-progress validation with 2 rows, 12 probe records, and zero errors.
- Qwen 10-row and 100-row stages are pilot validated. The 100-row cache has 100
  teacher rows, 602 legal probe records, zero errors, and completed in 586.2
  seconds.
- Complete Qwen VSR is pilot validated on the server: 340 rows, 2,150 probes,
  zero errors, and 1,325.9 seconds. The first local JSONL sync is truncated and
  must be repeated; the server experiment itself does not need rerunning.
- InternVL3-9B is downloaded and pinned on the documented Week 4 catch-up path;
  GPU validation/cache generation have not started. GQA-Relation remains
  scheduled for Week 7–8.

**Next Tasks (Week 4):**
1. Re-sync the completed Qwen VSR JSONL and verify its 340 rows and SHA-256.
2. Obtain separate explicit approval for the approximately 33.23 GPU-hour full
   Qwen/Gemma core run.
3. After approval, unlock and launch the four-shard core cache and monitor
   the high-cost four-GPU core run.

**Deviations from the Plan:**
- Week 3 core validation used two models over four active datasets; downloaded
  InternVL still requires its GPU catch-up run, and GQA-Relation remains
  scheduled for the catch-up window.
- Total Week 3 GPU-hours cannot be reconstructed exactly because several synced logs contain only resume/no-op or final fill segments. The available logs account for at least `2.9133` GPU-hours; future runs must retain complete start-to-finish logs.
- Week 4 policy-rollout and oracle-next partial subsets are explicitly deferred until those Week 5/6 artifacts exist; every current state records that dependency rather than fabricating unavailable trajectories.
