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
- Added 10 Week 4 unit/adversarial/integration tests. The complete local CPU suite passes: `168 passed in 2.75s`.
- Audited the combined manifest: 7,291 rows (951 HallusionBench, 3,000 POPE, 3,000 VizWiz, 340 VSR), including 110 relation-applicable rows. Qwen plus Gemma require 14,582 teacher rows and 102,294 clean/probe passes.

**Current Blocker:**
- `configs/experiments/teacher_core.yaml` is deliberately `DRAFT_REQUIRES_APPROVAL`. The owner must approve/revise the proposed class-collapse gate (`0.80`), per-slice bit minimum (5 positive and 5 negative), and 60-natural + 120-targeted audit composition.
- Exact immutable Qwen and Gemma revisions are still recorded as `main`. Run `scripts/inspect_model_revisions.py` on the server and review its output before changing the model YAMLs.
- The literal audit protocol requires all three models. The owner must decide whether Week 4 waits for InternVL or permits only an interim two-model packet while the documented catch-up shard runs.
- Full Qwen+Gemma generation is projected at about `33.23 GPU-hours` and is HIGH-COST. It requires staged 1/10/100/full-dataset evidence and explicit approval before pane commands are issued.

**Running or Awaiting Server Jobs:**
- None.
- Awaiting model-revision inspection and owner decisions; no GPU job should run yet.
- InternVL3-9B is on the documented Week 4 catch-up path. GQA-Relation remains scheduled for Week 7–8.

**Next Tasks (Week 4):**
1. Sync the Week 4 implementation, run the read-only server model-revision inspection, and return its complete log.
2. Resolve the open scientific/compute decisions, pin revisions, and pass the readiness gate.
3. Execute and review the mandatory 1/10/100/full-VSR stages before preparing the high-cost four-GPU approval card.

**Deviations from the Plan:**
- Week 3 core validation used two models over four active datasets; InternVL and GQA-Relation remain scheduled for the catch-up window.
- Total Week 3 GPU-hours cannot be reconstructed exactly because several synced logs contain only resume/no-op or final fill segments. The available logs account for at least `2.9133` GPU-hours; future runs must retain complete start-to-finish logs.
- Week 4 policy-rollout and oracle-next partial subsets are explicitly deferred until those Week 5/6 artifacts exist; every current state records that dependency rather than fabricating unavailable trajectories.
