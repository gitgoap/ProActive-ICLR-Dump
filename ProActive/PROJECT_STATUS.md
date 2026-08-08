# Project Status

**Current Phase:** Week 4 — Full teacher cache, labels, and partial states

**Status:** NOT STARTED

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

**Current Blocker:**
- Week 4 has not been approved for full GPU execution. Its requirements/run card must be completed before launching the 80–140 GPU-hour full teacher cache.
- Exact immutable model revisions are still recorded as `main`; revisions must be pinned before the full cache.
- `doc/docs/WEEK_04_REQUIREMENTS.md` is empty and must be populated from Plan §25.6 before implementation or scaling.

**Running or Awaiting Server Jobs:**
- None.
- InternVL3-9B and GQA-Relation remain on the documented catch-up path and must not delay the two-model core cache.

**Next Tasks (Week 4):**
1. Populate the Week 4 requirement matrix and conduct the pre-run leakage/schema audit.
2. Pin model revisions, freeze teacher-generation manifests/configuration, and prepare a measured full-run approval card for Qwen and Gemma.
3. After explicit approval, generate resumable full canonical teacher caches and validate counts/hashes daily.
4. Compute continuous signatures, source bits, six-way labels, grouped partial states, and checksum manifests.
5. Export the 180-example human-audit packet and run the strict leakage/class-balance completion gate.

**Deviations from the Plan:**
- Week 3 core validation used two models over four active datasets; InternVL and GQA-Relation remain scheduled for the catch-up window.
- Total Week 3 GPU-hours cannot be reconstructed exactly because several synced logs contain only resume/no-op or final fill segments. The available logs account for at least `2.9133` GPU-hours; future runs must retain complete start-to-finish logs.
