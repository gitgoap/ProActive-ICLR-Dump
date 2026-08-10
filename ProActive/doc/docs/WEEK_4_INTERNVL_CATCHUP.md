# Week 4 InternVL catch-up shard

**Status:** SCHEDULED — not yet executed

InternVL3-9B is the third-model catch-up path permitted by the Week 4
completion gate. It must use the same grouped manifest, frozen Week 3 probe
configuration, immutable model revision, seed 42, and four deterministic
shards as Qwen and Gemma. The cache is not part of the two-model core approval
until its server smoke test and exact revision are confirmed.

Before launch, replace the `main` revision in
`configs/models/internvl3_9b.yaml` with the immutable 40-character commit hash.
Then use the same `scripts/run_teacher.py` commands documented in
`doc/docs/WEEK_4_SERVER_EXECUTION.md`, substituting `--model internvl3_9b`.

Record start/end times, GPU ID, command, output files, row counts, and SHA-256
checksums in `RUN_REGISTRY.md` after execution.
