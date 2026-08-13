# Week 4 InternVL catch-up shard

**Status:** DOWNLOADED AND REVISION PINNED; GPU VALIDATION NOT STARTED

InternVL3-9B is the third-model catch-up path permitted by the Week 4
completion gate. It must use the same grouped manifest, frozen Week 3 probe
configuration, immutable model revision, seed 42, and four deterministic
shards as Qwen and Gemma. The cache is not part of the two-model core approval
until its server smoke test passes.

The model files are present on `bumblebee` at
`/home/models/InternVL3-9B`. Download availability is confirmed; this status
must not be confused with a successful adapter/teacher GPU validation.

Server evidence resolved immutable revision
`5f618513e35a9b85922341b8057feddfc8880e50`, now pinned in
`configs/models/internvl3_9b.yaml`. Use the same `scripts/run_teacher.py` commands documented in
`doc/docs/WEEK_4_SERVER_EXECUTION.md`, substituting `--model internvl3_9b`.

Record start/end times, GPU ID, command, output files, row counts, and SHA-256
checksums in `RUN_REGISTRY.md` after execution.
