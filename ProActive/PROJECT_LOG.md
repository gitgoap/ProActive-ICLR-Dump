# Project Log

## 2026-07-24
- Received server preflight logs from the user.
  - **Server:** `bumblebee.lcs2` (Ubuntu 24.04.1)
  - **RAM:** 314 GiB total, 306 GiB available.
  - **GPUs:** 4x NVIDIA RTX A6000 (48GB memory each). Topology shows no NVLink (all SYS/PCIe connected).
  - **CUDA/Driver:** Driver 550.163.01, CUDA 12.4 (nvcc 12.0).
  - **Storage:** `/home` has 3.4T available (81% used).
  - **Python Environment:** Python 3.13.12, `torch` 2.6.0+cu124, `transformers` 5.5.4, `accelerate` 1.13.0, `qwen-vl-utils` 0.0.14.
- This satisfies our compute requirements and removes the blocker for progressing with Week 1 tasks.

- Confirmed model availability on server at `/home/models/`:
  - `Qwen3-VL-8B-Instruct` — available
  - `gemma-4-E4B-it` — available
  - `InternVL3-9B` — **not available at this July 24 check; resolved on 2026-08-10**
- User has 4 datasets (VizWiz, VSR, POPE, HallusionBench) already on the server in another project (~40 GB). Will symlink or point configs to existing paths rather than re-downloading.
  - HallusionBench: `/home/aman/MMUQ/data/HallusionBench`
  - VSR: `/home/aman/MMUQ/data/VSR`
  - POPE: `/home/aman/MMUQ/data/POPE` (also `/home/aman/MMUQ/data/POPE_github_cloned_repo`)
  - VizWiz: `/home/aman/MMUQ/data/VizWiz`
- GQA relation slice not yet available — will need to be constructed as part of Week 1/3 tasks.
- InternVL3-9B download command provided: `huggingface-cli download OpenGVLab/InternVL3-9B --local-dir /home/models/InternVL3-9B`

## 2026-08-07 to 2026-08-09 — Week 3 completion

- Hardened the pilot cache against implicit append, duplicate keys, unsafe resume behavior, malformed outputs, and non-durable writes.
- Repaired the Qwen POPE cache after an earlier smoke run had been appended to the 100-example pilot and produced 31 duplicate records. The repaired output was regenerated and later passed directory-wide duplicate/schema validation.
- Completed canonical and compact three-level severity pilots for Qwen3-VL-8B-Instruct and Gemma-4-E4B-it over POPE, HallusionBench, VizWiz, and VSR.
- Final validated cache matrix:
  - 16 JSONL files;
  - 800 canonical records (100 per model/dataset pair);
  - 9,600 severity records (1,200 per model/dataset pair);
  - 10,400 total records;
  - zero duplicate rows and zero invalid schema rows.
- Generated the required analysis artifacts:
  - five distribution/source-bit plots;
  - 50 inspection images for each of blank, blur, crop, brightness, and noise (250 total);
  - `pilot_analysis_summary.json` and `schema_validation_report.json`.
- Human-labelled 50 non-exact VizWiz answer pairs from train/validation data only. Labels contained 17 semantic matches and 33 non-matches.
- Calibrated and froze the semantic similarity threshold at `0.50`, achieving precision `0.5926`, recall `0.9412`, and F1 `0.7273` at the required recall target of `0.90`.
- Reviewed and froze canonical severities: blur `8`, crop `0.65`, brightness `0.15`, and noise `25`.
- Created `configs/probes/frozen_week3_config.yaml` with status `FROZEN` and embedded semantic-calibration provenance.
- Full Week 3 artifact gate passed on 2026-08-09 against the server-synced outputs.
- Local CPU regression suite passed after artifact synchronization: `158 passed in 11.54s`.
- Synced logs preserve at least `2.9133` GPU-hours of measured execution. Several jobs were resumed from earlier partial results, so exact total Week 3 GPU-hours are not reconstructible from the retained logs and are explicitly marked as unavailable in `RUN_REGISTRY.md`.

## 2026-08-09 — Week 4 implementation

- Completed the local Week 4 code substrate without launching GPU work.
- Added deterministic four-way teacher sharding and strict resume validation. Existing rows are checked for duplicate model-instance keys, manifest/config hash drift, shard drift, validity, and legal probe completeness before append.
- Added independent CPU label reconstruction; invalid rows fail closed, and embedded signatures/bits/six-way labels must agree exactly with frozen Week 3 rules.
- Added leakage-safe pre-policy partial states and recursive validation that learner inputs contain no dataset/model/gold/teacher metadata or unacquired observations.
- Added the blinded 180-example human-audit exporter, private key, renamed lossless internal images, annotator instructions, and checksums.
- Added readiness, daily progress, and full Week 4 validators plus train/validation-only class/bit CSVs and artifact manifests.
- Removed user-specific data-root fallbacks and unpinned semantic-model fallback behavior.
- Combined manifest audit: 7,291 examples and 110 relation-applicable rows. Qwen and Gemma require 14,582 teacher rows and 102,294 passes.
- Week 3 measured throughput (`1.1694` seconds/pass) projects the core at about `33.23 GPU-hours`: idealized `33.23 h` on one GPU, `16.61 h` on two, or `8.31 h` on four, before overhead. This is HIGH-COST and unapproved.
- Local Week 4 tests: 10 passed. Full local CPU suite: `168 passed in 2.75s`.
- Readiness correctly fails closed on the intentional draft approval status and unpinned Qwen/Gemma `main` revisions.

## 2026-08-10 — Server environment procedure corrected

- An initial documentation update inferred from the `(base)` prompt that every
  pane should explicitly activate Conda. The owner clarified that the project
  uses the Python environment already present in the server shell and that an
  explicit activation step should be skipped.
- Updated `README.md`, `SERVER_RUNBOOK.md`, and the Week 4 server guide to use
  the existing shell environment. Each new shell or tmux pane verifies
  `which python`, the Python version, and required imports before execution.
- Environment variables still need to be exported independently in each pane.
  If verification fails, the run stops for review rather than installing
  packages or switching environments mid-experiment.

## 2026-08-10 — Week 4 revision inspection correction

- Server verification confirmed that the existing tmux shell resolves Python
  to `/home/aman/miniconda3/bin/python` with Python 3.13.12, Torch 2.6.0+cu124,
  CUDA 12.4, and Transformers 5.5.4. The Conda base environment is already
  active; a separate activation command is unnecessary in that pane.
- The first revision-inspection run reported all three local model directories
  as `AMBIGUOUS`. Review showed that the inspector extracted every 40-hex value
  from Hugging Face `.metadata` files, conflating the repository commit on the
  first line with per-file Git blob ETags on the second line.
- Corrected the parser to use only the first non-empty metadata line as the
  repository revision and to fail closed on malformed metadata. The inspection
  must be rerun before any model config is pinned or any Week 4 GPU generation
  begins.
- Added three regression tests covering Git blob and LFS ETags plus malformed
  metadata. Targeted tests passed `3/3`; the full local CPU suite passed
  `171/171` after using the repository's established sandbox-safe temporary
  directory procedure.
- The corrected server rerun exited `0` and reported exactly one consistent
  revision for every model: Qwen
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, Gemma
  `292a7e278a400932df35f9fd4b1501edd04133a5`, and InternVL
  `5f618513e35a9b85922341b8057feddfc8880e50`. All three model YAMLs were pinned
  to this local-file evidence; no GPU workload was launched.
- The owner confirmed that InternVL3-9B is fully downloaded at
  `/home/models/InternVL3-9B`. This closes the download/availability blocker;
  adapter GPU validation and the InternVL teacher catch-up cache are still not
  started and must not be reported as complete.
- Post-pin readiness inspected 7,291 manifest rows and 102,294 expected core
  forward passes; its only remaining error is the intentional owner-approval
  gate. The post-pin full local CPU suite passed `171/171` in 9.81 seconds.

## 2026-08-10 — Week 4 staged execution approved

- The owner approved the recommended `0.80` maximum class fraction, 5-positive/
  5-negative per-slice bit gate, 60-natural + 120-targeted human audit, and an
  interim two-model workflow whose final audit still requires InternVL.
- The owner approved only the mandatory 1/10/100/full-VSR staged GPU checks.
  The projected 33.23 GPU-hour full Qwen+Gemma core remains unapproved.
- Froze the scientific experiment metadata as `APPROVED` and added independent
  compute authorization. `run_teacher.py` allows at most 100 examples or a
  complete allowlisted VSR run and fails closed on full combined-manifest
  generation until `full_core_approved` is separately changed after approval.
- Added four compute-authorization regression tests. The local readiness gate
  now passes with 7,291 rows, 14,582 expected core teacher rows, 102,294 passes,
  and zero errors.
- Final post-approval local regression suite: `175 passed in 2.73s`.
- The first post-approval server readiness attempt correctly failed closed
  because `doc/docs/WEEK_04_REQUIREMENTS.md` was absent from the server sync.
  All substantive readiness values were otherwise correct: 7,291 manifest
  rows, 14,582 expected teacher rows, 102,294 passes, approved staged scope,
  and the full-core lock still false. No GPU workload started.
- At the readiness check, physical GPU 0 was occupied (12,840 MiB and active
  utilization); GPUs 1, 2, and 3 were free at 13 MiB and 0% utilization. GPU
  availability must be rechecked immediately before the one-row stage.
- After syncing `doc/docs/WEEK_04_REQUIREMENTS.md`, the server readiness rerun
  passed: `is_valid=true`, zero errors, 7,291 manifest rows, 14,582 expected
  Qwen/Gemma teacher rows, and 102,294 expected forward passes. Staged checks
  are approved and the full-core authorization remains false.
- The mandatory Qwen one-row stage selected one deterministic VizWiz validation
  instance and ran 7 clean/probe forward passes on physical GPU 1. It finished
  in 17.9 seconds with zero failed rows and exit code 0, writing
  `outputs/week4_staging/limit1/teacher_qwen3_vl_8b_all_all_shard00-of-01.jsonl`.
  Independent teacher-progress validation remains required before marking this
  stage `PILOT VALIDATED`.
- The run emitted non-blocking compatibility warnings: the local MiniLM load
  regenerated an unexpected `embeddings.position_ids` buffer, and Transformers
  ignored sampling-only `top_p`/`top_k` flags during deterministic generation.
  Model and semantic matcher loading otherwise succeeded.
- Independent validation accepted the Qwen one-row file with SHA-256
  `2bfe120ad6f0cba83b2e3114103c885cc1a6c176014e5b6cb33d0991d0833ec6`, one
  teacher row, six probe records, and zero errors.
- The Gemma one-row stage used the same deterministic VizWiz instance, ran 7
  passes in 22.4 seconds, and finished with zero failures. Post-run validation
  accepted both model files: 2 teacher rows, 12 probe records, zero errors;
  Gemma SHA-256 is
  `ed5ab6da3fc279637ecd019f5eb06d6cefc065ac00a0e6cb53396d7d6872c645`.
  Both one-row stages are now `PILOT VALIDATED`.

## 2026-08-12 — Week 4 Qwen 10/100-row stages

- The Qwen 10-row stage completed 70 forward passes in 70.2 seconds with zero
  failed rows. Independent validation accepted 10 teacher rows, 60 probe
  records, zero errors, and SHA-256
  `dd9dda34708a58457f441835df7dc7b4fe69a0ff7b14e196e2cab126a7642b00`.
- The Qwen 100-row stage completed in 586.2 seconds with zero failed rows and
  exit code 0. Independent validation accepted 100 teacher rows and 602 unique
  probe records with zero errors. The 602 probes are correct: two sampled VSR
  relation-applicable rows have a seventh probe, while the other 98 rows have
  six. The cache SHA-256 is
  `972f9dca8ced95d5fb27ac6c7bfaeaa75a089d6d46d0aa60faea9b86337f712a`.
- Observed 100-row throughput was 702 clean/probe passes in 586.2 seconds, or
  approximately 0.835 seconds/pass including model-load overhead. The next
  authorized gate is one complete Qwen VSR run; the full core remains locked.
- The complete Qwen VSR stage passed on the server: 340/340 teacher rows, 2,150
  legal probe records, zero failures/errors, and 1,325.9 seconds end-to-end.
  The 2,150 count is exact: 340 rows × six base probes plus 110 applicable
  relation probes. Server SHA-256:
  `d8a8996fc7e01855121369d99c402f42af057819788f1bfafb0431925877d2d6`.
- The first local synchronization of the VSR JSONL was truncated to 81 rows and
  524,288 bytes, producing local SHA-256
  `ceded69cf37ea1a86c781acba0a73c476413802de5b117b63b8d666f68cf69a1`.
  The server artifact and validation remain trustworthy, but the JSONL must be
  re-synced after completion until its local row count/hash match the server.
- A second local check on 2026-08-13 still found the VSR JSONL at exactly
  524,288 bytes, indicating a likely per-file synchronization cap rather than
  an inference/artifact failure. The recovery procedure is to create a gzip
  copy on the server without altering the validated JSONL, sync that compressed
  copy, and verify its decompressed SHA-256 against the server hash.
- The gzip transfer resolved the archival issue. The synced `.gz` is 286,141
  bytes and decompresses to 2,221,006 bytes containing exactly 340 valid JSON
  objects. Its decompressed SHA-256 is
  `d8a8996fc7e01855121369d99c402f42af057819788f1bfafb0431925877d2d6`,
  exactly matching the validated server artifact. No inference rerun was needed.

## 2026-08-13 — Full Week 4 core approved

- After all 1/10/100/full-VSR gates passed, the owner explicitly approved the
  conservative 33.23 GPU-hour full Qwen+Gemma core run.
- Updated `configs/experiments/teacher_core.yaml` to set
  `compute_authorization.full_core_approved: true` and record the approval date
  and estimate. Deterministic four-way sharding, immutable revisions, frozen
  probes, strict resume validation, and daily checksums remain enforced.
- GPU availability at approval: physical GPU 1 was free; GPU 0 held two Python
  processes using 37,967 MiB, while GPUs 2 and 3 were actively utilized by
  unrelated KVCompress jobs. Only shard 0 on GPU 1 is authorized for immediate
  launch; additional physical GPUs must be checked again before use.
- A local post-approval check found that `manifest_combined.jsonl` had also been
  truncated during synchronization (1,310,720 bytes locally versus the earlier
  recorded 2,933,173 bytes), and the current local shell lacked `src` on
  `PYTHONPATH`. These are local verification-environment issues: server
  readiness already validated the complete 7,291-row manifest. No server
  manifest rebuild or GPU rerun is required.
- Full-core authorization changes passed 14 focused Week 4 tests and the full
  local CPU regression suite (`175 passed in 3.77s`). The first full-suite
  attempt had four temporary-directory setup errors after 171 passes because
  the Windows shell failed to create its test directory; rerunning with an
  explicitly created directory resolved the environment issue.
- The owner overrode the idle-GPU-only recommendation and directed parallel
  launch of Qwen shards 0, 1, and 2 on physical GPUs 1, 2, and 3. GPUs 2/3 were
  already running unrelated jobs, so their timing is marked shared-GPU and is
  unsuitable as clean latency evidence. Separate output locks/files and strict
  resume make interruption or OOM recoverable without duplicating rows.

## 2026-08-16 — Qwen full-cache fail-closed recovery

- The four Qwen shards finished with 7,238/7,291 valid rows: 1,813, 1,805,
  1,785, and 1,835 rows. The 53 absent rows comprise 30 HallusionBench and 23
  VizWiz examples. They failed at mandatory grounding parsing and were
  correctly excluded rather than serialized with an invented answer or label.
- A repeated shard-0 resume reproduced its nine failures, confirming that an
  unchanged deterministic retry cannot resolve the formatting failures.
- Added a conservative parser fallback only for an explicit terminal
  `The answer is ...`/`Final answer is ...` construction. Bare free-form lines,
  empty tags, unknown outputs, and conflicting binary indicators still fail
  closed. Explicit VizWiz `unanswerable` remains a valid dataset answer.
- Resume now reparses every existing grounding raw output and refuses parser
  drift before appending. A local compatibility audit checked all 7,238 saved
  Qwen rows with zero validity or normalized-answer changes.
- Failed rows are now atomically upserted by `(model_id, instance_id)` into a
  per-shard `*.failures.jsonl` ledger containing the invalid raw teacher record,
  exception, attempt count, and manifest/config/model provenance. A later
  success removes the stale failure entry; retries cannot duplicate it.
- Added nine unit/adversarial/integration tests. Focused recovery tests passed
  `21/21`; the complete local CPU suite passed `184/184` in 2.44 seconds.
- Server validation remains pending. The new code must be synced before the
  four Qwen `--resume` commands; Gemma work should finish first on the occupied
  GPU. Any rows still unresolved after retry require review of the retained
  failure ledgers, not heuristic label fabrication.
- A local dry-resume correctly refused the Windows copies because CRLF line
  endings change byte-level hashes. Normalizing the three local YAMLs to LF
  exactly reproduces the hashes embedded in every server Qwen row, confirming
  content-preserving line-ending drift rather than scientific configuration
  drift. Recovery instructions therefore require syncing only the two Python
  runtime files and verifying the original server YAML/manifest hashes before
  resume.

## 2026-08-17 — Uniform grounding-only refresh

- The parser-only server retry completed normally but recovered zero rows;
  Qwen remained at 7,238 valid rows and all 53 invalid rows were preserved in
  four deduplicated failure ledgers.
- Raw-output inspection found the actual cause: 36 generations reached the
  256-token ceiling before emitting the requested tag. The remaining 17 are
  explicit behaviors: three terminal binary `the answer is no` conclusions,
  ten tagged VizWiz `Unknown` abstentions, and four short isolated VizWiz
  answers (`MVG`, `TLZ`, and `$1.00`).
- Extended the parser only for those auditable constructions. Tagged VizWiz
  `Unknown` maps to its benchmark's legitimate `unanswerable` class; prose,
  conflicting binary candidates, empty tags, and non-isolated free-form text
  remain invalid. All 7,238 existing Qwen grounding answers remain unchanged.
- Added `scripts/refresh_grounding_cache.py`. It never edits the original
  cache. It reconstructs each model-instance from a valid teacher row or its
  retained invalid record, runs exactly one grounding pass at a uniform
  512-token cap, recomputes all teacher labels, records source/effective-config
  hashes, and writes a separate resume-safe cache. Thus difficult examples do
  not receive a selectively larger budget.
- Added refresh reconstruction, resume drift, failure-ledger, parser
  adversarial, and validator-sidecar tests. Failure ledgers are excluded from
  teacher artifact collection while missing valid rows still fail the coverage
  gate. Final focused suite: `30 passed`; full CPU suite: `192 passed in
  1.70s`. Two-model server refresh and artifact validation remain pending.

## 2026-08-19 — HallusionBench answer-contract repair

- Audited the official `HallusionBench.json` after observing impossible
  all-false clean correctness. The 951 image-paired subset contains 937 binary
  rows and 14 released table questions whose natural answers are countries,
  states, months, or an explicit no-answer condition. Their semantic references
  live in `gt_answer_details`; their `gt_answer` value is only a benchmark-level
  indicator.
- Rejected deletion of the 14 rows. The primary dataset remains all 951
  image-paired examples; classification uses source question grammar and
  official annotation identity, never model output or failure status. A
  937-row binary-only result may be reported later only as a named sensitivity
  analysis.
- Added HallusionBench answer contract v1. Binary rows normalize
  `0/1/2 -> no/yes/uncertain`. Open rows preserve the official details and use
  an author-audited, versioned alias overlay with normalized exact matching.
  The VizWiz-calibrated semantic threshold is not used for gold correctness on
  these short entities. The loader requires
  exactly 14 open image rows and fails closed on missing, unused, or malformed
  references.
- Propagated answer type through clean inference, grounding parsing, probe
  matching, cache serialization, resume checks, and Week 4 validation. Removed
  the probe runner's silent semantic-exception fallback.
- Added `scripts/migrate_hallusion_answer_contract.py`. It requires an unchanged
  instance set, rejects any non-Hallusion manifest drift, recomputes binary
  normalization/correctness/probe matching/labels from saved raw generations,
  drops every one of the 14 prompt-changed rows for every model, and writes
  source file/record plus old/new manifest hashes into a separate cache. The
  original cache is never overwritten.
- Added a fail-closed manifest-builder gate so one failed active dataset cannot
  silently produce a partial combined manifest.
- Local verification: Hallusion-focused tests passed `84/84`; migration
  integration tests passed `4/4`; after the exact-alias hardening, the final
  full CPU suite passed `209 passed` in 3.61 seconds. No local GPU claim is
  made. Server manifest rebuild, migration, minimal pending-row rerun, uniform
  grounding refresh, and final checksum validation remain required.

## 2026-08-20 — VizWiz deterministic gold-selection repair

- The Hallusion contract migration stopped before writing outputs because 207
  VizWiz `gold_answer` values differed between the preserved and rebuilt
  manifests, while both contained the same 7,291 IDs and no other reported
  non-Hallusion field changed.
- Root cause: `max(set(answer_texts), key=answer_texts.count)` selected an
  arbitrary member of tied answer counts according to Python's randomized set
  iteration. The underlying dataset did not change.
- Replaced the rule with normalized majority voting and first-tied answer in
  released annotation order. The manifest records the normalized count table,
  tie size, policy ID, and contract version.
- Extended the audited migration to permit only the declared VizWiz answer
  contract fields, reuse raw inference, recompute clean correctness and labels,
  and continue rejecting changes to questions, IDs, images, groups, or splits.
- Added four focused tests including normalized aggregation, deterministic tie
  resolution, transition allowlisting, and CPU-only correctness recomputation.
  The complete local CPU suite passes `213 passed` in 2.25 seconds.
- Server rebuilding under `PYTHONHASHSEED=1` and `987654` produced the same
  7,291-row combined manifest file SHA-256
  `05fd0dbc554c5fb85664dd87e99f8eebdd995320e6b34e2fffe856867d3d0859`,
  validating the deterministic repair. Migration then stopped at the frozen
  probe-config byte-hash guard because a Windows sync converted its line
  endings from LF to CRLF. The scientific YAML content is unchanged; restore
  the original LF bytes and require recorded hash `5cfdbcde...` before retry.
- Exact-hash-guarded LF restoration resolved the provenance mismatch. The
  rerun completed with `is_valid: true`: 7,291 selected rows per model, exactly
  14 open Hallusion rows invalidated per model, 67 Qwen pending, and 88 Gemma
  pending. All eight teacher JSONL and eight failure-ledger SHA-256 values in
  the signed migration report match the synchronized artifacts.
- The first `run_teacher.py --resume --dry_run` then caught a remaining VizWiz
  parser drift (`cannot be determined` versus canonical `unanswerable`). The
  v1 migration report is therefore superseded despite its internal checksums.
  Migration schema v2 now refreshes parser-dependent VizWiz probe fields and
  embedding decisions only where normalization changed, recomputes labels,
  and runs the same grounding-parser compatibility check before writing a
  success report. An in-memory audit refreshed 78 of 5,971 valid two-model
  VizWiz rows and left every grounding row compatible with the current parser.
- Server migration schema v2 completed with `is_valid: true`. It refreshed 22
  Qwen and 56 Gemma VizWiz rows, retained pending totals of 67 and 88, and all
  16 signed artifact hashes match the synchronized files. Eight model/shard
  `run_teacher.py --resume --dry_run` checks passed, so minimal GPU recovery is
  authorized on newly verified-free devices.
- Parallel base recovery on physical GPUs 0 and 3 completed every shard. Qwen
  improved from 7,224 to 7,255 valid rows, leaving 36 failures; Gemma improved
  from 7,203 to 7,217, leaving 74. All 14 invalidated open Hallusion rows
  succeeded for both models. The remaining 110 rows are all auditable
  grounding-only failures with recoverable invalid records; coverage remains
  exactly 7,291 IDs/model with no exclusions. Uniform grounding refresh is next.

## 2026-08-23 — InternVL runtime incompatibility and adapter correction

- The first InternVL one-row smoke failed before model loading because `einops`
  was missing. After installing it, the checkpoint initialization progressed
  but failed with `InternVLChatModel` missing `all_tied_weights_keys`.
- Server version evidence records Python 3.13.12, PyTorch 2.6.0+cu124,
  Transformers 5.5.4, Accelerate 1.13.0, and einops 0.8.2. `pip check` passed,
  confirming this is a remote custom-code API incompatibility rather than a
  broken declared dependency.
- Preserved the base environment used for accepted Qwen/Gemma outputs. Added a
  separate `requirements-internvl.txt` runtime pinned to Transformers 4.37.2,
  matching the official InternVL custom-code dependency.
- Replaced the unvalidated generic InternVL adapter path with native dynamic
  image tiling, conversation/image-context construction, deterministic
  generation-score extraction, and teacher-forced scoring. Added a pre-load
  Transformers-version guard so an incompatible runtime fails immediately.
- Added focused CPU tests for version gating, deterministic tiling, and
  generation configuration. Syntax compilation passed locally; the local
  environment lacks pytest, so the focused/full pytest suites and the revised
  one-row GPU smoke remain pending. No GPU success is claimed.
