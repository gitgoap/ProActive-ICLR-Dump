# Project Log

## 2026-09-13 — Week 8 grouped statistical analysis completed

- The synchronized `analyze_week8.py` run completed with exit code zero in
  1:13:54 over 340,800 frozen held-out trajectory rows.
- It used 2,000 bootstrap resamples grouped by `group_id`, generated 710
  confidence-interval rows, 120 paired-comparison rows, 160 dataset/model slice
  rows, and 10 positive plus 10 negative deterministic qualitative cases.
- The analysis report self-hash `757908db...e191`, the source-trajectory hash,
  and all four output-artifact hashes independently match. No target-domain
  calibration or post-shift tuning was used.
- ProActive exceeds random at budgets 1--4. The predeclared budget-7 primary
  comparison is a small statistically significant negative difference
  (`-0.0063`, 95% CI `[-0.0087, -0.0041]`, Holm-adjusted `p=0.003`) while using
  about 0.42 fewer probes. The paper claim is therefore constrained-budget
  efficiency, not universal full-budget dominance.
- All mandatory Week 8 machine evidence is now complete. The remaining gate is
  three-person annotation, blinded adjudication/analysis, and CPU-only full
  Week 8 validation.
- Added `PROACTIVE_RESEARCH_HANDOFF_AND_RESULTS_INDEX.md` as the compact map
  from the paper story to signed result artifacts and independent-analysis
  instructions.
- Auditing against Plan §25.11 found that Week 9 packaging is still unbuilt:
  one-command paper assets, `repro_manifest.json`, independent leakage and
  calibration review, appendix/risk/release artifacts, and the fixed-seed
  rerun-or-documented-waiver decision. These do not block writing but do block
  calling the submission package reproducible and frozen.

## 2026-09-12 — Week 8 ablation resume incident isolated and repaired in code

- The synchronized run is not complete: the signed `ablations.json` and CSV do
  not exist. The first launch completed three component ablations, left two
  policy trainings partial at their declared timeout boundaries, and had not
  begun three trained ablations or the final reference/evidence stages.
- The resume stopped correctly at `no_budget_embedding` APS provenance
  validation. The preceding diagnostic `--resume` had incorrectly continued
  beyond an already completed early-stopping boundary (epoch 10), overwriting
  checkpoint SHA-256 `d2a80f3a...005050` with `424b0570...037a`.
- No test or held-out-shift data were accessed and no tuning decision changed.
  The original APS, VOI, policy, stack freeze, and frontier remain signed to
  the pre-incident checkpoint. The overwritten diagnostic checkpoint/report/
  history and diagnostic freeze are quarantined from scientific use.
- Diagnostic and policy trainers now treat a hash-valid final report as an
  immutable completion marker, verify provenance and the first early-stopping
  boundary, and return without loading a GPU or performing another epoch.
- `repair_week8_ablation_resume.sh` preserves the overwritten artifacts,
  rebuilds only the affected diagnostic from epoch zero, accounts its time in
  the existing eight-hour ledger, and permits continuation only if the
  checkpoint and diagnostic-freeze hashes exactly reproduce their original
  downstream bindings. Local syntax compilation passed; server focused and
  full pytest plus the guarded repair remain required.
- The synchronized ledger contains 18,185 seconds (`5.0514` GPU-hours), so
  10,615 seconds (`2.9486` GPU-hours) remain under the existing approval.
- The owner subsequently authorized running the bounded repair alongside idle
  resident processes. The repair launcher therefore has an explicit
  `--allow-shared` mode requiring at least 8 GiB free and at most 10% launch
  utilization. Exclusive-GPU behavior remains the default, other PIDs are
  never touched, and exact checkpoint reproduction remains mandatory.
- The shared repair ran on physical GPU 1 with 9,898 MiB free and 0% launch
  utilization. Training ended normally at epoch 10 in 227.92 seconds and
  reproduced every recorded train loss and both validation selection metrics
  exactly for epochs 0--10. Its raw checkpoint SHA nevertheless differed
  (`8bbef310...6802a`) because atomic PyTorch saves embed the randomly named
  temporary ZIP root; raw file-byte equality is therefore not a valid
  reserialization test. The script correctly stopped without touching the old
  descendants.
- Recovery now requires two independent scientific equivalence checks: exact
  history through the original first stopping boundary and exact regenerated
  APS thresholds/metrics, ignoring only checkpoint path/hash and report
  self-hash. If both pass, the stale downstream directories are moved to a
  recovery archive and rebuilt normally under the new hash. No provenance
  field is edited in place.

## 2026-09-11 — Week 8 held-out grounding recovery completed

- The owner-approved `concise_describe_then_answer_retry_v1` recovery ran on
  every ledger-defined held-out failure: 19 Qwen rows and 11 Gemma rows, with
  no exclusions and no changes to the other accepted observations.
- Both recovery processes exited zero. The separate recovered cache contains
  exactly 1,200 Qwen plus 1,200 Gemma rows, and both current failure ledgers
  contain zero rows. Qwen's 19 retries completed in 72.7 seconds with zero new
  or unresolved failures; Gemma likewise finished with zero unresolved rows.
- `outputs/week8_teacher_recovered` is now the authoritative input for the
  remaining Week 8 label, state, vectorization, frozen-shift, LOMO, ablation,
  latency, and statistical jobs. The original incomplete cache remains intact
  for provenance. No held-out result may tune the frozen Week 7 stack.
- Offline construction then produced all 2,400 labels and 63,687 partial
  states. The new state index stopped before vectorization because it looked
  for `split`, `model_id`, and `instance_id` at the JSONL top level, while the
  canonical `partial_state_v1` schema stores them under `metadata`. The state
  artifacts themselves are valid. The indexer now reads and validates the
  canonical metadata block and rejects missing, empty, or unknown identities;
  the focused server suite passed 10/10 after synchronization. The corrected
  index now binds all 63,687 states from all 2,400 teachers, and vectorization
  produced two hash-valid `shift` tensor shards with the same 63,687 rows.
  Frozen-stack shift evaluation is the next gate.
- The owner approved the complete frozen held-out shift frontier on physical
  GPU 3 when at least 2 GiB is free at launch, with a 0.5 GPU-hour ceiling and
  a hard 30-minute timeout. This approval permits evaluation only: the signed
  Week 7 stack and APS thresholds remain immutable, and no held-out tuning is
  authorized.

## 2026-09-10 — IllusionBench blank-probe normalizer fix and Gemma stage validation

- The first Gemma held-out one-row stage selected
  `illusionbench_10046007_2`, loaded the pinned model successfully, and then
  failed closed on the first visual probe with `Unknown dataset
  'illusionbench' ... Provide normalizer_type explicitly`.
- The frozen manifest row is valid and declares its row-specific
  `normalizer_type`. IllusionBench deliberately has no dataset-level fallback
  because the release mixes true/false and multiple-choice questions.
- Root cause: the special blank-image call in `run_all_probes` omitted
  `normalizer_type`, even though the field was already propagated to the clean,
  other visual, grounding, relation, and semantic paths. Since blank runs
  first, the row failed before later probes could execute.
- Corrected the blank call to pass the same row-specific normalizer. A
  dataset-level IllusionBench fallback remains deliberately absent because
  one release contains two different answer contracts.
- Added an end-to-end regression in `tests/test_week3_integration.py` that
  processes an IllusionBench multiple-choice row and requires normalized `B`
  for the clean answer and every applicable probe.
- After synchronization, the focused server regression suite passed 10 tests.
  Resumed Gemma held-out stages then produced 1/1 and 10/10 valid rows with
  zero unresolved failures. Qwen's matching stages also produced 1/1 and
  10/10 valid rows with zero unresolved failures. Both staged caches contain
  valid PRE-HAL and IllusionBench records, so the parser gate is complete and
  complete held-out inference now awaits explicit compute approval.
- The owner subsequently approved the complete two-model 1,200-row-per-model
  held-out cache on two GPUs, with no exclusions, an eight-GPU-hour combined
  ceiling, and a six-hour timeout per model. The Week 7 stack remains frozen
  and held-out results are prohibited from post-test tuning.
- At launch time only physical GPU 3 had sufficient free memory; Gemma occupied
  about 16.3 GiB while roughly 32 GiB remained free. To reduce wall-clock delay,
  the owner explicitly authorized Qwen to share that active GPU. This is an
  execution-placement deviation only: a 24-GiB free-memory launch guard, the
  existing six-hour timeout, resumable outputs, unchanged frozen settings, and
  the combined eight-GPU-hour ceiling remain in force.
- The failed output contains no valid teacher row and one transparent failure
  ledger entry. It is safe to rerun the same limit-1 stage with `--resume`
  after syncing the full normalization path; success will clear the ledger.

## 2026-09-10 — Complete held-out traversal and targeted recovery scope

- Both complete 1,200-row model traversals finished, but fail-closed accounting
  found 1,181 valid plus 19 failed Qwen rows and 1,189 valid plus 11 failed
  Gemma rows. Both model commands correctly returned exit code 1.
- Every rejected row is a malformed mandatory grounding response: the model
  omitted `FINAL_ANSWER` or supplied an out-of-domain terminal value. There
  were no missing images, CUDA failures, or selective data exclusions.
- Generalized the already approved concise grounding-recovery implementation
  to accept `shift` teacher failure ledgers and row-specific normalizers. It
  copies accepted rows unchanged into a separate cache and retries exactly the
  30 ledger-defined failures; the original cache remains immutable.
- Recovery execution remains pending focused server tests, complete dry-run
  accounting, current GPU inspection, and explicit owner approval for these 30
  retries.
- The owner approved `concise_describe_then_answer_retry_v1` for exactly the 19
  Qwen and 11 Gemma Week 8 ledger rows, with no exclusions. Two concurrent
  recovery processes may share physical GPU 2 only when at least 40 GiB is free
  at launch. The output remains separate from the immutable source cache.

## 2026-09-09 — Research progress brief for professor review

- Added `PROACTIVE_RESEARCH_PROGRESS_BRIEF.md`, a standalone account of the
  research question, completed weekly milestones, measured core results,
  outstanding evidence, and nine-day submission schedule.
- Checked the quoted encoder metrics against the signed Week 5 selection and
  the two-probe and full-budget comparisons against the locked Week 7 frontier.
  The brief distinguishes validation scores, locked-test results, operational
  labels, and pending human/transfer evidence.
- Incorporated the synced verified 600+600 held-out manifest and blank
  three-person audit packets. Recorded the outstanding server guide-sync
  issue without treating it as an inference failure.
- Documentation only; no scientific settings, source code, or outputs changed.

## 2026-09-09 — PRE-HAL cross-directory basename collision

- The authenticated pinned download completed all 6,471 selected repository
  files, but verification reported 6,346 image identities instead of 6,469.
- Root cause: the held-out loader used only `Path(image).stem`; PRE-HAL has
  identical basenames in different release subdirectories, so 123 distinct
  paths collapsed during inventory and grouping.
- PRE-HAL image identity now derives from the normalized complete relative
  path and includes a deterministic SHA-256 prefix in the manifest ID. Rows
  sharing one true image remain grouped, while equal basenames in different
  directories remain separate.
- Added an adversarial regression test with `source_a/1.png` and
  `source_b/1.png`. Server verification remains pending after synchronization.


## 2026-09-08 — Week 8 report integrity and closed-answer regression checks

- Corrected the quick-review finding: PRE-HAL and IllusionBench already occur
  in `BINARY_DATASETS`, so their existing dataset-based exact-match branch
  already prevented embedding matching for differing closed answers.
- Hardened `compute_semantic_match` so an explicit closed normalizer also
  enforces exact matching for new datasets. Free-form VizWiz and open-ended
  HallusionBench retain their existing embedding path.
- Week 8 report loading now rejects missing/non-string/empty/mismatched
  self-hashes, non-object JSON, and unreadable JSON before consuming a report.
- Added `tests/test_week8_integrity.py`: eight CPU regression tests passed
  locally using `python -B -m unittest tests.test_week8_integrity -v`.
  The initial sandbox run encountered temporary-directory permission errors;
  the approved unsandboxed rerun passed. No GPU was used. Pytest and full Week 8
  end-to-end validation remain pending in the server environment.


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
  generation configuration. The persistent server environment
  `proactive-internvl` was created with Python 3.11.15, PyTorch 2.6.0+cu124,
  Transformers 4.37.2, Accelerate 0.30.1, and einops 0.6.1; `pip check`
  reported no broken requirements.
- Server CPU validation passed all `11` focused InternVL adapter tests and all
  `225` repository tests. The one warning is a non-blocking Hugging Face
  `resume_download` deprecation.
- The corrected one-row GPU smoke then completed on physical A6000 GPU 0 in
  101.6 seconds with one valid VizWiz teacher row, six applicable probes,
  relation correctly marked not applicable, valid tagged grounding, and zero
  unresolved failures. Output SHA-256 is
  `fd813be00a5b37fa6fb75586afdbefb4b58869882ed99198f67cf8c0dcf8fca0`.
  This validates the native adapter on a real GPU; larger staged and catch-up
  runs remain pending.
- The InternVL 10-row stage then produced 10 unique valid rows and 60 legal
  probe observations across POPE and VizWiz with no invalid applicable probe
  and no failure ledger. It completed in 170.0 seconds; output SHA-256 is
  `635b8095415e669da6c725798c6ab4983a602a65b1d2c843e90cca59a3d661ef`.
- Parallel InternVL 100-row and complete-VSR stages completed on 2026-08-24
  with zero model failures. The 100-row cache contains all four datasets, 100
  valid rows, 602 probes, and two relation rows in 1,979.0 seconds (SHA-256
  `c1a447aebad945ac270fea11f8ad7d500b9e7e14db867aed292a44fc9f1369bd`).
  Complete VSR contains 340 valid rows, 2,150 probes, and all 110 relation rows
  in 5,682.5 seconds (SHA-256
  `929b563aa9302e943918e797b8204040f7e8363fc581c929dcb6bf836987ba2b`).
- The generic Week 3 schema validator marked only calibration/test rows as
  invalid (20 in the 100-row stage; 67 in VSR). This is an expected scope
  mismatch: that pilot validator intentionally admits train/val only, while
  Week 4 teacher generation requires train/val/cal/test. Independent Week 4
  integrity checks found zero actual invalid teacher rows, duplicates, probe
  errors, manifest mismatches, or provenance errors. No inference rerun is
  required.
- Implemented a non-destructive final grounding recovery path for the remaining
  six Qwen and three Gemma parse failures. The trigger is strictly membership
  in the 1,024-token malformed-grounding ledger; the same concise
  describe-then-answer prompt applies to every triggered row. All existing
  valid rows are copied unchanged except for immediate source provenance, and
  recovered labels are recomputed. Source file/record hashes, retry policy,
  prompt/generation hashes, attempts, and any continued failures remain
  auditable. Syntax compilation and a direct prompt-contract check pass
  locally. Server validation on 2026-08-24 passed all 26 focused
  recovery/refresh/parser tests and the complete 230-test repository suite;
  all eight real-source recovery dry-runs then passed. Their deterministic
  retry scope is exactly six Qwen rows (`1/1/3/1` by shard) and three Gemma
  rows (`0/1/1/1`), with no exclusions or unexpected records. GPU execution
  was authorized on 2026-08-24 when the owner explicitly approved
  `concise_describe_then_answer_retry_v1` for all nine remaining malformed
  rows with no exclusions.
- The two-model GPU recovery then completed successfully. All nine targeted
  rows recovered, all eight failure ledgers are empty, and the official Week 4
  teacher-progress report passes with 14,582 teacher rows, 87,712 unique legal
  probe records, exact 7,291-row coverage per core model, and zero errors.
  Independent recovery-metadata inspection found 14,573 unchanged copied rows,
  exactly nine recovered rows, and no duplicate or policy-drifted records.
- Before offline generation, preflight inspection found that label/state/audit
  discovery could include empty `*.failures.jsonl` sidecars. Those sidecars are
  now explicitly excluded and rejected as direct inputs. The focused
  sidecar/Week-4 suite passes 17 tests and the complete server suite passes
  236 tests.
- The first human-audit export attempt made no changes because the orchestration
  command pre-created an empty audit directory and then requested `--resume`.
  The exporter correctly refused the directory because it had no completed
  manifest. The nine teacher and nine label pool files are intact; the initial
  export must use `--overwrite`, after which `--resume` becomes valid.
- The corrected audit export selected exactly 180 rows, 30 per six-way label.
  The server full validator passed readiness, the complete 14,582-row teacher
  cache, all 14,582 labels, all 388,623 states, and the audit packet (180
  images, three models, four datasets) with zero artifact errors or warnings.
  The aggregate report remained false only because the server had an older
  InternVL catch-up document, so `catch_up.documented=false`; syncing the
  already updated document and rerunning the CPU validator is sufficient.
- Offline generation completed from the recovered core cache: 14,582 unique
  label records and 388,623 unique pre-policy partial states across all 14,582
  teacher keys. Independent inspection found zero duplicate state IDs, zero
  forbidden learner fields, exact observation/name alignment, all mandatory
  empty/random sources, and no phase/deferred-source drift. Train/validation
  label balance also passes preliminarily: the dominant `mixed` class is
  41.42% (gate 80%), and every dataset/model/bit slice has at least five
  positives and negatives. Official full validation remains pending the final
  audit packet.
- On 2026-08-26 the updated catch-up document was present on the server and the
  repeated full validator passed: readiness, teacher progress, labels/states,
  human-audit packet, and catch-up are all valid. Final evidence is 14,582
  teachers, 87,712 legal probe records, 14,582 labels, 388,623 states, 180
  audit rows/images, three models, four datasets, zero errors, and zero
  warnings. Week 4 is complete; human agreement scoring remains a parallel
  Week 8 study rather than a blocker on Week 5 implementation.

## 2026-08-26 — Shared Week 5–7 code implementation

- Implemented the Week 5 tensor substrate, four mandatory diagnostic
  encoders, shared heads/losses, validation-temporary APS, shortcut controls,
  permutation study, three-seed gate, and signed freeze. The clean-only
  baseline uses one empty state per model-instance for normalization/class
  weighting so relation-capable examples are not silently overweighted.
- Implemented exact offline realized VOI, an action-conditioned policy head,
  legal acquisition/STOP semantics, three-seed cost selection, and the full
  reviewer-facing baseline matrix. The uncertainty comparator predicts
  expected entropy reduction from the acquired state only; realized cached
  outcomes remain restricted to offline labels and explicit oracles.
- Added a validation-selected dataset-specific fixed schedule as a control.
  It is never an input to the main learner and is hash-frozen before test.
- Added separately calibrated APS for clean-only, scalar-confidence, and
  one-pass-distilled baselines to avoid applying the main model's conformal
  threshold to another model. Locked evaluation reuses every trajectory to
  report both 90% and 95% coverage.
- Reworked oracle-best-subset evaluation to batch all candidate subsets for an
  example/budget in one small-network forward call. The mandatory oracle is
  retained while avoiding hundreds of tiny GPU calls per row.
- Implemented stack freeze, calibration-only trajectories, final APS,
  locked-test authorization, complete permutation gates, and an automatic
  Week 7 go/no-go memo. Cal/test split access remains impossible before the
  preceding signed gates.
- Added unit, adversarial, configuration, and synthetic diagnostic→APS→VOI→
  policy integration tests. Local Python syntax compilation passes. The local
  runtime has no PyTorch/pytest, so server tests and staged runs are still
  required; no GPU behavior is claimed.
- The proposed optimizer, budget schedule, train-only uncertainty baseline,
  and 3-point undercoverage tolerance remain `PENDING` in `DECISIONS.md` and
  all non-pilot training fails closed until owner approval.
- Final implementation audit fixed budget projection through clean-only views,
  bounded VOI pilots before counterfactual expansion, made final test budgets
  originate from the frozen APS artifact, required the validation-selected
  dataset schedule to cover all five final budgets, and added row-level
  calibration hashes. Temporary APS now records validation set-size and
  singleton evidence for the plan-fixed RAPS gate; RAPS remains outside the
  critical path unless that gate fires.

## 2026-08-26 — Week 5 staging passed and Weeks 5–7 settings approved

- The server regression suite passed `279` tests with one non-blocking
  Hugging Face deprecation warning. Full CPU preprocessing validated eight
  source state files and vectorized all 388,623 states in 75.06 seconds;
  resume verification accepted the completed output.
- Deep Sets and canonical GRU each passed isolated 1/10/100-row GPU stages.
  All six jobs exited zero in 8.15–9.91 seconds and used approximately
  1.60–1.65 GB peak host RSS. Their class-deficient pilot metrics are marked
  scientifically invalid by design and cannot enter model selection.
- The owner approved AdamW `1e-3`, batch 512, maximum 30 epochs, patience 5;
  early budgets `1/2/4` and final budgets `1/2/3/4/7`; the train-only expected
  entropy-reduction baseline; maximum undercoverage gap `0.03`; and Week 5
  selection thresholds `0.01`, `1e-6`, `0.10`, and `0.015`.
- Approval is frozen in the three experiment YAMLs and `DECISIONS.md`. Since
  the vectorized manifest intentionally binds the complete config hash, the
  75-second preprocessing step must be rerun after syncing the approved YAML
  before full training.
- Inspection of the Week 4 blinded human-audit CSV found 180 prepared rows and
  images but zero completed fields in all three independent annotator blocks.
  Three-person annotation and later adjudication remain a parallel paper task;
  they do not block cached-data Week 5 training.
- The approved-hash refresh and Week 5 readiness subsequently passed. Full
  Deep Sets seed 42 exited zero in 6:09.83 at best epoch 12, with validation
  source-bit Macro-F1 `0.864124` and six-way Macro-F1 `0.502543`. Full
  canonical GRU seed 42 exited zero in 2:31.51 at best epoch 1, with
  corresponding scores `0.862133` and `0.467460`. Both reports cover 60,483
  validation projections and bind the same approved config/vector hashes.
- The next synchronized snapshot contains `11/15` complete checkpoint
  validation reports. Seed-42 temporary APS and 100-state permutation checks
  exited zero for Deep Sets and canonical GRU. Deep Sets was exactly invariant
  on every recorded drift measure; canonical GRU showed mean JS drift
  `0.000586`, relative hidden drift `0.07684`, bit-probability L1 drift
  `0.02585`, and prediction-set disagreement `0.08522`. This is the expected
  validation-only evidence that unordered evidence modeling removes artificial
  acquisition-order dependence.
- The final training/evaluation sync contains `15/15` checkpoint validation
  reports, `12/12` non-clean permutation reports, `12/12` temporary APS
  reports, and the identity shortcut-control report. Three-seed means are
  Deep Sets `0.864217/0.508031`, canonical GRU `0.863926/0.500815`, random-GRU
  `0.864333/0.494725`, masked-slot MLP `0.864605/0.499082`, and clean-only MLP
  `0.751657/0.272292` for source-bit/six-way Macro-F1. Pre-gate inspection
  predicts Deep Sets selection: it is within `0.000116` of the best GRU on the
  primary metric, exactly invariant, and `0.097711` above the identity-only
  shortcut. The Set Transformer trigger is not reached. The appendix-only
  RAPS trigger fires because mean 90%-APS size/singleton rate are
  `3.2087/0.2075`; this does not block the main APS stack.
- The signed CPU-only Week 5 selection gate then wrote report SHA-256
  `03a2e49f...36916` with status `SELECTED`, selected Deep Sets seed 42, and
  confirmed completion gate true, shortcut gate passed, Set Transformer not
  triggered, and no calibration/test access. RAPS remains a triggered
  appendix-only ablation. The deliberate nonzero exit requested explicit owner
  review before writing the freeze manifest.
- The owner explicitly approved Deep Sets seed 42 as the Week 5 diagnostic
  encoder and authorized the signed Week 5 freeze on 2026-08-26.
- The owner-approved Week 5 freeze was written with SHA-256
  `0512ca85...a7929`. Full Week 5 validation passed with vector status
  `COMPLETE`, zero errors, zero warnings, and report SHA-256
  `026ef289...207e`. Week 5 is COMPLETE; Week 6 readiness is unblocked.

## 2026-08-27 — Week 6 readiness passed

- The CPU-only Week 6 readiness report is valid with zero errors. VOI preflight
  verified eight frozen state/teacher sources, budgets `1/2/4`, cost
  multipliers `0/0.05/0.1/0.2/0.4`, and the Week 5 freeze/checkpoint/APS
  hashes. Calibration and test remain locked. A bounded 100-row VOI
  construction audit is the next execution gate.
- Added root `ICLR_DEFERRED_WORK_AND_EXTERNAL_SETUP.md` to track deadline
  deferrals, external dataset/model/human prerequisites, conditional ablations,
  and safe resume points without conflating them with the core critical path.
- The bounded Deep Sets VOI audit produced 100 valid train targets in 4.36
  seconds. Every cost multiplier had both positive and negative targets; STOP
  selections increased from 39 at cost 0 to 54 at cost 0.4. Calibration/test
  remained unused. The train-only prefix is valid for construction auditing
  but cannot support a policy pilot requiring validation rows, so complete
  architecture-specific VOI corpora are next.
- Complete Deep Sets VOI construction then passed in 10:41.57 with 496,912
  train/validation targets (`436,429/60,483`), mixed-sign targets at every
  cost, and no calibration/test access. The parallel GRU build failed closed
  before inference because `build_voi_targets.py` accepted only the primary
  freeze artifact even though the signed Week 5 freeze and Week 6 requirements
  include GRU/masked comparison diagnostics. The contract was corrected to
  accept any hash-bound `diagnostic_checkpoint*` artifact and reject unfrozen
  files; a regression test covers selected, comparison, unrelated, and
  wrong-prefix artifacts.
- The correction passed 14 focused tests and the complete server suite
  (`280 passed`, one non-blocking Hugging Face deprecation warning). Retried
  full VOI construction passed for random-order GRU in 10:46.19 and
  masked-slot MLP in 10:43.50. Together with Deep Sets, all three signed VOI
  manifests are `COMPLETE`, each records 496,912 train/validation targets
  (`436,429/60,483`), and none accesses calibration or test. The Week 6
  execution gate therefore advances to bounded policy-training pilots.
- The 100-row Deep Sets learned-VOI policy and train-only entropy-reduction
  policy pilots both exited zero in 40.94 and 30.24 seconds. Each consumed 100
  train plus 100 validation targets and explicitly records
  `calibration_used=false` and `test_used=false`. The learned-VOI pilot's best
  epoch/loss/agreement/STOP rate are `19/0.616179/0.72/0.40`; the uncertainty
  pilot's are `13/0.541400/0.66/0.46`. Both paths are pilot validated. One
  complete run of each is next so the remaining grid is scheduled from
  measured full-corpus throughput rather than extrapolated tiny-batch timing.
- Complete training then passed for both paths. The Deep Sets VOI policy at
  multiplier `0.1`, seed 42 trained on `436,429` rows, validated on `60,483`,
  selected epoch 16, and completed in 25:59.92 with validation loss
  `0.580301`, next-action agreement `0.797365`, and STOP rate `0.534960`. The
  train-only entropy-reduction baseline selected epoch 10 and completed in
  19:26.90 with loss `0.416474`, agreement `0.821586`, and STOP rate
  `0.406313`. Both reports are scientifically valid, hash-bound, and record no
  calibration/test access. These measurements project the remaining 14-policy
  grid at approximately six GPU-hours or three wall-clock hours on two GPUs.
- The owner approved all 14 remaining Deep Sets cost/seed policies on
  2026-08-28, restricted to physical GPU 0. The approved one-GPU sequential
  schedule preserves the full five-multiplier/three-seed grid and is expected
  to require about six GPU-hours and 6–7 wall-clock hours. The already valid
  multiplier-0.1/seed-42 checkpoint is excluded from the loop rather than
  overwritten.
- Server logs received on 2026-08-29 show every remaining run exited zero and
  wrote a scientifically valid `436,429/60,483` train/validation report with
  no calibration/test use. Aggregate measured runtime was 7.5586 GPU-hours
  (mean 32.39 minutes/run); multiplier 0/seed 42 was the slowest at 1:16:16
  but completed successfully. The log sync is complete, while the local
  checkpoint sync is not: only 4 of the expected 60 Deep Sets policy files
  are present. Do not rerun inference/training; sync the server checkpoint
  directory before hash and matrix validation.
- The checkpoint directory was resynchronized and independently re-audited.
  It now contains all 60 expected Deep Sets policy artifacts (15 each of best,
  last, history, and validation). The complete 5×3 seed/cost matrix is exact;
  every report is scientifically valid, contains `436,429/60,483`
  train/validation rows, excludes calibration/test, and matches the SHA-256 of
  its best checkpoint. No missing, extra, duplicate, malformed, or hash-drifted
  artifact remains. The policy-training portion of W6-10 is pilot validated;
  scalar/distilled controls and validation frontiers remain.
- The first scalar one-row baseline stage reached validation but stopped
  fail-closed because a one-row slice cannot contain both target classes for
  every source bit, making AUROC undefined. No report was written and no full
  baseline was launched. The correction leaves every full/scientific metric
  strict and permits only explicitly limited non-scientific pilots to record
  undefined AUROCs as JSON `null` with the observed class, reason, and
  `metrics_scientifically_valid=false`. A regression test preserves the
  default exception and covers the audited pilot representation. Local syntax
  compilation passes; local pytest is unavailable, so focused and full server
  tests are required before rerunning the one-row stage.
- The first focused server run reached the new regression test but failed
  because the assertion accidentally used two six-way reporting labels instead
  of the frozen source-bit names. Production output correctly used `visual`,
  `language`, and `alignment`. The test now derives its expectation directly
  from `SOURCE_BITS`; no production code or scientific contract changed.
- The corrected focused suite passed `7` tests and the complete server suite
  passed all `281` tests in 8:58. The sklearn single-label warning is expected
  only for the explicitly non-scientific pilot test; the Hugging Face warning
  is the existing deprecation notice. Baseline staging is unblocked, subject
  to choosing a GPU that is actually free at launch.
- The corrected scalar-confidence and one-pass-distilled baseline stages now
  pass at limits `1`, `10`, and `100` (six runs total). All six logs exit zero
  without tracebacks, all six best-checkpoint SHA-256 values match their
  reports, and no report accesses calibration or test. The one-row reports
  explicitly mark their metrics non-scientific and serialize all three
  mathematically undefined AUROCs as `null`; the 10- and 100-row reports have
  finite AUROCs. Each bounded run took 5.53--6.51 seconds. Complete scalar and
  distilled training is the next separately approved GPU step.
- The owner approved complete scalar-confidence and one-pass-distilled Week 6
  training on two GPUs verified free at launch, with a combined ceiling of one
  GPU-hour. Each job is capped at 30 minutes and uses resume-safe outputs; a
  timeout must be inspected rather than silently extending the authorization.
- Complete scalar and distilled training both passed on physical GPU 2 with
  exit status zero, no traceback, scientifically valid checkpoints, and no
  calibration/test access. Scalar selected epoch 1 and finished in 15.75
  seconds with validation source-bit Macro-F1 `0.739826`; distilled selected
  epoch 7 and finished in 24.53 seconds with Macro-F1 `0.757429`. Both reports
  cover all 1,404 validation model-instances and their checkpoint SHA-256
  values match. Running sequentially rather than on two devices is an
  operational deviation only; combined compute stayed far below the approved
  one-GPU-hour ceiling. Temporary validation APS for the three static controls
  is the next gate before frontier staging.
- Temporary validation APS passed for the frozen clean-only, scalar-confidence,
  and one-pass-distilled controls. All three jobs exit zero in 2.88--2.97
  seconds, fit only the 1,404 validation rows, exclude calibration/test, cover
  budgets `1/2/4` at targets `0.90/0.95`, match their checkpoint hashes, and
  pass independent report self-hash verification. Repeated per-budget static
  thresholds are expected because these controls acquire no probes. The next
  gate is a 100-row end-to-end validation frontier including both oracle
  controls before any complete frontier matrix.
- The 100-row Deep Sets cost-0.1/seed-42 validation frontier passed end to end
  on the correctly free physical GPU 2 in 1:08.45. It contains all 15 required
  conditions, budgets `1/2/4`, full-teacher budget 7, both mandatory oracles,
  100 rows per applicable cell, and a validation-selected dataset-specific
  schedule. The JSON self-hash and CSV, PNG, and schedule SHA-256 bindings all
  pass. Pilot completion indicators show ProActive beating random and fixed at
  at least one budget. This is encouraging but not a full-validation claim;
  all 15 seed/cost frontiers remain required. Measured linear projection is
  about 4.0 GPU-hours for that matrix before contingency, and only GPU 2 was
  free at the latest server check.
- The owner approved the exact 15-run complete Week 6 validation-frontier
  matrix on one or two GPUs verified free at launch, with both mandatory
  oracles retained, a combined `7.5` GPU-hour ceiling, and a 30-minute timeout
  per frontier. The latest device audit authorizes only physical GPU 2; GPUs
  0, 1, and 3 are occupied and must not be used without a fresh free-device
  check.
- The complete Deep Sets validation-frontier matrix was synchronized and
  independently audited on 2026-09-04. All `15/15` full validation reports
  (`5` cost multipliers x `3` seeds) are present, `is_valid=true`, cover all
  `1,404` validation model-instances, contain all `43` required frontier rows
  and both mandatory oracles, and have `limit=null`. All `60/60` referenced
  JSON/CSV/PNG/schedule artifacts exist and their recorded SHA-256 values
  match. All 15 logs end with exit status zero. Total measured single-GPU
  time was `0.930472` GPU-hours (mean `223.31` seconds/run; range
  `214.87--238.12` seconds), well below the approved `7.5` GPU-hour ceiling.
  A read-only recomputation predicts cost multiplier `0.0` as the selection
  winner with three-seed mean proactive source-bit Macro-F1 `0.942567`, mean
  cost `2.284742`, and mean set size `2.167221`. This is not yet the signed
  selection: `select_week6_policy.py` must now run CPU-only and the owner must
  review its report. GRU and masked-slot learned-policy comparison frontiers
  also remain required before the full Week 6 validator can pass.
- The formal CPU-only Week 6 selection review was synchronized and verified on
  2026-09-04. Report self-hash `c6554fb7...9073`, selected policy hash, and
  selected frontier hash all match. The predeclared validation-only rule
  selected cost multiplier `0.0`, primary seed `42`, with three-seed mean
  source-bit Macro-F1 `0.942567`, mean acquisition cost `2.284742`, and mean
  set size `2.167221`. It records no calibration/test access. All four gates
  pass: learned ProActive beats random, beats a fixed schedule, makes monotonic
  budget progress, and full evidence improves over clean-only by `30.0581%`
  versus the predeclared `10%` minimum. The review command correctly exited
  nonzero after writing the report because explicit owner approval is still
  required. Do not freeze Week 7 until that approval is recorded and the
  comparison-frontier/full-validator requirements are satisfied.
- On 2026-09-04 the owner explicitly approved cost multiplier `0.0` and seed
  `42` as the selected Week 6 operating point. This accepts the outcome of the
  predeclared validation rule; it does not yet declare Week 6 complete or
  unlock the test split. The next immediate action is the CPU-only
  `--approve_selection` acceptance rerun. After that, the mandatory GRU and
  masked-slot policy comparisons must be staged, completed under a separate
  bounded compute approval, and supplied to the full Week 6 validator.
- The approved CPU selection rerun completed without the owner-review
  `SystemExit`, reproducing the same self-hash `c6554fb7...9073`; the selected
  lambda-0.0/seed-42 boundary is now operationally recorded. Four mandatory
  architecture-comparison training pilots then passed on physical GPU 3:
  GRU VOI, GRU entropy reduction, masked-slot VOI, and masked-slot entropy
  reduction. All four logs exit zero, all checkpoint hashes and report
  self-hashes match, every report uses exactly 100 train plus 100 validation
  rows, and calibration/test flags are false. Their `is_valid=false` fields
  are intentional because `--limit 100` artifacts are bounded pilots rather
  than complete scientific checkpoints. Loss/agreement/STOP are finite for all
  four. Total measured pilot time was 129.85 seconds (`0.0361` GPU-hours), with
  individual runs taking 30.44--35.58 seconds. Complete architecture-specific
  training, controls, APS, and frontiers now require a bounded compute
  approval before full Week 6 validation.
- On 2026-09-05 the owner approved the complete random-permutation GRU and
  masked-slot MLP Week 6 comparison bundle on physical GPU 3. The authorization
  covers four complete policy jobs, four architecture-bound scalar/distilled
  controls, four temporary validation-only static APS jobs, and two complete
  validation frontiers. The combined ceiling is `3` GPU-hours, each training
  job is capped at `45` minutes, and each frontier is capped at `15` minutes.
  The launch must first verify that physical GPU 3 has no compute process; all
  jobs are sequential and fail closed at the first nonzero exit. Calibration
  and test remain locked.
- The first approved bundle invocation stopped correctly at its device
  preflight. Physical GPU 3 had live compute PIDs `3925159` and `3234270`
  holding `18,974` and `19,414` MiB respectively, despite reporting 0% momentary
  utilization. The bundle returned exit code `1` before launching any training
  job, so no scientific artifact or approved GPU-hour budget was consumed.
  The same resume-safe function may be relaunched after GPU 3 is genuinely free,
  or on another physical GPU only after an explicit authorization update.
- The complete comparison bundle was subsequently executed on the verified-free
  physical GPU 1 and synchronized on 2026-09-06. All 14 jobs exited zero. The
  four complete GRU/masked VOI and entropy-reduction policies use 436,429 train
  and 60,483 validation states, have `limit=null`, and record no calibration or
  test access. All four architecture-bound scalar/distilled checkpoints and all
  four temporary APS reports are valid and hash-consistent. Both complete
  validation frontiers contain 1,404 model-instances, 43 rows, 15 conditions,
  budgets `1/2/4`, and both mandatory oracles; every referenced checkpoint,
  CSV, PNG, and dataset-schedule hash matches. Both learned comparison policies
  beat random and fixed schedules at at least one budget. Measured sequential
  runtime was 6,815.44 seconds (`1.8932` GPU-hours), below the approved 3-hour
  ceiling. The remaining Week 6 action is the CPU-only full validator.
- The synchronized CPU-only full Week 6 validator passed on 2026-09-06:
  `is_valid=true`, `errors=[]`, configuration `APPROVED`, VOI status
  `COMPLETE`, and 496,912 VOI rows. The signed report self-hash independently
  reproduces as `fade7dc83ec2b94e3bbf3110457c18a800647026e7cc5de4ed05a8dedf29ff64`.
  Week 6 is COMPLETE. The next operation is not calibration or test: Week 7
  must first run the selected Deep Sets validation frontier at final budgets
  `1/2/3/4/7` so its dataset-specific schedule can be bound into the immutable
  main-stack freeze.
- The Week 7 static-control expansion completed and was synchronized on
  2026-09-06. Clean-only learned, scalar-confidence, and one-pass-distilled
  APS reports are complete validation-only artifacts with `limit=null`, budgets
  `1/2/3/4/7`, and both `0.90`/`0.95` targets. All three explicitly record
  `calibration_used=false` and `test_used=false`. The next computation is one
  selected Deep Sets validation frontier that materializes the final-budget
  dataset schedule; calibration and test remain locked until the resulting
  schedule is audited and the main stack is signed.
- On 2026-09-06 the owner approved one complete selected Deep Sets
  lambda-0.0/seed-42 validation frontier on physical GPU 1, conditional on a
  fresh empty-device check. The authorization ceiling is `0.25` GPU-hours with
  a fail-closed `15`-minute timeout. The run retains budgets `1/2/3/4/7`, all
  required static controls and both oracles, and may not access calibration or
  test.
- The approved final-budget validation frontier completed on physical GPU 1
  with exit code zero in `7:38.52` (`0.1274` GPU-hours). The synchronized
  report is valid over 1,404 validation model-instances, contains 71 rows and
  all 15 conditions, covers budgets `1/2/3/4/7`, retains both oracles, and
  records `limit=null`. ProActive passes both learned-versus-random and
  learned-versus-fixed indicators. Recorded CSV `16b1e56f...6fe0b`, figure
  `80f47890...6d697`, and dataset schedule `354aadbd...58c8` hashes match the
  synchronized files. The selected schedule is validation-only
  (`calibration_used=false`, `test_used=false`) and its self-hash is
  `0152171c...b6b1`. The next action is the CPU-only freeze preview; no
  calibration or test has been unlocked.
- The CPU-only Week 7 freeze preview passed on 2026-09-06 with exit code zero.
  It validated and proposed binding the frozen Deep Sets seed-42 diagnostic,
  selected lambda-0.0/seed-42 policy, signed Week 6 selection, uncertainty,
  scalar and distilled controls, and the five-budget validation-only dataset
  schedule. The preview wrote no freeze and explicitly reports
  `owner_approval_required=true`; calibration and test remain locked pending
  owner authorization of the immutable boundary.
- The owner explicitly approved the reviewed Week 7 main-stack freeze on
  2026-09-06 and authorized writing
  `outputs/week7_frozen/main_stack_freeze.json`. The approved boundary is Deep
  Sets seed 42, policy cost multiplier 0.0/seed 42, the signed Week 6
  selection, uncertainty/scalar/distilled controls, approved configuration
  files, and the validation-only five-budget dataset schedule. This approval
  permits writing the freeze; calibration/test remain locked until the written
  manifest is synchronized and hash-audited.
- The written main-stack freeze was synchronized and independently audited on
  2026-09-06. It records status `FROZEN`, `owner_approved=true`, 11 bound
  artifacts, and internal freeze hash `ec886b26...b21ef`; every bound artifact
  exists and its SHA-256 matches. The freeze file byte hash is
  `735b6b27...d2ed`. W7-01 is complete and the selection boundary is closed.
  The next allowed operation is calibration-only; locked test remains
  inaccessible until final APS exists.
- The CPU-only Week 7 readiness validator passed on 2026-09-06 with
  `is_valid=true`, approved configuration, no errors or warnings, and report
  self-hash `12e679f3...f136b`. The reported `NO-GO` is expected because this
  was readiness mode, not the final completion validator. Calibration-only
  execution is unlocked; test remains locked.
- The owner authorized the complete Week 7 calibration-only trajectory run on
  physical GPU 1 on 2026-09-06, conditional on a fresh empty-device check,
  with a `0.25` GPU-hour ceiling and fail-closed 15-minute timeout. The three
  frozen static-control APS calibration jobs are authorized to run on CPU in
  parallel. All jobs use budgets `1/2/3/4/7`, targets `0.90/0.95`, and the
  signed main-stack freeze. Test remains locked.
- The complete Week 7 calibration bundle was synchronized and audited on
  2026-09-06. The frozen policy generated 7,500 calibration-only trajectories,
  exactly 1,500 for every budget `1/2/3/4/7`; all row counts and file hashes
  match, the manifest is `COMPLETE`, and `test_used=false`. The GPU trajectory
  job exited zero in 49.71 seconds (`0.0138` GPU-hours). Main APS is
  `FINAL_FROZEN` with thresholds hash `3d70ec81...3177a`, and the three static
  APS controls are also final, complete, and freeze-bound. Every main
  target/budget cell meets the owner-approved 0.03 calibration-undercoverage
  tolerance. Test remains unaccessed pending an explicit one-time locked-test
  authorization.
- The owner explicitly authorized the one-time complete Week 7 locked-test
  bundle on 2026-09-06, acknowledged that test will be accessed once, and
  prohibited post-test tuning. The authorization assigns the complete frontier
  to physical GPU 1 and the primary Deep Sets plus canonical/random-order GRU
  permutation studies to physical GPU 2 after fresh empty-device checks. The
  combined ceiling is `1` GPU-hour. The frontier retains its approved
  20-minute cap; permutation caps are tightened from the approved maximum of
  15 to 13 minutes each so worst-case configured limits remain below the
  combined ceiling. Contract-only dry-runs must pass before test data load.
- The first locked-test frontier launch stopped safely before its contract
  dry-run on 2026-09-06 because physical GPU 1 had acquired two compute
  processes (PIDs `2651580` and `2652474`, using 24,658 and 11,336 MiB).
  Exit code `1` therefore records a device-preflight refusal, not an
  experimental failure. No test example was read, no frontier artifact was
  written, and none of the one-time locked-test or GPU-hour budget was
  consumed. The unchanged frontier may be retried under the existing approval
  only when a fresh check confirms physical GPU 1 has no compute processes.
- The owner then explicitly amended the execution mapping: run the larger
  locked-test frontier on physical GPU 3 and the smaller three-run permutation
  bundle on physical GPU 0, even when unrelated processes hold most VRAM.
  Because these jobs load only small diagnostic/policy checkpoints rather than
  an MLLM, the revised launch guard allows unrelated compute processes but
  requires at least 1,536 MiB free at launch and refuses a duplicate ProActive
  frontier/permutation process. The scientific boundary is unchanged: one
  complete locked-test pass, no post-test tuning, 20-minute frontier cap,
  13-minute cap for each permutation, and one combined GPU-hour ceiling.
- The amended locked-test bundle completed and was synchronized on 2026-09-07.
  All four contract checks and executions exit zero. The test frontier is
  `is_valid=true`, complete over 1,560 model-instances, 142 summary rows,
  budgets `1/2/3/4/7`, targets `0.90/0.95`, all 15 conditions, and both
  mandatory oracles. It reports ProActive beating random and fixed schedules;
  the 0.88 budget-1 coverage at target 0.90 passes the frozen 0.03 tolerance.
  The primary Deep Sets permutation report has 2,000 states and exactly zero
  JS, bit, hidden-state, set, action, coverage, and set-size drift. Both GRU
  comparison reports also have 2,000 states. All four report self-hashes and
  every CSV, PNG, checkpoint, APS, freeze, schedule, manifest, and eight
  teacher-cache bindings independently match. Runtime was 524.43 seconds for
  the frontier and 23.98 seconds across the three permutations, totaling
  548.41 seconds (`0.1523` GPU-hours), below the approved one-hour ceiling.
  The frozen test has now been read once; no post-test tuning or outcome-driven
  rerun is authorized. The CPU-only full Week 7 validator is the final gate.
- The synchronized full Week 7 validator then passed with `is_valid=true`,
  `errors=[]`, and `go_no_go=GO`. Its self-hash independently reproduces as
  `47fe3b99974f64be83d00a4b7c6838351d5e90ab65be97d392e7e2da14e9d0d7`,
  its log payload exactly matches the report, and the memo declares no blocking
  findings. The only warning is the validation-triggered optional RAPS appendix
  ablation; the locked APS main result remains unchanged. Week 7 is COMPLETE.
  Week 8 begins with cached robustness analyses and the already prepared but
  entirely unfilled three-person human audit; PRE-HAL/IllusionBench setup is a
  time-sensitive optional shift decision rather than a Week 7 dependency.

## 2026-09-11 — Frozen shift and latency complete; LOMO loader corrected

- The immutable Week 7 stack completed the held-out PRE-HAL/IllusionBench
  frontier over 2,400 model instances and budgets `1/2/3/4/7`, without
  target-domain calibration or post-shift tuning. The report is valid and all
  trajectory, CSV, and figure hashes match.
- Fixed-hardware RTX A6000 latency evidence is valid: 10 warm-ups, 100 CUDA-
  synchronized measurements, 1.355 ms mean controller overhead, and 9,163.951
  ms mean cached generation latency.
- The initial Qwen LOMO fold exited before training. The synchronized test log
  reported only three tests, proving the server ran the older builder. That
  version read `split` and `model_id` from the top level, while every canonical
  `partial_state_v1` source stores them under `metadata`; it therefore filtered
  every state row out. The corrected builder identifies itself as
  `metadata_identity_v2`, includes observed-identity diagnostics, and adds an
  end-to-end state-filter regression. Gemma was not attempted and no LOMO
  result was produced by the failed 12-second CPU job.
- The corrected server rerun passed seven focused tests and emitted both
  complete `metadata_identity_v2` fold manifests. Qwen-held-out contains
  194,249 states and Gemma-held-out 194,374; both use source-only development
  splits and held-out-only test. An independent local audit recomputed all
  eight referenced artifact hashes and confirmed both vector-manifest row sums.
  No fold training or held-out evaluation has occurred yet.
- The owner approved the complete two-fold LOMO run on 2026-09-11: seed 42,
  Deep Sets diagnostics, source-only calibration, and held-out-model evaluation
  on physical GPU 3 or two independently verified free GPUs. The combined
  ceiling is five GPU-hours. Held-out test evidence may not be used for tuning;
  this authorization also permits the two predeclared diagnostic/stack freeze
  manifests required to bind the source-trained checkpoints before evaluation.
- The complete LOMO bundle then finished in 2,944 seconds (`0.818` aggregate
  GPU-hours) with all 28 stages exiting zero. Both evaluation reports are valid
  over 780 held-out test examples; source-only calibration, held-out identity,
  and every bound artifact/self-hash independently reproduce. At budget 7,
  ProActive source-bit Macro-F1 exceeds clean-only and scalar controls in both
  directions: Qwen-held-out `0.9692` versus `0.6826`/`0.6964`, and
  Gemma-held-out `0.9948` versus `0.7956`/`0.7804`. Corresponding ProActive
  six-way Macro-F1 is `0.7927` and `0.8063`. Qwen-held-out 0.90-target
  coverage is `0.9910`, but Gemma-held-out coverage falls to `0.7115` at
  budget 7. This is recorded as cross-model calibration degradation rather
  than repaired through held-out tuning. The defensible claim is useful
  diagnostic transfer, not guaranteed conformal coverage across model shift.

## 2026-09-11 — Mandatory Week 8 ablation bundle authorized

- The owner approved all 15 predeclared Week 8 ablations at seed 42 using
  validation evidence only, on at most two GPUs verified free at launch.
- The hard compute boundary is eight combined GPU-hours, with 45 minutes per
  training job and 20 minutes per frontier. Core test and held-out shift are
  forbidden for tuning or ablation selection.
- The execution path is resumable and fail-closed. Feature ablations now share
  one transform across tensor construction, counterfactual VOI targets, and
  future rollout observations, preventing removed features from silently
  re-entering the policy pipeline.
- Authorization is not completion. The signed 15-item aggregate remains
  pending until `outputs/week8_reports/ablations.json` is produced and checked.
- The first launch gate passed 22 focused tests plus 16 subtests, then stopped
  before any experiment because physical GPU 1 had two compute processes. No
  ablation GPU time or scientific evidence was consumed. GPUs 2 and 3 were
  subsequently identified as free for an explicit guarded launch.

## 2026-09-12 — Ablation GPU work complete; aggregation CLI corrected

- The no-budget checkpoint rebuild passed the predeclared exact early-stopping
  history and APS scientific-equivalence gate. Its stale descendants were
  archived and its provenance chain was regenerated rather than edited.
- All remaining mandatory validation-only ablation stages completed. The GPU
  ledger records 28,275 conservative seconds (`7.8542` GPU-hours), below the
  approved eight-hour ceiling. It retains two earlier timeout exits and one
  APS provenance refusal; each was subsequently resolved by a successful
  resume/rebuild. All 15 evidence JSON files are present; their self-hashes and
  29 bound source-artifact hashes independently reproduce.
- The final CPU aggregator reported 14 evidence items missing because the
  launcher repeated `--evidence`, while the parser retained only the final
  occurrence. This was a command-interface defect, not missing experimental
  evidence. The option now accumulates repeated values and a regression test
  protects the 15-item invocation. Do not rerun training or frontiers; only the
  focused test and signed CPU aggregation remain.

## 2026-09-12 — Mandatory Week 8 ablation aggregate completed

- The repeated-evidence regression passed (`1 passed, 10 deselected`).
- CPU-only aggregation accepted all 15 unique mandatory evidence files and
  wrote 121 comparison rows. The signed report is valid and records no
  post-test tuning.
- Independent verification reproduced report SHA-256
  `ee0a3eb3374fa857b6981a920028cf3dac0d4ea9dd4a2a162de8c3356f7de14b`,
  CSV SHA-256
  `7dfcbbb7ae05e5395593c7479e3068c1b43d615d26e36e215867062bca222613`,
  and every referenced evidence-file hash. W8-04 is COMPLETE.
