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
  - `InternVL3-9B` — **NOT available, needs download**
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
