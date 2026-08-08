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
