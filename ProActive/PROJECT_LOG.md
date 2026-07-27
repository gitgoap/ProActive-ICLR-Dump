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
