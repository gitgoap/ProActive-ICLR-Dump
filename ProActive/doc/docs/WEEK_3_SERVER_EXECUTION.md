# Week 3 Server Execution — August 7, 2026

Code state: base commit `c3f68e68874e` plus the Week 3 sync bundle listed below. Local verification: `ruff` passed and `157` tests passed. GPU status remains **IMPLEMENTED, NOT VALIDATED** until the final gate passes on server artifacts.

## 1. Required server sync

Copy these files before running anything:

```text
src/proactive/data/loaders.py
src/proactive/utils/io.py
src/proactive/features/semantic.py
src/proactive/teacher/cache_builder.py
src/proactive/audits/schema_validator.py
src/proactive/audits/pilot_analysis.py
scripts/run_pilot_cache.py
scripts/validate_teacher_schema.py
scripts/analyze_pilot.py
scripts/calibrate_semantic_threshold.py
scripts/check_week_completion.py
tests/test_week3_pipeline_safety.py
```

Also sync these tracking documents when convenient:

```text
PROJECT_STATUS.md
WEEKLY_PROGRESS.md
doc/docs/WEEK_3_REQUIREMENTS_MATRIX.md
doc/docs/WEEK_3_SERVER_EXECUTION.md
```

Do not copy local `outputs/pilot_cache` over the server cache. The commands below safely repair/resume the existing server output.

## 2. CPU preflight and corrected manifests

Run from the server repository root in the existing project environment:

```bash
set -euo pipefail
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
pytest -q
python scripts/check_week_completion.py --mode readiness
python scripts/build_manifests.py \
  --config_dir configs/data \
  --data_root "$PROACTIVE_DATA_ROOT" \
  --output_dir outputs/manifests \
  --datasets hallusionbench pope vizwiz vsr \
  --overwrite --seed 42
python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/manifests/manifest_hallusionbench.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
assert len(rows) == 951, len(rows)
assert all(row.get("image_path") for row in rows)
print("HallusionBench manifest: 951/951 image-paired rows")
PY
mkdir -p outputs/logs/week3
```

No repeated 1- or 10-example adapter smoke is scheduled: Qwen and Gemma already passed real server smoke tests. The new optimized severity path is staged through the remaining 3 VizWiz examples and then 21 VSR examples before any new 100-example severity job.

## 3. Run W3-QWEN-CANONICAL-20260807

Purpose: remove the 31 duplicated POPE rows, fill VizWiz/VSR to exactly 100, and add HallusionBench.

```bash
set -euo pipefail
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
mkdir -p outputs/logs/week3

(CUDA_VISIBLE_DEVICES=0 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_pope.jsonl --model_config configs/models/qwen3_vl_8b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode canonical --overwrite 2>&1 | tee outputs/logs/week3/qwen_pope_canonical.log) & p0=$!
(CUDA_VISIBLE_DEVICES=1 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_vizwiz.jsonl --model_config configs/models/qwen3_vl_8b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode canonical --resume 2>&1 | tee outputs/logs/week3/qwen_vizwiz_canonical.log) & p1=$!
(CUDA_VISIBLE_DEVICES=2 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_vsr.jsonl --model_config configs/models/qwen3_vl_8b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode canonical --resume 2>&1 | tee outputs/logs/week3/qwen_vsr_canonical.log) & p2=$!
(CUDA_VISIBLE_DEVICES=3 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_hallusionbench.jsonl --model_config configs/models/qwen3_vl_8b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode canonical --resume 2>&1 | tee outputs/logs/week3/qwen_hallusionbench_canonical.log) & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
```

Assumptions: four available A6000 GPUs; one independent model replica per GPU; `dtype: auto`; batch size 1; deterministic generation; `max_new_tokens: 256`. Estimated work is about 1,600 forward passes, 0.3–0.7 GPU-hours, 10–25 minutes on four GPUs, and under 5 MB. On two GPUs run two commands at a time (about 20–45 minutes); on one GPU run sequentially (about 35–75 minutes).

Monitor with `watch -n 5 nvidia-smi` and `tail -f outputs/logs/week3/<log>.log`. Resume all commands with the same arguments except the POPE repair: after a successful POPE overwrite, change that command to `--resume`. Stop if any run reports failed instances, CUDA OOM, non-finite scores, semantic matcher failure, or projected runtime exceeds this estimate by more than 25%.

## 4. Run W3-QWEN-SEVERITY-BRIDGE-20260807

Run the 3-example VizWiz resume first. It is the nonredundant server validation of the optimized 15-generation severity path:

```bash
set -euo pipefail
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
CUDA_VISIBLE_DEVICES=0 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_vizwiz.jsonl --model_config configs/models/qwen3_vl_8b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode severity_grid --resume 2>&1 | tee outputs/logs/week3/qwen_vizwiz_severity.log
python scripts/validate_teacher_schema.py outputs/pilot_cache/qwen3_vl_8b_vizwiz_severity_pilot.jsonl
```

Success requires exactly 1,200 unique severity rows and no duplicate/schema errors. Then run the 21-example VSR bridge:

```bash
set -euo pipefail
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
CUDA_VISIBLE_DEVICES=0 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_vsr.jsonl --model_config configs/models/qwen3_vl_8b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode severity_grid --resume 2>&1 | tee outputs/logs/week3/qwen_vsr_severity.log
python scripts/validate_teacher_schema.py outputs/pilot_cache/qwen3_vl_8b_vsr_severity_pilot.jsonl
```

Estimated combined work: 360–380 forward passes, under 0.2 GPU-hours, about 5–12 minutes, and under 2 MB. Stop on any failed instance or validation error.

After both bridges pass, complete Qwen HallusionBench (POPE is already complete; its resume is a cheap no-op check):

```bash
set -euo pipefail
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
(CUDA_VISIBLE_DEVICES=0 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_pope.jsonl --model_config configs/models/qwen3_vl_8b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode severity_grid --resume 2>&1 | tee outputs/logs/week3/qwen_pope_severity.log) & p0=$!
(CUDA_VISIBLE_DEVICES=1 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_hallusionbench.jsonl --model_config configs/models/qwen3_vl_8b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode severity_grid --resume 2>&1 | tee outputs/logs/week3/qwen_hallusionbench_severity.log) & p1=$!
wait "$p0" "$p1"
```

Estimated HallusionBench work: about 1,500 forward passes, 0.3–0.7 GPU-hours, 15–35 minutes, and under 4 MB.

## 5. Run W3-GEMMA-CANONICAL-20260807

Purpose: obtain the required second-model 100-example canonical matrix.

```bash
set -euo pipefail
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
(CUDA_VISIBLE_DEVICES=0 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_pope.jsonl --model_config configs/models/gemma4_e4b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode canonical --resume 2>&1 | tee outputs/logs/week3/gemma_pope_canonical.log) & p0=$!
(CUDA_VISIBLE_DEVICES=1 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_vizwiz.jsonl --model_config configs/models/gemma4_e4b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode canonical --resume 2>&1 | tee outputs/logs/week3/gemma_vizwiz_canonical.log) & p1=$!
(CUDA_VISIBLE_DEVICES=2 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_vsr.jsonl --model_config configs/models/gemma4_e4b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode canonical --resume 2>&1 | tee outputs/logs/week3/gemma_vsr_canonical.log) & p2=$!
(CUDA_VISIBLE_DEVICES=3 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_hallusionbench.jsonl --model_config configs/models/gemma4_e4b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode canonical --resume 2>&1 | tee outputs/logs/week3/gemma_hallusionbench_canonical.log) & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
```

Assumptions: same precision/batch/generation settings as the Gemma YAML; about 2,800–2,900 forward passes. Because no complete Gemma pilot latency is available, budget a provisional 0.8–2.0 GPU-hours and 15–45 minutes on four GPUs (30–90 minutes on two, 1–3 hours on one). Pause if the first 10 completed records project more than 25% above the upper estimate.

## 6. Run W3-GEMMA-SEVERITY-20260807

Purpose: obtain the second-model three-level visual severity matrix.

```bash
set -euo pipefail
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
(CUDA_VISIBLE_DEVICES=0 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_pope.jsonl --model_config configs/models/gemma4_e4b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode severity_grid --resume 2>&1 | tee outputs/logs/week3/gemma_pope_severity.log) & p0=$!
(CUDA_VISIBLE_DEVICES=1 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_vizwiz.jsonl --model_config configs/models/gemma4_e4b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode severity_grid --resume 2>&1 | tee outputs/logs/week3/gemma_vizwiz_severity.log) & p1=$!
(CUDA_VISIBLE_DEVICES=2 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_vsr.jsonl --model_config configs/models/gemma4_e4b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode severity_grid --resume 2>&1 | tee outputs/logs/week3/gemma_vsr_severity.log) & p2=$!
(CUDA_VISIBLE_DEVICES=3 python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_hallusionbench.jsonl --model_config configs/models/gemma4_e4b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 100 --seed 42 --pilot_mode severity_grid --resume 2>&1 | tee outputs/logs/week3/gemma_hallusionbench_severity.log) & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
```

Estimated work: about 6,000–6,100 forward passes, provisionally 1.7–4.2 GPU-hours, 30–75 minutes on four GPUs (1–2.5 hours on two, 2–5 hours on one), and under 15 MB. Same monitoring, resume, and early-stop rules apply.

## 7. Validation, analysis, and mandatory human audit

```bash
set -euo pipefail
python scripts/validate_teacher_schema.py outputs/pilot_cache --output_report outputs/pilot_reports/schema_validation_report.json
python scripts/analyze_pilot.py --pilot_dir outputs/pilot_cache --output_dir outputs/pilot_reports --inspection_count 50
python scripts/calibrate_semantic_threshold.py --pilot_dir outputs/pilot_cache --output_csv outputs/pilot_reports/semantic_match_audit.csv --sample_count 50 --device cpu --overwrite
```

Inspect all five directories under `outputs/pilot_reports/inspection/`. Fill every `human_match` cell in `semantic_match_audit.csv` with `1` (same meaning) or `0` (different meaning), without changing other columns, then run:

```bash
python scripts/calibrate_semantic_threshold.py --labeled_csv outputs/pilot_reports/semantic_match_audit.csv --output_report outputs/pilot_reports/semantic_calibration_report.json --min_labels 50 --target_recall 0.90 --overwrite
```

Review `outputs/pilot_reports/pilot_analysis_summary.json`, the plots, image inspections, and semantic report. Only then freeze with all four reviewed severities:

```bash
python scripts/analyze_pilot.py \
  --pilot_dir outputs/pilot_cache \
  --output_dir outputs/pilot_reports \
  --freeze --confirm_freeze \
  --approved_severity blur=<REVIEWED_VALUE> \
  --approved_severity crop=<REVIEWED_VALUE> \
  --approved_severity brightness=<REVIEWED_VALUE> \
  --approved_severity noise=<REVIEWED_VALUE> \
  --semantic_report outputs/pilot_reports/semantic_calibration_report.json

python scripts/check_week_completion.py --mode full_week
```

The final gate must pass. Expected final artifacts include 16 model/dataset pilot JSONL files, the schema and semantic calibration reports, five plots, 250 transformed inspection images, `pilot_analysis_summary.json`, and `configs/probes/frozen_week3_config.yaml`.

## 8. Final execution outcome — August 9, 2026

- Qwen and Gemma canonical/severity matrices completed over HallusionBench, POPE, VizWiz, and VSR.
- Final directory: 16 JSONL files and 10,400 valid records (800 canonical + 9,600 severity).
- Schema validation: zero invalid rows and zero duplicates.
- Required analysis artifacts: five plots and 250 inspection images.
- Semantic audit: 50 labels; frozen threshold `0.50`.
- Frozen severities: blur `8`, crop `0.65`, brightness `0.15`, noise `25`.
- `configs/probes/frozen_week3_config.yaml` exists with `metadata.status: FROZEN`.
- Full Week 3 completion gate passed against the synced server artifacts.
- Local regression suite passed: 158 tests.
