# Week 4 Server Execution

These commands are staged deliberately. Do not start the full four-GPU cache
until the model revisions and draft scientific settings are approved.

## 1. Environment and revision evidence

```bash
cd ~/ProActive
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004
mkdir -p outputs/logs/week4 outputs/teacher_core outputs/labels_core outputs/states_v1 outputs/week4_reports

python scripts/inspect_model_revisions.py \
  --model_configs configs/models/qwen3_vl_8b.yaml configs/models/gemma4_e4b.yaml configs/models/internvl3_9b.yaml \
  2>&1 | tee outputs/logs/week4/model_revision_inspection.log
```

Send the complete revision-inspection output back before editing the model
YAMLs. An unresolved path is a blocker; do not replace it with the current
remote `main` hash.

## 2. Pre-approval dry run

This loads no GPU model and is allowed while the config is still draft.

```bash
python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --num_shards 4 \
  --shard_id 0 \
  --dry_run \
  --allow_unapproved_smoke
```

## 3. Staged GPU checks after revision evidence and approval cards

After the owner-approved config is synced, run a one-row check in a separate
staging directory, then a ten-row check. These are validation artifacts, not
part of the full cache.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --limit 1 \
  --output_dir outputs/week4_staging/limit1 \
  2>&1 | tee outputs/logs/week4/qwen_limit1.log

CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --limit 10 \
  --output_dir outputs/week4_staging/limit10 \
  2>&1 | tee outputs/logs/week4/qwen_limit10.log
```

Repeat the one-row check with `--model gemma4_e4b` before the full launch.

The next mandatory stages are a 100-row run and one complete model-dataset
validation. Do not skip them even when the 1/10-row stages pass.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --limit 100 \
  --output_dir outputs/week4_staging/limit100 \
  2>&1 | tee outputs/logs/week4/qwen_limit100.log

CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_vsr.jsonl \
  --model qwen3_vl_8b \
  --dataset vsr \
  --device cuda:0 \
  --output_dir outputs/week4_staging/qwen_vsr_complete \
  --resume \
  2>&1 | tee outputs/logs/week4/qwen_vsr_complete.log
```

## 4. Full four-pane launch (withheld until explicit high-cost approval)

Use four tmux panes. Run one model across all four GPUs; after all four Qwen
shards pass, replace the model with Gemma and run the same four commands.

The exact four commands are intentionally withheld until the 1/10/100/full-VSR
logs have passed review and the owner explicitly approves the high-cost run.
At that point the run approval card will provide exact pane commands, commit
state, pinned revisions, measured wall-time estimate, monitoring, early-stop
criteria, and expected artifacts.

## 5. Daily/cache-end integrity check

```bash
python scripts/validate_week4.py \
  --mode teacher_progress \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --teacher_path outputs/teacher_core \
  --output_dir outputs/week4_reports \
  --resume \
  2>&1 | tee outputs/logs/week4/teacher_progress.log
```

## 6. Offline labels and states

Run only after both core model caches are complete.

```bash
python scripts/build_labels.py \
  --config configs/experiments/teacher_core.yaml \
  --teacher_path outputs/teacher_core \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --output_dir outputs/labels_core \
  --resume

python scripts/sample_states.py \
  --config configs/experiments/teacher_core.yaml \
  --teacher_path outputs/teacher_core \
  --labels_path outputs/labels_core \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --output_dir outputs/states_v1 \
  --resume
```

## 7. Human-audit packet

The literal plan requires all three models. Run this after InternVL is present.
For an explicitly interim two-model packet only, add
`--allow_incomplete_model_coverage`; it cannot pass the final gate.

```bash
python scripts/export_human_audit.py \
  --config configs/experiments/teacher_core.yaml \
  --teacher_path outputs/teacher_core \
  --labels_path outputs/labels_core \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --output_dir outputs/human_audit \
  --resume
```

## 8. Full completion gate

```bash
python scripts/validate_week4.py \
  --mode full \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --teacher_path outputs/teacher_core \
  --labels_path outputs/labels_core \
  --states_path outputs/states_v1 \
  --audit_dir outputs/human_audit \
  --output_dir outputs/week4_reports \
  --resume \
  2>&1 | tee outputs/logs/week4/full_validation.log
```

Do not delete or merge shard files manually. Resume the exact same command when
a pane stops; the script validates every existing row before appending only
missing model-instance keys.
