# Week 4 Server Execution

These commands are staged deliberately. Do not start the full four-GPU cache
until the model revisions and draft scientific settings are approved.

## 1. Existing environment verification and revision evidence

Use the Python environment already present in the tmux pane. No explicit Conda
activation is required. Run the verification lines again in each new pane
because environment variables are local to that shell.

```bash
cd ~/ProActive

which python
python --version
python -c "import torch, transformers, yaml; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"

export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004
mkdir -p outputs/logs/week4 outputs/teacher_core outputs/labels_core outputs/states_v1 outputs/week4_reports
set -o pipefail

python scripts/inspect_model_revisions.py \
  --model_configs configs/models/qwen3_vl_8b.yaml configs/models/gemma4_e4b.yaml configs/models/internvl3_9b.yaml \
  2>&1 | tee outputs/logs/week4/model_revision_inspection.log
```

Send the complete revision-inspection output back before editing the model
YAMLs. The corrected inspector reads repository commits separately from
per-file Hugging Face ETags. The expected result is `RESOLVED` with exactly one
revision candidate per model. A genuinely unresolved or ambiguous path is a
blocker; do not replace it with the current remote `main` hash.

Repeat the verification and `export` lines in every new tmux pane. No
environment activation command is needed, but exports set in one pane do not
propagate to the other panes.

After the pinned and approved configs are synced, run the CPU readiness gate:

```bash
python scripts/validate_week4.py \
  --mode readiness \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --output_dir outputs/week4_reports \
  --overwrite \
  2>&1 | tee outputs/logs/week4/readiness.log
```

It must report `"is_valid": true`, `"approval_status": "APPROVED"`, and an
empty `errors` list.

## 2. No separate dry run required

The local and server readiness gates cover the CPU-only contract checks. Skip a
separate teacher dry run and proceed to the mandatory one-row GPU stage only
after readiness reports no errors.

## 3. Staged GPU checks after revision evidence and approval cards

After the owner-approved config is synced, run a one-row check in a separate
staging directory, then a ten-row check. These are validation artifacts, not
part of the full cache.

Approved staged-card estimate: approximately 3,274 MLLM passes across the
Qwen 1/10/100/full-VSR checks and Gemma one-row check, using the Week 3 measured
rate of 1.1694 seconds/pass. Expected cost is about 1.06 GPU-hours, about 1.1
hours sequential wall time on one A6000 before loading overhead, and under
10 MB of JSONL/log output. The 33.23 GPU-hour full core remains unapproved and
is blocked by `compute_authorization.full_core_approved: false`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --limit 1 \
  --output_dir outputs/week4_staging/limit1 \
  --resume \
  2>&1 | tee outputs/logs/week4/qwen_limit1.log
```

Review this output, then run the corresponding one-row Gemma check:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model gemma4_e4b \
  --device cuda:0 \
  --limit 1 \
  --output_dir outputs/week4_staging/limit1 \
  --resume \
  2>&1 | tee outputs/logs/week4/gemma_limit1.log
```

Only after both one-row outputs are reviewed, run the Qwen ten-row check:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --limit 10 \
  --output_dir outputs/week4_staging/limit10 \
  --resume \
  2>&1 | tee outputs/logs/week4/qwen_limit10.log
```

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
  --resume \
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

## 4. Full deterministic-shard launch (approved 2026-08-13)

All staged gates passed and the owner approved the conservative 33.23 GPU-hour
Qwen+Gemma core. Each model has four deterministic shards. Start a shard only
on a physical GPU that `nvidia-smi` shows as free immediately before launch.
`CUDA_VISIBLE_DEVICES=N` exposes physical GPU N as logical `cuda:0`, so every
command below uses `--device cuda:0`.

At approval time only physical GPU 1 was free. The immediate launch is Qwen
shard 0:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --num_shards 4 \
  --shard_id 0 \
  --output_dir outputs/teacher_core \
  --resume \
  2>&1 | tee outputs/logs/week4/full_core/qwen_shard00-of-04.log
```

When other GPUs become free, launch Qwen shard IDs 1, 2, and 3 in separate
panes, changing both `CUDA_VISIBLE_DEVICES` and `--shard_id` and using distinct
log filenames. Never run two processes for the same model/shard output. After
all four Qwen shards validate, run the same four shard IDs for
`--model gemma4_e4b`. Interrupted shards resume with the identical command.

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
