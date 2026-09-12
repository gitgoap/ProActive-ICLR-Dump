#!/usr/bin/env bash
# Restore the one Week 8 ablation checkpoint overwritten by the 2026-09-12
# resume bug, then prove that its frozen downstream chain is unchanged.

set -u
set -o pipefail

ALLOW_SHARED=0
if [[ $# -eq 2 && "$1" == "--allow-shared" ]]; then
  ALLOW_SHARED=1
  shift
fi
if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/repair_week8_ablation_resume.sh [--allow-shared] PHYSICAL_GPU_ID"
  exit 2
fi
if [[ ! -f configs/experiments/week8_evaluation.yaml ]]; then
  echo "Run this script from the ProActive repository root."
  exit 2
fi
if ! python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "CUDA-enabled PyTorch is unavailable in the active environment."
  exit 2
fi

GPU_ID="$1"
GPU_UUID=$(nvidia-smi -i "$GPU_ID" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
FREE_MIB=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
GPU_UTIL=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
GPU_JOBS=$(
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits |
    awk -F',' -v target="$GPU_UUID" '
      {gsub(/[[:space:]]/, "", $1); if ($1 == target) print $0}'
)
if [[ "$ALLOW_SHARED" -eq 1 ]]; then
  if [[ -z "$FREE_MIB" || "$FREE_MIB" -lt 8192 ]]; then
    echo "STOP: shared mode requires at least 8192 MiB free; observed ${FREE_MIB:-unknown} MiB."
    exit 2
  fi
  if [[ -z "$GPU_UTIL" || "$GPU_UTIL" -gt 10 ]]; then
    echo "STOP: shared mode requires launch utilization at or below 10%; observed ${GPU_UTIL:-unknown}%."
    exit 2
  fi
  echo "Shared mode approved for physical GPU $GPU_ID: ${FREE_MIB} MiB free, utilization ${GPU_UTIL}%."
  if [[ -n "$GPU_JOBS" ]]; then
    echo "Existing compute processes (left untouched):"
    echo "$GPU_JOBS"
  fi
else
  if [[ -n "$GPU_JOBS" ]]; then
    echo "STOP: physical GPU $GPU_ID already has compute processes:"
    echo "$GPU_JOBS"
    exit 2
  fi
  if [[ -z "$FREE_MIB" || "$FREE_MIB" -lt 2048 ]]; then
    echo "STOP: physical GPU $GPU_ID has only ${FREE_MIB:-unknown} MiB free."
    exit 2
  fi
  echo "Physical GPU $GPU_ID verified exclusive with $FREE_MIB MiB available."
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
ROOT=outputs/week8_ablations/no_budget_embedding
CHECKPOINT="$ROOT/diagnostic_checkpoints/diag_deep_sets_standard_seed42.best.pt"
REPORT="$ROOT/diagnostic_checkpoints/diag_deep_sets_standard_seed42.validation.json"
HISTORY="$ROOT/diagnostic_checkpoints/diag_deep_sets_standard_seed42.history.json"
LAST="$ROOT/diagnostic_checkpoints/diag_deep_sets_standard_seed42.last.pt"
APS="$ROOT/temporary_aps/temporary_aps_deep_sets_standard_seed42.json"
DIAG_FREEZE="$ROOT/freeze/week8_diagnostic_freeze.json"
VOI="$ROOT/voi/voi_manifest.json"
POLICY="$ROOT/policy_checkpoints/policy_deep_sets_standard_lambda0_seed42.best.pt"
STACK_FREEZE="$ROOT/freeze/week8_stack_freeze.json"
LOG_ROOT=outputs/logs/week8/ablations
LEDGER="$LOG_ROOT/gpu_time_ledger.tsv"
ARCHIVE_ROOT=outputs/week8_recovery_archive
mkdir -p "$LOG_ROOT" "$ARCHIVE_ROOT"

for required in "$APS" "$VOI" "$POLICY" "$STACK_FREEZE"; do
  if [[ ! -s "$required" ]]; then
    echo "STOP: required pre-incident evidence is missing: $required"
    exit 2
  fi
done

EXPECTED_CHECKPOINT_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$APS")
EXPECTED_FREEZE_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["freeze_manifest_sha256"])' "$VOI")
if [[ ! "$EXPECTED_CHECKPOINT_SHA" =~ ^[0-9a-f]{64}$ || ! "$EXPECTED_FREEZE_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "STOP: invalid pre-incident provenance hash."
  exit 2
fi
echo "Expected pre-incident diagnostic checkpoint: $EXPECTED_CHECKPOINT_SHA"
echo "Expected pre-incident diagnostic freeze:     $EXPECTED_FREEZE_SHA"

CURRENT_CHECKPOINT_SHA="missing"
if [[ -f "$CHECKPOINT" ]]; then
  CURRENT_CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
fi

if [[ "$CURRENT_CHECKPOINT_SHA" != "$EXPECTED_CHECKPOINT_SHA" ]]; then
  PRIOR_SECONDS=0
  if [[ -s "$LEDGER" ]]; then
    PRIOR_SECONDS=$(awk -F'\t' '{sum += $1} END {print int(sum)}' "$LEDGER")
  fi
  REMAINING_SECONDS=$((28800 - PRIOR_SECONDS))
  if (( REMAINING_SECONDS <= 60 )); then
    echo "STOP: less than 60 seconds remain under the approved eight GPU-hour ceiling."
    exit 2
  fi
  ALLOWED_SECONDS=2700
  if (( ALLOWED_SECONDS > REMAINING_SECONDS )); then ALLOWED_SECONDS=$REMAINING_SECONDS; fi

  ARCHIVE="$ARCHIVE_ROOT/no_budget_embedding_early_stop_drift_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$ARCHIVE"
  for artifact in "$CHECKPOINT" "$LAST" "$REPORT" "$HISTORY" "$DIAG_FREEZE" "$ROOT/freeze/predeclared_selection.json"; do
    if [[ -f "$artifact" ]]; then cp -p "$artifact" "$ARCHIVE/"; fi
  done
  echo "Archived the overwritten resume artifacts at $ARCHIVE"
  echo "Rebuilding the diagnostic run from epoch zero; timeout ${ALLOWED_SECONDS}s."

  STAGE_START=$SECONDS
  /usr/bin/time -v timeout --signal=TERM --kill-after=5m "${ALLOWED_SECONDS}s" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/train_diag.py \
      --config outputs/week8_data/ablation_configs/no_budget_embedding_diag.yaml \
      --manifest_path outputs/week5_data/vectorized_manifest.json \
      --encoder deep_sets --seed 42 --device cuda:0 \
      --output_dir "$ROOT/diagnostic_checkpoints" --overwrite \
      2>&1 | tee "$LOG_ROOT/no_budget_embedding_checkpoint_repair.log"
  REPAIR_RC=${PIPESTATUS[0]}
  REPAIR_SECONDS=$((SECONDS - STAGE_START))
  printf '%s\t%s\t%s\n' "$REPAIR_SECONDS" "$REPAIR_RC" \
    "no_budget_embedding deterministic checkpoint repair" >> "$LEDGER"
  if [[ "$REPAIR_RC" -ne 0 ]]; then
    echo "STOP: deterministic checkpoint repair failed with exit code $REPAIR_RC."
    exit "$REPAIR_RC"
  fi
fi

RESTORED_CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
if [[ "$RESTORED_CHECKPOINT_SHA" != "$EXPECTED_CHECKPOINT_SHA" ]]; then
  echo "STOP: rebuilt checkpoint does not reproduce the pre-incident checkpoint."
  echo "Expected: $EXPECTED_CHECKPOINT_SHA"
  echo "Observed: $RESTORED_CHECKPOINT_SHA"
  echo "Do not resume the ablation bundle; sync the repair log and recovery archive for review."
  exit 1
fi
echo "Checkpoint restored exactly: $RESTORED_CHECKPOINT_SHA"

# Prove the new resume guard recognizes this run as already complete.
python scripts/train_diag.py \
  --config outputs/week8_data/ablation_configs/no_budget_embedding_diag.yaml \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --encoder deep_sets --seed 42 --device cpu \
  --output_dir "$ROOT/diagnostic_checkpoints" --resume || exit $?

# Recreate only the two small metadata files overwritten during the failed
# resume, then require their byte-level hash to match the original VOI binding.
python scripts/freeze_week8_diagnostic.py \
  --week8_config configs/experiments/week8_evaluation.yaml \
  --ablation_id no_budget_embedding \
  --diag_config outputs/week8_data/ablation_configs/no_budget_embedding_diag.yaml \
  --policy_config outputs/week8_data/ablation_configs/no_budget_embedding_policy.yaml \
  --vector_manifest outputs/week5_data/vectorized_manifest.json \
  --diagnostic_checkpoint "$CHECKPOINT" \
  --output_dir "$ROOT/freeze" --approve_freeze --overwrite || exit $?

RESTORED_FREEZE_SHA=$(sha256sum "$DIAG_FREEZE" | awk '{print $1}')
if [[ "$RESTORED_FREEZE_SHA" != "$EXPECTED_FREEZE_SHA" ]]; then
  echo "STOP: restored diagnostic freeze does not match the pre-incident VOI binding."
  echo "Expected: $EXPECTED_FREEZE_SHA"
  echo "Observed: $RESTORED_FREEZE_SHA"
  exit 1
fi
echo "Diagnostic freeze restored exactly: $RESTORED_FREEZE_SHA"

# These --resume calls are CPU-only integrity checks. They must return without
# recomputation if the original APS, VOI, policy and frontier remain compatible.
python scripts/fit_temporary_aps.py \
  --config outputs/week8_data/ablation_configs/no_budget_embedding_diag.yaml \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --checkpoint "$CHECKPOINT" --output_dir "$ROOT/temporary_aps" \
  --seed 42 --device cpu --resume || exit $?

python scripts/build_voi_targets.py \
  --config outputs/week8_data/ablation_configs/no_budget_embedding_policy.yaml \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --state_manifest outputs/week4_reports/final/state_manifest.json \
  --checkpoint "$CHECKPOINT" --freeze_manifest "$DIAG_FREEZE" \
  --temporary_aps "$APS" --teacher_path outputs/teacher_core_contract_v1_recovered \
  --output_dir "$ROOT/voi" --seed 42 --inference_batch_size 512 \
  --device cpu --resume || exit $?

python scripts/train_policy.py \
  --config outputs/week8_data/ablation_configs/no_budget_embedding_policy.yaml \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --voi_manifest "$VOI" --diagnostic_checkpoint "$CHECKPOINT" \
  --freeze_manifest "$DIAG_FREEZE" --cost_multiplier 0.0 --target_kind voi \
  --seed 42 --device cpu --output_dir "$ROOT/policy_checkpoints" --resume || exit $?

python scripts/eval_week8_ablation.py \
  --week8_config configs/experiments/week8_evaluation.yaml \
  --ablation_id no_budget_embedding \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --diagnostic_checkpoint "$CHECKPOINT" --policy_checkpoint "$POLICY" \
  --aps_report "$APS" --freeze_manifest "$STACK_FREEZE" \
  --teacher_path outputs/teacher_core_contract_v1_recovered \
  --output_dir "$ROOT/frontier" --budgets 1 2 3 4 7 \
  --coverages 0.90 0.95 --seed 42 --device cpu --resume || exit $?

echo "WEEK8_NO_BUDGET_RESUME_REPAIR_ALL_OK=1"
echo "It is now safe to resume scripts/run_week8_ablations.sh."
