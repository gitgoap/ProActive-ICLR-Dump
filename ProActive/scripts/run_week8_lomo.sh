#!/usr/bin/env bash
# Run the approved two-fold Week 8 LOMO experiment on one physical GPU.
# Both source-only stacks are frozen and calibrated before either held-out test
# fold is evaluated. Every expensive stage is resumable and timeout bounded.

set -u
set -o pipefail

GPU_ID="${1:-}"
if [[ -z "$GPU_ID" || ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/run_week8_lomo.sh PHYSICAL_GPU_ID"
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

FREE_MIB=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if [[ -z "$FREE_MIB" || "$FREE_MIB" -lt 2048 ]]; then
  echo "STOP: physical GPU $GPU_ID has only ${FREE_MIB:-unknown} MiB free; at least 2048 MiB is required."
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

LOG_ROOT=outputs/logs/week8/lomo_full
REPORT_ROOT=outputs/week8_reports/lomo
BUNDLE_START=$SECONDS
BUNDLE_LIMIT_SECONDS=18000
mkdir -p "$LOG_ROOT" "$REPORT_ROOT"

run_stage() {
  local timeout_value="$1"
  local label="$2"
  local logfile="$3"
  local requested_seconds
  local remaining_seconds
  local stage_limit_seconds
  shift 3

  case "$timeout_value" in
    *m) requested_seconds=$((10#${timeout_value%m} * 60)) ;;
    *h) requested_seconds=$((10#${timeout_value%h} * 3600)) ;;
    *s) requested_seconds=$((10#${timeout_value%s})) ;;
    *) echo "Unsupported timeout value: $timeout_value"; return 2 ;;
  esac
  remaining_seconds=$((BUNDLE_LIMIT_SECONDS - (SECONDS - BUNDLE_START)))
  if (( remaining_seconds <= 0 )); then
    echo "STOP: the approved five-hour one-GPU bundle ceiling was reached before $label."
    return 124
  fi
  if (( requested_seconds < remaining_seconds )); then
    stage_limit_seconds=$requested_seconds
  else
    stage_limit_seconds=$remaining_seconds
  fi

  echo "===== $label =====" | tee "$logfile"
  echo "Timeout: ${stage_limit_seconds}s; bundle remaining: ${remaining_seconds}s" | tee -a "$logfile"
  /usr/bin/time -v timeout --signal=TERM --kill-after=5m "${stage_limit_seconds}s" "$@" \
    2>&1 | tee -a "$logfile"
  local rc=${PIPESTATUS[0]}
  echo "$label exit code: $rc" | tee -a "$logfile"
  return "$rc"
}

prepare_fold() {
  local model="$1"
  local root="outputs/week8_lomo/$model"
  local logs="$LOG_ROOT/$model"
  local vector="$root/vectorized/vectorized_manifest.json"
  local fold="$root/lomo_fold_manifest.json"
  local states="$root/states"
  local teachers="$root/teachers"
  local state_manifest="$root/state_manifest.json"
  local diag_dir="$root/checkpoints/diagnostic"
  local policy_dir="$root/checkpoints/policy"
  local baseline_dir="$root/checkpoints/baselines"
  local freeze_dir="$root/freeze"
  local temp_dir="$root/temporary_aps"
  local voi_dir="$root/voi"
  local trajectory_dir="$root/calibration/trajectories"
  local aps_dir="$root/calibration/aps"
  local static_dir="$root/calibration/static_aps"
  local diag="$diag_dir/diag_deep_sets_standard_seed42.best.pt"
  local clean="$diag_dir/diag_clean_mlp_standard_seed42.best.pt"
  local diag_freeze="$freeze_dir/lomo_diagnostic_freeze.json"
  local temp_aps="$temp_dir/temporary_aps_deep_sets_standard_seed42.json"
  local policy="$policy_dir/policy_deep_sets_standard_lambda0_seed42.best.pt"
  local scalar="$baseline_dir/baseline_scalar_seed42.best.pt"
  local stack_freeze="$freeze_dir/lomo_stack_freeze.json"

  mkdir -p "$logs" "$diag_dir" "$policy_dir" "$baseline_dir" "$freeze_dir" \
    "$temp_dir" "$voi_dir" "$trajectory_dir" "$aps_dir" "$static_dir"

  if [[ ! -s "$fold" || ! -s "$vector" ]]; then
    echo "Missing complete LOMO fold inputs for $model"
    return 2
  fi

  run_stage 10m "$model state-manifest index" "$logs/01_state_manifest.log" \
    python scripts/index_state_manifest.py \
      --state_dir "$states" \
      --output_path "$state_manifest" \
      --expected_teacher_rows 7291 \
      --resume || return $?

  run_stage 12m "$model Deep Sets diagnostic" "$logs/02_deep_sets.log" \
    python scripts/train_diag.py \
      --config configs/experiments/diag_bakeoff.yaml \
      --manifest_path "$vector" \
      --encoder deep_sets \
      --seed 42 \
      --device cuda:0 \
      --output_dir "$diag_dir" \
      --resume || return $?

  run_stage 5m "$model clean diagnostic" "$logs/03_clean_mlp.log" \
    python scripts/train_diag.py \
      --config configs/experiments/diag_bakeoff.yaml \
      --manifest_path "$vector" \
      --encoder clean_mlp \
      --seed 42 \
      --device cuda:0 \
      --output_dir "$diag_dir" \
      --resume || return $?

  run_stage 5m "$model diagnostic freeze" "$logs/04_diagnostic_freeze.log" \
    python scripts/freeze_lomo_stack.py \
      --stage diagnostic \
      --fold_manifest "$fold" \
      --diagnostic_checkpoint "$diag" \
      --clean_checkpoint "$clean" \
      --output_dir "$freeze_dir" \
      --approve_freeze \
      --overwrite || return $?

  run_stage 3m "$model temporary validation APS" "$logs/05_temporary_aps.log" \
    python scripts/fit_temporary_aps.py \
      --config configs/experiments/diag_bakeoff.yaml \
      --manifest_path "$vector" \
      --checkpoint "$diag" \
      --output_dir "$temp_dir" \
      --seed 42 \
      --device cuda:0 \
      --resume || return $?

  run_stage 18m "$model VOI targets" "$logs/06_voi_targets.log" \
    python scripts/build_voi_targets.py \
      --config configs/experiments/policy_train.yaml \
      --manifest_path "$vector" \
      --state_manifest "$state_manifest" \
      --state_dir "$states" \
      --checkpoint "$diag" \
      --freeze_manifest "$diag_freeze" \
      --temporary_aps "$temp_aps" \
      --teacher_path "$teachers" \
      --output_dir "$voi_dir" \
      --seed 42 \
      --inference_batch_size 512 \
      --device cuda:0 \
      --resume || return $?

  run_stage 75m "$model VOI policy" "$logs/07_policy.log" \
    python scripts/train_policy.py \
      --config configs/experiments/policy_train.yaml \
      --manifest_path "$vector" \
      --voi_manifest "$voi_dir/voi_manifest.json" \
      --diagnostic_checkpoint "$diag" \
      --freeze_manifest "$diag_freeze" \
      --cost_multiplier 0.0 \
      --target_kind voi \
      --seed 42 \
      --device cuda:0 \
      --output_dir "$policy_dir" \
      --resume || return $?

  run_stage 5m "$model scalar baseline" "$logs/08_scalar.log" \
    python scripts/train_one_pass_baselines.py \
      --config configs/experiments/policy_train.yaml \
      --manifest_path "$vector" \
      --diagnostic_checkpoint "$diag" \
      --freeze_manifest "$diag_freeze" \
      --baseline scalar \
      --seed 42 \
      --device cuda:0 \
      --output_dir "$baseline_dir" \
      --resume || return $?

  run_stage 5m "$model complete-stack freeze" "$logs/09_stack_freeze.log" \
    python scripts/freeze_lomo_stack.py \
      --stage stack \
      --fold_manifest "$fold" \
      --diagnostic_checkpoint "$diag" \
      --clean_checkpoint "$clean" \
      --policy_checkpoint "$policy" \
      --scalar_checkpoint "$scalar" \
      --output_dir "$freeze_dir" \
      --approve_freeze \
      --overwrite || return $?

  run_stage 5m "$model source calibration trajectories" "$logs/10_calibration_trajectories.log" \
    python scripts/build_policy_trajectories.py \
      --config configs/experiments/calibrate_aps.yaml \
      --manifest_path "$vector" \
      --freeze_manifest "$stack_freeze" \
      --teacher_path "$teachers" \
      --output_dir "$trajectory_dir" \
      --seed 42 \
      --device cuda:0 \
      --resume || return $?

  run_stage 5m "$model source APS calibration" "$logs/11_aps.log" \
    python scripts/calibrate_aps.py \
      --config configs/experiments/calibrate_aps.yaml \
      --manifest_path "$trajectory_dir/calibration_trajectory_manifest.json" \
      --freeze_manifest "$stack_freeze" \
      --output_dir "$aps_dir" \
      --seed 42 \
      --device cpu \
      --resume || return $?

  run_stage 5m "$model clean source APS" "$logs/12_clean_aps.log" \
    python scripts/calibrate_static_baseline_aps.py \
      --approval_config configs/experiments/calibrate_aps.yaml \
      --manifest_path "$vector" \
      --checkpoint "$clean" \
      --phase calibration \
      --freeze_manifest "$stack_freeze" \
      --budgets 1 2 3 4 7 \
      --coverages 0.90 0.95 \
      --output_dir "$static_dir" \
      --seed 42 \
      --device cpu \
      --resume || return $?

  run_stage 5m "$model scalar source APS" "$logs/13_scalar_aps.log" \
    python scripts/calibrate_static_baseline_aps.py \
      --approval_config configs/experiments/calibrate_aps.yaml \
      --manifest_path "$vector" \
      --checkpoint "$scalar" \
      --phase calibration \
      --freeze_manifest "$stack_freeze" \
      --budgets 1 2 3 4 7 \
      --coverages 0.90 0.95 \
      --output_dir "$static_dir" \
      --seed 42 \
      --device cpu \
      --resume || return $?

  echo "$model source-only stack is frozen and calibrated; held-out test is still unopened."
}

evaluate_fold() {
  local model="$1"
  local root="outputs/week8_lomo/$model"
  local logs="$LOG_ROOT/$model"
  local report="$REPORT_ROOT/$model"
  local freeze="$root/freeze/lomo_stack_freeze.json"

  mkdir -p "$report"
  run_stage 12m "$model held-out evaluation" "$logs/14_heldout_evaluation.log" \
    python scripts/eval_lomo.py \
      --fold_manifest "$root/lomo_fold_manifest.json" \
      --freeze_manifest "$freeze" \
      --aps_report "$root/calibration/aps/aps_thresholds.json" \
      --clean_aps_report "$root/calibration/static_aps/static_aps_clean_only_learned_calibration.json" \
      --scalar_aps_report "$root/calibration/static_aps/static_aps_scalar_confidence_calibration.json" \
      --teacher_path "$root/teachers" \
      --output_dir "$report" \
      --seed 42 \
      --device cuda:0 \
      --resume
}

echo "Physical GPU $GPU_ID has $FREE_MIB MiB free."
echo "Preparing both source-only folds before any held-out test access."

PREPARATION_OK=1
prepare_fold qwen3_vl_8b || PREPARATION_OK=0
if [[ "$PREPARATION_OK" -eq 1 ]]; then
  prepare_fold gemma4_e4b || PREPARATION_OK=0
fi

EVALUATION_OK=0
if [[ "$PREPARATION_OK" -eq 1 ]]; then
  echo "Both folds are frozen and source-calibrated. Beginning locked held-out evaluation."
  EVALUATION_OK=1
  evaluate_fold qwen3_vl_8b || EVALUATION_OK=0
  if [[ "$EVALUATION_OK" -eq 1 ]]; then
    evaluate_fold gemma4_e4b || EVALUATION_OK=0
  fi
else
  echo "Held-out evaluation skipped because source-only preparation did not complete."
fi

ELAPSED_SECONDS=$((SECONDS - BUNDLE_START))
echo "LOMO_PREPARATION_ALL_OK=$PREPARATION_OK"
echo "LOMO_EVALUATION_ALL_OK=$EVALUATION_OK"
if [[ "$PREPARATION_OK" -eq 1 && "$EVALUATION_OK" -eq 1 ]]; then
  echo "LOMO_FULL_ALL_OK=1"
else
  echo "LOMO_FULL_ALL_OK=0"
fi
echo "LOMO bundle wall seconds: $ELAPSED_SECONDS"
