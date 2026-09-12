#!/usr/bin/env bash
# Run the owner-approved, validation-only Week 8 mandatory ablation bundle.

set -u
set -o pipefail

ALLOW_SHARED=0
if [[ $# -ge 1 && "$1" == "--allow-shared" ]]; then
  ALLOW_SHARED=1
  shift
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_week8_ablations.sh auto"
  echo "   or: bash scripts/run_week8_ablations.sh [--allow-shared] PHYSICAL_GPU_ID [PHYSICAL_GPU_ID]"
  exit 2
fi
if [[ ! -f configs/experiments/week8_evaluation.yaml ]]; then
  echo "Run this script from the ProActive repository root."
  exit 2
fi

GPUS=()
if [[ $# -eq 1 && "$1" == "auto" ]]; then
  while IFS= read -r candidate; do
    candidate=$(echo "$candidate" | tr -d '[:space:]')
    [[ -z "$candidate" ]] && continue
    candidate_uuid=$(nvidia-smi -i "$candidate" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
    candidate_free=$(nvidia-smi -i "$candidate" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
    candidate_jobs=$(
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits |
        awk -F',' -v target="$candidate_uuid" '
          {gsub(/[[:space:]]/, "", $1); if ($1 == target) count += 1}
          END {print count + 0}'
    )
    if [[ "$candidate_jobs" -eq 0 && -n "$candidate_free" && "$candidate_free" -ge 2048 ]]; then
      GPUS+=("$candidate")
      [[ "${#GPUS[@]}" -eq 2 ]] && break
    fi
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "STOP: no physical GPU is free with at least 2048 MiB available."
    exit 2
  fi
  echo "Auto-selected physical GPU(s): ${GPUS[*]}"
else
  GPUS=("$@")
fi
if [[ "${#GPUS[@]}" -eq 2 && "${GPUS[0]}" == "${GPUS[1]}" ]]; then
  echo "The two physical GPU IDs must be distinct."
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid physical GPU ID: $gpu"
    exit 2
  fi
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
  free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
  gpu_util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
  jobs=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits |
    awk -F',' -v target="$uuid" '{gsub(/ /,"",$1); if ($1==target) print $0}')
  if [[ "$ALLOW_SHARED" -eq 1 ]]; then
    if [[ -z "$free_mib" || "$free_mib" -lt 8192 ]]; then
      echo "STOP: shared mode requires at least 8192 MiB free on GPU $gpu; observed ${free_mib:-unknown} MiB."
      exit 2
    fi
    if [[ -z "$gpu_util" || "$gpu_util" -gt 10 ]]; then
      echo "STOP: shared mode requires launch utilization at or below 10% on GPU $gpu; observed ${gpu_util:-unknown}%."
      exit 2
    fi
    echo "Physical GPU $gpu accepted in shared mode: $free_mib MiB free, utilization $gpu_util%."
    if [[ -n "$jobs" ]]; then
      echo "Existing compute processes on GPU $gpu are left untouched:"
      echo "$jobs"
    fi
  else
    if [[ -n "$jobs" ]]; then
      echo "STOP: physical GPU $gpu is not free at launch:"
      echo "$jobs"
      exit 2
    fi
    if [[ -z "$free_mib" || "$free_mib" -lt 2048 ]]; then
      echo "STOP: physical GPU $gpu has only ${free_mib:-unknown} MiB free."
      exit 2
    fi
    echo "Physical GPU $gpu verified free with $free_mib MiB available."
  fi
done

if ! python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "CUDA-enabled PyTorch is unavailable in the active environment."
  exit 2
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
ROOT=outputs/week8_ablations
CONFIG_ROOT=outputs/week8_data/ablation_configs
LOG_ROOT=outputs/logs/week8/ablations
REPORT_ROOT=outputs/week8_reports/ablation_evidence
mkdir -p "$ROOT" "$CONFIG_ROOT" "$LOG_ROOT" "$REPORT_ROOT"

BUNDLE_START=$SECONDS
GPU_COUNT=${#GPUS[@]}
USAGE_LEDGER="$LOG_ROOT/gpu_time_ledger.tsv"
PRIOR_GPU_SECONDS=0
if [[ -s "$USAGE_LEDGER" ]]; then
  PRIOR_GPU_SECONDS=$(awk -F'\t' '{sum += $1} END {print int(sum)}' "$USAGE_LEDGER")
fi
REMAINING_APPROVED_SECONDS=$((28800 - PRIOR_GPU_SECONDS))
if (( REMAINING_APPROVED_SECONDS <= 0 )); then
  echo "STOP: the cumulative eight GPU-hour approval is already exhausted."
  exit 2
fi
PER_WORKER_SECONDS=$((REMAINING_APPROVED_SECONDS / GPU_COUNT))
echo "Previously recorded conservative GPU seconds: $PRIOR_GPU_SECONDS"
echo "Approved GPU seconds remaining: $REMAINING_APPROVED_SECONDS"

run_stage() {
  local limit="$1"
  local worker_start="$2"
  local label="$3"
  local logfile="$4"
  shift 4
  local requested=$((10#${limit%m} * 60))
  local remaining=$((PER_WORKER_SECONDS - (SECONDS - worker_start)))
  local allowed="$requested"
  local stage_start=$SECONDS
  if (( remaining <= 0 )); then
    echo "STOP: approved per-worker share of the 8 GPU-hour ceiling reached before $label" | tee "$logfile"
    return 124
  fi
  if (( allowed > remaining )); then allowed="$remaining"; fi
  echo "===== $label =====" | tee "$logfile"
  echo "Timeout ${allowed}s; worker ceiling remaining ${remaining}s" | tee -a "$logfile"
  /usr/bin/time -v timeout --signal=TERM --kill-after=5m "${allowed}s" "$@" \
    2>&1 | tee -a "$logfile"
  local rc=${PIPESTATUS[0]}
  local elapsed=$((SECONDS - stage_start))
  printf '%s\t%s\t%s\n' "$elapsed" "$rc" "$label" >> "$USAGE_LEDGER"
  echo "$label exit code: $rc" | tee -a "$logfile"
  return "$rc"
}

echo "Materializing approval-bound configs and feature manifests."
python scripts/materialize_week8_ablation_configs.py \
  --week8_config configs/experiments/week8_evaluation.yaml \
  --output_dir "$CONFIG_ROOT" \
  --overwrite || exit $?

for ablation in no_answer_flip no_confidence_shift no_semantic_match no_relation_probe; do
  python scripts/build_feature_ablation_manifest.py \
    --manifest_path outputs/week5_data/vectorized_manifest.json \
    --ablation "$ablation" \
    --output_dir "$ROOT/$ablation/vectorized" \
    --resume || exit $?
done

run_trained_ablation() {
  local ablation="$1"
  local gpu="$2"
  local worker_start="$3"
  local root="$ROOT/$ablation"
  local logs="$LOG_ROOT/$ablation"
  local diag_config="$CONFIG_ROOT/${ablation}_diag.yaml"
  local policy_config="$CONFIG_ROOT/${ablation}_policy.yaml"
  local vector=outputs/week5_data/vectorized_manifest.json
  if [[ "$ablation" =~ ^no_(answer_flip|confidence_shift|semantic_match|relation_probe)$ ]]; then
    vector="$root/vectorized/vectorized_manifest.json"
  fi
  local diag="$root/diagnostic_checkpoints/diag_deep_sets_standard_seed42.best.pt"
  local temp="$root/temporary_aps/temporary_aps_deep_sets_standard_seed42.json"
  local diag_freeze="$root/freeze/week8_diagnostic_freeze.json"
  local policy="$root/policy_checkpoints/policy_deep_sets_standard_lambda0_seed42.best.pt"
  local stack_freeze="$root/freeze/week8_stack_freeze.json"
  mkdir -p "$logs" "$root/diagnostic_checkpoints" "$root/temporary_aps" \
    "$root/freeze" "$root/voi" "$root/policy_checkpoints" "$root/frontier"

  run_stage 45m "$worker_start" "$ablation diagnostic training" "$logs/01_diagnostic.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_diag.py \
      --config "$diag_config" --manifest_path "$vector" --encoder deep_sets \
      --seed 42 --device cuda:0 --output_dir "$root/diagnostic_checkpoints" --resume || return $?
  run_stage 10m "$worker_start" "$ablation diagnostic freeze" "$logs/02_diagnostic_freeze.log" \
    python scripts/freeze_week8_diagnostic.py \
      --week8_config configs/experiments/week8_evaluation.yaml --ablation_id "$ablation" \
      --diag_config "$diag_config" --policy_config "$policy_config" \
      --vector_manifest "$vector" --diagnostic_checkpoint "$diag" \
      --output_dir "$root/freeze" --approve_freeze --overwrite || return $?
  run_stage 15m "$worker_start" "$ablation temporary validation APS" "$logs/03_temporary_aps.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" python scripts/fit_temporary_aps.py \
      --config "$diag_config" --manifest_path "$vector" --checkpoint "$diag" \
      --output_dir "$root/temporary_aps" --seed 42 --device cuda:0 --resume || return $?
  run_stage 45m "$worker_start" "$ablation VOI targets" "$logs/04_voi.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" python scripts/build_voi_targets.py \
      --config "$policy_config" --manifest_path "$vector" \
      --state_manifest outputs/week4_reports/final/state_manifest.json \
      --checkpoint "$diag" --freeze_manifest "$diag_freeze" --temporary_aps "$temp" \
      --teacher_path outputs/teacher_core_contract_v1_recovered --output_dir "$root/voi" \
      --seed 42 --inference_batch_size 512 --device cuda:0 --resume || return $?
  run_stage 45m "$worker_start" "$ablation policy training" "$logs/05_policy.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_policy.py \
      --config "$policy_config" --manifest_path "$vector" --voi_manifest "$root/voi/voi_manifest.json" \
      --diagnostic_checkpoint "$diag" --freeze_manifest "$diag_freeze" \
      --cost_multiplier 0.0 --target_kind voi --seed 42 --device cuda:0 \
      --output_dir "$root/policy_checkpoints" --resume || return $?
  run_stage 10m "$worker_start" "$ablation stack freeze" "$logs/06_stack_freeze.log" \
    python scripts/freeze_week8_ablation_stack.py \
      --week8_config configs/experiments/week8_evaluation.yaml --ablation_id "$ablation" \
      --diag_config "$diag_config" --policy_config "$policy_config" --vector_manifest "$vector" \
      --diagnostic_checkpoint "$diag" --policy_checkpoint "$policy" \
      --output_dir "$root/freeze" --approve_freeze --overwrite || return $?
  run_stage 20m "$worker_start" "$ablation validation frontier" "$logs/07_frontier.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_week8_ablation.py \
      --week8_config configs/experiments/week8_evaluation.yaml --ablation_id "$ablation" \
      --manifest_path "$vector" --diagnostic_checkpoint "$diag" --policy_checkpoint "$policy" \
      --aps_report "$temp" --freeze_manifest "$stack_freeze" \
      --teacher_path outputs/teacher_core_contract_v1_recovered --output_dir "$root/frontier" \
      --budgets 1 2 3 4 7 --coverages 0.90 0.95 --seed 42 --device cuda:0 --resume || return $?
}

run_loss_only() {
  local gpu="$1"
  local worker_start="$2"
  local root="$ROOT/loss_only_voi"
  local logs="$LOG_ROOT/loss_only_voi"
  local diag=outputs/week5_checkpoints/diag_deep_sets_standard_seed42.best.pt
  local policy="$root/policy_checkpoints/loss_only_voi_deep_sets_standard_lambda0_seed42.best.pt"
  mkdir -p "$logs" "$root/policy_checkpoints" "$root/freeze" "$root/frontier"
  run_stage 45m "$worker_start" "loss-only VOI policy training" "$logs/01_policy.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_policy.py \
      --config configs/experiments/policy_train.yaml \
      --manifest_path outputs/week5_data/vectorized_manifest.json \
      --voi_manifest outputs/week6_voi/voi_manifest.json --diagnostic_checkpoint "$diag" \
      --freeze_manifest outputs/week5_reports/week5_encoder_freeze.json \
      --cost_multiplier 0.0 --target_kind loss_only_voi --seed 42 --device cuda:0 \
      --output_dir "$root/policy_checkpoints" --resume || return $?
  run_stage 10m "$worker_start" "loss-only VOI stack freeze" "$logs/02_stack_freeze.log" \
    python scripts/freeze_week8_ablation_stack.py \
      --week8_config configs/experiments/week8_evaluation.yaml --ablation_id loss_only_voi \
      --diag_config configs/experiments/diag_bakeoff.yaml \
      --policy_config configs/experiments/policy_train.yaml \
      --vector_manifest outputs/week5_data/vectorized_manifest.json \
      --diagnostic_checkpoint "$diag" --policy_checkpoint "$policy" \
      --output_dir "$root/freeze" --approve_freeze --overwrite || return $?
  run_stage 20m "$worker_start" "loss-only VOI validation frontier" "$logs/03_frontier.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_week8_ablation.py \
      --week8_config configs/experiments/week8_evaluation.yaml --ablation_id loss_only_voi \
      --manifest_path outputs/week5_data/vectorized_manifest.json \
      --diagnostic_checkpoint "$diag" --policy_checkpoint "$policy" \
      --aps_report outputs/week5_reports/temporary_aps_deep_sets_standard_seed42.json \
      --freeze_manifest "$root/freeze/week8_stack_freeze.json" \
      --teacher_path outputs/teacher_core_contract_v1_recovered --output_dir "$root/frontier" \
      --budgets 1 2 3 4 7 --coverages 0.90 0.95 --seed 42 --device cuda:0 --resume || return $?
}

worker() {
  local worker_index="$1"
  local gpu="$2"
  local worker_start=$SECONDS
  shift 2
  for ablation in "$@"; do
    if [[ "$ablation" == "loss_only_voi" ]]; then
      run_loss_only "$gpu" "$worker_start" || return $?
    else
      run_trained_ablation "$ablation" "$gpu" "$worker_start" || return $?
    fi
  done
  echo "WORKER_${worker_index}_ALL_OK=1"
}

ALL=(no_budget_embedding no_answer_flip no_confidence_shift no_semantic_match no_relation_probe no_signature_regression shared_vs_independent_heads loss_only_voi)
QUEUE0=()
QUEUE1=()
for index in "${!ALL[@]}"; do
  if [[ "$GPU_COUNT" -eq 1 || $((index % 2)) -eq 0 ]]; then
    QUEUE0+=("${ALL[$index]}")
  else
    QUEUE1+=("${ALL[$index]}")
  fi
done

worker 0 "${GPUS[0]}" "${QUEUE0[@]}" >"$LOG_ROOT/worker0.log" 2>&1 &
PID0=$!
PID1=""
if [[ "$GPU_COUNT" -eq 2 ]]; then
  worker 1 "${GPUS[1]}" "${QUEUE1[@]}" >"$LOG_ROOT/worker1.log" 2>&1 &
  PID1=$!
fi
wait "$PID0"; RC0=$?
RC1=0
if [[ -n "$PID1" ]]; then wait "$PID1"; RC1=$?; fi
if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then
  echo "A training worker failed: worker0=$RC0 worker1=$RC1"
  echo "Inspect $LOG_ROOT/worker0.log and worker1.log. The usage ledger preserves the cumulative approval before a resume."
  exit 1
fi

REFERENCE="$ROOT/reference"
mkdir -p "$REFERENCE" "$ROOT/no_voi_training/frontier" "$LOG_ROOT/reference" "$LOG_ROOT/no_voi_training"
run_reference_frontier() {
  run_stage 20m "$BUNDLE_START" "reference validation frontier" "$LOG_ROOT/reference/frontier.log" \
    env CUDA_VISIBLE_DEVICES="${GPUS[0]}" python scripts/eval_week8_ablation.py \
    --week8_config configs/experiments/week8_evaluation.yaml --ablation_id reference \
    --manifest_path outputs/week5_data/vectorized_manifest.json \
    --diagnostic_checkpoint outputs/week5_checkpoints/diag_deep_sets_standard_seed42.best.pt \
    --policy_checkpoint outputs/week6_checkpoints/policy_deep_sets_standard_lambda0_seed42.best.pt \
    --aps_report outputs/week5_reports/temporary_aps_deep_sets_standard_seed42.json \
    --freeze_manifest outputs/week7_frozen/main_stack_freeze.json \
    --teacher_path outputs/teacher_core_contract_v1_recovered --output_dir "$REFERENCE" \
    --budgets 1 2 3 4 7 --coverages 0.90 0.95 --seed 42 --device cuda:0 --resume
}

run_no_voi_frontier() {
  local last_gpu_index=$((GPU_COUNT - 1))
  run_stage 20m "$BUNDLE_START" "no-VOI validation frontier" "$LOG_ROOT/no_voi_training/frontier.log" \
    env CUDA_VISIBLE_DEVICES="${GPUS[$last_gpu_index]}" python scripts/eval_week8_ablation.py \
    --week8_config configs/experiments/week8_evaluation.yaml --ablation_id no_voi_training \
    --manifest_path outputs/week5_data/vectorized_manifest.json \
    --diagnostic_checkpoint outputs/week5_checkpoints/diag_deep_sets_standard_seed42.best.pt \
    --policy_checkpoint outputs/week6_checkpoints/uncertainty_deep_sets_standard_lambda0_seed42.best.pt \
    --aps_report outputs/week5_reports/temporary_aps_deep_sets_standard_seed42.json \
    --freeze_manifest outputs/week7_frozen/main_stack_freeze.json \
    --teacher_path outputs/teacher_core_contract_v1_recovered --output_dir "$ROOT/no_voi_training/frontier" \
    --budgets 1 2 3 4 7 --coverages 0.90 0.95 --seed 42 --device cuda:0 --resume
}

if [[ "$GPU_COUNT" -eq 2 ]]; then
  run_reference_frontier & REF_PID=$!
  run_no_voi_frontier & NOVOI_PID=$!
  wait "$REF_PID"; REF_RC=$?
  wait "$NOVOI_PID"; NOVOI_RC=$?
else
  run_reference_frontier; REF_RC=$?
  run_no_voi_frontier; NOVOI_RC=$?
fi
if [[ "$REF_RC" -ne 0 || "$NOVOI_RC" -ne 0 ]]; then
  echo "Reference/no-VOI frontier failed: reference=$REF_RC no_voi=$NOVOI_RC"
  exit 1
fi

run_stage 20m "$BUNDLE_START" "validation latency measurement" "$LOG_ROOT/reference/latency.log" \
  env CUDA_VISIBLE_DEVICES="${GPUS[0]}" python scripts/measure_latency.py \
    --config configs/experiments/week8_evaluation.yaml --split val \
    --trajectory_path "$REFERENCE/trajectories_validation.jsonl" \
    --teacher_path outputs/teacher_core_contract_v1_recovered \
    --freeze_manifest outputs/week7_frozen/main_stack_freeze.json \
    --vector_manifest outputs/week5_data/vectorized_manifest.json \
    --output_dir "$ROOT/pass_count_vs_latency" --condition proactive --budget 7 \
    --target_coverage 0.90 --device cuda:0 --overwrite || exit $?

compare() {
  local id="$1"; local ref="$2"; local cmp="$3"; local ref_condition="${4:-proactive}"; local cmp_condition="${5:-proactive}"
  python scripts/compare_ablation_frontiers.py \
    --ablation_id "$id" --reference_csv "$ref" --comparison_csv "$cmp" \
    --reference_condition "$ref_condition" --comparison_condition "$cmp_condition" \
    --evaluation_split val --output_path "$REPORT_ROOT/$id.json" --overwrite
}

compare deep_sets_vs_gru \
  outputs/week6_frontiers/deep_lambda0_seed42/frontier_validation.csv \
  outputs/week6_architecture_frontiers/gru_random/frontier_validation.csv
compare deep_sets_vs_masked_slot_mlp \
  outputs/week6_frontiers/deep_lambda0_seed42/frontier_validation.csv \
  outputs/week6_architecture_frontiers/masked_slot/frontier_validation.csv
compare no_stop_action "$REFERENCE/frontier_validation.csv" "$REFERENCE/frontier_validation.csv" proactive proactive_no_stop
compare no_voi_training "$REFERENCE/frontier_validation.csv" "$ROOT/no_voi_training/frontier/frontier_validation.csv"
compare global_vs_per_budget_aps "$REFERENCE/frontier_validation.csv" "$REFERENCE/frontier_validation.csv" proactive proactive_global_aps
for id in no_budget_embedding loss_only_voi no_answer_flip no_confidence_shift no_semantic_match no_relation_probe no_signature_regression shared_vs_independent_heads; do
  compare "$id" "$REFERENCE/frontier_validation.csv" "$ROOT/$id/frontier/frontier_validation.csv"
done

python scripts/build_week8_auxiliary_ablation_evidence.py \
  --ablation_id pass_count_vs_latency \
  --reference_report "$ROOT/pass_count_vs_latency/latency_report.json" \
  --output_path "$REPORT_ROOT/pass_count_vs_latency.json" --overwrite
python scripts/build_week8_auxiliary_ablation_evidence.py \
  --ablation_id gru_canonical_vs_permutation_augmentation \
  --reference_report outputs/week5_reports/permutation_week5_gru_canonical_seed42.json \
  --comparison_report outputs/week5_reports/permutation_week5_gru_random_permutation_seed42.json \
  --output_path "$REPORT_ROOT/gru_canonical_vs_permutation_augmentation.json" --overwrite

EVIDENCE_ARGS=()
for id in deep_sets_vs_gru deep_sets_vs_masked_slot_mlp no_budget_embedding no_stop_action no_voi_training loss_only_voi no_answer_flip no_confidence_shift no_semantic_match no_relation_probe no_signature_regression shared_vs_independent_heads global_vs_per_budget_aps pass_count_vs_latency gru_canonical_vs_permutation_augmentation; do
  EVIDENCE_ARGS+=(--evidence "$id=$REPORT_ROOT/$id.json")
done
python scripts/aggregate_week8_ablations.py \
  --config configs/experiments/week8_evaluation.yaml "${EVIDENCE_ARGS[@]}" \
  --output_dir outputs/week8_reports --overwrite || exit $?

echo "WEEK8_ABLATION_BUNDLE_ALL_OK=1"
echo "Wall seconds: $((SECONDS - BUNDLE_START))"
