#!/usr/bin/env bash
# Prove semantic equivalence of the rebuilt no-budget diagnostic, then archive
# its stale hash-bound descendants so the main bundle can rebuild them.

set -u
set -o pipefail

if [[ $# -ne 2 || "$1" != "--allow-shared" || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/prepare_week8_ablation_chain_rebuild.sh --allow-shared PHYSICAL_GPU_ID"
  exit 2
fi
if [[ ! -f configs/experiments/week8_evaluation.yaml ]]; then
  echo "Run this script from the ProActive repository root."
  exit 2
fi

GPU_ID="$2"
GPU_UUID=$(nvidia-smi -i "$GPU_ID" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
FREE_MIB=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
GPU_UTIL=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
GPU_JOBS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits |
  awk -F',' -v target="$GPU_UUID" '{gsub(/[[:space:]]/, "", $1); if ($1 == target) print $0}')
if [[ -z "$FREE_MIB" || "$FREE_MIB" -lt 8192 ]]; then
  echo "STOP: shared mode requires at least 8192 MiB free; observed ${FREE_MIB:-unknown} MiB."
  exit 2
fi
if [[ -z "$GPU_UTIL" || "$GPU_UTIL" -gt 10 ]]; then
  echo "STOP: shared mode requires launch utilization at or below 10%; observed ${GPU_UTIL:-unknown}%."
  exit 2
fi
echo "Shared mode approved for physical GPU $GPU_ID: ${FREE_MIB} MiB free, utilization ${GPU_UTIL}%."
if [[ -n "$GPU_JOBS" ]]; then echo "$GPU_JOBS"; fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
ROOT=outputs/week8_ablations/no_budget_embedding
DIAG="$ROOT/diagnostic_checkpoints/diag_deep_sets_standard_seed42.best.pt"
HISTORY="$ROOT/diagnostic_checkpoints/diag_deep_sets_standard_seed42.history.json"
REFERENCE_APS="$ROOT/temporary_aps/temporary_aps_deep_sets_standard_seed42.json"
REFERENCE_VOI="$ROOT/voi/voi_manifest.json"
VERIFY_ROOT=outputs/week8_repair_validation/no_budget_embedding
VERIFY_APS_DIR="$VERIFY_ROOT/temporary_aps"
VERIFY_APS="$VERIFY_APS_DIR/temporary_aps_deep_sets_standard_seed42.json"
LEDGER=outputs/logs/week8/ablations/gpu_time_ledger.tsv
mkdir -p "$VERIFY_APS_DIR" outputs/logs/week8/ablations outputs/week8_recovery_archive

ARCHIVED_HISTORY=$(find outputs/week8_recovery_archive -type f \
  -name 'diag_deep_sets_standard_seed42.history.json' -printf '%T@ %p\n' |
  sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z "$ARCHIVED_HISTORY" || ! -f "$ARCHIVED_HISTORY" ]]; then
  echo "STOP: archived pre-repair history is missing."
  exit 2
fi

# The new completion guard must accept the rebuilt 0--10 history and reject
# any accidental continuation before equivalence testing proceeds.
python scripts/train_diag.py \
  --config outputs/week8_data/ablation_configs/no_budget_embedding_diag.yaml \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --encoder deep_sets --seed 42 --device cpu \
  --output_dir "$ROOT/diagnostic_checkpoints" --resume || exit $?

STAGE_START=$SECONDS
env CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/fit_temporary_aps.py \
  --config outputs/week8_data/ablation_configs/no_budget_embedding_diag.yaml \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --checkpoint "$DIAG" --output_dir "$VERIFY_APS_DIR" \
  --seed 42 --device cuda:0 --overwrite \
  2>&1 | tee outputs/logs/week8/ablations/no_budget_embedding_equivalence_aps.log
APS_RC=${PIPESTATUS[0]}
APS_SECONDS=$((SECONDS - STAGE_START))
printf '%s\t%s\t%s\n' "$APS_SECONDS" "$APS_RC" \
  "no_budget_embedding rebuild equivalence APS" >> "$LEDGER"
if [[ "$APS_RC" -ne 0 ]]; then exit "$APS_RC"; fi

python scripts/validate_ablation_rebuild_equivalence.py \
  --rebuilt_history "$HISTORY" --archived_history "$ARCHIVED_HISTORY" \
  --reference_aps "$REFERENCE_APS" --reference_voi "$REFERENCE_VOI" \
  --rebuilt_aps "$VERIFY_APS" \
  --output_report "$VERIFY_ROOT/equivalence_report.json" \
  --patience 5 --overwrite || exit $?

ARCHIVE=outputs/week8_recovery_archive/no_budget_embedding_stale_chain_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$ARCHIVE"
for name in temporary_aps voi policy_checkpoints freeze frontier; do
  if [[ -e "$ROOT/$name" ]]; then mv "$ROOT/$name" "$ARCHIVE/$name"; fi
done
mv "$VERIFY_APS_DIR" "$ROOT/temporary_aps"
mkdir -p "$ROOT/voi" "$ROOT/policy_checkpoints" "$ROOT/freeze" "$ROOT/frontier"

echo "WEEK8_NO_BUDGET_EQUIVALENCE_ALL_OK=1"
echo "Stale downstream chain archived at $ARCHIVE"
echo "The main ablation bundle may now rebuild the hash-bound descendants."
