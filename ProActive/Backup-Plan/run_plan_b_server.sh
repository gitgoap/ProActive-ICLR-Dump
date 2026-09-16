#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: bash Backup-Plan/run_plan_b_server.sh {preflight|audit|prepare|evaluate|validate}"
  exit 2
fi

cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

mkdir -p Backup-Plan/outputs/logs

python - <<'PY'
import numpy
import sklearn
print("NumPy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
PY

COMMON=(
  --config Backup-Plan/config/plan_b_config.json
  --repo-root .
  --output-dir Backup-Plan/outputs
  --device cpu
)

# Capture stage failures ourselves so the wrapper always prints the real
# Python exit code (including validation-gate code 2) despite `set -e`.
set +e
case "$MODE" in
  preflight)
    python Backup-Plan/run_plan_b.py preflight "${COMMON[@]}" \
      2>&1 | tee Backup-Plan/outputs/logs/preflight.log
    ;;
  audit)
    python Backup-Plan/run_plan_b.py audit "${COMMON[@]}" --resume \
      2>&1 | tee Backup-Plan/outputs/logs/audit.log
    ;;
  prepare)
    /usr/bin/time -v python Backup-Plan/run_plan_b.py prepare "${COMMON[@]}" --resume \
      2>&1 | tee Backup-Plan/outputs/logs/prepare.log
    ;;
  evaluate)
    /usr/bin/time -v python Backup-Plan/run_plan_b.py evaluate "${COMMON[@]}" \
      --resume --confirm-locked-evaluation \
      2>&1 | tee Backup-Plan/outputs/logs/evaluate.log
    ;;
  validate)
    python Backup-Plan/run_plan_b.py validate "${COMMON[@]}" \
      --require-evaluation \
      2>&1 | tee Backup-Plan/outputs/logs/validate.log
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 2
    ;;
esac

RC=${PIPESTATUS[0]}
echo "Plan B ${MODE} exit code: ${RC}"
exit "$RC"
