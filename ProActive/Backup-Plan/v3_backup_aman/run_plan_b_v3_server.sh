#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: bash Backup-Plan/v3_backup_aman/run_plan_b_v3_server.sh {preflight|prepare|evaluate|validate}"
  exit 2
fi

cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

V3="Backup-Plan/v3_backup_aman"
mkdir -p "$V3/outputs/logs"

COMMON=(
  --config "$V3/config/plan_b_v3_config.json"
  --repo-root .
  --output-dir "$V3/outputs"
  --device cpu
)

python - <<'PY'
import importlib
for name in ("numpy", "sklearn", "joblib", "xgboost", "lightgbm"):
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'unknown')}")
    except ImportError:
        print(f"MISSING: {name}")
PY

set +e
case "$MODE" in
  preflight)
    python "$V3/run_plan_b_v3.py" preflight "${COMMON[@]}" \
      2>&1 | tee "$V3/outputs/logs/preflight.log"
    ;;
  prepare)
    /usr/bin/time -v python "$V3/run_plan_b_v3.py" prepare "${COMMON[@]}" --resume \
      2>&1 | tee "$V3/outputs/logs/prepare.log"
    ;;
  evaluate)
    /usr/bin/time -v python "$V3/run_plan_b_v3.py" evaluate "${COMMON[@]}" \
      --resume --confirm-locked-evaluation \
      2>&1 | tee "$V3/outputs/logs/evaluate.log"
    ;;
  validate)
    python "$V3/run_plan_b_v3.py" validate "${COMMON[@]}" \
      2>&1 | tee "$V3/outputs/logs/validate.log"
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 2
    ;;
esac

RC=${PIPESTATUS[0]}
echo "Plan B V3 ${MODE} exit code: ${RC}"
exit "$RC"
