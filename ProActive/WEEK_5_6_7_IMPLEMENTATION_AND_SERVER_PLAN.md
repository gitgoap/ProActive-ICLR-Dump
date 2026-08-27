# Weeks 5–7 Implementation and Server Plan

**Last updated:** 2026-08-27  
**Code status:** Week 5 COMPLETE; Weeks 6–7 IMPLEMENTED, NOT VALIDATED  
**Execution status:** Week 5 owner-approved freeze and full validation passed; Week 6 readiness is next

This document records what was implemented, what may run now, and which gates
must pass before later splits can be touched. It is deliberately separate from
the aspirational master plan and from generated output reports.

## What is now implemented

| Week | Implemented substrate | Gate before execution can advance |
|---|---|---|
| 5 | Strict vectorization; clean MLP; canonical/random GRU; masked-slot MLP; Deep Sets; shared heads/losses; three seeds; validation-only APS; shortcut and permutation controls; signed selection/freeze | Full train/validation evidence under the approved 2026-08-26 settings |
| 6 | Exact cached VOI; action-conditioned policy; legal masks and STOP; three-seed cost grid; main/GRU/masked policy support; complete baseline/oracle matrix; matched-cost frontier | Signed Week 5 freeze and a validation frontier that passes the Week 6 gate |
| 7 | Main-stack freeze; calibration-only trajectories; 90%/95% finite-sample APS at all budgets; locked test; full permutation study; go/no-go memo | Signed Week 6 selection, explicit owner freeze, and final calibration before test unlock |

The main learner never receives dataset ID or model ID. Dataset identity is
used only by an explicitly labelled reviewer control. Realized unacquired
outcomes are used only for offline VOI labels and explicit oracle conditions.

## Approved scientific settings

The owner approved these entries on 2026-08-26 and they are frozen in
`DECISIONS.md` plus the three experiment YAMLs:

1. AdamW `1e-3`, batch 512, maximum 30 epochs, patience 5.
2. Budgets `{1,2,4}` for Weeks 5–6, expanding to `{1,2,3,4,full}` for Week 7.
3. A separate train-only expected entropy-reduction network for the
   uncertainty-greedy baseline.
4. A 0.03 maximum acceptable undercoverage gap at final evaluation.
5. The Week 5 selection gate: 0.01 source-bit Macro-F1 tolerance, `1e-6`
   invariant tolerance, 0.10 substantial-drift ratio, and a 0.015
   Set-Transformer trigger gap.

The plan-fixed RAPS gate is also encoded: RAPS stays outside the critical path
unless validation APS mean set size exceeds 3.2, or singleton rate is below
35% while coverage is within 0.03 of target.

The completed `--limit <= 100` artifacts remain scientifically invalid pilots
and cannot enter a freeze. Full jobs must not use `--limit` or
`--allow_unapproved_pilot`.

## Server environment

Weeks 5–7 contain only cached-data preprocessing and small PyTorch networks;
they do not call Qwen, Gemma, or InternVL. Use the already validated
`proactive-internvl` environment because it has Python 3.11, PyTorch 2.6,
pytest, scikit-learn, and the editable ProActive package. This does not change
or downgrade the accepted `(base)` teacher runtime.

Every new tmux pane must begin with:

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

which python
python --version
python -c "import torch, sklearn, yaml, matplotlib; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'sklearn', sklearn.__version__, 'matplotlib', matplotlib.__version__)"
```

Expected Python path:
`/home/aman/miniconda3/envs/proactive-internvl/bin/python`.

## Stage A — completed on the server

All Stage A gates passed on 2026-08-26: `279` tests, complete vectorization,
and six successful GPU pilots. The commands remain below for provenance.

Run `nvidia-smi` immediately before setting `GPU_A` and `GPU_B`; low
utilization alone is not enough if another process already owns most memory.

Sync these code paths before running Stage A:

- `configs/experiments/diag_bakeoff.yaml`
- `configs/experiments/policy_train.yaml`
- `configs/experiments/calibrate_aps.yaml`
- `src/proactive/conformal/`, `src/proactive/eval/`,
  `src/proactive/policy/`, and `src/proactive/train/`
- `src/proactive/networks/encoders/` plus `controls.py`, `diagnostic.py`,
  `losses.py`, and `voi.py`
- the Week 5–7 scripts named in the requirement documents
- the new Week 5–7 tests, `pyproject.toml`, and `requirements-internvl.txt`

The exact copy checklist is also stored in `WEEK_5_6_7_SYNC_FILES.txt`.

Do not copy local `outputs/` back to the server and do not sync
`__pycache__/` directories.

After syncing the code, refresh only the small-network environment dependencies
and editable package. This does not change the accepted Qwen/Gemma base
environment:

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

python -m pip install -r requirements-internvl.txt
python -m pip install -e . --no-deps
python -m pip check
python -c "import torch, sklearn, yaml, matplotlib; print('DEPENDENCIES_OK', torch.__version__, sklearn.__version__, matplotlib.__version__)"
```

### A1. CPU test and full vectorization

This is a CPU gate. It does not consume either free GPU.

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

mkdir -p outputs/logs/week5 outputs/week5_data
set -o pipefail

PYTHONPATH=src python -m pytest -q \
  2>&1 | tee outputs/logs/week5/pre_gpu_full_pytest.log
TEST_RC=${PIPESTATUS[0]}
echo "Pre-GPU pytest exit code: $TEST_RC"
test "$TEST_RC" -eq 0 || exit "$TEST_RC"

/usr/bin/time -v python scripts/prepare_week5_data.py \
  --config configs/experiments/diag_bakeoff.yaml \
  --manifest_path outputs/week4_reports/final/state_manifest.json \
  --state_path outputs/states_v1 \
  --output_dir outputs/week5_data \
  --device cpu \
  --overwrite \
  2>&1 | tee outputs/logs/week5/vectorize_full.log
VECTOR_RC=${PIPESTATUS[0]}
echo "Full vectorization exit code: $VECTOR_RC"
test "$VECTOR_RC" -eq 0 || exit "$VECTOR_RC"

python scripts/prepare_week5_data.py \
  --config configs/experiments/diag_bakeoff.yaml \
  --manifest_path outputs/week4_reports/final/state_manifest.json \
  --state_path outputs/states_v1 \
  --output_dir outputs/week5_data \
  --device cpu \
  --resume \
  2>&1 | tee outputs/logs/week5/vectorize_resume_check.log
VECTOR_VERIFY_RC=${PIPESTATUS[0]}
echo "Vectorization verification exit code: $VECTOR_VERIFY_RC"
```

### A2. First free physical GPU — Deep Sets 1/10/100 stages

One line: trains the main invariant diagnostic network on increasingly larger
pilot slices, with each size isolated so resume cannot cross pilot boundaries.

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

mkdir -p outputs/logs/week5 outputs/week5_staging
set -o pipefail

# Set this to the first GPU that nvidia-smi shows as genuinely free.
GPU_A=0

for LIMIT in 1 10 100; do
  OUT="outputs/week5_staging/deep_sets_limit${LIMIT}"
  LOG="outputs/logs/week5/deep_sets_limit${LIMIT}.log"
  mkdir -p "$OUT"

  echo "===== Deep Sets limit ${LIMIT} ====="
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU_A" python scripts/train_diag.py \
    --config configs/experiments/diag_bakeoff.yaml \
    --manifest_path outputs/week5_data/vectorized_manifest.json \
    --encoder deep_sets \
    --seed 42 \
    --device cuda:0 \
    --limit "$LIMIT" \
    --output_dir "$OUT" \
    --overwrite \
    --allow_unapproved_pilot \
    2>&1 | tee "$LOG"

  RC=${PIPESTATUS[0]}
  echo "Deep Sets limit ${LIMIT} exit code: ${RC}"
  test "$RC" -eq 0 || exit "$RC"
done

echo "DEEP_SETS_STAGES_ALL_OK=1"
```

### A3. Second free physical GPU — GRU 1/10/100 stages

One line: trains the mandatory order-sensitive canonical GRU baseline on the
same staged slices for a fair timing and interface comparison.

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

mkdir -p outputs/logs/week5 outputs/week5_staging
set -o pipefail

# Set this to the second GPU that nvidia-smi shows as genuinely free.
GPU_B=2

for LIMIT in 1 10 100; do
  OUT="outputs/week5_staging/gru_canonical_limit${LIMIT}"
  LOG="outputs/logs/week5/gru_canonical_limit${LIMIT}.log"
  mkdir -p "$OUT"

  echo "===== GRU canonical limit ${LIMIT} ====="
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU_B" python scripts/train_diag.py \
    --config configs/experiments/diag_bakeoff.yaml \
    --manifest_path outputs/week5_data/vectorized_manifest.json \
    --encoder gru \
    --gru_condition canonical \
    --seed 42 \
    --device cuda:0 \
    --limit "$LIMIT" \
    --output_dir "$OUT" \
    --overwrite \
    --allow_unapproved_pilot \
    2>&1 | tee "$LOG"

  RC=${PIPESTATUS[0]}
  echo "GRU limit ${LIMIT} exit code: ${RC}"
  test "$RC" -eq 0 || exit "$RC"
done

echo "GRU_STAGES_ALL_OK=1"
```

## Timing policy

No full-run wall-time claim is made before Stage A. The logs capture elapsed
time and peak memory for every step. Use the 100-row measurements plus the
actual vectorized train/validation row counts to calculate a conservative full
bound. If a one-row small-network stage produces no epoch log within 15
minutes, treat it as abnormal and return the process/log output rather than
leaving it overnight.

Planning bounds only—not promises:

| Work | Expected order | Stop/review trigger |
|---|---:|---|
| Full CPU vectorization of 388,623 states | minutes to tens of minutes | no progress/output for 30 minutes |
| Each 1/10/100 small-network pane | minutes, not hours | no epoch log for 15 minutes or nonzero exit |
| Full Week 5 bake-off | calculated only from the 100-row logs | do not launch before owner approval |
| Week 6 VOI/frontier | calculated after Week 5 freeze; oracle subsets are batched | pilot first and inspect target counts/signs |
| Week 7 calibration/test | calculated after Week 6 freeze | locked until all hashes and APS gates pass |

## What to return after Stage A

Synchronize only:

- `outputs/logs/week5/`
- `outputs/week5_data/`
- `outputs/week5_staging/`

Do not synchronize Python `__pycache__` directories. Also paste the final 20
lines from both GPU pane logs if synchronization is delayed.

## Stage B — approved Week 5 full-run timing pair

Approval changes the diagnostic-config hash, so regenerate the vectorized
manifest once before full training. This is mandatory provenance refresh, not
a repeated pilot, and the measured server time is about 75 seconds.

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

mkdir -p outputs/logs/week5 outputs/week5_data outputs/week5_reports
set -o pipefail

/usr/bin/time -v python scripts/prepare_week5_data.py \
  --config configs/experiments/diag_bakeoff.yaml \
  --manifest_path outputs/week4_reports/final/state_manifest.json \
  --state_path outputs/states_v1 \
  --output_dir outputs/week5_data \
  --device cpu \
  --overwrite \
  2>&1 | tee outputs/logs/week5/vectorize_approved_full.log

VECTOR_RC=${PIPESTATUS[0]}
echo "Approved vectorization exit code: $VECTOR_RC"
test "$VECTOR_RC" -eq 0 || exit "$VECTOR_RC"

python scripts/validate_week5.py \
  --mode readiness \
  --config configs/experiments/diag_bakeoff.yaml \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --output_dir outputs/week5_reports \
  --overwrite \
  2>&1 | tee outputs/logs/week5/approved_readiness.log

echo "Week 5 readiness exit code: ${PIPESTATUS[0]}"
```

After readiness exits zero, run one mandatory full Deep Sets seed and one
mandatory full canonical-GRU seed in parallel on genuinely free GPUs. These
are the first two of the required 15 checkpoints and provide real full-corpus
timing without wasting compute.

Deep Sets pane:

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

mkdir -p outputs/logs/week5/full outputs/week5_checkpoints
set -o pipefail
GPU_A=0

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU_A" python scripts/train_diag.py \
  --config configs/experiments/diag_bakeoff.yaml \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --encoder deep_sets \
  --seed 42 \
  --device cuda:0 \
  --output_dir outputs/week5_checkpoints \
  --overwrite \
  2>&1 | tee outputs/logs/week5/full/deep_sets_seed42.log

echo "Full Deep Sets seed 42 exit code: ${PIPESTATUS[0]}"
```

Canonical-GRU pane:

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

mkdir -p outputs/logs/week5/full outputs/week5_checkpoints
set -o pipefail
GPU_B=2

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU_B" python scripts/train_diag.py \
  --config configs/experiments/diag_bakeoff.yaml \
  --manifest_path outputs/week5_data/vectorized_manifest.json \
  --encoder gru \
  --gru_condition canonical \
  --seed 42 \
  --device cuda:0 \
  --output_dir outputs/week5_checkpoints \
  --overwrite \
  2>&1 | tee outputs/logs/week5/full/gru_canonical_seed42.log

echo "Full canonical GRU seed 42 exit code: ${PIPESTATUS[0]}"
```

Physical GPU IDs `0` and `2` are examples. Change only `GPU_A`/`GPU_B` after
checking current memory ownership with `nvidia-smi`; keep `--device cuda:0`.

## Later execution order

| Order | Stage | Why it cannot move earlier |
|---:|---|---|
| 1 | Week 5 full three-seed encoder bake-off and validation-only selection | Needs Stage A timing and owner-approved training settings |
| 2 | Week 5 signed diagnostic freeze | Needs all mandatory checkpoints, permutation reports, shortcut controls, and temporary APS |
| 3 | Week 6 VOI corpus and policy/baseline training | Requires the frozen Week 5 checkpoint |
| 4 | Week 6 validation frontier and signed policy selection | Uses train/validation only; calibration/test remain locked |
| 5 | Week 7 main-stack freeze | Requires owner review of Week 6 evidence |
| 6 | Calibration trajectories and 90%/95% APS | May occur only after the stack is frozen |
| 7 | One locked test frontier and full permutation run | Requires freeze-bound final APS; no post-test tuning |

Set Transformer, RAPS, full InternVL caching, and new datasets remain outside
the critical path unless their declared validation gates fire.
