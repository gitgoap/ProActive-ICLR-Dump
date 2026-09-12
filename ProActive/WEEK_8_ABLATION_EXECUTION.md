# Week 8 mandatory ablation execution

## Purpose

This bundle measures which parts of ProActive actually matter. It covers all
15 predeclared comparisons while preserving the experimental firewall: seed 42
only, core validation evidence only, and no access to core test or held-out
PRE-HAL/IllusionBench results for fitting or selection.

## Approved boundary

- At most two physical GPUs, each verified free at launch.
- Eight combined GPU-hours maximum.
- Forty-five minutes maximum per training job.
- Twenty minutes maximum per frontier.
- Every stage is resumable and hash-bound.
- A failure stops the bundle; it does not silently drop a component or row.

## Server environment

Always activate the project environment first:

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
```

## Run

`auto` selects up to two GPUs that have no compute process and at least 2 GiB
free. It refuses to launch when none is safe.

```bash
mkdir -p outputs/logs/week8
set -o pipefail

PYTHONPATH=src python -m pytest -q \
  tests/test_week8_ablation_models.py \
  tests/test_voi_targets.py \
  tests/test_week8_integrity.py \
  2>&1 | tee outputs/logs/week8/ablation_focused_tests.log

TEST_RC=${PIPESTATUS[0]}
echo "Week 8 ablation focused tests exit code: $TEST_RC"

if [ "$TEST_RC" -eq 0 ]; then
  bash scripts/run_week8_ablations.sh auto \
    2>&1 | tee outputs/logs/week8/ablation_bundle.log
  BUNDLE_RC=${PIPESTATUS[0]}
  echo "Week 8 ablation bundle exit code: $BUNDLE_RC"
else
  echo "Ablation bundle skipped because focused tests failed."
fi
```

The focused tests are a short safety gate. The full repository test suite can
run after the expensive bundle, before Week 8 is declared complete.

## Monitoring and recovery

```bash
tail -f outputs/logs/week8/ablation_bundle.log
```

The per-worker details are in:

```text
outputs/logs/week8/ablations/worker0.log
outputs/logs/week8/ablations/worker1.log
outputs/logs/week8/ablations/<ablation>/
```

If a stage stops or times out, synchronize those logs and the entire
`outputs/week8_ablations/` directory. Also preserve
`outputs/logs/week8/ablations/gpu_time_ledger.tsv`: it makes the eight-hour
ceiling cumulative across retries. Re-running the same command resumes
completed artifacts after checking their hashes and refuses to exceed the
remaining authorization.

## Success artifacts

The terminal must end with `WEEK8_ABLATION_BUNDLE_ALL_OK=1`. The main evidence
is:

```text
outputs/week8_reports/ablations.json
outputs/week8_reports/ablations.csv
outputs/week8_reports/ablation_evidence/
```

These artifacts complete the ablation requirement only. Week 8 still needs the
three-person human audit and final statistical/full validation gates.
