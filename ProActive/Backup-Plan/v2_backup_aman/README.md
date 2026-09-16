# Plan B V2: safe cached-answer correction

> **Validation status:** `FROZEN_VALIDATION_GATE_FAILED`. The run completed
> correctly, but the best ≤1% BreakRate point achieved only +0.049 percentage
> points six-cell macro gain versus the required +1 point. Test/shift remain
> closed. See [`PLAN_B_V2_VALIDATION_CONCLUSION.md`](PLAN_B_V2_VALIDATION_CONCLUSION.md).

Plan B V2 is a **post-hoc exploratory** follow-up to the negative V1 validation
result. It reuses V1's pinned cached teacher records, cohort construction,
image-overlap purge, fixed probe order, and locked-split rules. It performs no
new MLLM inference and requires no GPU.

## What changed

1. **Richer evidence:** vote disagreement, vote margins, pairwise consistency,
   transform consistency, and whether grounding has independent support.
2. **Explicit safety objective:** two models estimate repair probability and
   damage probability for each alternative. A correction is made only when its
   repair probability and `repair - penalty × damage` score pass frozen
   validation thresholds. Otherwise the selector abstains from correction and
   keeps the original answer.
3. **Validation-only ablations:** logistic regression, default XGBoost, default
   LightGBM, and three small MLP configurations. Each is tested with a trusted
   structural feature set and a separately labelled cached-confidence proxy.

The validation gate remains demanding: six-cell macro net repair gain must be
at least +1 percentage point while pooled BreakRate is at most 1%. Failure is a
scientific result and keeps test/shift closed.

## Server run

Sync this whole directory first. Then run:

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

python -m pip install -r Backup-Plan/v2_backup_aman/requirements-plan-b-v2.txt

PYTHONPATH=src python -m pytest -q \
  Backup-Plan/v1_backup_sourish_sir_plan/tests/test_plan_b.py \
  Backup-Plan/v2_backup_aman/tests/test_plan_b_v2.py

bash Backup-Plan/v2_backup_aman/run_plan_b_v2_server.sh preflight
bash Backup-Plan/v2_backup_aman/run_plan_b_v2_server.sh prepare
```

`prepare` returning exit code `2` means the validation safety/scientific gate
failed, not that the program crashed. This is the observed V2 result. Do not
run `evaluate`; the freeze manifest says `FROZEN_VALIDATION_GATE_FAILED`.

Expected compute: 0 GPU-hours and approximately 10–45 CPU minutes for the full
24 fitted heads plus threshold search. Package installation time is separate;
actual wall time depends on available CPU threads.

## Primary evidence

- `outputs/training/freeze_manifest.json` — signed gate and selected variant.
- `outputs/training/validation_ablation_leaderboard.csv` — every frozen
  feature/model/threshold operating point.
- `outputs/training/validation_predictions.csv` — selected predictions and
  repair/damage decisions.
- `outputs/training/validation_metrics.csv` — pooled and sliced metrics.
- `outputs/logs/prepare.log` — execution record.

See [`PLAN_B_V2_PROTOCOL.md`](PLAN_B_V2_PROTOCOL.md) for the scientific contract.
