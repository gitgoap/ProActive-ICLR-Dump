# Plan B V3: selective correction with active verification

V3 is a frozen, post-hoc exploratory follow-up to the failed V1 and V2
validation gates. It makes no new MLLM calls and needs no GPU. It reuses the
pinned cached answers, but changes the decision rule:

1. XGBoost, LightGBM, and a regularized MLP must unanimously propose the same
   correction under cost-sensitive repair/damage heads.
2. Hard false corrections are mined from grouped out-of-fold **training**
   predictions, never from observed validation errors.
3. Only after unanimous proposal does the policy acquire one additional visual
   probe. It switches only when that verifier agrees; otherwise it keeps the
   original answer.
4. Validation freezes one risk/coverage operating point. Calibration is then a
   no-retuning confirmation gate. Test/shift remain closed unless both pass.

See [`PLAN_B_V3_PROTOCOL.md`](PLAN_B_V3_PROTOCOL.md) for the complete contract.

## Server sequence

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

python -m pip install -r Backup-Plan/v3_backup_aman/requirements-plan-b-v3.txt

PYTHONPATH=src python -m pytest -q \
  Backup-Plan/v3_backup_aman/tests/test_plan_b_v3.py

bash Backup-Plan/v3_backup_aman/run_plan_b_v3_server.sh preflight
bash Backup-Plan/v3_backup_aman/run_plan_b_v3_server.sh prepare
```

Preparation uses CPU only. Exit code `2` means a scientific validation or
confirmation gate failed; it is not a software crash. Do not run locked
evaluation unless the signed freeze status is
`FROZEN_CONFIRMATION_GATE_PASSED`.
