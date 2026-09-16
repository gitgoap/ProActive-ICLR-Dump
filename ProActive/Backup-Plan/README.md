# Backup experiments index

These experiments are separate from the frozen ProActive Week 7/8 main stack.
They use cached answers only and make no new multimodal-model calls.

| Directory | Meaning | Current state |
|---|---|---|
| [`v1_backup_sourish_sir_plan/`](v1_backup_sourish_sir_plan/) | Original cached-answer selector | Validation gate failed; test and shift remained closed |
| [`v2_backup_aman/`](v2_backup_aman/) | Post-hoc safe-correction follow-up with richer features, abstention, and model ablations | Validation gate failed; test and shift remained closed |
| [`v3_backup_aman/`](v3_backup_aman/) | Frozen post-hoc selective-correction study with grouped-OOF hard negatives, unanimous XGBoost/LightGBM/MLP proposals, and one train-selected verification probe | Implemented locally; server validation and confirmation pending |

V2 is explicitly post-hoc because it was designed after observing V1 validation.
It must not be described as preregistered or used to alter the frozen main-stack
results.

V3 is also explicitly post-hoc. It was frozen only after reviewing V1 and V2
validation, and therefore remains a secondary/appendix experiment. Its
calibration, test, and shift gates are deliberately sequential and fail closed.
