# Week 5 Requirements — Shared-state encoder bake-off

**Status:** COMPLETE

**Source of truth:** Plan §§5, 7, 8, 16, 19–21, and 25.7.

## Completion gate

Deep Sets must equal or exceed GRU on validation frontier quality, or remain
within one source-bit Macro-F1 point while showing substantially lower
permutation drift. Calibration and test splits remain locked throughout Week
5. Set Transformer remains unimplemented unless its predeclared gate fires.

## Requirement traceability

| ID | Requirement | Code location | Unit/adversarial test | Integration test | Required artifact | Status |
|---|---|---|---|---|---|---|
| W5-01 | One strict, metadata-free tensor contract for every encoder | `src/proactive/train/state_data.py` | `tests/test_week5_state_data.py` | Week 5 readiness validation | `outputs/week5_data/vectorized_manifest.json` | COMPLETE |
| W5-02 | Clean-only MLP using clean features only | `src/proactive/networks/encoders/clean_mlp.py` | `tests/test_week5_encoders.py` | synthetic diagnostic training | checkpoint + validation CSV | COMPLETE |
| W5-03 | Canonical/random-augmentation GRU baseline | `src/proactive/networks/encoders/gru.py` | `tests/test_week5_encoders.py` | permutation pilot | checkpoint + drift CSV | COMPLETE |
| W5-04 | Exact invariant masked-slot MLP | `src/proactive/networks/encoders/masked_slot_mlp.py` | `tests/test_week5_encoders.py` | permutation pilot | checkpoint + drift CSV | COMPLETE |
| W5-05 | Sum-pooled Deep Sets main encoder | `src/proactive/networks/encoders/deep_sets.py` | `tests/test_week5_encoders.py` | permutation pilot | checkpoint + drift CSV | COMPLETE |
| W5-06 | Shared source-bit, six-way, and signature heads | `src/proactive/networks/diagnostic.py`, `src/proactive/networks/losses.py` | `tests/test_week5_training.py` | synthetic diagnostic training | loss/metric history | COMPLETE |
| W5-07 | Weighted BCE/CE weights fitted from train only | `src/proactive/train/diagnostic.py` | `tests/test_week5_training.py` | split-guard integration | checkpoint provenance | COMPLETE |
| W5-08 | Train on `train`, select on `val`; never load `cal`/`test` for selection | `scripts/train_diag.py` | `tests/test_week5_split_guards.py` | Week 5 readiness/full validation | run manifest | COMPLETE |
| W5-09 | Three seeds for all four mandatory encoders, including both mandatory GRU order conditions | `configs/experiments/diag_bakeoff.yaml` | configuration test | full Week 5 validator | 15 checkpoints | COMPLETE |
| W5-10 | Temporary APS fitted on validation only | `src/proactive/conformal/aps.py`, `scripts/fit_temporary_aps.py` | `tests/test_aps.py` | temporary-APS integration | validation thresholds JSON | COMPLETE |
| W5-11 | Source-bit and six-way metrics, including within-dataset reporting | `src/proactive/eval/diagnostic_metrics.py` | `tests/test_diagnostic_metrics.py` | diagnostic evaluation | metrics CSV | COMPLETE |
| W5-12 | Dataset/model identity controls never feed the main learner | `scripts/eval_shortcut_controls.py` | `tests/test_week5_split_guards.py` | shortcut-control integration | shortcut-control CSV | COMPLETE |
| W5-13 | First permutation pilot with exact invariant checks | `src/proactive/eval/permutation.py`, `scripts/eval_permutation.py` | `tests/test_permutation_metrics.py` | permutation pilot | permutation CSV | COMPLETE |
| W5-14 | Set Transformer, RAPS, and identity-shortcut gates are validation-only and predeclared | `scripts/select_week5_encoder.py` | configuration test | architecture/APS/shortcut decision gate | signed selection report | COMPLETE |
| W5-15 | Resume-safe, hash-bound preprocessing/training outputs | Week 5 scripts and `src/proactive/train/checkpoints.py` | resume/hash tests | interrupted-run simulation | signed manifests | COMPLETE |

## Non-negotiable leakage rules

- `dataset`, `model_id`, `group_id`, correctness, teacher labels, and
  unacquired observations never enter the main model input.
- Normalization statistics and class weights are fitted from `train` only.
- Week 5 evaluation reads `val` only. `cal` and `test` are rejected.
- A budget-conditioned copy may reveal only a subset whose acquisition cost is
  at most that budget; remaining budget is recomputed, never trusted from a
  stale serialized value.

## Approval and execution status

The owner approved the optimizer schedule, batch size, early stopping, and
selection thresholds on 2026-08-26. The approved values are frozen in
`configs/experiments/diag_bakeoff.yaml`. The full 15-checkpoint training matrix
and all required validation-only APS, permutation, and shortcut controls are
now present. The signed report selected Deep Sets seed 42, and the owner
approved that measured selection and authorized the freeze on 2026-08-26.
The owner-approved freeze was written and full validation passed with zero
errors and warnings on 2026-08-26. Freeze SHA-256 is `0512ca85...a7929` and
full-validation report SHA-256 is `026ef289...207e`.
