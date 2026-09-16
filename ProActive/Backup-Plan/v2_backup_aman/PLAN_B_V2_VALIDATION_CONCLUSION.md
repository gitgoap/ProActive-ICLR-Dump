# Plan B V2 validation conclusion

**Decision:** the run completed correctly, but the frozen validation gate
failed. Do not open core test or held-out shift.

## What happened

All pinned-input audits passed. The run fitted 24 repair/damage heads across
two feature families and six learner variants, then evaluated all 2,400 frozen
validation operating points. It finished in 2 minutes 23 seconds with maximum
resident memory below 700 MiB. The process returned exit code `2` by design:
this means a scientific gate failure, not a software crash.

The repeated LightGBM message
`X does not have valid feature names` is a scikit-learn warning caused by
predicting from positional arrays. Training and prediction used the same frozen
column order; it did not abort the run or invalidate the signed artifacts.

## Frozen safety-compliant result

The selected operating point was XGBoost with the
`structural_plus_cached_confidence_proxy_v2` features, repair minimum `0.85`,
damage penalty `8.0`, and safe-score threshold `0.8`.

| Quantity | Result |
|---|---:|
| Validation model-example pairs | 314 |
| Corrections attempted | 2 |
| Repairs | 1 |
| Damage | 1 |
| Correction abstentions / keep-original decisions | 312 |
| BreakRate | 0.415% |
| Pooled accuracy gain | 0.000 percentage points |
| Six-cell macro gain | +0.049 percentage points |
| Required six-cell macro gain | +1.000 percentage point |

The safety constraint passed, but the gain constraint did not. The one repair
was on Qwen/HallusionBench; the one damaged answer was on Gemma/POPE.

## What the ablation says

The strongest unconstrained structural operating point produced 18 repairs and
5 damages. This was a substantial Pareto improvement over V1's aggressive
21-repair/18-damage behavior, but its 2.075% BreakRate still exceeded the hard
1% ceiling. Its six-cell macro gain was +4.023 percentage points.

Therefore richer structural evidence did help distinguish corrections, but
not reliably enough to satisfy the predeclared safety limit. At ≤1% BreakRate,
the learned selector could not obtain a meaningful accuracy improvement.

The frozen result does **not** support presenting safe cached-answer correction
as a successful paper contribution. It does support an honest secondary
finding: probe evidence contains correction opportunities, and richer features
improve the repair/damage frontier, but strict safe correction remains
unresolved.

## Evidence

- `outputs/training/freeze_manifest.json` — signed selected point and failed gate.
- `outputs/training/validation_ablation_leaderboard.csv` — all 2,400 points.
- `outputs/training/validation_predictions.csv` — exact two selected corrections.
- `outputs/training/validation_metrics.csv` — pooled and sliced metrics.
- `outputs/logs/prepare.log` — runtime and exit status.

Artifact validation passes with no hash errors. Calibration, core test, and
held-out shift were not used for V2 selection and remain unopened by V2.
