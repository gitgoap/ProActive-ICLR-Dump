# Plan B validation conclusion

**Decision:** validation gate failed; stop before core test and held-out shift.

The preparation run completed normally. The logistic solver converged in 54
iterations, all input/cache audits passed, and the signed preparation bundle
passes artifact validation. The failure is scientific, not technical.

## Frozen validation result

The validation-selected simple comparator was always-ground:

| Method | Six-cell macro accuracy | Pooled accuracy | Repairs | Damage | BreakRate |
|---|---:|---:|---:|---:|---:|
| Keep original | 0.7511 | 0.7675 | 0 | 0 | 0.00% |
| Always ground | 0.7531 | 0.7771 | 21 | 18 | 7.47% |
| Majority | 0.7129 | 0.7357 | 3 | 13 | 5.39% |
| Supported ground | 0.7129 | 0.7357 | 3 | 13 | 5.39% |

The learned selector behaved in three regimes:

| Margin | Six-cell macro accuracy | Pooled accuracy | Repairs | Damage | BreakRate | Safety eligible? |
|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.7531 | 0.7771 | 21 | 18 | 7.47% | No |
| 0.05 | 0.7531 | 0.7771 | 21 | 18 | 7.47% | No |
| 0.10 | 0.7129 | 0.7357 | 3 | 13 | 5.39% | No |
| 0.20 | 0.7511 | 0.7675 | 0 | 0 | 0.00% | Yes |
| 1.01 | 0.7511 | 0.7675 | 0 | 0 | 0.00% | Yes; selected by the frozen tie rule |

The safety-compliant selector therefore kept every original answer. Its
six-cell macro accuracy was 0.7511, below the always-ground comparator's
0.7531. The observed gain was -0.0020, while the gate required at least
+0.0100 with BreakRate at most 1%. Relative to the required threshold, the
selector missed by approximately 1.20 percentage points.

At aggressive margins, correction opportunities were concentrated in
HallusionBench, but damage occurred across HallusionBench, POPE and VSR. The
ten permitted answer-provenance/vote features did not separate helpful from
harmful switches reliably enough to satisfy the safety constraint. At safe
margins the selector collapsed to keep-original.

## Scientific interpretation

This negative result argues against presenting cached-answer selection as a
successful correction contribution. It supports keeping the paper focused on
ProActive's original contribution: efficient behavioural diagnosis and
calibrated diagnostic sets. Probe answers contain correction opportunities,
but simple confidence-free candidate selection cannot exploit them safely
under this protocol.

No calibration, core-test or held-out-shift rows were accessed during fitting
or validation. Do not run the locked evaluation, change the margin grid, add
features, or relax the BreakRate gate under this experiment ID. Any future
correction study must be declared as a new protocol rather than a continuation
of this frozen run.

## Evidence

- `outputs/training/freeze_manifest.json` — signed model, margins, baselines and gate.
- `outputs/training/validation_metrics.csv` — validation slices and denominators.
- `outputs/training/validation_predictions.csv` — paired per-example outcomes.
- `outputs/audit/audit_report.json` — signed 16,982-row cache/cohort audit.
- `outputs/logs/prepare.log` — execution record and exit code 2 for the failed gate.
