# Plan B run summary

**Final status:** completed with a negative validation result.

## What we did

- Verified all ten cached Qwen and Gemma files: 16,982 records, correct hashes and zero answer-correctness drift.
- Reconstructed the leakage-safe cohorts: 1,898 training pairs, 314 validation pairs, 914 core-test pairs and 2,400 held-out-shift pairs.
- Trained one small L2 logistic-regression answer selector using 12,808 candidate rows and ten answer-source/voting features.
- The selector converged successfully in 54 iterations. Qwen and Gemma were not trained, fine-tuned or queried again.
- Selected the comparator and switching margin using validation data only.

## What failed

The model failed the predeclared validation gate, not the software run.

- Frozen comparator: always-ground, with 75.31% six-cell macro accuracy.
- Safety-compliant selector: 75.11% accuracy, zero repairs and zero damage.
- Selector gain: -0.20 percentage points; required gain: at least +1.00 point.
- Aggressive selector settings repaired 21 answers but damaged 18 originally correct answers, giving a 7.47% BreakRate against the 1% limit.

At safe settings the selector therefore made no switches and became equivalent to keep-original. The available features could not reliably distinguish helpful corrections from harmful ones.

## Why evaluation stopped

The frozen protocol required stopping when validation failed. Core-test and held-out-shift evaluation were therefore not opened, and the `evaluate` command must not be run for this experiment. This is the correct terminal outcome for the Plan B protocol.

The result does not support answer correction as a paper contribution. It supports keeping ProActive focused on efficient behavioural diagnosis and calibrated diagnostic sets.

Primary evidence: `outputs/training/freeze_manifest.json`, `outputs/training/validation_metrics.csv`, and `outputs/logs/prepare.log`.
