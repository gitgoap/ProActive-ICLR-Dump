# Agent Execution and Audit Rules

## 1. Plan fidelity

Before coding, read:

- instructions.md
- v3.5_ProActive_Complete_Super_Implementation_Plan.md
- PROJECT_STATUS.md
- WEEKLY_PROGRESS.md

Do not replace explicit plan requirements with placeholders, simplified
implementations, or "temporary" fallbacks unless clearly approved.

## 2. Requirement traceability

For every task, maintain:

requirement → code location → unit test → integration test → output artifact

A requirement is incomplete if any item is missing.

## 3. Fail-closed behaviour

Never silently convert:

- missing scores to zero,
- missing probes to False,
- malformed answers to flips,
- unknown outputs to valid labels,
- failed parsing to empty strings.

Raise an error or mark the example invalid.

## 4. Testing

Passing unit tests is insufficient.

Every major component must have:

- unit tests,
- adversarial edge-case tests,
- an end-to-end integration test.

## 5. Data leakage

Before any experiment, verify:

- no dataset ID or model ID enters the learner,
- train/validation only are used for threshold selection,
- test data are never used for severity or hyperparameter selection,
- proxy features and label distributions are audited by dataset.

## 6. Completion rule

Do not declare a week complete until all required:

- code,
- tests,
- pilot outputs,
- plots,
- inspections,
- schemas,
- frozen configurations,
- documentation

exist and have been reviewed.

Use these statuses only:

- NOT STARTED
- IMPLEMENTED, NOT VALIDATED
- PILOT VALIDATED
- COMPLETE
