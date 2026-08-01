Act as an adversarial ML research-code reviewer.

Do not modify code initially. Compare the implementation line by line against:

1. v3.5_ProActive_Complete_Super_Implementation_Plan.md
2. AGENT_EXECUTION_AND_AUDIT.md
3. docs/WEEK_3_REQUIREMENTS_MATRIX.md

Search specifically for:

- missing requirements,
- placeholders and TODOs,
- simplified formulas,
- silent defaults,
- missing-data-to-zero conversions,
- train/test leakage,
- dataset proxy confounds,
- nondeterministic behaviour,
- unit tests that merely mirror the implementation,
- missing integration tests.

Provide concrete file and line references. Do not accept passing tests as proof
of correctness.
