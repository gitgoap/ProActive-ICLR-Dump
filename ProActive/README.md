# ProActive

ProActive studies whether a multimodal large language model can be diagnosed reliably without running every possible diagnostic probe. The system begins with one clean model response, decides which additional probe is most useful, observes the probe result, updates its diagnostic state, and either acquires another probe or stops. The final output is not a single overconfident diagnosis. It is a **calibrated set of plausible behavioural failure states**.

## Key Contributions
1. **Active diagnostic measurement:** learn which behavioural probe to acquire next under a limited forward-pass budget.
2. **Unordered evidence modelling:** represent the acquired probe outcomes as a set rather than as an artificial sequence.
3. **Calibrated diagnostic sets:** return a set with target marginal coverage after the complete acquisition policy has been frozen.
4. **Robust evaluation:** compare cost–diagnostic frontiers, permutation stability, leave-one-model-out transfer, held-out dataset shift, and strong fixed or oracle baselines.

See `PROJECT_STATUS.md` for current progress and `SERVER_RUNBOOK.md` for server execution instructions.
