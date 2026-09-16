# Plan B V3 frozen protocol

## Research question

Can ProActive turn cached diagnostic evidence into **selective corrections**
without exceeding 1% empirical BreakRate by spending one extra probe only when
a conservative ensemble proposes a correction?

V3 was designed after observing V1 and V2 validation. It is explicitly
post-hoc and cannot be described as preregistered. Its purpose is a secondary
or appendix analysis, not a replacement for the frozen main ProActive stack.

## Information flow

The initial decision sees only clean, grounding, and blur observations (budget
2). XGBoost, LightGBM, and MLP repair/damage heads must unanimously select the
same alternative. Only then is one train-selected visual verifier acquired
from crop, brightness, or noise. Blank is forbidden as the primary verifier
because it removes visual evidence. A switch occurs only if the verifier's
normalized answer agrees with the proposed correction.

The additional verifier costs one probe action even when its cached result
originated later in the old fixed-order cache. This simulates ProActive's legal
ability to choose any next probe; unselected intermediate cache entries are not
revealed to the decision rule.

## Hard-negative and cost-sensitive training

- Use five deterministic folds grouped by image identity.
- For every fold, train on the other four and predict only the held-out fold.
- A damage example is a hard negative when at least two OOF models would have
  proposed it under the fixed mining thresholds.
- Base damage examples receive weight 10; mined hard negatives receive an
  additional multiplier of 3.
- Repeat grouped OOF fitting with those frozen hard-negative weights; this
  second OOF pass, rather than the preliminary mining pass, selects the
  verifier.
- Refit all three learners on complete training data only after OOF mining.
- Dataset ID, model ID, gold answer, correctness, fold ID, and future verifier
  outcomes never enter model features.

## Selection and gates

Training OOF evidence selects one verifier using the frozen
`repairs - 10 × damage` utility. Validation searches only the finite threshold
grid in the signed config and freezes one unanimous ensemble operating point.

Validation must achieve:

- BreakRate ≤1%;
- at least one repair;
- six-cell macro accuracy gain over keep-original ≥1 percentage point.

Only if validation passes is calibration opened as an independent confirmation
set. No parameter may change after seeing confirmation. Confirmation must pass
the same three criteria before locked test/shift evaluation can be authorized.

This reject/keep mechanism follows selective-prediction principles, but V3 is
not the end-to-end SelectiveNet architecture and must not be described as such.

## Compute and outputs

All work is CPU-only and uses existing cached probe answers. Expected runtime
on Bumblebee is roughly 5–30 minutes, depending on available CPU threads.
Primary evidence will be written under `outputs/training/`, including OOF fold
assignments, hard negatives, verifier selection, validation leaderboard,
predictions, metrics, model hashes, and the signed freeze manifest.
