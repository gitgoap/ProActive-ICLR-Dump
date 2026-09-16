# Plan B V2 frozen protocol

## Status and scope

This is a post-hoc exploratory experiment designed after V1 validation showed
that an aggressive selector repaired 21 examples but damaged 18 (7.47%
BreakRate), while its safety-compliant setting made no corrections. V2 asks a
narrow question: **can cached probe evidence identify a small subset of safe
answer corrections under BreakRate ≤1%?**

It is not a new main-paper contribution unless the result is independently
interpretable and clearly disclosed as post-hoc. The frozen ProActive Week 7/8
stack is unchanged.

## Data and split contract

- Reuse the SHA-256-pinned V1 cached files and exact cohort audit.
- Fit learners on purged closed-answer training examples only.
- Choose model, feature family, damage penalty, repair threshold, and abstention
  threshold on purged validation examples only.
- Do not access calibration.
- Keep core test and held-out shift closed unless the validation gate passes.
- Never tune after locked evaluation.

## Learner target

For every acquired prefix and every answer candidate different from the clean
answer, train two binary heads:

- `repair = 1` when the original answer is wrong and the candidate is correct;
- `damage = 1` when the original answer is correct and the candidate is wrong.

At inference, calculate

`safe_score = P(repair) - damage_penalty × P(damage)`.

Switch only when both `P(repair)` and `safe_score` exceed their validation-frozen
thresholds. Otherwise **abstain from correction and keep the original answer**.
This abstention is a decision not to edit; it is not an unanswered prediction.

## Features

The trusted `structural_v2` set contains only acquired-prefix evidence:

- answer/source support and vote share;
- top-vote and top-two margins;
- normalized disagreement entropy and pairwise agreement;
- clean/grounding agreement;
- independent transform support for the grounding candidate;
- usable-evidence and acquired-budget fractions.

`structural_plus_cached_confidence_proxy_v2` adds clean confidence and probe
confidence reconstructed as `clean value + cached shift`. Those scores were not
independently re-extracted, so this family is a labelled proxy ablation. If it
wins, the limitation must appear in any claim.

Dataset ID, model ID, gold answer, correctness, and future probes are forbidden
from features.

## Frozen ablations and selection

- Logistic regression with L2 regularization.
- XGBoost defaults, except deterministic seed 42.
- LightGBM defaults, except deterministic seed 42 and quiet logging.
- MLP `(32)`, MLP `(64,32)`, and regularized MLP `(32)`.
- Two feature families and separate repair/damage heads: 24 fitted estimators.

Validation search is fully enumerated in the config. Among points with
BreakRate ≤1%, choose maximum six-cell macro net repair gain over keep-original.
Ties prefer more repairs, less damage, and stricter abstention settings.

## Go/no-go gate

Locked evaluation opens only if validation satisfies all of:

- pooled BreakRate ≤1%;
- at least one repair;
- six-cell macro accuracy gain over keep-original ≥1 percentage point.

If the gate fails, exit code `2` is expected and test/shift remain unopened.
