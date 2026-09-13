# ProActive: Research Progress and Submission Plan

**Progress snapshot: 13 September 2026.** Weeks 1–7: COMPLETE. Week 8:
IMPLEMENTED, NOT VALIDATED. Every mandatory machine experiment and statistical
analysis is complete; only the three-person human audit and final validator
remain. Paper writing should proceed now.

## Research question and approach

Can we identify plausible failure modes of a vision-language model using a small number of diagnostic tests, while expressing uncertainty about the diagnosis?

ProActive extends HalluPrism's behavioural probes into selective diagnosis. It observes answer changes under visual corruption, image removal, and grounding/relation prompts. Deep Sets combines observations independently of order; a learned policy chooses the next probe or stops. Adaptive Prediction Sets (APS), calibrated after freezing the system, return plausible diagnoses.

The indicators are **visual fragility, language-prior persistence, and grounding/alignment instability**. Reporting labels are no-failure, visual, language-prior, alignment, mixed, and unclear. These describe observed behaviour, without establishing internal causal mechanisms.

## Progress against the implementation plan

| Week | Status | Completed work and evidence |
|---|---|---|
| 1 — Data and contracts | COMPLETE | Defined schemas, grouped splits, dataset loading, and answer normalization. |
| 2 — Model inference | COMPLETE | Implemented and server-checked generation/scoring interfaces for Qwen, Gemma, and InternVL. |
| 3 — Probe calibration | COMPLETE | Validated 10,400 pilot records. Froze visual severities and semantic threshold 0.50 after a 50-pair training/validation human audit. |
| 4 — Diagnostic supervision | COMPLETE | 7,291 examples per main model: 14,582 teacher rows/labels, 87,712 probe observations, and 388,623 partial evidence states. No unresolved core failures. |
| 5 — Evidence encoder | COMPLETE | 15 runs across three seeds comparing clean-only, masked-slot MLP, canonical/random-order GRU, and Deep Sets. Froze Deep Sets seed 42. |
| 6 — Active policy | COMPLETE | 496,912 value-of-information targets per evaluated encoder; 15 Deep Sets policies and required controls evaluated. Validation selected cost multiplier 0.0, seed 42. |
| 7 — Calibration and core test | COMPLETE | Frozen stack, 90%/95% APS, and 1,560 test model–example pairs at budgets 1/2/3/4/7. Controls, oracles, permutations, and final validation passed. |
| 8 — Robustness and audit | IMPLEMENTED, NOT VALIDATED | Held-out shift, two-fold leave-one-model-out transfer, latency, all 15 ablations, grouped statistics, slices, and qualitative cases are complete. Only the three-person audit and final validator remain. |
| 9 — Paper and reproducibility | NOT STARTED | The evidence map exists and writing can start, but one-command paper assets, reproducibility manifest, independent review, and release freeze remain. |

The core datasets are POPE, VizWiz, HallusionBench, and VSR. Main training uses Qwen3-VL-8B and Gemma-4-E4B. InternVL contributes staged evidence and complete VSR coverage; a full three-model corpus is not claimed.

## Results and the emerging paper story

**Order stability with comparable accuracy.** Week 5 mean validation source-indicator Macro-F1 was 0.8642 for Deep Sets versus 0.8643 for the best GRU. The locked Deep Sets permutation study showed zero measured representation, prediction, action, and diagnosis-set drift across 2,000 states.

**Active acquisition is promising under limited budgets.** On the locked core test, the two-probe-budget comparison at a nominal 90% coverage target is:

| Method | Source-indicator Macro-F1 | Six-label Macro-F1 | Observed coverage | Mean diagnosis-set size |
|---|---:|---:|---:|---:|
| ProActive | 0.9392 | 0.6841 | 89.62% | 2.61 |
| Dataset-specific fixed schedule | 0.9200 | 0.5669 | 90.38% | 3.16 |
| Random acquisition | 0.8719 | 0.5257 | 92.88% | 3.27 |

Mean cost is approximately two probes for each method. Macro-F1 measures
balanced classification performance; smaller sets give more specific
diagnoses. At budget 7, ProActive uses 5.10 probes versus 6.01 for full
evidence, with source-indicator Macro-F1 0.9942 versus 0.9958 and slightly
larger diagnosis sets. Advantages therefore depend on budget.

The completed held-out PRE-HAL/IllusionBench analysis confirms this shape.
ProActive-minus-random source-bit Macro-F1 is `+0.0458`, `+0.1189`, `+0.1474`,
and `+0.1178` at budgets 1--4. At the predeclared budget-7 primary comparison,
it is `-0.0063` (95% CI `[-0.0087, -0.0041]`; Holm-adjusted `p=0.003`) while
using about `0.42` fewer probes. The paper will report this full-budget
saturation result directly and claim benefit under constrained budgets.

**The paper's current story:** selective evidence acquisition and an
order-invariant representation enable accurate behavioural diagnosis with few
additional model calls. The advantage is strongest at constrained budgets and
largely disappears when almost all probes are affordable. Calibration is
reliable in the frozen core setting but degrades in one cross-model transfer
fold, so the paper separates diagnostic transfer from conformal guarantees.

## Credibility and remaining evidence

Selection used training/validation data; calibration followed freezing, and test-based tuning is prohibited. Dataset/model identities are excluded from the main learner. Duplicate handling, answer-contract corrections, and malformed-generation recovery are documented. The synced server suite passed 303 tests.

PRE-HAL and IllusionBench are verified; the 1,200-row sample was fixed before
inference. IllusionBench's 381 release-integrity exclusions are documented.
The recovered two-model held-out cache covers all 2,400 requested rows without
selective exclusions. Shift, LOMO, latency, ablations, grouped intervals,
slices, and qualitative outputs are complete and hash-bound.

Three independent annotators must each judge the same 180 examples; packets
are ready but unfilled. At budget 7, ProActive source-bit Macro-F1 exceeds
clean-only and scalar controls for both held-out models, but the Gemma fold
undercovers at the nominal 0.90 target (`0.7115`). The paper will therefore
claim useful diagnostic transfer while treating cross-model calibration as a
limitation. Human agreement is the only missing evidence. New-dataset coverage
is reported empirically.

## Final submission sprint

| Window | Work and deliverable |
|---|---|
| Now | Distribute all three 180-row annotation packets and draft the complete paper except the audit paragraph. |
| When packets return | Merge, adjudicate disagreements while blinded, run agreement/rule-match analysis, then run the full Week 8 validator. |
| Next 2–3 days | Convert signed CSV/JSON results into final tables and figures; write results, limitations, and abstract. |
| Final buffer | Professor/coauthor claim review, number-to-artifact trace, formatting, reproducibility check, and upload. |

No additional mandatory GPU experiment remains. Human turnaround and writing
bandwidth are now the main risks. Optional GQA, leave-one-dataset-out, full
InternVL expansion, and new architecture searches remain deferred. Missing
human evidence must be disclosed and claims narrowed; submission readiness
does not predict acceptance.

**Evidence entry point:** `PROACTIVE_RESEARCH_HANDOFF_AND_RESULTS_INDEX.md`.
Week 5 scores are validation results; the table above reports locked-test
results; Week 8 coverage is empirical shift behavior.
