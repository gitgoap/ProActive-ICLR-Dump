# ProActive: Research Progress and Submission Plan

**Progress snapshot: 11 September 2026.** Weeks 1–7: COMPLETE. Week 8: IMPLEMENTED, PARTIALLY VALIDATED. Submission planning assumes the remaining deadline window currently available.

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
| 8 — Robustness and audit | IMPLEMENTED, PARTIALLY VALIDATED | Held-out shift, fixed-hardware latency, and two-fold leave-one-model-out transfer are complete. Mandatory ablations, signed aggregate analysis, and the three-person audit remain. |
| 9 — Paper and reproducibility | NOT STARTED | Final evidence packaging, claim review, and submission checks; writing proceeds alongside experiments. |

The core datasets are POPE, VizWiz, HallusionBench, and VSR. Main training uses Qwen3-VL-8B and Gemma-4-E4B. InternVL contributes staged evidence and complete VSR coverage; a full three-model corpus is not claimed.

## Results and the emerging paper story

**Order stability with comparable accuracy.** Week 5 mean validation source-indicator Macro-F1 was 0.8642 for Deep Sets versus 0.8643 for the best GRU. The locked Deep Sets permutation study showed zero measured representation, prediction, action, and diagnosis-set drift across 2,000 states.

**Active acquisition is promising under limited budgets.** On the locked core test, the two-probe-budget comparison at a nominal 90% coverage target is:

| Method | Source-indicator Macro-F1 | Six-label Macro-F1 | Observed coverage | Mean diagnosis-set size |
|---|---:|---:|---:|---:|
| ProActive | 0.9392 | 0.6841 | 89.62% | 2.61 |
| Dataset-specific fixed schedule | 0.9200 | 0.5669 | 90.38% | 3.16 |
| Random acquisition | 0.8719 | 0.5257 | 92.88% | 3.27 |

Mean cost is approximately two probes for each method. Macro-F1 measures balanced classification performance; smaller sets give more specific diagnoses. Coverage differs, and statistical intervals remain pending. At budget 7, ProActive uses 5.10 probes versus 6.01 for full evidence, with source-indicator Macro-F1 0.9942 versus 0.9958 and slightly larger diagnosis sets. Advantages therefore depend on budget.

**The paper's current story:** selective evidence acquisition and an order-invariant representation enable accurate, calibrated behavioural diagnosis under limited budgets. Core results support this direction; external transfer and human agreement must establish its relevance beyond recovering probe-derived targets.

## Credibility and remaining evidence

Selection used training/validation data; calibration followed freezing, and test-based tuning is prohibited. Dataset/model identities are excluded from the main learner. Duplicate handling, answer-contract corrections, and malformed-generation recovery are documented. The synced server suite passed 303 tests.

PRE-HAL and IllusionBench are verified; the 1,200-row sample precedes inference. IllusionBench's 381 release-integrity exclusions are documented. Implementation validation needs rerunning after one missing server document is synced. External-shift results remain pending.

Three independent annotators must each judge the same 180 examples; packets are ready but unfilled. Leave-one-model-out transfer and latency are complete. At budget 7, ProActive source-bit Macro-F1 exceeds clean-only and scalar controls for both held-out models, but the Gemma fold undercovers at the nominal 0.90 target (`0.7115`). The paper will therefore claim useful diagnostic transfer while treating cross-model calibration as a limitation. Remaining evidence is the mandatory ablation bundle, signed aggregate analysis, and human agreement. New-dataset coverage is reported empirically.

## Nine-day completion schedule

| Window | Work and deliverable |
|---|---|
| Days 1–2 | Distribute annotation packets immediately; request return within 48 hours. Complete held-out parser checks and launch the approved two-model cache. Draft methods and core results concurrently. |
| Days 3–5 | Complete shift evaluation, transfer folds, mandatory ablations, latency, and statistics. Merge annotations and adjudicate disagreements while blinded; produce the human-agreement report. |
| Days 6–7 | Finalize tables, figures, limitations, and abstract; obtain professor/coauthor review and trace each reported number to its saved artifact. |
| Days 8–9 | Complete reproducibility and submission checks, with a buffer for corrections and upload. |

The core corpus, model selection, and locked evaluation already exist, making this schedule plausible. Held-out inference provisionally needs 6–12 uninterrupted hours on two A6000 GPUs, subject to pilot timing and recovery. Human availability and GPU access are the main risks. Optional GQA, full InternVL expansion, and new architecture searches are deferred. Missing required evidence must be disclosed and claims narrowed; submission readiness does not predict acceptance.

**Evidence sources:** `outputs/week5_reports/week5_encoder_selection.json`; `outputs/week6_reports/week6_policy_selection.json`; `outputs/week7_reports/frontier_test.json` and the corresponding CSV; `outputs/week7_reports/week7_full_validation.json`; `outputs/week8_data/manifests/heldout_manifest_bundle.json`. Week 5 scores are validation results; the table above reports locked-test results.
