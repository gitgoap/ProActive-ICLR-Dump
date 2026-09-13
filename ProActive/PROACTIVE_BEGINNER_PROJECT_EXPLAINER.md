# ProActive: Beginner-Friendly Project Explainer

**Current snapshot:** 13 September 2026  
**Audience:** New collaborators, supervisors, and readers who want the complete
idea before inspecting detailed results.

For current metrics and direct links to signed artifacts, use
[`PROACTIVE_RESEARCH_HANDOFF_AND_RESULTS_INDEX.md`](PROACTIVE_RESEARCH_HANDOFF_AND_RESULTS_INDEX.md).
That index is the result source of truth; this document explains the project.

## The problem in one paragraph

A vision-language model can give a wrong answer for different reasons. It may
fail to read the image, rely too strongly on language patterns, or fail to
connect the question to the correct object or relation. A single confidence
score does not distinguish these behaviours. ProActive runs a small number of
controlled diagnostic probes, learns which probe is useful next, stops when
more evidence is not worth its cost, and returns a calibrated set of plausible
failure modes.

## A simple analogy

ProActive works like a doctor choosing tests. The doctor does not order every
test or report only one confidence number. They keep several explanations in
mind, request the most informative next test, update the diagnosis, and stop
when another test is unlikely to help.

| Diagnostic setting | ProActive equivalent |
|---|---|
| Patient | One image-question example evaluated by a model |
| Symptom | An incorrect or unstable answer |
| Test | A controlled image or prompt probe |
| Test cost | An additional model inference |
| Updated diagnosis | New failure-mode probabilities |
| Stop decision | No remaining probe is worth its cost |
| Differential diagnosis | A calibrated set of plausible failure modes |

## What ProActive diagnoses

The framework derives three operational behavioural signals:

- **Visual fragility:** the answer is sensitive to controlled image changes.
- **Language-prior persistence:** the answer persists when visual support is
  removed or contradicted.
- **Grounding/alignment instability:** the model struggles to connect the
  question to the relevant object, attribute, or relation.

These signals produce six reporting classes: `visual`, `language-prior`,
`alignment`, `mixed`, `unclear`, and `no-failure`. They are reproducible
probe-induced diagnostics, not proof of a model's internal causal mechanism.

## How the system works

1. Normalize examples from the source datasets and create group-safe
   train/validation/calibration/test splits.
2. Cache clean and probed answers from the teacher vision-language models.
3. Convert answer changes into diagnostic targets and partial evidence states.
4. Use a Deep Sets encoder to summarize the probes observed so far without
   imposing an arbitrary order.
5. Predict the value of each legal next probe. The policy chooses a probe or
   `STOP`.
6. Use Adaptive Prediction Sets (APS) to return a calibrated set of plausible
   diagnoses.
7. Evaluate diagnostic quality, set coverage, set size, probe cost, order
   stability, model transfer, and dataset shift.

The data firewall is essential: training fits weights, validation selects the
system, calibration fixes APS thresholds after the stack is frozen, and the
locked test reports the final core result once. Held-out datasets are used for
stress testing, not retuning.

## Progress by research phase

| Phase | Status | Main outcome |
|---|---|---|
| Weeks 1–2: contracts and infrastructure | COMPLETE | Common schemas, deterministic grouped splits, model adapters, answer normalization, hashes, and fail-closed validation. |
| Week 3: probe calibration | COMPLETE | 10,400 valid pilot records; frozen visual severities and semantic-match threshold `0.50`. |
| Week 4: supervision corpus | COMPLETE | 14,582 core teacher rows, 87,712 probe observations, 14,582 labels, and 388,623 partial states across Qwen and Gemma. |
| Week 5: diagnostic encoder | COMPLETE | Deep Sets seed 42 selected and frozen; strong diagnostic accuracy with zero measured permutation drift. |
| Week 6: active policy | COMPLETE | VOI targets, policy grid, controls, matched-cost frontiers, and validation-only selection of cost multiplier `0.0`, seed 42. |
| Week 7: calibration and locked core test | COMPLETE | Frozen stack, APS thresholds, one-time locked-test frontier, order studies, and signed `GO`. |
| Week 8: robustness and audit | IMPLEMENTED, NOT VALIDATED | All mandatory machine experiments and statistics are complete. The three-person blinded human audit and final validator remain. |
| Week 9: paper and reproducibility | NOT STARTED | Paper assets, reproducibility manifest, claim audit, and release freeze remain. |

## The paper story supported so far

The strongest result is budget dependent. When only a few probes are allowed,
learned acquisition improves diagnosis over random and fixed schedules. On the
locked core test at budget 2, ProActive reaches source-bit Macro-F1 `0.9392`,
compared with `0.9200` for the dataset-specific fixed schedule and `0.8719`
for random acquisition. Its prediction sets are also smaller.

The held-out PRE-HAL and IllusionBench frontier has the same shape: ProActive
outperforms random at budgets 1–4, with the largest observed advantage at
budget 3. At budget 7, random is slightly more accurate while ProActive uses
fewer probes. This supports the precise claim that active selection is most
useful under constrained budgets; it does not support universal dominance.

Deep Sets supplies the second important result. It is competitive with the
order-sensitive GRU while showing zero measured permutation drift in the
locked study. This matches the scientific structure of the input: acquired
probe observations form a set, not a meaningful sequence.

The robustness results also establish important boundaries. Diagnostic
transfer remains useful when either Qwen or Gemma is held out, but APS coverage
degrades in the Gemma-held-out fold. The paper should therefore claim useful
diagnostic transfer, not a distribution-free cross-model coverage guarantee.

## What the ablations and latency show

- Removing semantic matching causes the clearest measured ablation loss,
  supporting semantic comparison for free-form answers.
- VOI supervision contributes positively, although its effect is smaller.
- Removing `STOP` can slightly improve saturated-budget accuracy but requires
  more probes, supporting `STOP` as an efficiency mechanism.
- Mean controller overhead is about `1.355 ms`, while mean cached generation
  latency is about `9163.951 ms`. The extra model calls dominate runtime, not
  the policy network.

## Why errors occurred during development

| Error family | Plain explanation | Resolution principle |
|---|---|---|
| Duplicate pilot rows | A resumed append reused samples from an earlier run. | Validate identities and use overwrite/resume contracts. |
| Malformed grounding answers | A generative model ignored the requested output format. | Keep the row visible and retry only under an approved, documented policy. |
| Manifest or parser drift | Code or source data changed after an artifact was cached. | Refuse unsafe resume and migrate or rebuild with hashes. |
| GPU OOM/contention | Other processes already occupied memory on a shared server. | Verify physical-GPU processes before launch and preserve resumable outputs. |
| Environment mismatch | InternVL required compatible Python, Transformers, and auxiliary packages. | Use the documented `proactive-internvl` Conda environment. |
| Validator exit code 1 | A prerequisite, approval, hash, schema, or scientific metric was unsafe. | Find the first substantive error and repair it; do not bypass the gate. |
| Incomplete sync | Local files were copied while a server run was unfinished. | Check process state, final log line, row counts, and hashes after syncing. |

These failures are part of the audit history. The final core caches have full
accounting and no unresolved rows; saying that development had “zero errors”
would be inaccurate.

## What remains before the evidence package is final

1. Three independent annotators must each complete the same 180 blinded rows.
2. Merge the packets, adjudicate genuine disagreements while still blinded,
   and compute agreement and automatic-rule match statistics.
3. Run the CPU-only full Week 8 validator and archive its signed report.
4. Generate paper tables and figures from the signed JSON/CSV artifacts.
5. Build the reproducibility manifest, trace every paper number to evidence,
   complete the limitations/claim audit, and freeze the release.

No additional mandatory GPU experiment is currently required. Optional full
InternVL coverage, GQA-Relation, leave-one-dataset-out evaluation, broad model
search, and RAPS extensions remain deferred so they do not displace the audit,
paper, or reproducibility work.

## Language that keeps the claims accurate

- Say **operational behavioural signals**, not “true causal failure sources.”
- Say **complete two-model core cache**, not complete three-model coverage.
- Separate **locked core-test coverage** from empirical held-out-shift coverage.
- Say active acquisition helps **under constrained budgets**, not at every
  possible budget.
- Report the budget-7 saturation result and Gemma transfer undercoverage.

## Where to continue

| Need | Read |
|---|---|
| Current claims, numbers, and artifact links | [`PROACTIVE_RESEARCH_HANDOFF_AND_RESULTS_INDEX.md`](PROACTIVE_RESEARCH_HANDOFF_AND_RESULTS_INDEX.md) |
| Short supervisor-facing status | [`PROACTIVE_RESEARCH_PROGRESS_BRIEF.md`](PROACTIVE_RESEARCH_PROGRESS_BRIEF.md) |
| Detailed chronological record | [`WEEKLY_PROGRESS.md`](WEEKLY_PROGRESS.md) |
| Current blockers and next actions | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) |
| Approved choices | [`DECISIONS.md`](DECISIONS.md) |
| Failures and recoveries | [`FAILURE_LOG.md`](FAILURE_LOG.md) |
| Human annotation procedure | [`human_annotation/README.md`](human_annotation/README.md) |
| Exact completion gates | [`doc/docs/WEEK_08_REQUIREMENTS.md`](doc/docs/WEEK_08_REQUIREMENTS.md) and [`doc/docs/WEEK_09_REQUIREMENTS.md`](doc/docs/WEEK_09_REQUIREMENTS.md) |
