# ProActive Research Handoff and Results Index

> **Plan B cached-answer correction experiment:** see
> [`Backup-Plan/PLAN_B_IMPLEMENTATION_AND_EXECUTION.md`](Backup-Plan/PLAN_B_IMPLEMENTATION_AND_EXECUTION.md)
> for the self-contained protocol, implementation map, server commands, gates,
> and expected evidence. It is an exploratory secondary experiment and does not
> replace the frozen Week 7/8 results indexed below. Its validation gate failed
> on 16 September 2026, so locked test/shift evaluation was not opened; see
> [`Backup-Plan/PLAN_B_VALIDATION_CONCLUSION.md`](Backup-Plan/PLAN_B_VALIDATION_CONCLUSION.md).

**Snapshot:** 13 September 2026  
**Submission status:** Weeks 1–7 are COMPLETE. All mandatory Week 8 machine
experiments and statistical analyses are complete. The three-person blinded
human audit and the resulting final Week 8 validation are still pending. Week
9 paper-asset and reproducibility packaging has not yet been completed.

This is the shortest reliable entry point for a new researcher or LLM-assisted
analysis session. It explains the current paper story and links claims directly
to evidence. JSON reports are the signed source of truth, CSV files are for
tabulation, PNG files are views, and logs establish execution history.

## Read in this order

1. [`AGENTS.md`](AGENTS.md) and [`instructions.md`](instructions.md) — non-negotiable scientific and workspace rules.
2. This file — current result map and honest paper story.
3. [`PROJECT_STATUS.md`](PROJECT_STATUS.md) and [`WEEKLY_PROGRESS.md`](WEEKLY_PROGRESS.md) — current gate and chronological record.
4. [`v3.5_ProActive_Complete_Super_Implementation_Plan.md`](v3.5_ProActive_Complete_Super_Implementation_Plan.md) — formulas, protocols, and original scope.
5. [`DECISIONS.md`](DECISIONS.md), [`FAILURE_LOG.md`](FAILURE_LOG.md), and [`RUN_REGISTRY.md`](RUN_REGISTRY.md) — approvals, failures, commands, compute, and provenance.
6. [`doc/docs/WEEK_08_REQUIREMENTS.md`](doc/docs/WEEK_08_REQUIREMENTS.md) — exact remaining completion gate.
7. [`doc/docs/WEEK_09_REQUIREMENTS.md`](doc/docs/WEEK_09_REQUIREMENTS.md) — final paper/reproducibility package still to build.

Do not tune any model, threshold, probe, budget, or claim using locked core-test
or held-out-shift outcomes. Treat source bits as operational behavioural
signals, not proven internal causal mechanisms.

## Current scientific story

ProActive diagnoses *why* a multimodal model may be unreliable while limiting
the number of additional model calls. A Deep Sets encoder aggregates acquired
probe evidence without imposing an arbitrary order. A learned policy chooses
the next legal probe or STOP, and frozen APS calibration returns a set of
plausible failure modes rather than one overconfident label.

The evidence supports a precise claim: **active acquisition is most useful in
the constrained-budget regime**. On the locked core test at budget 2,
ProActive reaches source-bit Macro-F1 `0.9392`, compared with `0.9200` for the
dataset-specific fixed schedule and `0.8719` for random acquisition, at about
two probes. Its diagnosis sets are also smaller (`2.61` versus `3.16` and
`3.27`).

Held-out PRE-HAL/IllusionBench shift shows the same budget-dependent pattern:

| Budget | ProActive F1 | Random F1 | Paired difference |
|---:|---:|---:|---:|
| 1 | 0.6659 | 0.6201 | +0.0458 |
| 2 | 0.8191 | 0.7002 | +0.1189 |
| 3 | 0.9184 | 0.7711 | +0.1474 |
| 4 | 0.9625 | 0.8447 | +0.1178 |
| 7 | 0.9882 | 0.9945 | -0.0063 |

The budget-7 comparison was the predeclared primary test. Random is slightly
but significantly higher there (Holm-adjusted `p=0.003`). This is a useful
saturation result, not something to hide: when almost every probe can be used,
active selection has little room to help. ProActive still uses fewer probes on
average (`5.58` versus `6.00`). The paper should lead with the frontier and
limited-budget value, not claim universal dominance at every budget.

Additional evidence sharpens the claim:

- Deep Sets is competitive with the GRU and has exactly zero measured
  permutation drift across 2,000 locked-test states.
- At held-out budget 7, ProActive substantially exceeds clean-only/scalar
  controls when Qwen is held out (`0.9692` F1) and when Gemma is held out
  (`0.9948` F1).
- Cross-model APS coverage does not transfer uniformly: the Gemma-held-out
  0.90-target coverage falls to `0.7115`. Claim diagnostic transfer, not a
  distribution-free cross-model coverage guarantee.
- The largest clear ablation loss comes from removing semantic matching
  (`-0.0413` F1 relative to the reference at budget 7). VOI training has a
  smaller positive contribution. Removing STOP can slightly raise saturated
  F1 but costs about `0.95` extra probes, supporting STOP as an efficiency
  mechanism.
- Mean controller overhead is `1.355 ms`, compared with `9163.951 ms` mean
  cached generation latency; probe inference, not controller logic, dominates
  wall time.

## Which supporting document answers which question

| Question | Document |
|---|---|
| What are the strongest paper results, ranked by wow factor? | [`wow-result.md`](wow-result.md) |
| What was completed each week? | [`WEEKLY_PROGRESS.md`](WEEKLY_PROGRESS.md) |
| What is the short professor-facing summary? | [`PROACTIVE_RESEARCH_PROGRESS_BRIEF.md`](PROACTIVE_RESEARCH_PROGRESS_BRIEF.md) |
| How does the complete system work in beginner-friendly language? | [`PROACTIVE_BEGINNER_PROJECT_EXPLAINER.md`](PROACTIVE_BEGINNER_PROJECT_EXPLAINER.md) |
| Which settings and scientific choices were approved? | [`DECISIONS.md`](DECISIONS.md) |
| Which runs produced which outputs and compute costs? | [`RUN_REGISTRY.md`](RUN_REGISTRY.md) |
| Which failures occurred and how were they resolved? | [`FAILURE_LOG.md`](FAILURE_LOG.md) and [`PROJECT_LOG.md`](PROJECT_LOG.md) |
| How are the two held-out datasets reproduced? | [`WEEK_8_DATASET_SETUP_AND_EXECUTION.md`](WEEK_8_DATASET_SETUP_AND_EXECUTION.md) |
| How was the ablation bundle executed and audited? | [`WEEK_8_ABLATION_EXECUTION.md`](WEEK_8_ABLATION_EXECUTION.md) |
| What must the annotators do? | [`human_annotation/README.md`](human_annotation/README.md) |
| What was deliberately postponed? | [`ICLR_DEFERRED_WORK_AND_EXTERNAL_SETUP.md`](ICLR_DEFERRED_WORK_AND_EXTERNAL_SETUP.md) |
| How should server environments and GPU jobs be launched? | [`SERVER_RUNBOOK.md`](SERVER_RUNBOOK.md) |
| What is still required for the paper-ready release? | [`doc/docs/WEEK_09_REQUIREMENTS.md`](doc/docs/WEEK_09_REQUIREMENTS.md) |

## Result artifact index

| Stage | Primary artifact(s) | What to learn from it |
|---|---|---|
| Week 3 probe pilot | [Pilot summary](outputs/pilot_reports/pilot_analysis_summary.json); [semantic calibration](outputs/pilot_reports/semantic_calibration_report.json); [frozen probes](configs/probes/frozen_week3_config.yaml) | Frozen probe severities and semantic-match threshold, with pilot counts and calibration metrics. |
| Week 4 supervision | [Full report](outputs/week4_reports/final/week4_full_report.json); [teacher manifest](outputs/week4_reports/final/teacher_manifest.json); [label manifest](outputs/week4_reports/final/label_manifest.json); [state manifest](outputs/week4_reports/final/state_manifest.json) | Completeness and hashes for 14,582 teacher rows/labels and 388,623 leakage-safe partial states. |
| Week 5 encoder choice | [Selection](outputs/week5_reports/week5_encoder_selection.json); [freeze](outputs/week5_reports/week5_encoder_freeze.json); [full validation](outputs/week5_reports/week5_full_validation.json) | Why Deep Sets seed 42 was selected and frozen using validation only. |
| Week 5 order study | [Deep Sets report](outputs/week5_reports/permutation_week5_deep_sets_standard_seed42.json) and the adjacent GRU/masked-slot reports | Encoder accuracy and sensitivity to evidence order. |
| Week 6 VOI corpus | [VOI manifest](outputs/week6_voi/voi_manifest.json) | Integrity and split accounting for the complete value-of-information targets. |
| Week 6 policy choice | [Selection](outputs/week6_reports/week6_policy_selection.json); [selected frontier](outputs/week6_frontiers/deep_lambda0_seed42/frontier_validation.json) | Validation-only selection of cost multiplier 0.0 and seed 42, plus its budget frontier. |
| Week 6 architecture controls | [Random-order GRU](outputs/week6_architecture_frontiers/gru_random/frontier_validation.json); [masked-slot MLP](outputs/week6_architecture_frontiers/masked_slot/frontier_validation.json) | Matched downstream comparisons for order-sensitive and fixed-slot encoders. |
| Week 6 completion | [Full validation](outputs/week6_reports/final/week6_full_validation.json) | Signed Week 6 completion gate. |
| Week 7 frozen stack | [Main freeze](outputs/week7_frozen/main_stack_freeze.json); [APS thresholds](outputs/week7_calibration/aps_thresholds.json) | Exact diagnostic/policy/checkpoint/calibration boundary used for test and shift. |
| Week 7 locked core test | [JSON](outputs/week7_reports/frontier_test.json); [CSV](outputs/week7_reports/frontier_test.csv); [figure](outputs/week7_reports/frontier_test.png) | Main in-distribution cost–quality–coverage frontier and all controls/oracles. |
| Week 7 order robustness | [Deep Sets report](outputs/week7_reports/permutation_week7_deep_sets_standard_seed42.json) and adjacent GRU reports | Locked-test permutation stability. |
| Week 7 completion | [Full validation](outputs/week7_reports/week7_full_validation.json); [GO/NO-GO memo](outputs/week7_reports/week7_go_no_go.md) | Signed `GO` and frozen limitations. |
| Week 8 release audit | [Setup report](outputs/week8_data/dataset_setup_report.json); [manifest bundle](outputs/week8_data/manifests/heldout_manifest_bundle.json) | Pinned PRE-HAL/IllusionBench releases, deterministic sampling, and uniform exclusions. |
| Week 8 shift substrate | [Vector manifest](outputs/week8_vectorized/vectorized_manifest.json); [trajectories](outputs/week8_reports/shift/trajectories_shift.jsonl) | Hash-bound held-out tensors and 340,800 frozen-policy trajectory rows. |
| Week 8 shift frontier | [JSON](outputs/week8_reports/shift/frontier_shift.json); [CSV](outputs/week8_reports/shift/frontier_shift.csv); [figure](outputs/week8_reports/shift/frontier_shift.png) | Empirical held-out performance and coverage; no target-domain calibration. |
| Week 8 LOMO | [Qwen held out](outputs/week8_reports/lomo/qwen3_vl_8b/lomo.json); [Gemma held out](outputs/week8_reports/lomo/gemma4_e4b/lomo.json) | Transfer to each unseen model using source-only fitting and calibration. |
| Week 8 latency | [Report](outputs/week8_reports/latency_report.json); [measurements](outputs/week8_reports/latency.csv) | Fixed-A6000 controller and cached-generation timing. |
| Week 8 ablations | [Aggregate](outputs/week8_reports/ablations.json); [table](outputs/week8_reports/ablations.csv); [per-ablation evidence](outputs/week8_reports/ablation_evidence/) | Signed 15-item component study, 121 aggregate rows, and per-ablation provenance. |
| Week 8 uncertainty | [Confidence intervals](outputs/week8_reports/confidence_intervals.csv); [paired tests](outputs/week8_reports/paired_comparisons.csv) | Grouped-bootstrap intervals and paired tests; use `primary_comparison=True` for the predeclared family. |
| Week 8 slices/cases | [Slices](outputs/week8_reports/slices.csv); [qualitative cases](outputs/week8_reports/qualitative_examples.json) | Dataset/model breakdowns and deterministic positive/negative examples. |
| Week 8 analysis manifest | [Signed report](outputs/week8_reports/week8_analysis.json); [full log](outputs/logs/week8/week8_analysis_full.log) | Hashes, statistical protocol, trajectory count, and successful execution record. |
| Human audit | [Prepared packets](outputs/human_annotation_packets/); final target `outputs/week8_reports/human_audit/human_audit_summary.json` | Three independent judgements over 180 blinded cases; final file is not yet present. |

The 13 September audit found all three packet CSVs present with 180 rows each,
but all 1,080 required annotation cells per annotator are still blank.

## How to independently analyze the results

1. Verify each JSON self-hash and every referenced file hash before reading a
   metric. Use `RUN_REGISTRY.md` to connect commands, approvals, and outputs.
2. Use `frontier_test.csv` for locked in-distribution claims and
   `frontier_shift.csv` for held-out empirical claims. Never merge the two.
3. Use `confidence_intervals.csv` for uncertainty and
   `paired_comparisons.csv` for matched method differences. The bootstrap unit
   is `group_id`; do not treat 340,800 trajectory rows as independent samples.
4. Inspect `slices.csv` before making a global robustness claim and use
   `qualitative_examples.json` only for explanation, not selection.
5. Read individual `ablation_evidence/*.json` when interpreting a component;
   `ablations.csv` is the compact comparison table.
6. Preserve negative results: full-budget random saturation and Gemma LOMO
   undercoverage are part of the defensible story.

## What remains

There is no further experiment that must finish before writing begins. Draft
the paper now. The items below are required before calling the evidence and
submission package final.

| Item | Compute | Blocking? |
|---|---:|---|
| Three annotators independently complete all 180 blinded rows | Human only | Yes |
| Merge, blinded adjudication, and agreement/rule-match analysis | CPU, minutes | Yes |
| Run `validate_week8.py --mode full` and archive its signed report | CPU, seconds/minutes | Yes, after audit |
| Build one-command paper tables/figures and trace every number to a CSV/JSON | CPU/engineering | Yes for submission quality |
| Write `repro_manifest.json`; verify hashes, seeds, splits, leakage, and calibration | CPU/review | Yes for reproducibility |
| Archive logs/decisions and finish appendix, limitations, risk memo, and release checklist | Manual/CPU | Yes for submission quality |
| Decide whether to rerun the main fixed-seed small networks or formally record a deadline-driven waiver using the already signed runs | GPU or documented decision | Required plan decision |
| Optional GQA-Relation, LODO, or full InternVL corpus | High/variable | No; keep deferred under the deadline |

Paper writing should proceed now in parallel. Introduction, related work,
method, core experiments, shift/LOMO/ablation results, and limitations no
longer depend on a GPU run. The human-audit paragraph and final completion
statement are the only scientific evidence sections that must wait; Week 9 is
packaging and review, not a new modeling phase.
