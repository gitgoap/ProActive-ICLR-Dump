# ProActive Wow Results

**Snapshot:** 15 September 2026  
**Purpose:** ranked, evidence-linked results for framing the ICLR paper.  
**Status:** Weeks 1--7 are complete. The mandatory Week 8 machine experiments
and statistical analyses are complete, but the three-person human audit and
final Week 8 validator are still pending. This is therefore a paper-story
brief, not final claim sign-off.

## The paper in one sentence

**Multimodal failure diagnosis is a budgeted active-measurement problem: with
only a few additional model calls, ProActive selects useful behavioural probes,
combines their outcomes as an unordered evidence set, and returns a compact
set of plausible failure modes; its advantage is largest when probes are
scarce and naturally disappears near full-information saturation.**

## How to read the numbers

- **Source-bit Macro-F1** is balanced performance on the three operational
  behavioural indicators: visual fragility, language-prior persistence, and
  grounding/alignment instability. These are not claimed to be internal causal
  mechanisms.
- **Six-way Macro-F1** is balanced performance on the final six diagnostic
  families.
- **Coverage** is the fraction of examples whose target family is contained in
  the returned diagnosis set. **Smaller set size** means a more specific answer.
- **Cost** is the mean number of additional MLLM forward passes.
- Locked core-test numbers, held-out-shift numbers, LOMO numbers, and
  validation-only ablations are kept separate below.

## Results ranked by paper-level wow factor

### 1. The constrained-budget advantage survives held-out diagnostic dataset shift

The frozen Week 7 stack was evaluated without target-domain calibration on
PRE-HAL and IllusionBench, which were held out from ProActive training,
selection, and calibration: 1,200 examples, two MLLMs, and 2,400 model--example
rows. Against random acquisition, ProActive improves source-bit Macro-F1 at
every constrained budget:

| Budget | ProActive | Random | ProActive minus random | Six-way gain | Set-size change |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6659 | 0.6201 | **+4.58 pp** | +5.45 pp | -0.38 |
| 2 | 0.8191 | 0.7002 | **+11.89 pp** | +8.91 pp | -0.38 |
| 3 | 0.9184 | 0.7711 | **+14.73 pp** | **+15.15 pp** | -0.45 |
| 4 | 0.9625 | 0.8447 | **+11.78 pp** | **+14.49 pp** | -0.17 |
| 7 | 0.9882 | 0.9945 | -0.63 pp | -2.41 pp | +0.03 |

At the strongest point, budget 3, the mean costs are effectively matched
(2.987 versus 3.000 probes). The grouped-bootstrap 95% interval for the
source-bit gain is **[+13.16, +16.36] percentage points**. The gain is positive
on both held-out datasets and both models: +13.88 pp on IllusionBench, +15.68
pp on PRE-HAL, +9.11 pp on Qwen, and +22.00 pp on Gemma.

This produces a clean scientific shape rather than a universal-winner claim:
active selection matters when only 1--4 probes are affordable. At budget 7,
random is slightly better by 0.63 pp (the predeclared primary comparison,
Holm-adjusted `p=0.003`), while ProActive still uses 0.42 fewer probes. The
budget-1--4 comparisons are strong secondary/exploratory evidence; they were
not members of the predeclared Holm family.

**Reviewer-safe claim:** ProActive generalizes its advantage over random
acquisition to diagnostic datasets held out from ProActive in the
constrained-budget regime. Public-dataset presence in the base MLLMs' pretraining
is unknown, and the result is not dominance over every shift baseline.

**Evidence:** [shift frontier](outputs/week8_reports/shift/frontier_shift.csv),
[paired comparisons](outputs/week8_reports/paired_comparisons.csv),
[slices](outputs/week8_reports/slices.csv), and
[signed analysis manifest](outputs/week8_reports/week8_analysis.json).

### 2. Exact order invariance is obtained for essentially zero diagnostic-performance cost

The representation result is unusually clean. Mean Week 5 validation
source-bit Macro-F1 is 0.864217 for Deep Sets and 0.864333 for the best GRU--a
difference of only **0.0116 percentage points**--while Deep Sets is 1.33 points
better on six-way Macro-F1. This is a practical quality tie.

The mandatory masked-slot MLP is also exactly invariant in its Week 5 audits.
Across three seeds it reaches 0.864605 source-bit Macro-F1, only 0.0388 points
above Deep Sets, while Deep Sets is 0.895 points higher on six-way Macro-F1.
This control strengthens the case for invariant evidence modeling, but means
Deep Sets should not be presented as the only invariant or universally best
architecture.

On 2,000 locked-test states, Deep Sets has exactly zero measured drift in its
hidden state, bit probabilities, six-way probabilities, prediction sets, and
calibration outcomes; action agreement is exactly 1.0. In contrast:

| Encoder | Mean bit drift | Mean hidden drift | Mean diagnosis-set disagreement | Worst set disagreement |
|---|---:|---:|---:|---:|
| **Deep Sets** | **0** | **0** | **0%** | **0%** |
| Canonical GRU | 0.0278 | 0.0751 | 12.67% | 66.67% |
| Permutation-trained GRU | 0.0109 | 0.0417 | 15.20% | 100% |

Permutation augmentation reduces some internal GRU drift but does not provide
structural invariance. This is exactly the intended architecture story: the
probes are independently applied to the same image--question pair, so their
accumulated outcomes are a set rather than a semantically ordered sequence.

**Reviewer-safe claim:** Deep Sets matches the GRU's diagnostic quality while
eliminating an artificial order-dependence and remains competitive with the
exact-invariant masked-slot control. Do not claim that it is universally more
powerful than either comparator.

**Evidence:** [encoder selection](outputs/week5_reports/week5_encoder_selection.json),
[Deep Sets locked permutation](outputs/week7_reports/permutation_week7_deep_sets_standard_seed42.json),
[canonical-GRU permutation](outputs/week7_reports/permutation_week7_gru_canonical_seed42.json),
[permutation-trained GRU](outputs/week7_reports/permutation_week7_gru_random_permutation_seed42.json),
and [masked-slot validation audits](outputs/week5_checkpoints/).

### 3. On the locked core test, two selected probes give a much sharper diagnosis than random or dataset-specific acquisition

At the nominal 90% coverage target on 1,560 locked model--example pairs, the
budget-2 comparison is:

| Method | Mean probes | Source-bit F1 | Six-way F1 | Coverage | Mean set size |
|---|---:|---:|---:|---:|---:|
| **ProActive** | 1.997 | **0.9392** | **0.6841** | 0.8962 | **2.61** |
| Dataset-specific fixed | 2.000 | 0.9200 | 0.5669 | 0.9038 | 3.16 |
| Random | 2.000 | 0.8719 | 0.5257 | 0.9288 | 3.27 |

At effectively equal cost, ProActive gains **+6.74 points** of source-bit F1
and **+15.84 points** of six-way F1 over random, while returning diagnosis sets
that are **20.4% smaller**. Against the dataset-specific fixed schedule, the
gains are +1.92 and +11.72 points, with **17.6% smaller** sets.

A strong blank-first heuristic reaches 0.9414 source-bit F1 at budget 2, only
0.22 points above ProActive, but ProActive is +6.38 points better on six-way F1
and returns sets smaller by 0.24 labels. This is evidence for a better
quality--specificity package, not raw-metric dominance over every heuristic.

**Evidence:** [locked core frontier CSV](outputs/week7_reports/frontier_test.csv)
and [signed frontier report](outputs/week7_reports/frontier_test.json).

### 4. ProActive recovers 99.84% of full-teacher F1 while saving 15.2% of the probe calls

On the locked core test at maximum budget 7, ProActive uses 5.099 probes rather
than the full teacher's 6.013. Its source-bit Macro-F1 is 0.9942 versus 0.9958:
only 0.159 percentage points lower, or **99.84% of the full-teacher score**, at
**15.2% fewer forward passes**. Coverage is 0.9994 versus 0.9981.

The same saturation-efficiency pattern appears in LOMO: at budget 7 ProActive
uses 18.3% fewer probes in the Qwen-held-out fold and 24.2% fewer in the
Gemma-held-out fold, with source-bit F1 gaps of only -0.86 and -0.07 points
relative to random/full acquisition.

**Reviewer-safe claim:** STOP and selective acquisition recover nearly all
full-information diagnostic quality with fewer calls. Do not call the small
remaining quality gaps improvements.

**Evidence:** [core frontier](outputs/week7_reports/frontier_test.csv),
[Qwen-held-out LOMO](outputs/week8_reports/lomo/qwen3_vl_8b/lomo.csv), and
[Gemma-held-out LOMO](outputs/week8_reports/lomo/gemma4_e4b/lomo.csv).

### 5. Diagnostic quality transfers when the target MLLM is held out

Each of the two LOMO folds retrains and calibrates only on the other model, then
evaluates 780 examples from the MLLM held out from diagnostic training. Against random acquisition, ProActive's
source-bit F1 gain at budget 3 is **+9.55 points for held-out Qwen** and **+6.78
points for held-out Gemma**. The gains remain positive at budgets 1--4 in both
folds.

At budget 7, ProActive reaches 0.9692 F1 on held-out Qwen and 0.9948 on held-out
Gemma, far above their clean-only controls (0.6826 and 0.7956) and scalar
confidence controls (0.6964 and 0.7804). This satisfies the predeclared
diagnostic-transfer gate.

The calibration result is intentionally narrower: 90%-target empirical
coverage is 0.9910 for Qwen but only 0.7115 for Gemma at budget 7. The wow is
**diagnostic transfer**, not universal cross-model conformal coverage.

**Evidence:** [Qwen LOMO report](outputs/week8_reports/lomo/qwen3_vl_8b/lomo.json),
[Gemma LOMO report](outputs/week8_reports/lomo/gemma4_e4b/lomo.json), and
[Week 8 requirements audit](doc/docs/WEEK_08_REQUIREMENTS.md).

### 6. The confirmatory held-out analysis shows that evaluated one-pass controls miss substantial diagnostic signal

In the predeclared budget-7 primary family, ProActive exceeds clean-only by
**42.25 F1 points** and scalar confidence by **46.13 points**. Both grouped
comparisons have Holm-adjusted `p=0.003`. This does not isolate acquisition
strategy--the controls use zero extra probes--but it strongly validates the
premise that behavioural measurements capture operational full-probe teacher
signals not captured by the evaluated clean-only and scalar controls. It is
not an information-theoretic claim that clean responses can never contain the
same information.

The validation-stage active-signal gate tells the same story: the full teacher
has mean source-bit F1 0.9970 versus 0.7666 without acquisition, a 30.1%
relative gain.

**Evidence:** [paired held-out tests](outputs/week8_reports/paired_comparisons.csv)
and [Week 6 selection](outputs/week6_reports/week6_policy_selection.json).

### 7. Probe evidence clears the predeclared identity-shortcut gate by 9.77 points

The Week 5 shortcut audit exposes a real confounding risk and then tests it
directly. Mean Deep Sets validation source-bit Macro-F1 is 0.864217, compared
with 0.766506 for a control using dataset/model identity plus clean-response
features: a **+9.77-point gap**, far above the predeclared +2-point gate. The
main learner itself receives neither dataset identity nor model identity.

This does not prove that every possible confound is absent, and it should not
be misreported as a comparison against the privileged identity-plus-probe
control. It does show that the main result is not reducible to memorizing which
dataset or model produced a clean response.

**Evidence:** [encoder selection and shortcut gate](outputs/week5_reports/week5_encoder_selection.json)
and [full shortcut-control audit](outputs/week5_reports/shortcut_controls.json).

### 8. Semantic matching has the largest consistent measured component effect

Among the 15 mandatory, validation-only ablations, removing semantic matching
causes the largest consistent diagnostic loss. At budget 2 and effectively
equal cost, source-bit F1 falls from 0.9499 to 0.9133 (**-3.67 points**). At
budget 7 it falls from 0.9948 to 0.9536 (**-4.13 points**), and six-way F1 falls
by 2.66 points. The loss is present at every evaluated budget.

This is a useful mechanism result: free-form answer equivalence is not a minor
preprocessing detail; it materially determines whether behavioural changes are
recognized correctly.

The VOI contribution is smaller and more mixed: removing VOI training lowers
source-bit F1 by 1.31 points at budget 2 and 0.65 points at budget 7, but also
changes cost and six-way performance. This supports a modest VOI contribution,
not universal superiority of every VOI objective.

**Evidence:** [ablation table](outputs/week8_reports/ablations.csv) and
[semantic-match evidence](outputs/week8_reports/ablation_evidence/no_semantic_match.json),
plus the [no-VOI control](outputs/week8_reports/ablation_evidence/no_voi_training.json).

### 9. The frozen stopped pipeline is approximately calibrated on the grouped in-distribution test

At a nominal 90% target, ProActive's locked-test coverage is 0.8962, 0.8987,
and 0.9032 at budgets 2, 3, and 4--all within 0.4 percentage points of target.
Budget 1 is 0.8808, still within the predeclared 3-point tolerance. This is
important because calibration was performed only after the diagnostic encoder,
policy, STOP rule, and budget schedule were frozen.

At the 95% target, budget-1 coverage is 0.9526; larger budgets are conservative
(0.998--1.000), so the paper should report both coverage and set size rather
than presenting coverage alone.

**Reviewer-safe claim:** approximately nominal **marginal in-distribution**
coverage for the frozen stopped pipeline. Shift and LOMO coverage are empirical
stress results, not guarantees.

**Evidence:** [core frontier](outputs/week7_reports/frontier_test.csv),
[main-stack freeze](outputs/week7_frozen/main_stack_freeze.json), and
[Week 7 GO memo](outputs/week7_reports/week7_go_no_go.md).

### 10. The learned controller is not a single disguised global probe order

A direct audit of the immutable held-out trajectory file finds that the
budget-7 controller produces **189 distinct probe sequences** over 2,400
model--example rows. Its observed first action varies among blank, grounding,
and blur, and 714/2,400 cases (**29.75%**) stop before exhausting all six legal
probes. At budget 1 it chooses four distinct first-probe types. The resulting
mean budget-7 cost is 5.577 rather than 6.000.

This derived audit is reproducible by grouping `condition=proactive`,
`target_coverage=0.9` rows by the `actions` field. It should support a policy
behaviour figure or appendix table, not replace the primary frontier.
It rules out one global fixed order; sequence diversity alone does not prove
fine-grained state dependence rather than dependence on broad subpopulations.

**Evidence:** [hash-bound held-out trajectories](outputs/week8_reports/shift/trajectories_shift.jsonl)
and their [signed analysis manifest](outputs/week8_reports/week8_analysis.json).

### 11. STOP buys meaningful efficiency for almost no saturated-quality loss

In the validation-only STOP ablation at budget 7, the full controller uses
5.061 probes versus 6.011 without STOP: **0.95 fewer calls, or 15.8% fewer
than no STOP**. The no-STOP model is only 0.22 points higher in source-bit F1
and 0.90 points higher in six-way F1. This cleanly supports STOP as an
efficiency mechanism rather than an accuracy booster.

**Evidence:** [STOP ablation](outputs/week8_reports/ablation_evidence/no_stop_action.json)
and [aggregate ablations](outputs/week8_reports/ablations.csv).

### 12. Controller overhead is negligible relative to multimodal generation

With 10 warmups and 100 RTX A6000 measurements, one complete controller
decision--diagnostic encoder plus action-conditioned VOI head--takes **1.355
ms** on average. The matched cached records contain **9,163.951 ms** of mean
clean-plus-acquired MLLM generation time per budget-7 example. The sampled
trajectories average 5.68 acquired probes plus one STOP decision; multiplying
the per-decision measurement by 6.68 gives an estimated **9.05 ms** of total
controller compute, about **0.10%** of the recorded generation time. This is
an explicit approximate accounting because controller and cached generation
times were measured separately, not one end-to-end wall-clock trace.

**Evidence:** [latency report](outputs/week8_reports/latency_report.json) and
[per-example measurements](outputs/week8_reports/latency.csv).

### 13. The framework exposes a smooth, usable cost--quality control knob

Across three validation seeds, increasing the cost multiplier from 0.0 to 0.05
reduces mean acquisition cost from 2.285 to 1.711 (**25.1% fewer probes**) while
source-bit Macro-F1 moves only from 0.9426 to 0.9373 (**-0.53 points**). Larger
multipliers continue reducing cost with progressively larger quality loss.

The frozen main result uses multiplier 0.0 because the predeclared selection
rule prioritized F1. The other settings are useful frontier evidence, not
post-test alternatives.

**Evidence:** [three-seed policy selection](outputs/week6_reports/week6_policy_selection.json)
and the linked [validation frontiers](outputs/week6_frontiers/).

### 14. The machine-result chain is broad and audit-friendly

The headline numbers sit on top of:

- 10,400 deduplicated pilot records across four core datasets and two models;
- 14,582 complete teacher rows, 87,712 unique probe observations, and 388,623
  leakage-safe partial states;
- 496,912 value-of-information targets;
- 1,560 locked core-test model--example pairs;
- 2,400 held-out model--example pairs and 340,800 frozen-policy trajectory rows;
- two complete 780-example LOMO folds; and
- all 15 mandatory ablations, producing 121 aggregate comparison rows.

Core and held-out recovery ledgers finish with no unresolved requested rows.
Manifests, frozen configurations, checkpoints, reports, and CSVs are connected
by hashes. Scale is not itself a scientific contribution, but it makes the
result much harder to dismiss as a small pilot or hand-picked example.

**Evidence:** [pilot summary](outputs/pilot_reports/pilot_analysis_summary.json),
[Week 4 full report](outputs/week4_reports/final/week4_full_report.json),
[Week 5 validation](outputs/week5_reports/week5_full_validation.json),
[Week 6 validation](outputs/week6_reports/final/week6_full_validation.json),
[Week 8 analysis](outputs/week8_reports/week8_analysis.json), and
[ablation aggregate](outputs/week8_reports/ablations.json).

### 15. The pilot already showed heterogeneous probe trigger rates

On 800 canonical pilot records, answer-flip rates range from 16.5% for
brightness and 21.4% for noise to 48.5% for blur and 57.8% for blank-image.
The relation probe flips 61.7% of its 60 applicable cases. Different probes
therefore have substantially different empirical trigger rates--the premise
that makes selective acquisition plausible. Flip-rate differences alone do
not establish that the probes identify distinct causal mechanisms.

This is supporting motivation, not a standalone generalization claim.

**Evidence:** [Week 3 pilot analysis](outputs/pilot_reports/pilot_analysis_summary.json)
and [frozen probe settings](configs/probes/frozen_week3_config.yaml).

## The strongest coherent paper story

1. **Problem:** diagnosing *why* an MLLM response is unreliable is not a
   one-pass classification problem; it is an active measurement problem.
2. **Representation:** independently acquired probe outcomes form a set. Deep
   Sets removes arbitrary-order drift without sacrificing meaningful accuracy.
3. **Main result:** on the locked core test, selected probes produce more
   accurate and more specific diagnoses than random and the dataset-specific
   schedule at constrained budgets.
4. **Robustness result:** the same large low-budget gain over random appears on
   two datasets held out from ProActive and in both leave-one-model-out folds.
5. **Saturation result:** when almost every probe is available, the advantage
   disappears while STOP retains some cost saving. This is the expected limit
   of active acquisition and defines the claim precisely.
6. **Trustworthiness:** in-distribution APS is approximately calibrated;
   ablations identify semantic matching and STOP as important; controller
   overhead is negligible; shift/LOMO calibration failures are reported rather
   than hidden.

## Reviewer traps that must remain explicit

| Tempting overclaim | What the evidence actually says |
|---|---|
| “The held-out datasets were necessarily unseen by the base MLLMs.” | Unknown. They were held out from all ProActive training, selection, and calibration; their presence in base-model pretraining cannot be established here. |
| “ProActive beats every baseline or metric under shift.” | No. Blank-first exceeds ProActive source-bit F1 at budgets 1--4, and uncertainty-greedy is slightly higher at budgets 1 and 2. The large robust shift result is specifically against random acquisition. |
| “ProActive wins at every budget.” | No. At held-out budget 7, random/full acquisition is 0.63 F1 points better, significantly, while using 0.42 more probes. |
| “Conformal coverage transfers across models.” | No. Gemma-held-out empirical coverage falls to 0.7115 at a 0.90 target. Claim diagnostic transfer, not universal coverage. |
| “Deep Sets is more accurate than GRU.” | Not materially. The validation F1 difference is 0.0116 points in the GRU's favor. The contribution is exact invariance at comparable quality. |
| “Deep Sets is the only successful invariant encoder.” | No. Masked-slot MLP is also exactly invariant and competitive. Deep Sets is the predeclared variable-cardinality set encoder, not a universally dominant architecture. |
| “The composite VOI objective is universally superior.” | No. The validation-only loss-only VOI variant has slightly higher F1 and lower cost at several budgets, although its diagnosis sets are consistently larger. |
| “The learned policy reaches the oracle ceiling.” | No. Oracle-best-subset results retain constrained-budget source-F1 headroom, although they do not dominate coverage and are not an implementable per-example policy. |
| “The final results establish broad seed robustness.” | Not yet. Week 5 and Week 6 selection use three seeds, but the frozen main stack, LOMO, and mandatory ablations use seed 42. |
| “The labels are independent causal ground truth.” | No. They are operational targets produced from the full behavioural-probe teacher. The pending three-person audit provides an independent validity check, not causal identification. |
| “The main paper evaluates three MLLMs end to end.” | No. The complete primary corpus and transfer evidence cover Qwen and Gemma. InternVL has staged and complete-VSR catch-up evidence, not a full third-model main result. |
| “APS efficiency work is fully exhausted.” | No. The optional RAPS appendix gate triggered at mean validation set size 3.2087, but RAPS remains deferred. Locked APS must not be replaced using test outcomes. |
| “The indicators reveal true internal causes.” | No. They are operational behavioural failure signals defined by controlled probes. |
| “Week 8 is complete.” | Not yet. Machine evidence is complete, but the three-person human audit and final full validator are still required. |

## Recommended main-paper evidence order

1. One two-panel frontier figure: locked core and held-out shift, budgets 1--7.
2. One permutation-stability figure: Deep Sets at zero versus the two GRUs.
3. One compact table: budget-2 core quality, coverage, set size, and cost.
4. One LOMO table that shows both diagnostic transfer and the Gemma coverage
   failure.
5. One ablation/latency table led by semantic matching, STOP, and controller
   overhead.

The human audit should be added as an independent validity result when all
three packets are complete. Until then, the strongest accurate status is:
**machine evidence complete; final Week 8 validation pending human evidence.**
