# ProActive: Beginner-Friendly Project Explainer, Progress, Errors, and 9-Day ICLR Sprint

**Meeting guide for:** Professor discussion and ICLR planning  
**Last updated:** 13 September 2026
**Current project stage:** Week 8 external-shift, generalization, ablation, latency, and human-audit validation
**Submission horizon:** Final submission sprint

This document explains ProActive from the beginning and records the detailed
history. For the current result map, claim boundary, and direct links to signed
artifacts, start with `PROACTIVE_RESEARCH_HANDOFF_AND_RESULTS_INDEX.md`. The
filename is retained so existing links do not break.

## If you remember only ten things

1. ProActive diagnoses *why* a multimodal language model may be unreliable, rather than producing only one uncertainty score.
2. It actively chooses a small number of diagnostic image probes, similar to a doctor ordering useful tests instead of ordering every possible test.
3. The probes test visual fragility, language-prior dependence, and image-text grounding/relation instability.
4. Teacher multimodal models generated the supervision; Qwen3-VL-8B and Gemma-4-E4B produced 14,582 complete core teacher rows.
5. The project uses strict train/validation/calibration/test separation to prevent leakage.
6. Week 5 selected Deep Sets as the diagnostic encoder because it was accurate and exactly invariant to probe order.
7. Week 6 learns which probe to request next, or when to stop, using value-of-information targets.
8. Week 7 froze the system, calibrated diagnosis sets, and completed the one-time locked test evaluation without post-test tuning.
9. The repeated errors were mostly integrity checks, malformed model outputs, environment mismatches, GPU contention, or stale/incomplete synced artifacts—not evidence that the research idea failed.
10. Weeks 1–7 are complete. All mandatory Week 8 machine experiments and
    statistics are complete. The only unfinished evidence is the three-person
    human audit, followed by the CPU-only full validator and paper artifact
    freeze.

## The 30-second explanation

Modern vision-language models can answer incorrectly for different reasons. Sometimes the image is degraded, sometimes the model relies on language stereotypes instead of the image, and sometimes it fails to connect the words in the question to the correct visual objects or relations. Ordinary confidence scores usually do not tell these cases apart.

ProActive treats diagnosis as a sequential decision problem. It observes the model's clean answer, applies carefully designed probes to the image or prompt, watches how the answer changes, and chooses the next most useful probe. It can stop early when additional probes are not worth their cost. Finally, it returns a calibrated *set* of plausible failure diagnoses instead of pretending that one label is always certain.

## The beginner analogy: a doctor ordering tests

Imagine a patient with an unclear symptom. A poor strategy is to order every medical test. Another poor strategy is to report only “confidence: 62%.” A better doctor forms several possible explanations, orders the test most likely to distinguish them, updates the diagnosis, stops when another test adds little value, and reports the remaining plausible diagnoses honestly.

| Medical diagnosis | ProActive |
|---|---|
| Patient | One image-question instance evaluated by a multimodal model |
| Symptom | A suspicious, wrong, or unstable model answer |
| Diagnostic test | A controlled image or prompt probe |
| Test cost | Extra model inference and possible image degradation |
| Belief update | Diagnostic encoder updates failure probabilities |
| Stop decision | Policy decides no further probe is worth its cost |
| Differential diagnosis | Calibrated set of plausible failure types |

## What changed from HalluPrism to ProActive

The earlier direction mainly studied multimodal uncertainty. ProActive turns that foundation into an active diagnostic system.

| Earlier emphasis | ProActive emphasis |
|---|---|
| “Is this answer uncertain?” | “What kind of failure is likely, and what evidence should we request next?” |
| One-pass score | Sequential evidence collection |
| Fixed evaluation | Learned probe selection with a STOP action |
| Single prediction | Calibrated set of plausible diagnoses |
| Accuracy alone | Accuracy, calibration, probe cost, order invariance, and leakage controls |

## Core vocabulary

| Term | Plain meaning |
|---|---|
| **Multimodal LLM / VLM** | A model that receives images and text and generates a textual answer. |
| **Instance** | One image, one question or claim, and the associated metadata. |
| **Clean answer** | The model's answer before applying a diagnostic probe. |
| **Probe** | A controlled intervention such as blur, crop, brightness, noise, blank image, or grounding/relation prompt. |
| **Teacher cache** | Stored model answers and features produced once so later experiments do not repeatedly run expensive VLM inference. |
| **Source bit** | A binary indicator for visual, language-prior, or grounding/alignment evidence. These are operational diagnostic signals, not proven independent causal mechanisms. |
| **Partial state** | The evidence available after only some probes have been observed. |
| **Diagnostic encoder** | A small neural model that converts observed probe evidence into failure probabilities. |
| **Deep Sets** | An encoder designed for unordered sets, so changing probe order should not change its result. |
| **GRU** | An order-sensitive sequence encoder used as a control in the Week 5 comparison. |
| **VOI** | Value of information: expected diagnostic improvement minus probe cost. |
| **Policy** | The model that chooses the next legal probe or STOP. |
| **APS** | Adaptive Prediction Sets, a conformal method used to return a calibrated set of diagnoses. |
| **Coverage** | How often the true diagnosis is included in the returned set. |
| **Permutation test** | Reordering the same evidence to verify that an invariant model does not change its prediction. |

## What exactly is being diagnosed?

ProActive first represents evidence using three source bits:

- **Visual fragility (`bV`)**: the answer changes or fails under controlled visual changes.
- **Language-prior persistence (`bL`)**: the model appears to preserve a language-driven answer even when visual support is removed or contradicted.
- **Grounding/alignment instability (`bA`)**: the model struggles to connect the question to the correct object, attribute, or relation in the image. Here “alignment” means image-text grounding, not AI safety alignment.

These bits are mapped into six reporting classes:

| Class | Interpretation |
|---|---|
| Visual | Evidence mainly points to visual fragility. |
| Language-prior | Evidence mainly points to language-driven behavior. |
| Alignment | Evidence mainly points to grounding/relation failure. |
| Mixed | Multiple mechanisms appear together. |
| Unclear | Evidence is insufficient or internally ambiguous. |
| No-failure | No meaningful failure signal is detected. |

The system also preserves a continuous diagnostic signature rather than relying only on a hard class. This lets downstream analysis express uncertainty and mixed evidence.

## End-to-end pipeline

1. **Build grouped dataset manifests.** Normalize HallusionBench, POPE, VizWiz, and VSR into a common schema while keeping related examples in the same split.
2. **Run clean inference.** Record each teacher model's unmodified answer.
3. **Apply legal probes.** Generate controlled image and grounding observations using a frozen probe specification.
4. **Build teacher labels.** Convert observed changes into source-bit targets, six-way labels, and continuous features.
5. **Enumerate partial states.** Create examples representing what is known after only a subset of probes.
6. **Train diagnostic encoders.** Compare Deep Sets with an order-sensitive GRU and check permutation behavior.
7. **Construct VOI targets.** Measure how much each possible next probe would improve diagnosis after accounting for cost.
8. **Train active policies.** Learn which legal probe to request next, or choose STOP.
9. **Build validation frontiers.** Compare ProActive with random, fixed-order, uncertainty-only, scalar, distilled, and oracle controls across budgets and costs.
10. **Freeze, calibrate, and test.** Use calibration data for final APS thresholds and evaluate the locked test set once.

### How value of information works

For every partial state and legal next probe, Week 6 asks: “If I pay for this probe, how much should diagnosis improve?” The implemented target combines:

`VOI = reduction in prediction-set size + 0.25 × reduction in diagnostic loss − probe cost`

The policy chooses the legal probe with the highest predicted VOI. If every remaining probe has non-positive value, it chooses **STOP**. This is what makes ProActive budget-aware rather than merely a classifier over all cached evidence.

### Why APS returns a set

A model can have genuinely mixed or ambiguous failure evidence. Returning only one label would hide that uncertainty. APS converts diagnostic scores into a set such as `{visual, mixed}` and uses held-out calibration data to target a chosen coverage level. The desired scientific behavior is: higher coverage usually creates larger sets, while better evidence allows those sets to shrink without undercoverage.

## The split firewall: why four data splits exist

| Split | Allowed use | Forbidden use |
|---|---|---|
| Train | Fit neural weights and train-only baselines | Final model selection or claimed test result |
| Validation | Select architecture, cost, seed, temperature, and provisional settings | Final conformal calibration |
| Calibration | Compute final APS/conformal thresholds after the stack is frozen | Retraining or tuning the system |
| Test | One locked final evaluation | Repeated debugging, tuning, or threshold selection |

This firewall is central to the paper. A strong-looking test number is not credible if test data influenced model or threshold choices.

## Week-by-week progress

### Week 1 — specification, contracts, and skeleton: complete

**Question addressed:** What exactly are we building, and how will we know that the evidence is scientifically valid?

Work completed:

- Defined the three source bits, the six reporting classes, diagnostic signatures, probe families, STOP action, and conformal output.
- Created canonical schemas, experiment/configuration structure, provenance fields, audit rules, and failure-ledger behavior.
- Established the split firewall and grouped-splitting requirements before running experiments.
- Added a dry-run-first compute policy so large GPU jobs require explicit scope and authorization.

**Why this matters:** Without a written contract, labels and metrics can quietly change while experiments are running. Week 1 made later results auditable.

**Professor sentence:** “Week 1 converted the research idea into frozen data, label, evaluation, and compute contracts.”

### Week 2 — data manifests and clean inference foundations: complete

**Question addressed:** Can all datasets and models be represented consistently and reproducibly?

Work completed:

- Implemented loaders and normalized manifests for HallusionBench, POPE, VizWiz, and VSR.
- Built group-aware train/validation/calibration/test splits to avoid related-example leakage.
- Added model adapters and clean-answer caching for Qwen, Gemma, and optional InternVL catch-up.
- Added hashes, model-revision inspection, resume safety, schema validation, and deterministic sampling.

**Why this matters:** Every later experiment depends on stable instance identities and reproducible clean answers.

**Professor sentence:** “Week 2 built the common data and model interface that all later experiments reuse.”

### Week 3 — probe pilot and frozen probe specification: complete

**Question addressed:** Which probes and severity levels are informative without destroying the task?

Work completed:

- Ran canonical pilots across two core models and four datasets.
- Produced **800 canonical records** and **9,600 compact severity-grid records**, for **10,400 valid pilot records** in total.
- Confirmed no duplicate records and no schema failures in the final pilot artifacts.
- Materialized and checked 250 transformed images.
- Human-labeled 50 semantic answer pairs: 17 matches and 33 non-matches.
- Calibrated semantic matching at threshold **0.50**, with precision **0.5926**, recall **0.9412**, and F1 **0.7273**. Recall was prioritized so semantically equivalent answers were less likely to be treated as changes.
- Froze the main image severities: blur `8`, crop `0.65`, brightness `0.15`, and noise `25`.

**Why this matters:** A probe must create useful evidence, not simply make every image impossible. The pilot fixed a defensible operating point before expensive teacher generation.

**Professor sentence:** “Week 3 calibrated and froze a compact set of diagnostic interventions using 10,400 pilot observations and a human semantic audit.”

### Week 4 — full teacher cache, labels, and partial states: complete

**Question addressed:** Can we create a complete, leakage-safe supervision corpus for active diagnosis?

Final core evidence:

| Artifact | Count |
|---|---:|
| Manifest instances | 7,291 |
| Qwen teacher rows | 7,291 |
| Gemma teacher rows | 7,291 |
| Total teacher rows | 14,582 |
| Unique probe observations | 87,712 |
| Diagnostic labels | 14,582 |
| Leakage-safe partial states | 388,623 |
| Human-audit packet rows | 180 |
| Final unresolved core failure-ledger rows | 0 |

What happened:

- Ran the complete Qwen3-VL-8B and Gemma-4-E4B teacher caches over all four core datasets.
- Repaired the HallusionBench answer contract instead of selectively excluding difficult examples.
- Resolved 207 nondeterministic VizWiz majority-answer ties through a deterministic manifest contract.
- Recovered nine malformed grounding rows under an explicitly approved concise retry policy, with no exclusions.
- Rebuilt labels and 388,623 partial states only after the corrected contracts were frozen.
- Exported a blinded 180-row human-audit packet spanning three observed models and four datasets.
- Passed full Week 4 validation with complete model coverage, no final artifact errors, and no warnings.

**Important nuance:** InternVL is an optional catch-up/robustness model, not part of the two-model core cache. Its 1/10/100-row stages and full 340-row VSR cache passed. A full four-dataset 7,291-row InternVL cache has not been completed and should not be described as core evidence.

**Why this matters:** Week 4 transformed expensive VLM behavior into a reusable offline dataset. Weeks 5–7 can train small models without repeating roughly 102,294 teacher forward passes.

**Professor sentence:** “Week 4 completed the two-model supervision corpus: 14,582 teacher examples and 388,623 partial diagnostic states, with every core row accounted for.”

### Week 5 — diagnostic encoder bake-off and freeze: complete

**Question addressed:** What architecture best summarizes an unordered set of observed probes?

Work completed:

- Trained the complete 15-checkpoint matrix across architectures, variants, and seeds.
- Compared Deep Sets with GRU controls and tested multiple random evidence permutations.
- Selected and signed **Deep Sets, seed 42** as the frozen Week 5 encoder.

Key selection evidence:

| Measure | Result | Interpretation |
|---|---:|---|
| Aggregate source-bit Macro-F1 | 0.864217 | Strong multilabel diagnostic performance |
| Gap from best GRU source-bit result | 0.000116 | Essentially matched the best order-sensitive control |
| Best six-way Macro-F1 | 0.508031 | Best selected six-class result |
| Deep Sets permutation drift | 0.000000 | Exactly invariant in the tested permutations |
| Identity shortcut advantage | 0.097711 | Probe identity carries useful information and must be reported as a control |

The Set Transformer contingency was not triggered because the predeclared conditions did not require it. This is a correct omission, not missing work. The RAPS appendix trigger did fire and is tracked as deferred work.

**Why Deep Sets won:** Its diagnostic quality was practically tied with the best GRU, but it produced exactly zero order drift. Probe evidence is conceptually a set, so that inductive bias is cleaner and easier to defend.

**Professor sentence:** “Week 5 selected Deep Sets because it retained the best diagnostic quality while eliminating dependence on arbitrary probe order.”

### Week 6 — value-of-information policies and validation frontier: complete

**Question addressed:** Can the system learn to request useful evidence and stop early, beating simple baselines at matched cost?

Completed work:

- Built three complete VOI corpora, each containing **496,912 train/validation rows**.
- Trained the complete Deep Sets policy grid: **5 probe-cost settings × 3 seeds = 15 policies**.
- Verified 60 policy artifacts and their recorded hashes.
- Completed the train-only entropy-reduction uncertainty baseline.
- Completed scalar and distilled one-pass baselines. Their source-bit Macro-F1 values were **0.739826** and **0.757429**, respectively, over 1,404 validation rows without calibration/test leakage.
- Completed clean, scalar, and distilled APS validation artifacts.
- Passed a signed 100-row validation-frontier pilot covering all 15 conditions, both oracles, budgets 1/2/4 and full budget 7.
- The pilot completed in about 68 seconds. At budget 4, ProActive reached Macro-F1 **0.9512** with mean cost **2.22**, versus random at Macro-F1 **0.9265** with cost **4.00**.

Current gate:

- The owner approved the complete 15-frontier validation matrix with a combined ceiling of 7.5 GPU-hours and a 30-minute timeout per frontier.
- The complete matrix is now synchronized: all 15 reports and 60 referenced artifacts are present and hash-valid, every log exits zero, and measured compute was 0.930472 GPU-hours.
- The formal CPU-only report selects cost multiplier 0.0 and seed 42, with three-seed mean source-bit Macro-F1 0.942567, mean acquisition cost 2.284742, and mean prediction-set size 2.167221. It passes the random, fixed, budget-progress, and active-signal gates without calibration/test access.
- The owner approved cost multiplier 0.0 and seed 42 on 4 September 2026, and the CPU acceptance rerun recorded the boundary. Four 100-row GRU/masked VOI and uncertainty-policy pilots passed with hash-valid artifacts and no calibration/test access before the complete comparison launch.
- On 5 September 2026 the owner authorized the complete GRU/masked comparison bundle with a three-GPU-hour combined ceiling and strict per-job timeouts. This compares the already selected Deep Sets system against the mandatory order-sensitive GRU and exact-invariant masked-slot control on the same validation-only protocol.
- The first GPU-3 launch was safely refused because two live processes held about 38.4 GiB. The unchanged bundle then ran on verified-free physical GPU 1 and completed all 14 jobs in 1.8932 GPU-hours. Both 1,404-instance comparison frontiers are complete and hash-valid, include all 15 conditions and both oracles, and show the learned GRU/masked policies beating random and fixed schedules at at least one budget.
- The CPU-only full validator passed on 6 September 2026 with `is_valid=true`, zero errors, 496,912 complete VOI rows, and report SHA-256 `fade7dc83ec2b94e3bbf3110457c18a800647026e7cc5de4ed05a8dedf29ff64`.

**Why this matters:** The Week 6 frontier is the central adaptivity test. The paper needs evidence that learned probe selection beats random and at least two fixed schedules at comparable cost—not merely that the diagnostic encoder works when given all evidence.

**Professor sentence:** “Week 6 is complete: the selected Deep Sets policy, all required baselines and architecture comparisons, and the final fail-closed validator have passed; Week 7 now freezes that choice before calibration or test.”

### Week 7 — frozen stack, calibration, and locked test: active

**Question addressed:** Does the selected system remain calibrated and effective on untouched data?

Code already implemented for:

- a signed end-to-end freeze manifest;
- calibration-only trajectory generation;
- APS at target coverage 90% and 95%;
- budgets 1/2/3/4/7;
- one locked test evaluation;
- full permutation checks; and
- final go/no-go reporting.

Week 6 selection is now complete. The next required step is a validation-only expansion from budgets 1/2/4 to the approved final grid 1/2/3/4/7, followed by the signed stack freeze. Only then may calibration and the single locked test evaluation run.

The three static controls completed that five-budget validation expansion with both 90% and 95% targets and no calibration/test access. The selected Deep Sets frontier then passed over 1,404 validation model-instances with all 15 conditions, both oracles, and all five budgets in 7:38.52. Its hash-valid dataset schedule was bound into the owner-approved main-stack freeze; all 11 frozen artifact hashes match. Calibration then produced 7,500 frozen-policy trajectories and final main/static APS thresholds at both coverage targets. Every main calibration cell passes the 0.03 undercoverage tolerance. The one-time locked test subsequently passed over 1,560 model-instances, and the final validator emitted `GO` with zero errors. Week 7 is complete; no post-test tuning is allowed.

### Week 8 — robustness, ablations, latency, and human audit: partially validated

The Week 8 code covers two-fold leave-one-model-out evaluation, pinned PRE-HAL
and IllusionBench stress data, the 15 mandatory ablations, fixed-hardware
latency, grouped bootstrap intervals, dataset/model slices, qualitative
examples, and a three-person blinded audit. Held-out shift, latency, and both
LOMO folds are now complete and hash-verified. LOMO diagnostic performance
beats the clean-only and scalar controls in both directions, while the Gemma
fold reveals cross-model calibration undercoverage at larger budgets. This is
retained as a limitation; the held-out result will not be tuned. All 15
mandatory ablations and the final grouped statistical analysis are also
complete. Only completed human annotations and final validation remain.

### Week 9 — reproducibility and paper asset freeze: planned

The final phase packages tables and figures, regenerates checksums and run metadata, performs an independent leakage/reproducibility audit, freezes artifacts, and prepares the abstract and paper narrative. Week 9 is not a new modeling week; it turns verified evidence into a submission-ready package.

## Current status at a glance

| Stage | Status | Remaining gate |
|---|---|---|
| Week 1 contracts | Complete | None |
| Week 2 manifests/clean foundations | Complete | None |
| Week 3 probe pilot/freeze | Complete | None |
| Week 4 teacher cache/labels/states | Complete | Optional independent human annotations remain useful |
| Week 5 encoder bake-off/freeze | Complete | RAPS appendix is deferred |
| Week 6 policies/frontier | Complete | Full validator passed with zero errors and 496,912 complete VOI rows |
| Week 7 calibration/test | Complete | Signed freeze, final APS, one-time locked test, permutations, and final `GO` all passed |
| Week 8 robustness/ablations/audit | Implemented, not validated | All machine evidence complete; three-person audit and full validator remain |
| Week 9 assets/reproducibility | Not started | Evidence index exists; one-command assets, reproducibility manifest, review, and release freeze remain |

## What the frequent errors actually meant

The project did encounter real errors. “Zero failures” referred only to the final recovered core teacher artifacts, not to the entire development history. The important scientific distinction is between a failed attempt that was detected and repaired, versus silent corruption that enters a reported result.

| Error seen | Beginner explanation | Resolution or lesson |
|---|---|---|
| 31 duplicate POPE rows | A 100-row run appended to an older smoke file, and the same seed repeated early samples. | Added duplicate-aware validation and clean overwrite/resume discipline. |
| `--limit: command not found` | A space followed the Bash continuation backslash, so the next line became a new command. | The backslash must be the final character on the line. |
| `Permission denied` on a YAML path | A configuration file was typed as though it were an executable command. | Pass it to a Python script with `--config`; do not execute YAML. |
| `pytest: command not found` | The active Python environment did not have pytest installed. | Testing can be skipped for a time-critical server launch only after equivalent local tests pass; the isolated environment later passed the full suite. |
| CUDA out-of-memory | Another user's processes already occupied most GPU memory, even if utilization looked low earlier. | Check per-process memory, use only genuinely free GPUs, and stage before full runs. |
| Missing Week 4 prerequisite document | The server sync omitted a required audit/requirements file. | Sync the required Markdown contract before readiness validation. |
| Hash changed after sync | CRLF/LF line-ending conversion or a regenerated artifact changed the exact bytes. | Treat hashes as fingerprints of bytes; preserve frozen inputs and compare explicit old/new hashes. |
| 14 HallusionBench contract cases | Some examples did not fit the original binary/open-answer assumption. | Corrected and documented the data contract; did not selectively drop difficult examples. |
| 207 VizWiz answer changes | Multiple human answers tied, and earlier tie resolution depended on unstable ordering. | Added a deterministic canonical-answer rule and migrated affected caches. |
| Missing `FINAL_ANSWER` / empty grounding answer | A generative model ignored the requested output format or produced unusable text. | Increased grounding allowance and used a signed concise recovery prompt for exactly the remaining malformed rows. |
| `einops` missing | InternVL's remote model code required a package absent from the base environment. | Created a dedicated InternVL environment with its required dependencies. |
| InternVL + Transformers incompatibility | InternVL remote code was incompatible with the newer Transformers 5.x base environment. | Used Python 3.11, Transformers 4.37.2, Accelerate 0.30.1, and verified focused/full tests. |
| Pip CA certificate/environment vanished | A failed installation left a partially created environment with a bad certificate path. | Recreated the Conda environment cleanly and verified PyTorch/CUDA and `pip check`. |
| FlashAttention warning | The optional faster attention kernel was absent, so inference used the slower eager implementation. | Safe for correctness; only runtime is affected. |
| InternVL VSR “67 invalid” | A Week 3-oriented schema checker rejected 36 calibration + 31 test rows, not bad generations. | Used the correct Week 4 validation contract; all 340 teacher rows were present. |
| Audit resume refused: manifest missing | Output rows existed, but the provenance manifest required for safe resume did not. | Re-export with a complete audit bundle rather than trusting partial files. |
| Undefined AUROC in tiny pilot | A 1/10-row slice contained only one class, so ranking metrics were mathematically undefined. | Pilot mode records `null`; scientific full runs require enough class diversity. |
| One focused test label mismatch | Test code used older bit names (`grounding`, `language-prior`) than the current schema (`alignment`, `language`). | Updated the test to the canonical labels; 281 tests then passed. |
| Synced output looked truncated or absent | Large files or ongoing runs were copied before completion. | Verify line counts, final log line, hashes, and active processes after every sync. |
| Terminal exited with code 1 | Usually a fail-closed validator stopped a pipeline because one prerequisite or artifact was unsafe. | Read the first real error above the final exit line; do not rerun blindly. |

### Why fail-closed errors are useful

Several scripts deliberately exit with code 1 when an artifact is incomplete, a resume boundary is unsafe, a hash changes, a scientific metric is undefined, or a required approval is missing. This is inconvenient operationally, but it protects the paper from silently mixing incompatible results. The lesson is to reduce wasted wall-clock time with staging and monitoring—not to remove integrity checks.

## Scientific safeguards already built in

- **Deterministic manifests and hashes:** establish exactly which rows and files produced a result.
- **Group-aware splitting:** prevents related examples from leaking across train and evaluation splits.
- **Train/validation/calibration/test firewall:** separates fitting, choosing, calibrating, and reporting.
- **Append/resume validation:** refuses to mix incompatible cache generations.
- **Failure sidecars:** every unresolved row remains visible rather than disappearing from counts.
- **No selective exclusions:** difficult rows are repaired under documented contracts or reported explicitly.
- **Permutation controls:** distinguish a true set encoder from an order-sensitive shortcut.
- **Identity controls:** measure how much performance comes from knowing the probe name alone.
- **Matched-cost baselines:** compare quality at similar probe cost rather than giving ProActive more evidence.
- **Signed freezes:** record the approved architecture, seed, costs, thresholds, and hashes before final evaluation.

## Claims we can and cannot make today

### Supported now

- A complete two-model, four-dataset diagnostic teacher corpus exists.
- The probe specification and semantic matching threshold were calibrated and frozen.
- Deep Sets gives strong diagnostic performance with zero observed permutation drift.
- The full Week 6 policy grid and one-pass baselines were trained.
- The complete Week 6 validation matrix selected cost multiplier 0.0/seed 42 and passed its full validator.
- The frozen Week 7 stack, source-calibration APS, locked test frontier, and permutation studies passed with final `GO`.
- Held-out PRE-HAL/IllusionBench shift, both source-only LOMO folds, fixed-hardware latency, all 15 ablations, grouped intervals, paired tests, slices, and qualitative cases are complete.
- Held-out results support active acquisition at budgets 1--4; the small budget-7 random advantage is retained as an honest saturation result.

### Not yet supported

- We cannot claim all automatic labels are human-plausible until three
  independent annotators complete the blinded audit.
- We cannot claim full four-dataset InternVL generalization; only staged runs
  and complete VSR are available.
- We cannot claim universal active-policy dominance: random is slightly better
  at the saturated held-out budget 7, although ProActive is better at budgets
  1--4 and uses fewer probes at budget 7.
- We cannot claim a distribution-free cross-model coverage guarantee because
  the Gemma LOMO fold undercovers.
- We should not describe the source bits as uniquely identified causal mechanisms; they are operational diagnostics induced by defined probes.

## Original 9-day plan and current compressed closeout

The table below records the original plan for auditability. Its machine-work
items are now complete. The live closeout is: distribute/finish annotations,
draft the paper immediately, merge and analyze returned packets, run the final
CPU validator, then freeze tables/figures/claims and submit. Optional work is
deferred.

| Day | Human track | Experiment and engineering track | Paper track / completion evidence |
|---|---|---|---|
| **1 — today** | Give the three ready packets to three independent people. Each person labels the same 180 rows; target return within 48 hours. | Sync `WEEK_8_DATASET_SETUP_AND_EXECUTION.md`, rerun implementation validation, then run the 1-row and 10-row held-out parser gates for Qwen and Gemma on two verified-free GPUs. | Record verified dataset hashes, 600+600 frozen manifest counts, pilot logs, and exact exclusions. |
| **2** | Annotators continue independently; answer only procedural questions without discussing labels. | Review the four short parser outputs. If valid, obtain explicit full-run approval and launch Qwen/Gemma held-out caches concurrently with resume-safe outputs. Start cached CPU analyses in parallel. | Draft methods/data-shift and human-audit protocol sections using no pending result numbers. |
| **3** | Collect all three CSVs and check that every required cell is filled. | Finish/recover held-out cache; build labels and partial states. Run shift frontier with the immutable Week 7 stack and source APS only. | Populate the held-out result table and state clearly that shift coverage is empirical, not a renewed guarantee. |
| **4** | Merge annotations. Adjudicate only genuine three-way label disagreements while still blinded. | Run the highest-value cached ablations and fixed-hardware latency. Begin feasible two-fold LOMO; stop it if runtime threatens Days 6–9. | Run agreement/rule-match analysis and write the audit result honestly even if thresholds are weak. |
| **5** | Human work should be closed; preserve original files unchanged. | Finish mandatory cached statistics: grouped bootstrap, paired tests with Holm correction, dataset/model slices, action/STOP behavior, oracle gaps, and qualitative cases. | Freeze the main results inventory and identify unsupported claims. |
| **6** | No further label changes except documented correction of file-format mistakes. | Run Week 8 full validation on the evidence that exists. Resume only missing artifacts; do not add new tuning. | Complete abstract draft, main tables, figures, captions, limitations, and related-work gap. |
| **7** | Archive blinded packets, merged file, adjudication, and analysis hashes. | Independent leakage, count, checksum, and command audit. | Professor/coauthor review; every number must point to a JSON/CSV artifact. |
| **8** | None. | Reproducibility reruns only if cheap and necessary; no optional new model or dataset. | Revise paper, freeze figures/tables, check formatting and supplementary inventory. |
| **9 — buffer** | None. | Correct only verified implementation/reporting defects; no test-driven tuning. | Final abstract/paper check and submission buffer. |

### Exact human-annotation timing and ownership

- **Start:** Day 1, immediately. The packets already exist and all three
  180-row CSVs are currently blank.
- **Return target:** End of Day 2, or Day 3 morning at the latest.
- **Who:** Three genuinely different people. They need ordinary image/question
  judgement and careful reading; they do not need to know the model internals.
- **Work allocation:** Every annotator completes all 180 rows. Do not split the
  rows 60/60/60, because agreement can only be measured when all three judge
  the same items.
- **Independence:** Give each person only their own folder. They must not see
  model names, dataset names, teacher labels/scores, the private key, or another
  person's answers, and they must not discuss cases before submission.
- **Expected effort:** About 2–4 hours per person, all in parallel. No GPU or
  external dataset is involved.
- **Why it is needed:** It tests whether the automatic six-way behavioural
  labels are understandable to humans. It is required for the Week 8 completion
  gate and the human-validity claim, but it does not block held-out inference.
- **After return:** Check completeness, merge the three files, adjudicate only
  three-way disagreements while blinded, then unblind once to compute Fleiss'
  kappa, majority agreement, and automatic-rule match rates.

### The Week 6 decision branch

Proceed with the full active-diagnosis claim only if the complete validation evidence shows that the learned policy:

1. beats random selection at matched cost;
2. beats at least two fixed schedules;
3. is not matched by a clean-only or one-pass shortcut;
4. reduces prediction-set size while maintaining target coverage;
5. preserves the Deep Sets order-invariance advantage;
6. remains useful across meaningful dataset/model slices; and
7. is not explained only by probe identity.

If the adaptivity result is weak, do **not** spend days forcing it. Pivot the paper to the strongest verified contribution: an invariant diagnostic representation, calibrated diagnosis sets, one-pass distillation/fixed schedules, and a careful negative result about when adaptive probes do or do not help. That is scientifically cleaner than tuning on the final test set.

## Priority tiers for the remaining time

| Priority | Do before submission | Examples |
|---|---|---|
| **P0 — mandatory** | Yes | Three-person audit, full Week 8 validation, primary figures/tables, claim audit, reproducibility, abstract and paper |
| **P1 — high value if cached/short** | Already complete | Dataset/model slices, ablations, latency, action traces, oracle gaps, grouped statistics |
| **P2 — defer if it threatens P0** | No | Full InternVL four-dataset cache, new GQA construction, optional LODO, broad hyperparameter search, Set Transformer, extensive RAPS appendix |

## What is needed from the project owner

### Needed now

1. Recruit three genuinely independent annotators and send the three existing
   180-row packets today. All three annotation files are currently empty.
2. Draft the paper from `PROACTIVE_RESEARCH_HANDOFF_AND_RESULTS_INDEX.md` while
   annotation proceeds; no mandatory GPU run remains.
3. After all packets return, merge, adjudicate while blinded, analyze agreement,
   and run the CPU-only full Week 8 validator.
4. Trace every final table, figure, and claim back to its signed JSON/CSV source.

### External setup that is optional or deferred

- **InternVL:** The server already has `/home/models/InternVL3-9B` and the dedicated `proactive-internvl` Conda environment. Do not download it again. Full four-dataset catch-up is optional.
- **GQA:** Requires images plus scene graphs and a construction script that is not yet present. Start only with explicit scientific need and schedule approval.
- **PRE-HAL and IllusionBench:** Both pinned snapshots are downloaded and
  verified. PRE-HAL has 10,000 eligible source rows; IllusionBench has 5,196
  eligible rows after 381 deterministic release-integrity exclusions. The
  frozen held-out manifest contains 600 rows from each release and was selected
  without model outputs or target-domain calibration. IllusionBench declares no
  license on its current Hugging Face card, so do not redistribute it without
  author confirmation.
- **Human audit:** No completed set of three independent annotation blocks was recorded at the last local inspection. This is the most useful non-GPU owner action if annotators are available.

## Suggested meeting flow for tomorrow

### Minute 0–2: motivation

“A wrong multimodal answer can arise for different reasons, but current uncertainty methods mostly give one confidence score. ProActive actively requests a few diagnostic probes, decides when to stop, and returns a calibrated set of plausible failure types.”

### Minute 2–5: method

Use the doctor analogy. Explain the three source bits, six reporting classes, Deep Sets encoder, VOI policy, STOP action, and APS output. Emphasize that the bits are operational diagnostics, not proven causal decomposition.

### Minute 5–9: completed evidence

Show the Week 3 probe freeze, Week 4 counts, Week 5 Deep Sets decision, the completed Week 6 policy frontier, and the frozen/calibrated Week 7 locked-test result. The four most memorable numbers are:

- 10,400 valid Week 3 pilot records;
- 14,582 complete core teacher rows;
- 388,623 partial states;
- Deep Sets Macro-F1 0.864217 with zero permutation drift.

Then state that Weeks 6 and 7 passed their full validation gates. The earlier 100-row policy pilot was only staging evidence and must not be described as the final result.

### Minute 9–12: current gate

“The Deep Sets diagnostic encoder, the selected cost-zero policy, the conformal calibration, and the one-time core test are now frozen and complete. Week 8 stress-tests that unchanged stack on two external datasets, leave-one-model-out transfer, ablations, latency, and a blinded three-person human audit. None of those Week 8 results will be used to retune the locked core system.”

### Minute 12–15: 9-day plan and decisions requested

Ask the professor to confirm:

1. whether the active-policy result or the diagnostic representation should be the headline;
2. how prominently to position the PRE-HAL/IllusionBench external-shift result;
3. whether three independent annotators can be arranged immediately; and
4. which optional Week 8 ablations may be deferred if they threaten the submission schedule.

## Likely professor questions and short answers

**How is this different from ordinary uncertainty estimation?**  
Ordinary uncertainty asks how unsure the model is. ProActive asks which failure mechanism is plausible and actively chooses evidence that can distinguish mechanisms.

**Why perturb images at all?**  
Controlled changes reveal whether an answer depends on visual evidence. Severity was calibrated in Week 3 so the probes are informative without simply destroying the task.

**Why Deep Sets instead of a Transformer or GRU?**  
The observations form a set. Deep Sets essentially matched the best GRU diagnostic score while showing zero tested order drift, which is a cleaner inductive bias. The predeclared Set Transformer trigger did not fire.

**Where do the labels come from?**  
They are derived from the clean and probed behavior of two teacher VLMs under a frozen operational contract, then checked through schemas, failure ledgers, semantic calibration, and selected human audits.

**Are the labels ground-truth causal explanations?**  
No. They are reproducible operational diagnostic targets induced by controlled probes. The paper must avoid overstating them as uniquely identified causes.

**Why is Week 4 so large?**  
Each of 7,291 instances was evaluated by two core models under clean and legal probe conditions. Storing this once makes later policy experiments cheap and reproducible.

**Why does the policy need STOP?**  
Without STOP, it would always spend the full budget even when the diagnosis is already clear. STOP turns diagnostic quality into an accuracy-cost tradeoff.

**Why not use the test set now?**  
Because selecting settings after seeing test performance would leak information and make the final claim unreliable. Validation chooses; calibration calibrates; test reports once.

**Did the project have errors?**  
Yes. They were detected, documented, and repaired. The final core artifacts have complete accounting, but the development history includes duplicate appends, malformed generations, environment conflicts, GPU OOMs, and incomplete syncs.

**What happens if adaptive selection does not win?**  
We pivot to the verified invariant diagnosis and calibrated prediction-set contribution, report the adaptivity result honestly, and avoid wasting the remaining submission window on test-driven tuning.

## Language to use carefully in the paper and meeting

Prefer:

- “operational failure-source indicators” rather than “true causal sources”;
- “image-text grounding/alignment” rather than an ambiguous use of “alignment”;
- “complete two-model core cache” rather than implying complete InternVL coverage;
- “completed Week 6 validation frontier” and “frozen Week 7 locked-test result,” while clearly calling earlier 1/10/100-row runs staging checks;
- “calibrated diagnosis set” rather than “guaranteed correct diagnosis”; and
- “no unresolved failures in the final core cache” rather than “we faced zero errors.”

## Where to find authoritative details

| File | Purpose |
|---|---|
| `AGENTS.md` | Repository overview, constraints, reading order, and audit rules for coding work |
| `PROJECT_STATUS.md` | Current completion snapshot and immediate gates |
| `WEEKLY_PROGRESS.md` | Detailed chronological progress |
| `PROJECT_LOG.md` | Operational decisions, runs, and important changes |
| `DECISIONS.md` | Approved scientific and compute decisions |
| `FAILURE_LOG.md` | Failure history and recovery evidence |
| `SERVER_RUNBOOK.md` | Server environment and execution instructions |
| `ICLR_DEFERRED_WORK_AND_EXTERNAL_SETUP.md` | Deferred experiments, model/data setup, and human-audit status |
| `v3.5_ProActive_Complete_Super_Implementation_Plan.md` | Full research implementation plan and go/no-go criteria |

## Final takeaway

ProActive now has a frozen and tested core stack, a complete 2,400-row held-out shift evaluation, fixed-hardware latency evidence, and two complete leave-one-model-out transfer folds. The transfer result supports diagnostic generalization but also exposes Gemma-side calibration degradation that must be reported honestly. The decisive remaining work is the mandatory ablation bundle, signed Week 8 analysis, the three-person audit, and conversion of verified artifacts into the ICLR paper. Optional model catch-up, GQA construction, LODO, broad hyperparameter searches, and cosmetic analyses must not displace that critical path.
