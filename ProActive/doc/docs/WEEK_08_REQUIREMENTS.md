# Week 8 Requirements — Generalization, shift, ablations, latency, and human audit

**Status:** IMPLEMENTED, NOT VALIDATED

**Prerequisite:** Week 7 is COMPLETE. The immutable main stack, final APS, and
one-time locked core test are closed; no Week 8 result may be used to tune them.

**Source of truth:** Plan §25.10 and the human-audit protocol in §15.

## Execution order under the submission deadline

1. Start the three-person human audit immediately. It needs no GPU, but human
   availability is the slowest dependency.
2. Run cached-data analyses first: dataset/model slices, mandatory ablations,
   action and STOP behavior, oracle gaps, latency, bootstrap confidence
   intervals, and qualitative examples.
3. In parallel, decide whether to include the two held-out stress datasets.
   If yes, obtain and checksum PRE-HAL and IllusionBench immediately; their
   loader and shift-evaluation code must still be implemented and tested.
4. Run held-out shift only with the already frozen stack. Report empirical
   coverage under shift, not a new conformal guarantee.
5. Freeze paper tables, plots, claims, and provenance after the audits pass.

## Requirement traceability

| ID | Requirement | Required implementation/evidence | Status |
|---|---|---|---|
| W8-01 | Leave-one-model-out transfer for feasible model families | `build_lomo_fold.py`, `freeze_lomo_stack.py`, `eval_lomo.py`, `test_week8_lomo_and_human.py`, `lomo.csv` | COMPLETE |
| W8-02 | Held-out shift on PRE-HAL and IllusionBench | `heldout.py`, setup/manifest scripts, `eval_frontier.py --phase shift`, `test_week8_heldout.py`, frozen manifest and `shift.csv` | COMPLETE |
| W8-03 | Optional leave-one-dataset-out | Run only if W8-01/02 and paper-critical work are secure | NOT STARTED |
| W8-04 | Mandatory component ablations | Feature-manifest builder, real zero-budget/independent-source architectures, loss-only VOI, no-STOP rollout, signed comparison/aggregation scripts | COMPLETE |
| W8-05 | Fixed-hardware latency and cost accounting | `measure_latency.py`: full controller, CUDA synchronization, 10 warm-ups/100 measurements, cached pass accounting | COMPLETE |
| W8-06 | Three-person blinded human audit | `human_annotation/`, packet/merge/analyze scripts, `test_week8_lomo_and_human.py`; final agreement output pending people | IMPLEMENTED, NOT VALIDATED |
| W8-07 | Grouped bootstrap confidence intervals and paired primary tests | `statistics.py`, `analyze_week8.py`, `test_week8_statistics.py`; 2,000-resample grouped output and Holm-corrected primary family | COMPLETE |
| W8-08 | Within-dataset and within-model controls | Frozen slice generation in `analyze_week8.py`; 160 signed slice rows | COMPLETE |
| W8-09 | Positive and negative qualitative cases | Deterministic frozen-outcome ranking in `analyze_week8.py`; 10 positive and 10 negative cases | COMPLETE |
| W8-10 | Shift/calibration claim audit | Frozen source APS and empirical-only shift claims verified; full validator waits for the human-audit report | IMPLEMENTED, NOT VALIDATED |

## External dataset setup

The two held-out datasets are **PRE-HAL** and **IllusionBench**. Their expected
server locations are:

```text
/home/aman/MMUQ/data/PREHAL/
/home/aman/MMUQ/data/IllusionBench/
```

Before implementation or evaluation, record for each release: official URL,
license, version or commit, download date, archive SHA-256, extracted-data
SHA-256, schema, image layout, answer fields, and any filtering. These datasets
must not affect model selection, thresholds, costs, or stopping behavior.

GQA-Relation is a separate third setup item, not one of these two held-out
datasets. It requires GQA images, `sceneGraphs/val_sceneGraphs.json`, a missing
construction script, and manual inspection of at least 100 reversible-relation
pairs. Under the current deadline it is secondary to cached analyses and the
human audit unless the paper explicitly needs a stronger relation claim.

## Human audit execution contract

The source packet already exists at `outputs/human_audit/` with 180 blinded rows
and 180 materialized images. `prepare_human_annotation_packets.py` creates one
private working copy for each of three annotators. As of 2026-09-13, all three
packet CSVs contain 180 rows, but all required annotation cells remain blank.

1. Recruit three genuinely independent annotators now.
2. Give them only `human_audit_blinded.csv`, `images/`, and `README.md`.
3. Each annotator completes all 180 rows in exactly one column block:
   `ann1_*`, `ann2_*`, or `ann3_*`.
4. Do not expose `human_audit_private_key.jsonl`, dataset/model identities,
   teacher scores, or teacher labels during independent annotation.
5. After all three blocks are complete, adjudicate disagreements while still
   blinded; only then unblind for rule-label comparison.

Minimum acceptance targets:

| Metric | Minimum |
|---|---:|
| Fleiss' kappa for six-way label | 0.20 |
| At least two of three annotators agree | 75% |
| Rule label matches adjudicated label | 55% overall |
| Rule label matches singleton-label cases | 65% |

If the targets fail, report the disagreement structure honestly. Do not merge
mixed/unclear cases into cleaner classes and never imitate three annotators
with one person.

## Completion gate

- cached mandatory ablations and latency evidence are complete;
- LOMO remains above clean-only and scalar controls, or the claim is narrowed;
- at least one meaningful held-out stress result exists; if external access
  makes this impossible, Week 8 remains incomplete and the paper must disclose
  the missing evidence and narrow its generalization claims;
- the three-person human audit is complete; if annotators are unavailable,
  Week 8 remains incomplete and the paper must disclose that limitation;
- no leakage, post-test tuning, or unsupported coverage claim remains.

## LOMO result recorded on 2026-09-11

The approved two-fold run completed in 2,944 seconds (0.818 aggregate
GPU-hours on one GPU), below the five-GPU-hour ceiling. All 28 preparation and
evaluation stages exited zero. Both reports contain 780 held-out test examples,
use source-model calibration only, and pass every recorded artifact and
self-hash check.

At budget 7, ProActive source-bit Macro-F1 is `0.9692` when Qwen is held out,
versus `0.6826` for clean-only and `0.6964` for scalar confidence. When Gemma
is held out it is `0.9948`, versus `0.7956` and `0.7804`. The corresponding
six-way Macro-F1 values for ProActive are `0.7927` and `0.8063`. This satisfies
the predeclared diagnostic-quality transfer gate. Cross-model conformal coverage
is not uniformly preserved: Qwen-held-out coverage at the 0.90 target is
`0.9910`, while Gemma-held-out coverage is `0.7115` at budget 7 (and `0.9205`
at budget 1). Therefore the paper may claim useful diagnostic transfer, but
must explicitly report calibration degradation on the Gemma transfer fold and
must not claim a distribution-free cross-model coverage guarantee.

## Ablation execution authorization recorded on 2026-09-11

All 15 mandatory ablations are approved at seed 42 on validation evidence
only. At most two GPUs may be used after a free-device check. The aggregate
ceiling is eight GPU-hours; each training job is capped at 45 minutes and each
frontier at 20 minutes. No core test or held-out-shift result may select,
calibrate, or tune an ablation. `scripts/run_week8_ablations.sh` enforces these
limits and writes the final signed bundle to
`outputs/week8_reports/ablations.json`. Status remains IMPLEMENTED, NOT
VALIDATED until that artifact passes the Week 8 validator.

## Ablation execution result recorded on 2026-09-12

All expensive stages have completed within the authorization: the conservative
ledger totals 28,275 seconds (`7.8542` GPU-hours), including two earlier
bounded timeouts and one APS provenance refusal that were later resolved. The no-budget recovery passed
exact early-stopping-history and APS scientific-equivalence checks before its
downstream provenance chain was rebuilt. All 15 mandatory evidence JSON files
are present, and an independent local audit reproduced every evidence self-hash
and all 29 bound input hashes. The combined report was not written because the
aggregator retained only the last of repeated `--evidence` CLI arguments. The
parser now accumulates those arguments and is covered by a regression test.
No training, frontier, or other GPU stage was rerun for this issue; the only
remaining actions at that point were the focused test and CPU-only aggregation.

The corrected server rerun passed its focused regression and produced the
signed aggregate on 2026-09-12. It binds 15/15 unique evidence reports and 121
comparison rows, with report SHA-256 `ee0a3eb...e14b` and CSV SHA-256
`7dfcbbb7...2613`. Independent verification reproduced both hashes and every
evidence binding. W8-04 is therefore COMPLETE.

## Statistical analysis result recorded on 2026-09-13

The final CPU analysis completed with exit code zero in 1:13:54. It read
340,800 immutable held-out trajectory rows and used 2,000 bootstrap resamples
grouped by `group_id`; target-domain calibration was not used. The signed
outputs contain 710 confidence-interval rows, 120 paired comparisons, 160
dataset/model slice rows, and 20 deterministic qualitative examples. The
analysis self-hash is `757908db...e191`, and all referenced trajectory, CSV,
and qualitative-file hashes independently match.

At target coverage 0.90, ProActive exceeds random in source-bit Macro-F1 at
budgets 1--4 by `0.0458`, `0.1189`, `0.1474`, and `0.1178`. The predeclared
budget-7 primary comparison is a small negative result: ProActive-minus-random
is `-0.0063` with 95% CI `[-0.0087, -0.0041]` and Holm-adjusted `p=0.003`.
ProActive uses about 0.42 fewer probes there. The paper must describe this as
full-budget saturation and limit the active-acquisition advantage to constrained
budgets. W8-07, W8-08, and W8-09 are COMPLETE. W8-10 remains open only because
the human-audit report and subsequent full validator are pending.
