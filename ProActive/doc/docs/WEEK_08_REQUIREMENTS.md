# Week 8 Requirements — Generalization, shift, ablations, latency, and human audit

**Status:** NOT STARTED

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
| W8-01 | Leave-one-model-out transfer for feasible model families | LOMO cache/evaluator, matched controls, CSV and confidence intervals | NOT STARTED |
| W8-02 | Held-out shift on PRE-HAL and IllusionBench | Official releases, provenance, tested loaders, frozen manifests, identical cost accounting, `shift.csv` | NOT STARTED |
| W8-03 | Optional leave-one-dataset-out | Run only if W8-01/02 and paper-critical work are secure | NOT STARTED |
| W8-04 | Mandatory component ablations | Frozen no-identity, clean-only, no-grounding/no-visual where scientifically defined, and one-pass comparisons | NOT STARTED |
| W8-05 | Fixed-hardware latency and cost accounting | Warm-up policy, synchronized timing, hardware/runtime record, `latency.csv` | NOT STARTED |
| W8-06 | Three-person blinded human audit | Three independent 180-row blocks, adjudication, agreement/label-match summary | NOT STARTED |
| W8-07 | Grouped bootstrap confidence intervals | Resample by base instance, not partial state; fixed seed and interval report | NOT STARTED |
| W8-08 | Within-dataset and within-model controls | Frozen aggregate and slice tables with sample counts | NOT STARTED |
| W8-09 | Positive and negative qualitative cases | Provenance-bound examples chosen without changing the model | NOT STARTED |
| W8-10 | Shift/calibration claim audit | No target-domain tuning and no formal shift-coverage claim without target-like calibration | NOT STARTED |

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

The packet already exists at `outputs/human_audit/` with 180 blinded rows and
180 materialized images. As of 2026-09-07, all three annotator blocks are empty.

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
