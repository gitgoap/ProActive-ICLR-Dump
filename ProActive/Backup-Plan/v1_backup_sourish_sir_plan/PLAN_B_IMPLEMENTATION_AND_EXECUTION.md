# ProActive Plan B: cached-answer correction experiment

**Status:** VALIDATION GATE FAILED on 16 September 2026; locked core-test and
held-out-shift evaluation are prohibited by the frozen protocol.

Source protocol: [`ProActive_Plan_B_Student_Instructions.pdf`](ProActive_Plan_B_Student_Instructions.pdf).

The implementation was audited locally against all ten synchronized caches:
all SHA-256 hashes and all frozen counts matched, including 16,982 total rows,
1,898 train and 314 validation model-example pairs, 914 closed core-test rows,
2,400 held-out rows, 654 metadata image identities crossing original splits,
675 unusable probe answers, and zero correctness drift in all 16,982 cached
records under their declared exact/alias contracts.

## What this experiment asks

The main ProActive result shows that the learned acquisition policy is not yet a strong answer-correction mechanism. Plan B tests a narrower and auditable question: **after a fixed prefix of already-cached probes has been acquired, can a small learned selector choose a better answer than simple correction rules without damaging originally correct answers?**

This is a secondary, exploratory experiment. It does not replace the frozen Week 7/8 study and it makes **zero new multimodal-model calls**. It reads the existing Qwen and Gemma teacher caches only.

## Frozen protocol

- Cohort: closed-answer POPE, HallusionBench and VSR for core train/validation/test, plus the existing 2,400 held-out model-example pairs for descriptive shift evaluation. VizWiz and open-ended HallusionBench items are excluded from the primary cohort.
- Evidence order: `grounding -> blur -> crop -> brightness -> noise -> blank`.
- Primary budget: two acquired probes. Reported frontier budgets: `0, 1, 2, 3, 5, 6`.
- Selector: one L2 logistic regression over ten provenance/vote features. It never receives dataset ID, model ID, gold answer, raw question text or image content.
- Training: train split only. Hyperparameters and switching margin: validation only. Calibration is untouched.
- Comparator: selected on validation from keep-original, always-ground, majority and supported-ground using six-cell dataset/model macro accuracy.
- Validation gate: at least +1 percentage point over that comparator and at most 1% BreakRate. A failed gate blocks test/shift access in code.
- Locked evaluation: must be started by a separate command with an explicit confirmation flag. No tuning is allowed afterward.
- Statistics: image-cluster bootstrap (5,000 draws), paired cluster randomization (10,000 draws) and Holm correction.

## Why the three missing bundle files are not required

The implementation does not import `ProActive_PlanB_Audit_Bundle.zip`, `audit_answer_correction.py`, or `reference_results/cohort_index.csv`. Instead it reconstructs the cohort index from the ten pinned JSONL caches, applies the PDF's image-group exclusion rule, recomputes clean correctness, records unusable probe answers, and fails if any expected row count or SHA-256 hash differs.

The generated replacements live under `Backup-Plan/outputs/audit/` and are signed by the run manifest after evaluation.

## Files

- `config/plan_b_config.json`: frozen hashes, counts, features, solver, gates and seeds.
- `PLAN_B_DECISIONS.md`: pre-fit resolutions of the PDF's explicitly unspecified details.
- `plan_b/core.py`: fail-closed answer/candidate/cohort contracts.
- `plan_b/pipeline.py`: audit, train/validation freeze, locked evaluation and statistics.
- `run_plan_b.py`: command-line interface.
- `run_plan_b_server.sh`: safe server wrapper.
- `tests/test_plan_b.py`: unit/adversarial/cache-audit coverage.
- `outputs/`: generated evidence only; sync this directory back after each stage.

## Requirement traceability

| Requirement | Code | Test/check | Output evidence |
|---|---|---|---|
| Pinned-input and answer-contract audit | `plan_b/core.py`, `run_audit` | real-cache integration test | `audit/audit_report.json`, `input_inventory.csv` |
| Image leakage purge | `build_cohort_index`, `_image_overlap_report` | cross-dataset COCO/adversarial tests | `cohort_index.csv`, `image_exclusions.csv`, `image_overlap_report.json` |
| Ten-feature candidate selector | `candidate_features`, `_training_rows` | feature-order and unacquired-evidence tests | `selector_model.json`, `freeze_manifest.json` |
| Validation-only selection and stop gate | `run_prepare` | frozen cohort/count checks | `validation_predictions.csv`, `freeze_manifest.json` |
| One-time paired evaluation | `run_evaluate`, `_cluster_statistics` | reference-baseline and statistics execution checks | `final_report.json`, `paired_statistics.csv`, `run_manifest.json` |

## Server procedure

Sync the entire `Backup-Plan/` directory to `~/ProActive/Backup-Plan/`, except local generated outputs if the server is the authoritative run location. This experiment is CPU-only; do not reserve a GPU.

Run the preparation stage first:

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

PYTHONPATH=src python -m pytest -q Backup-Plan/tests/test_plan_b.py
bash Backup-Plan/run_plan_b_server.sh preflight
bash Backup-Plan/run_plan_b_server.sh prepare
```

Then inspect the frozen validation decision:

```bash
python -m json.tool Backup-Plan/outputs/training/freeze_manifest.json
```

Proceed only if its status is `FROZEN_VALIDATION_GATE_PASSED`. The separate locked command is:

```bash
bash Backup-Plan/run_plan_b_server.sh evaluate
bash Backup-Plan/run_plan_b_server.sh validate
```

Sync back the complete `Backup-Plan/outputs/` directory. The decisive files are:

- `training/freeze_manifest.json`: validation-selected comparator/margin and gate.
- `training/environment_preflight.json`: Python/NumPy/scikit-learn versions captured before fitting.
- `audit/input_inventory.csv`: verified paths, hashes, bytes and row counts.
- `audit/cohort_index.csv` and `audit/image_overlap_report.json`: the independently reconstructed inclusion and leakage-protection evidence.
- `evaluation/final_report.json`: locked core-test result and gate.
- `evaluation/metrics.csv`: all test/shift frontiers and slices.
- `evaluation/oracle_bound.csv`: separately labeled gold-privileged opportunity bound.
- `evaluation/paired_statistics.csv`: CIs and corrected tests.
- `evaluation/conclusion.md`: concise, deliberately qualified conclusion.
- `run_manifest.json`: hashes of the complete evidence bundle.

## Expected compute and time

GPU usage is zero. On the server CPU, hashing/audit plus logistic fitting should normally take roughly 5–20 minutes; the 5,000-draw bootstrap and 10,000-draw randomization should normally take roughly 10–40 minutes. Allow about one hour wall-clock for the complete protocol. Actual timing is recorded in the logs.

## Interpretation rule

Call the result supportive only if both the validation and locked test gates pass. If the validation gate fails, stop and report a negative result. If the test gate fails, retain the outputs and report the pre-registered negative result—do not change margins, features, cohorts, or thresholds after seeing test/shift.

The completed run failed at validation. See
[`PLAN_B_VALIDATION_CONCLUSION.md`](PLAN_B_VALIDATION_CONCLUSION.md). Do not run
`run_plan_b_server.sh evaluate` for this frozen experiment.
