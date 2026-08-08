# ProActive — Agent Onboarding

## What is ProActive?

ProActive is an **active diagnostic framework** for multimodal large language models (MLLMs). Instead of running every possible diagnostic probe on a model response (expensive), the system learns a sequential policy that selects the most informative probe at each step, observes the result, updates a diagnostic state, and stops early when confident — outputting a **calibrated set of plausible behavioral failure modes**.

**Target conference:** ICLR 2026.

## Prior Work: HalluPrism (EMNLP)

The repo owner has a paper submitted to **EMNLP** called **HalluPrism**, which is the direct predecessor to ProActive. Read `Important_papers/HalluPrism-my_paper.pdf` before writing any code — it explains the probe taxonomy, the failure-mode ontology, and the dataset-confounding problem that ProActive is designed to fix.

## Core Research Contributions

1. **Active probe acquisition** — learn which behavioral probe to run next under a limited forward-pass budget.
2. **Unordered evidence modeling** — represent acquired probe outcomes as a set (Set Transformer), not an artificial sequence.
3. **Calibrated diagnostic sets** — return a prediction set with target marginal coverage via conformal prediction.
4. **Robust evaluation** — cost–diagnostic frontiers, permutation stability, leave-one-model-out transfer, held-out dataset shift, and oracle baselines.

## Reading Order for New Agents

Read these files **in this order** before writing any code:

| Priority | File                                                   | What it tells you                                                                                      |
| -------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| 1        | `instructions.md`                                      | Workspace rules, environment constraints, strict coding policies                                       |
| 2        | `v3.5_ProActive_Complete_Super_Implementation_Plan.md` | The scientific and engineering source of truth — every formula, every threshold, every design decision |
| 3        | `PROJECT_STATUS.md`                                    | Current phase and blockers                                                                             |
| 4        | `WEEKLY_PROGRESS.md`                                   | What has actually been built and validated so far                                                      |
| 5        | `doc/docs/WEEK_*_REQUIREMENTS.md`                      | Per-week completion gates and requirement matrices                                                     |
| 6        | `CODE_REVIEW.md`                                       | Adversarial review checklist — use before declaring anything complete                                  |
| 7        | `SERVER_RUNBOOK.md`                                    | How to run GPU workloads on the remote server                                                          |

Do not replace explicit plan requirements with placeholders, simplified
implementations, or "temporary" fallbacks unless clearly approved.

## Key Constraints

- **No GPU locally.** Code is edited locally; all GPU work runs on a remote server (`bumblebee`). Never claim something works on GPU without server logs.
- **Data paths are configurable.** Use `PROACTIVE_DATA_ROOT` env var. Never hardcode paths.
- **Fail closed.** Missing scores are errors, not zeros. Malformed outputs are invalid, not flips. See §3 below.
- **No data leakage.** Train/val only for thresholds. Cal/test never touch severity or hyperparameter selection.
- **Deterministic seeds.** All transforms derive seeds from `SHA-256(global_seed | instance_id | probe_name | severity)`.
- **Run `pytest` before declaring anything done.** The test suite must remain green (currently 150 tests).

## Datasets

| Dataset        | Type       | Normalizer            | Relation Probe? | Status              |
| -------------- | ---------- | --------------------- | --------------- | ------------------- |
| POPE           | Yes/No     | `yes_no`              | No              | Active              |
| VSR            | True/False | `true_false`          | Yes (spatial)   | Active              |
| VizWiz         | Free-form  | `freeform` + semantic | No              | Active              |
| HallusionBench | Yes/No     | `yes_no`              | No              | Active              |
| GQA-Relation   | Yes/No     | `yes_no`              | Yes             | Pending (Week 7-8)  |
| PreHal         | —          | —                     | No              | Held-out (Week 7-8) |
| IllusionBench  | —          | —                     | No              | Held-out (Week 7-8) |

## Quick Commands

```bash
# Build manifests
python scripts/build_manifests.py --config_dir configs/data --data_root data --output_dir outputs/manifests --overwrite --seed 42

# Run pilot (canonical mode, 5 samples, smoke test)
python scripts/run_pilot_cache.py --manifest_path outputs/manifests/manifest_pope.jsonl --model_config configs/models/qwen3_vl_8b.yaml --output_dir outputs/pilot_cache --device cuda:0 --limit 5 --pilot_mode canonical

# Run tests
pytest
```

## 2. Requirement traceability

For every task, maintain:

requirement → code location → unit test → integration test → output artifact

A requirement is incomplete if any item is missing.

## 3. Fail-closed behaviour

Never silently convert:

- missing scores to zero,
- missing probes to False,
- malformed answers to flips,
- unknown outputs to valid labels,
- failed parsing to empty strings.

Raise an error or mark the example invalid.

## 4. Testing

Passing unit tests is insufficient.

Every major component must have:

- unit tests,
- adversarial edge-case tests,
- an end-to-end integration test.

## 5. Data leakage

Before any experiment, verify:

- no dataset ID or model ID enters the learner,
- train/validation only are used for threshold selection,
- test data are never used for severity or hyperparameter selection,
- proxy features and label distributions are audited by dataset.

## 6. Completion rule

Do not declare a week complete until all required:

- code,
- tests,
- pilot outputs,
- plots,
- inspections,
- schemas,
- frozen configurations,
- documentation

exist and have been reviewed.

Use these statuses only:

- NOT STARTED
- IMPLEMENTED, NOT VALIDATED
- PILOT VALIDATED
- COMPLETE
