# Week 8 Ablation Resume Recovery

**Status:** RESUME FIX VERIFIED; SCIENTIFIC-EQUIVALENCE GATE PENDING  
**Incident date:** 2026-09-12  
**Scope:** Validation-only Week 8 ablations; no calibration, test, or held-out-shift tuning

## What failed

The first ablation run completed three component ablations and timed out safely
during two policy jobs. On resume, `train_diag.py` continued the already
completed `no_budget_embedding` diagnostic from its `.last.pt` file. The
original run had reached its first patience-5 stopping boundary at epoch 10;
the resume improperly ran epochs 11--16 and replaced the selected checkpoint.

The temporary APS report then rejected the changed checkpoint SHA-256. This is
the intended fail-closed behavior. The signed Week 8 ablation aggregate does
not exist and must not be claimed complete.

## Artifact decision

| Artifact | Decision |
| :--- | :--- |
| Original `no_budget_embedding` APS, VOI, policy, stack freeze, and frontier | Retain; each is bound to the original checkpoint hash |
| Current overwritten `no_budget_embedding` diagnostic checkpoint, report, history, and diagnostic freeze | Do not use as scientific evidence |
| Completed `no_confidence_shift` and `no_relation_probe` chains | Retain |
| Partial `no_answer_flip` and `shared_vs_independent_heads` policy checkpoints | Resume |
| Test, calibration, and held-out-shift artifacts | Unaffected; none were accessed by the ablation run |

## Code-level prevention

Completed diagnostic and policy validation reports are now immutable completion
markers. A resume verifies the report self-hash, checkpoint hash, configuration
and input provenance, best epoch, and first early-stopping boundary before
returning without training. Histories that continue after that boundary are
rejected.

Traceability:

| Requirement | Code | Tests | Server evidence |
| :--- | :--- | :--- | :--- |
| Completed training is idempotent on resume | `src/proactive/train/checkpoints.py`, `scripts/train_diag.py`, `scripts/train_policy.py` | `tests/test_week5_training.py` plus full suite | `outputs/logs/week8/ablation_resume_fix_tests.log`, `ablation_resume_fix_full_pytest.log` |
| Restore only the overwritten artifact and fail if scientific evidence changes | `scripts/repair_week8_ablation_resume.sh`, `scripts/validate_ablation_rebuild_equivalence.py`, `scripts/prepare_week8_ablation_chain_rebuild.sh` | Hash/provenance and equivalence tests plus script preflight | Repair log and `outputs/week8_repair_validation/no_budget_embedding/equivalence_report.json` |
| Preserve the approved compute boundary | Repair and main launchers share `gpu_time_ledger.tsv` | Ledger inspected before launch | Cumulative limit: 28,800 GPU-seconds |

## Required execution order

1. Synchronize the changed code and this record to the server.
2. Run focused and full pytest in `proactive-internvl`.
3. The epoch-zero repair has run. It reproduced all epoch metrics exactly but,
   as expected for an atomic PyTorch save through a randomly named ZIP file,
   did not reproduce raw file bytes. Do not run it again.
   If no exclusive device is available, the owner-approved shared-device form
   is `bash scripts/repair_week8_ablation_resume.sh --allow-shared ID`. It
   permits existing processes only when at least 8,192 MiB is free and launch
   utilization is at most 10%; it never kills or modifies another process.
4. Run `prepare_week8_ablation_chain_rebuild.sh`; continue only if it prints
   `WEEK8_NO_BUDGET_EQUIVALENCE_ALL_OK=1`. This requires exact history and APS
   scientific content, archives the stale descendants, and never edits their
   provenance in place.
5. Resume with `bash scripts/run_week8_ablations.sh --allow-shared 1` when the
   explicitly authorized shared-device gate passes.
6. Do not declare completion until the launcher prints
   `WEEK8_ABLATION_BUNDLE_ALL_OK=1` and both
   `outputs/week8_reports/ablations.json` and `.csv` pass the Week 8 validator.

Before repair, the ledger contains 18,185 seconds (`5.0514` GPU-hours), leaving
10,615 seconds (`2.9486` GPU-hours) under the approved eight-hour ceiling. The
repair itself is charged to the same ledger.
