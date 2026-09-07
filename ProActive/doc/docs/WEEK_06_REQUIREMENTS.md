# Week 6 Requirements — VOI targets, policies, and baseline frontier

**Status:** COMPLETE

**Source of truth:** Plan §§9, 16, 18–21, and 25.8.

## Completion gate

The validation-selected Deep Sets policy must beat random at one or more
matched budgets and at least one fixed schedule. Full teacher must remain
stronger than clean-only, and increasing budget must shrink diagnostic sets or
improve diagnostic quality.

## Requirement traceability

| ID | Requirement | Code location | Unit/adversarial test | Integration test | Required artifact | Status |
|---|---|---|---|---|---|---|
| W6-01 | Accept only a frozen Week 5 diagnostic checkpoint | `src/proactive/train/checkpoints.py` | `tests/test_freeze_contracts.py` | Three complete signed VOI builds | checkpoint freeze manifest | COMPLETE |
| W6-02 | Realized VOI uses cached legal counterfactuals and validation-temporary APS only | `src/proactive/train/voi.py`, `scripts/build_voi_targets.py` | `tests/test_voi_targets.py` | Three complete architecture-specific VOI builds | `outputs/week6_voi*/voi_manifest.json` | COMPLETE |
| W6-03 | Exact Plan §9.1 VOI formula and sign/scale audit | `src/proactive/train/voi.py` | `tests/test_voi_targets.py` | Complete VOI manifest sign/action audits | `outputs/week6_voi*/voi_manifest.json` | COMPLETE |
| W6-04 | Action-conditioned VOI head | `src/proactive/networks/voi.py` | `tests/test_policy.py` | Complete main and comparison policy checkpoints | policy checkpoint | COMPLETE |
| W6-05 | Ranking + 0.25 MSE + 0.5 action CE objective | `src/proactive/networks/losses.py` | `tests/test_policy.py` | Complete finite GPU training histories | loss history | COMPLETE |
| W6-06 | Legal mask, no repeated action, budget accounting, and STOP at nonpositive value | `src/proactive/policy/controller.py` | `tests/test_policy.py` | Complete Deep Sets and comparison frontiers | action trace JSONL | COMPLETE |
| W6-07 | Random, four fixed schedules, and a validation-selected dataset-specific fixed control at matched cost | `src/proactive/eval/baselines.py`, `scripts/eval_frontier.py` | `tests/test_baselines.py` | Complete Deep Sets and comparison frontiers | baseline CSV + frozen schedule | COMPLETE |
| W6-08 | Scalar, clean-only, one-pass distilled, full-teacher, oracle-next, and batched oracle-subset controls with separate APS | `scripts/train_one_pass_baselines.py`, `scripts/calibrate_static_baseline_aps.py`, `scripts/eval_frontier.py` | `tests/test_baselines.py`, `tests/test_policy_rollout.py` | Complete Deep Sets and comparison frontiers | baseline CSV | COMPLETE |
| W6-09 | Uncertainty-greedy baseline cannot inspect an unacquired outcome | `src/proactive/eval/baselines.py` | `tests/test_baselines.py` | Complete train-only entropy-reduction checkpoints/frontiers | baseline contract JSON | COMPLETE |
| W6-10 | Cost multipliers selected over three seeds using validation only | `scripts/train_policy.py`, `scripts/select_week6_policy.py` | split-guard/config tests | completed and owner-approved validation-only selection | selection JSON | COMPLETE; lambda `0.0`, seed `42` |
| W6-11 | Frontier, action/STOP frequencies, and oracle gap | `scripts/eval_frontier.py` | metric tests | Complete Deep Sets and comparison frontiers | frontier/action/oracle CSVs | COMPLETE |
| W6-12 | Resume-safe and provenance-bound outputs | Week 6 scripts | resume/hash tests | Completed/resumed builds plus full validator | manifests | COMPLETE |
| W6-13 | Main, GRU, and masked-slot learned-policy frontiers | generic `scripts/train_policy.py` and `scripts/eval_frontier.py` | encoder/policy tests | complete Deep Sets, GRU-random, and masked-slot validation frontiers | comparison frontiers | COMPLETE |
| W6-14 | Full evidence provides at least the plan risk-register's 10% relative source-bit Macro-F1 gain over clean-only | `scripts/select_week6_policy.py` | configuration/gate tests | three-seed validation selection | signed active-signal gate | COMPLETE; measured gain 30.0581% |

## Hard gates

- `cal` and `test` are forbidden during VOI construction and policy selection.
- A cached next-probe outcome may be used only to form an offline target or an
  explicitly labelled oracle; it may not enter the learned policy input.
- Week 6 refuses to run until a Week 5 freeze manifest names the selected
  checkpoint, config hash, state-manifest hash, and validation evidence.

## Owner-approved training settings

The VOI loss formula and `eta_loss=0.25` are plan-fixed. The owner approved the
validation cost grid, shared optimizer schedule, and non-leaking train-only
expected entropy-reduction baseline on 2026-08-26. The signed Week 5 freeze now
exists and passed full validation; Week 6 readiness and bounded VOI staging are
authorized.
