# Week 7 Requirements — Stack freeze, final APS, and locked evaluation

**Status:** COMPLETE

**Source of truth:** Plan §§10, 19–21, 25.9, and 30.

## Completion gate

On locked evaluation, ProActive must beat random and at least two fixed
schedules on the aggregate diagnostic frontier, or show a clearly superior
set-size frontier. Deep Sets must beat GRU or match it with materially lower
drift. Empirical 90% and 95% in-distribution coverage must be acceptably close
to target without post-test tuning.

## Requirement traceability

| ID | Requirement | Code location | Unit/adversarial test | Integration test | Required artifact | Status |
|---|---|---|---|---|---|---|
| W7-01 | Freeze encoder, heads, policy, configs, controls, dataset schedule, and hashes before calibration | `scripts/freeze_main_stack.py` | `tests/test_freeze_contracts.py` | freeze integration | `main_stack_freeze.json` | COMPLETE |
| W7-02 | Frozen-policy trajectories for calibration only | `src/proactive/policy/rollout.py` | `tests/test_policy_rollout.py` | calibration trajectory build | trajectory JSONL | COMPLETE |
| W7-03 | Finite-sample APS thresholds per budget at 90% and 95%, including static controls | `src/proactive/conformal/aps.py`, `scripts/calibrate_aps.py`, `scripts/calibrate_static_baseline_aps.py` | `tests/test_aps.py` | calibration integration | APS threshold JSONs | COMPLETE |
| W7-04 | Test remains inaccessible before signed freeze + APS artifacts | `scripts/eval_frontier.py` | `tests/test_week7_lock.py` | locked-test adversary | evaluation manifest | COMPLETE |
| W7-05 | Main quality/cost/set-size frontier at both coverage targets | `src/proactive/eval/frontier.py`, `scripts/eval_frontier.py` | `tests/test_frontier.py` | locked evaluation | main frontier CSV | COMPLETE |
| W7-06 | Full 2/3/4-token permutation protocol with up to 10 permutations and at least 2,000 states | `src/proactive/eval/permutation.py` | `tests/test_permutation_metrics.py` | full permutation run | logs + CSV | COMPLETE |
| W7-07 | JS, bit, set, action, hidden-state, coverage, and set-size drift | `src/proactive/eval/permutation.py` | `tests/test_permutation_metrics.py` | full permutation run | drift CSV | COMPLETE |
| W7-08 | GRU canonical and random-augmentation comparison | `scripts/eval_permutation.py` | encoder tests | full permutation run | comparison CSV | COMPLETE |
| W7-09 | Set Transformer and RAPS decisions use predeclared validation gates only | `scripts/select_week5_encoder.py`, `scripts/validate_week7.py` | configuration/gate tests | validation-only selection + go/no-go report | signed decision fields | COMPLETE |
| W7-10 | No post-test hyperparameter selection | freeze/validation modules | `tests/test_week7_lock.py` | locked-test adversary | immutable freeze manifest | COMPLETE |
| W7-11 | Reproducible paper-facing CSVs and plots | `scripts/eval_frontier.py`, `scripts/eval_permutation.py` | artifact tests | Week 7 validation | CSV + figures | COMPLETE |

## Split firewall

1. Train fits encoders and policy.
2. Validation selects architecture, hyperparameters, cost multipliers, and
   temporary APS used only for VOI labels.
3. Calibration fits final APS after the stack is frozen.
4. Test is evaluated once through a signed freeze/threshold contract.

No script may silently substitute one split for another.

The final CPU-only validator passed on 2026-09-07 with `is_valid=true`, no
errors, and `go_no_go=GO`. Its self-hash is
`47fe3b99974f64be83d00a4b7c6838351d5e90ab65be97d392e7e2da14e9d0d7`.
The optional RAPS appendix gate remains recorded as a warning; it does not
change the frozen APS main result.

The owner approved the five-budget APS grid, 90%/95% targets, and maximum
undercoverage gap `0.03` on 2026-08-26. Week 6 passed its full validator on
2026-09-06. Calibration and test remain locked until the final-budget
validation schedule and signed main-stack freeze exist.
