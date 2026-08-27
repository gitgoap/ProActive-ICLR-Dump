# ProActive Weekly Experiment Summary

**Last updated:** 2026-08-27  
**Purpose:** one paper-facing index of frozen settings, measured results, and
remaining human work. Generated reports and signed manifests remain the
authoritative machine-readable evidence.

## At a glance

| Week | Status | Frozen or approved settings | Measured evidence | Next gate |
|---|---|---|---|---|
| 1–2 | COMPLETE | Seed `42`; grouped split construction; configurable data root; Qwen, Gemma, and InternVL adapters | Four active datasets and deterministic manifests established | None |
| 3 | COMPLETE | Blur `8`; crop `0.65`; brightness `0.15`; noise `25`; semantic threshold `0.50` selected at target recall `0.90` | 10,400 valid pilot records; no duplicates/schema failures; semantic precision `0.5926`, recall `0.9412`, F1 `0.7273` from 50 human labels | None |
| 4 | COMPLETE | Two-family core plus InternVL audit catch-up; collapse gate `0.80`; minimum bit balance `5/5`; human-audit design `60` natural + `120` targeted | 14,582 Qwen/Gemma teachers; 87,712 probe observations; 14,582 labels; 388,623 states; final validator passed with zero errors/warnings | Three-person annotation and agreement analysis continue in parallel |
| 5 | COMPLETE | AdamW `1e-3`; weight decay `1e-4`; batch `512`; maximum `30` epochs; patience `5`; budgets `1/2/4`; seeds `42/43/44`; selected Deep Sets seed 42 | Owner-approved freeze `0512ca85...a7929`; full validator passed with zero errors/warnings; calibration/test remained locked | None |
| 6 | IMPLEMENTED, NOT VALIDATED | Same optimizer schedule; budgets `1/2/4`; cost grid `0/0.05/0.1/0.2/0.4`; train-only expected entropy-reduction baseline; active-signal gate `0.10` | Code and tests are present; Week 5 prerequisite passed | CPU readiness and bounded VOI staging |
| 7 | IMPLEMENTED, NOT VALIDATED | Final budgets `1/2/3/4/7`; APS coverage `0.90/0.95`; maximum undercoverage gap `0.03`; permutation sizes `2/3/4`, at least 2,000 states, up to 10 permutations | Code and tests are present; calibration/test are locked | Signed Week 6 selection and explicit main-stack freeze |

## Week 5 approved architecture matrix

The mandatory comparison contains 15 full checkpoints:

| Condition | Seeds | Full runs |
|---|---|---:|
| Clean-only MLP | 42, 43, 44 | 3 |
| Canonical-order GRU | 42, 43, 44 | 3 |
| Random-permutation-trained GRU | 42, 43, 44 | 3 |
| Masked-slot MLP | 42, 43, 44 | 3 |
| Deep Sets | 42, 43, 44 | 3 |

As of the latest 2026-08-26 sync, all `15/15` validation reports are complete,
covering the five declared conditions over seeds 42/43/44. All `12/12`
non-clean permutation studies, all `12/12` associated temporary APS reports,
and the identity shortcut-control report are also present. The completed
checkpoints bind approved config SHA-256 `0293d3bd...f3d74` and approved
vector-manifest file SHA-256 `1fac2c34...201ae`.

The seed-42 permutation reports are both valid over 100 sampled states. Deep
Sets has exactly zero probability, hidden-state, bit-prediction, and prediction-
set drift across all tested permutations. Canonical GRU is order-sensitive, as
expected: mean Jensen-Shannon drift `0.000586`, mean relative hidden drift
`0.07684`, mean bit-probability L1 drift `0.02585`, and mean prediction-set
disagreement `0.08522`. These are validation diagnostics, not final test
results; the checks must still be repeated for every applicable seed/condition.

The completed three-seed validation means are: Deep Sets source-bit/six-way
Macro-F1 `0.864217/0.508031`; canonical GRU `0.863926/0.500815`; random-order
GRU `0.864333/0.494725`; masked-slot MLP `0.864605/0.499082`; and clean-only
MLP `0.751657/0.272292`. Deep Sets and masked-slot MLP have zero permutation
drift at every seed. The best GRU by the primary metric is random-order GRU,
whose mean JS drift is `0.000442`; its primary advantage over Deep Sets is only
`0.000116`, well inside the approved `0.01` tolerance. The identity-only
source-bit Macro-F1 is `0.766506`, giving Deep Sets a `0.097711` advantage over
the shortcut-control threshold of `0.02`. The evidence therefore predicts a
passed Deep Sets selection gate and no Set Transformer trigger, subject to the
signed selection script's hash checks. The predeclared RAPS appendix gate is
triggered (mean 90%-APS set size `3.2087`, singleton rate `0.2075`); it is an
appendix efficiency ablation and does not replace or block the main APS stack.

The signed CPU-only selection report then passed with status `SELECTED`, chose
Deep Sets seed 42, and recorded report SHA-256 `03a2e49f...36916`. It confirms
`calibration_used=false`, `test_used=false`, Set Transformer `NOT_TRIGGERED`,
shortcut gate `PASSED`, and RAPS `TRIGGERED_APPENDIX_ABLATION`. Explicit owner
approval was granted on 2026-08-26. The resulting freeze and full validator
both passed; Week 5 is COMPLETE.

Model selection uses validation only. Calibration and test are unavailable to
Week 5 scripts. The Set Transformer is not run unless the predeclared
validation trigger fires; RAPS remains an appendix-only conditional ablation.

## Human-audit status

The Week 4 audit packet is structurally complete but not human-annotated:

- packet: `outputs/human_audit/`;
- rows/images: `180/180`;
- independent annotator blocks completed: `0/540`;
- adjudicated labels completed: `0/180`.

Use only `outputs/human_audit/human_audit_blinded.csv` and its `images/`
directory while annotating. Do not open the private key. Three different
people independently fill `ann1_*`, `ann2_*`, and `ann3_*`; one person must
not simulate three annotators. After all three blocks are frozen, disagreements
are adjudicated without viewing the private key. Agreement with the hidden
rule-backed labels is computed only afterward.

This annotation can run in parallel with Weeks 5–7 and uses no GPU. It is
needed for the paper's human-validity/agreement evidence, but it does not
authorize changing already frozen labels after observing model results.

## Evidence locations

| Evidence | Location |
|---|---|
| Scientific decisions | `DECISIONS.md` |
| Chronological engineering record | `PROJECT_LOG.md` |
| Current phase and blockers | `PROJECT_STATUS.md` |
| Week 3 frozen probe settings | `configs/probes/frozen_week3_config.yaml` |
| Week 4 final report | `outputs/week4_reports/final/week4_full_report.json` |
| Week 5 configuration | `configs/experiments/diag_bakeoff.yaml` |
| Week 5 vectorized manifest | `outputs/week5_data/vectorized_manifest.json` |
| Weeks 6–7 configurations | `configs/experiments/policy_train.yaml`, `configs/experiments/calibrate_aps.yaml` |

Update this file only from validated reports/logs. Never copy a pilot metric
into the final-result column or report a GPU result without its server log.
