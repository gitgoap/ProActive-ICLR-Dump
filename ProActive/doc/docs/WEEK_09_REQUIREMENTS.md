# Week 9 Requirements — Paper assets, reproducibility, and release freeze

**Status:** NOT STARTED

**Prerequisites:** The frozen Week 7 stack is immutable. All mandatory Week 8
machine evidence is complete. Human-audit results and full Week 8 validation
remain pending. Week 9 must not introduce new scientific tuning.

**Source of truth:** Plan §23 and §25.11.

## Deadline-first execution order

1. Start writing from the signed results now; do not wait for the human audit.
2. Build a single command that regenerates every main table and figure from
   JSON/CSV evidence without manual spreadsheet editing.
3. Create `repro_manifest.json` binding configs, checkpoints, datasets, result
   tables, figures, seeds, commands, and SHA-256 hashes.
4. Independently review split leakage, calibration provenance, cost accounting,
   missing seeds, and all paper-number-to-artifact links.
5. Insert the returned human-audit results and run full Week 8 validation.
6. Freeze paper assets, appendix, logs, decisions, limitations, risk memo, and
   release checklist.

## Requirement traceability

| ID | Requirement | Evidence required | Status |
|---|---|---|---|
| W9-01 | One-command paper asset generation | `scripts/make_paper_assets.py`, generated main tables/figures, source CSV/JSON hashes | NOT STARTED |
| W9-02 | Reproducibility manifest | `repro_manifest.json` covering data, configs, checkpoints, reports, commands, seeds, and hashes | NOT STARTED |
| W9-03 | Fixed-seed reproducibility decision | Rerun result within tolerance, or an explicit recorded waiver using existing signed multi-seed runs because of the submission deadline | NOT STARTED |
| W9-04 | Independent leakage/calibration/cost audit | Reviewer checklist with no unresolved critical issue | NOT STARTED |
| W9-05 | Paper appendix and limitations | Implementation details, per-slice results, audit, failures, nonclaims, and shift/calibration limitations | NOT STARTED |
| W9-06 | Archive and release freeze | Logs, decisions, named checkpoints, risk memo, release checklist, frozen artifact inventory | NOT STARTED |
| W9-07 | Independent regeneration | A second person can follow the README and regenerate primary tables/figures without hidden settings | NOT STARTED |

## Mandatory main-paper assets

1. Method overview diagram.
2. Cost–diagnostic frontier including clean, fixed, random, architecture, and
   full-teacher controls.
3. Permutation-drift comparison.
4. Leave-one-model-out table.
5. Calibration/coverage/set-size table at 90% and 95% targets.
6. Probe-use heatmap by dataset and predicted source bits.

## Completion gate

Week 9 is COMPLETE only when every table traces to a generated CSV, every
figure traces to a script, hashes and seeds are recorded, no calibration/test
contamination or unresolved critical issue remains, and a second person can
regenerate the primary result package from documented commands.

No new architecture search, dataset expansion, or test/shift tuning is allowed
in this phase. Optional GQA, LODO, full InternVL, Set Transformer, and broad
RAPS work remain deferred unless the submission package is already secure.
