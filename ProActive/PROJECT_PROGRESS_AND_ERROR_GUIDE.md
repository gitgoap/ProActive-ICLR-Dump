# ProActive progress and recurring-error guide

**Last updated:** 2026-08-26
**Current phase:** Week 5 — shared-state encoder bake-off
**Current status:** Week 4 COMPLETE; Week 5 NOT STARTED

This is the plain-language companion to `PROJECT_STATUS.md`, `PROJECT_LOG.md`,
and `RUN_REGISTRY.md`. Use it to remember what happened, why an error appeared,
whether it damaged an experiment, and what remains.

## What ProActive is doing

ProActive diagnoses why a multimodal model fails. For each image/question, it
first records the model's normal answer. It then applies controlled probes such
as blanking, blur, crop, brightness, noise, grounding, and (where meaningful)
relation changes. Changes in answers and confidence are converted into visual,
language-prior, and alignment evidence. Later weeks train a policy to request
only the most useful probes and stop early.

The project fails closed: an unparseable answer, missing score, missing probe,
duplicate row, provenance mismatch, or incomplete cache is reported as an
error. It is never silently converted into a valid label.

## Work completed so far

| Period | Work completed | Evidence |
| --- | --- | --- |
| Week 1 | Repository structure, schemas, four active dataset loaders, deterministic grouped train/val/cal/test splits, configurable data paths | Manifests and tests |
| Week 2 | Qwen, Gemma, and InternVL adapter work; clean inference and token-score extraction | Qwen/Gemma GPU smoke; InternVL corrected GPU smoke in Week 4 |
| Week 3 | Seven diagnostic probes, deterministic visual transforms, grounding parsing, semantic matching, source bits, six-way labels, pilot runner, reports, inspections, and freeze gates | 10,400 valid pilot rows, zero duplicates, five plots, 250 inspection images, 50 human semantic labels, threshold `0.50`, frozen severities, Week 3 gate passed |
| Week 4 code | Four-way resumable teacher generation, failure ledgers, independent label reconstruction, partial-state sampling, blinded human-audit export, revision inspection, and readiness/progress/full validators | Full server CPU suite: `236 passed` |
| Week 4 data repair | Preserved all 951 HallusionBench image rows, added a correct contract for its 14 open-ended rows, fixed nondeterministic VizWiz gold-answer ties, migrated cached inference with hashes, and passed eight resume-boundary checks | Corrected combined manifest SHA-256 `05fd0dbc...`; no selective exclusions |
| Qwen/Gemma core | Generated every selected model-instance slot, uniformly refreshed grounding at a 1,024-token ceiling, then applied the approved concise recovery to every remaining format failure | Qwen: 7,291 valid; Gemma: 7,291 valid; 14,582 total, zero failures and zero exclusions |
| InternVL runtime | Kept Qwen/Gemma in server `(base)` and created an isolated `proactive-internvl` environment because the checkpoint requires Transformers 4.37.2 | Python 3.11.15, PyTorch 2.6.0+cu124, 11 focused adapter tests and all 236 repository tests passed |
| InternVL GPU validation | Corrected native image tiling/token injection passed 1-row, 10-row, 100-row, and complete-VSR runs | 1/10/100/340 rows, zero generation failures in every stage |
| Offline labels and states | Recomputed labels independently from the frozen Week 3 thresholds and serialized only acquired evidence into pre-policy states | 14,582 unique labels; 388,623 unique states; no duplicate IDs or forbidden learner features; preliminary balance gates pass |
| Human-audit packet | Deterministically selected a blinded packet from all six labels with InternVL catch-up coverage | 180 rows/images; 30 per label; all three models and all four datasets; packet validator passes; human annotations remain |

## Current verified Week 4 numbers

### InternVL

| Stage | Valid rows | Legal probes | Failures | Runtime | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| One row | 1 | 6 | 0 | 101.6 s | `fd813be00a5...` |
| Ten rows | 10 | 60 | 0 | 170.0 s | `635b8095415e...` |
| 100 rows | 100 | 602 | 0 | 1,979.0 s | `c1a447aebad9...` |
| Complete VSR | 340 | 2,150 | 0 | 5,682.5 s | `929b563aa930...` |

The 100-row stage covers HallusionBench, POPE, VizWiz, and VSR. The VSR stage
contains all 340 VSR rows and all 110 relation-applicable rows. Independent
audits found no duplicate IDs, manifest mismatches, invalid legal probes, or
missing provenance hashes.

### Why the complete-VSR command still printed 67 errors

"Zero failures" applies to **InternVL generation**: all 340 requested examples
were generated and written, and the model failure ledger is empty. A separate
schema command then reported 67 errors and exited with code 1. Those 67 are not
67 broken generations:

| VSR split | Rows | Week 3 pilot validator | Week 4 teacher cache |
| --- | ---: | --- | --- |
| train | 244 | accepted | required |
| val | 29 | accepted | required |
| cal | 36 | rejected by design | required |
| test | 31 | rejected by design | required |

The command used `schema_validator.py`, whose Week 3 leakage guard explicitly
allows only `train` and `val`. Therefore it rejected exactly `36 + 31 = 67`
legitimate Week 4 `cal`/`test` rows. The Week 4 validator explicitly allows all
four splits. Keep the report as evidence of a validator-scope mismatch, do not
delete the 67 rows, and do not rerun InternVL because of this message.

### Final Qwen/Gemma recovery

| Model | Valid | Failure ledger | Accounted | Meaning |
| --- | ---: | ---: | ---: | --- |
| Qwen3-VL-8B | 7,291 | 0 | 7,291 | All six prior format failures recovered under the approved uniform prompt |
| Gemma-4-E4B | 7,291 | 0 | 7,291 | All three prior format failures recovered under the same policy |

No datapoint was deleted. The recovered cache records 14,573 unchanged copied
rows and exactly nine recovered rows with source hashes and policy provenance.
Official teacher-progress validation reports 14,582 rows, 87,712 unique probes,
and zero errors.

## The errors and warnings we have seen

| Message or symptom | Simple explanation | Did it damage accepted data? | Resolution/status |
| --- | --- | --- | --- |
| Duplicate pilot rows | An early 32-row smoke file was opened in append mode and the later seeded 100-row run repeated 31 IDs | The original file was invalid, but the final Week 3 cache was repaired | Append is now refused unless explicit safe resume is used; final pilot has zero duplicates |
| `pytest: command not found` | The active server environment did not expose pytest | No | Pytest was skipped temporarily; it is now installed in the InternVL environment and all 236 tests pass |
| `--limit: command not found` | A backslash had a trailing space, so Bash ended the Python command early and treated the next line as a new command | No model run occurred | Backslash must be the final character on the line |
| `Permission denied` after typing a YAML path | A configuration filename was entered as though it were an executable command | No | YAML files are inputs to Python commands; they are not run directly |
| Readiness missing `WEEK_04_REQUIREMENTS.md` | A required documentation file had not yet been synced to the server | No | File was synced; readiness passed |
| GPU OOM | A GPU looked partly utilized but another process had already reserved most VRAM | Failed rows remained in ledgers; completed rows survived | Check process-level VRAM, use only genuinely free GPUs, and resume safely |
| Terminal exits with code 1 | Usually a fail-closed command found an incomplete cache, invalid row, missing file, or failed model row | This protects data rather than corrupting it | Read the preceding error; do not treat every exit 1 as the same problem |
| HallusionBench answer-contract problem | Fourteen released image questions are genuinely open-ended, but the old loader treated a benchmark indicator as a literal yes/no answer | Old Hallusion correctness was invalidated, not accepted | Retained all 951 rows, added official references/aliases, migrated unaffected inference, reran changed rows |
| Migration said non-Hallusion rows changed | VizWiz used an unordered Python set to break tied answers, so different hash seeds selected different gold text for 207 rows | The first migration was rejected before acceptance | Replaced it with normalized majority plus released source-order tie breaking; two rebuild seeds now hash identically |
| Frozen-config hash mismatch | Windows synchronization changed LF line endings to CRLF, changing the byte-level SHA-256 even though YAML meaning was unchanged | Run correctly stopped | Restored the exact LF bytes and verified the recorded hash before migration |
| `Missing FINAL_ANSWER` / malformed grounding | The model produced long reasoning or an unknown answer but did not finish the required tagged final line | Yes, that individual teacher row is invalid and stays outside labels | Parser recovery plus uniform 512/1,024-token grounding refresh reduced the problem to nine explicit rows |
| Grounding refresh exit code 1 | At least one row in that shard still failed the mandatory grounding contract | Successful rows remained valid; failures remained auditable | Resolved: all nine final failures recovered uniformly and the final ledgers are empty |
| `No module named einops` | InternVL custom code required a package absent from base | No row was produced | Built a separate pinned InternVL environment |
| `all_tied_weights_keys` attribute error | InternVL's downloaded custom code is incompatible with Transformers 5.5.4 in base | No row was produced | Preserved base and used Transformers 4.37.2 in `proactive-internvl` |
| Pip CA certificate error and disappearing environment | The first unpinned environment creation selected a pip/certificate combination with a broken CA path and did not leave a registered environment | No experiment data affected | Recreated with Python 3.11, pip 25.2, certifi, CA certificates, and OpenSSL pinned |
| `FlashAttention2 is not installed` | InternVL used standard eager attention instead of an optional acceleration library | No; this is a warning, not a failure | Eager attention is the documented deterministic fallback; outputs are valid but slower |
| Zero-byte `.lock` file | The writer leaves a lock filename used to prevent simultaneous writes; the operating-system lock is released when the process exits | No | Harmless when no writer process is running; do not manually merge JSONL files |
| InternVL schema report says `cal`/`test` invalid | We accidentally ran the Week 3 pilot validator, which deliberately accepts only train/val, on a Week 4 teacher cache that must contain all four splits | No. All 100 and 340 teacher rows independently pass Week 4 integrity checks | Do not rerun inference. Use the Week 4 validator for full caches; retain this report as a tooling-scope warning |
| `Unsafe audit resume refused: Audit manifest is missing` | The audit output directory was created before the first export, but `--resume` means “validate an already completed packet” | No audit data were written or lost | The first export uses `--overwrite`; use `--resume` only after `human_audit_manifest.json` exists |
| Full Week 4 report false although every artifact passes | The server used an older InternVL catch-up Markdown file, so the validator could not find its documented completion status | No; teacher, labels, states, and audit all independently reported valid | Resolved after syncing the document; the repeated full report is valid |

## Environment rule to remember

| Work | Environment |
| --- | --- |
| Qwen, Gemma, manifests, migration, labels, states, and normal validators | Existing server `(base)` |
| InternVL only | `source ~/miniconda3/etc/profile.d/conda.sh` then `conda activate proactive-internvl` |

Never downgrade `(base)`. Each new tmux pane must activate
`proactive-internvl` independently before an InternVL command.

## Week 4 closure and what comes next

Week 4 is complete. The full validator passes with 14,582 teachers, 14,582
labels, 388,623 states, and a valid 180-row audit packet. The next engineering
work is Week 5 encoder training. Human annotation of the packet should begin in
parallel and must finish before reporting agreement metrics, but it does not
block cached-feature encoder implementation.

Do not run a full 7,291-row InternVL cache for Week 4. The documented catch-up
and validated audit coverage satisfy the frozen completion rule.
