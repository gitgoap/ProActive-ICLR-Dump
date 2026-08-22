# HallusionBench answer contract and Week 4 cache repair

**Status:** implemented locally; server migration pending  
**Contract version:** 1  
**Source audit date:** 2026-08-19  
**Primary population:** all 951 released image-paired records

## Why this document exists

HallusionBench is predominantly exposed as a binary benchmark, but the released
`HallusionBench.json` contains 14 image-paired table questions that ask for a
country, state, month, or explicit no-answer conclusion. For these rows,
`gt_answer` is a benchmark-level `0/1` indicator rather than the literal answer;
the natural-language reference is stored in `gt_answer_details`.

The original ProActive manifest retained `gt_answer` but not
`gt_answer_details`. This made ordinary binary rows compare model answers such
as `yes` against `1`, and made the 14 open questions impossible to score
correctly. The defect affects answer correctness and downstream labels; it does
not imply that the saved images or raw model generations are corrupt.

## Non-selective dataset policy

ProActive retains all 951 image-paired HallusionBench examples as the primary
evaluation population. The 14 rows are not removed after viewing model
failures. Their type is determined solely from released source question grammar
and official annotation identity before any prediction is consulted.

The paper may additionally report a clearly named 937-row binary-only
sensitivity analysis. It must remain secondary and must not replace the
all-951 primary result.

## Contract v1

For 937 binary rows:

- `answer_type = binary`;
- `answer_match_mode = binary_exact`;
- official indicators normalize deterministically as `0 -> no`, `1 -> yes`,
  and `2 -> uncertain`;
- correctness is exact equality after binary normalization.

For 14 open-ended rows:

- `answer_type = open_ended`;
- `answer_match_mode = exact_alias`;
- the official `gt_answer_details` is preserved;
- versioned canonical aliases are derived from that official detail and stored
  in `configs/data/hallusionbench_open_ended_references.json`;
- correctness is exact equality after free-form normalization against any
  canonical alias or the complete official detail.

The open-ended gold correctness rule deliberately does not use the
VizWiz-calibrated embedding threshold. This avoids accepting semantically
related but factually different short entities. The frozen semantic matcher is
still used for probe-to-clean response stability, which is a different signal
from gold correctness.

## Audited open-ended records

| ProActive instance | Official source key | Question target | Canonical reference |
|---|---|---|---|
| `hallusionbench_218_218` | `VS|table|1|1|0` | largest 2023 population growth | Syria |
| `hallusionbench_221_221` | `VS|table|1|2|0` | largest 2023 population growth | Niger |
| `hallusionbench_227_227` | `VS|table|2|1|0` | highest European GDP in 2021 | Germany |
| `hallusionbench_230_230` | `VS|table|2|2|0` | highest European GDP in 2021 | France |
| `hallusionbench_233_233` | `VS|table|2|3|0` | highest European GDP in 2021 | France |
| `hallusionbench_248_248` | `VS|table|4|1|0` | highest average maximum temperature | Florida |
| `hallusionbench_251_251` | `VS|table|4|2|0` | inconsistent temperature table | unanswerable / no answer / table inconsistent |
| `hallusionbench_254_254` | `VS|table|4|3|0` | highest average maximum temperature | South Carolina |
| `hallusionbench_261_261` | `VS|table|5|1|0` | lowest personal-income change | November 2022, December 2022, and July 2023 |
| `hallusionbench_262_262` | `VS|table|5|1|1` | highest personal-income change | January 2023 |
| `hallusionbench_265_265` | `VS|table|5|2|0` | lowest personal-income change | December 2022 and July 2023 |
| `hallusionbench_266_266` | `VS|table|5|2|1` | highest personal-income change | November 2022 |
| `hallusionbench_269_269` | `VS|table|5|3|0` | lowest personal-income change | unanswerable / no answer |
| `hallusionbench_270_270` | `VS|table|5|3|1` | highest personal-income change | unanswerable / no answer |

The released source also contains four related text-only controls with no image.
They are outside ProActive's preregistered 951 image-paired subset, which
explains why the source search returns 18 related rows but this contract lists
14.

## Cache migration policy

The old cache is immutable evidence. Migration writes a separate directory and
enforces the following rules:

1. The old and new manifests must contain exactly the same instance IDs.
2. Any non-HallusionBench record change stops the migration for inspection.
3. Binary HallusionBench raw generations are reused; normalization,
   correctness, answer-comparison features, and labels are recomputed.
4. All cached observations for every open-ended row are invalidated for every
   model because the prompt and answer domain changed.
5. Existing fail-closed rows remain failures rather than fabricated labels.
6. Every migrated row records source file and record hashes; the migration
   report records both manifest hashes and per-shard counts.
7. A later uniform grounding refresh applies the same maximum generation budget
   to every row, not only difficult examples.

This policy is independent of whether Qwen or Gemma originally answered a row
correctly or whether a row appeared in a failure ledger.

## Paper-facing methods text

> We retained all 951 image-paired HallusionBench examples. Inspection of the
> released annotations identified 14 table questions that are open-ended despite
> the benchmark's predominantly binary interface. For the remaining 937 rows,
> we mapped the released 0/1/2 indicators to no/yes/uncertain and used exact
> normalized matching. For the 14 open-ended rows, we used the released
> `gt_answer_details` field and source-derived canonical aliases with
> deterministic normalized exact matching. Answer type was assigned from source
> question structure and annotation identity without consulting model outputs.

The paper should disclose that this is a deterministic adaptation of the
released annotations rather than the benchmark's optional LLM-as-judge
evaluation. Report the all-951 result first and, if useful, the predeclared
937-row binary-only sensitivity result second.

## Required evidence before final claims

- New HallusionBench manifest contains exactly 951 rows.
- New combined manifest contains exactly 7,291 rows.
- Exactly 14 rows have `answer_type=open_ended`.
- Migration report is valid and records exactly 14 invalidations per model.
- Corrected teacher cache contains exactly 7,291 valid rows per core model with
  zero unresolved failures before label construction.
- The 14 corrected predictions and aliases receive a focused human inspection
  before paper tables are frozen; inspection may discover annotation issues but
  must not silently modify aliases based on model identity.
- Primary and sensitivity denominators are printed in all paper tables.

## Traceability

- Source classification: `src/proactive/data/hallusion_contract.py`
- Loader and provenance: `src/proactive/data/loaders.py`
- Canonical aliases: `configs/data/hallusionbench_open_ended_references.json`
- Correctness matching: `src/proactive/features/semantic.py`
- Cache migration: `scripts/migrate_hallusion_answer_contract.py`
- Validation: `src/proactive/audits/week4_validation.py`
- Server procedure: `doc/docs/WEEK_4_SERVER_EXECUTION.md` §11
- Decision record: `DECISIONS.md`
- Failure chronology: `FAILURE_LOG.md`
- Experiment chronology: `PROJECT_LOG.md`
