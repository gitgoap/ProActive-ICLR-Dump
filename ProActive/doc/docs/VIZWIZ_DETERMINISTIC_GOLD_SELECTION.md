# VizWiz deterministic gold-answer selection

**Status:** implemented locally; server manifest rebuild pending  
**Contract version:** 1  
**Policy:** `normalized_majority_source_order_tiebreak_v1`

## Problem found by the migration guard

VizWiz supplies multiple annotator answers per question. The original loader
selected a majority answer with:

```python
max(set(answer_texts), key=answer_texts.count)
```

When multiple answers had the same maximum count, the result depended on
Python's randomized set iteration. Rebuilding the same 3,000-row source on
2026-08-20 changed 207 `gold_answer` fields while instance IDs, questions,
groups, splits, and all other non-Hallusion fields remained fixed. The strict
cache migration correctly stopped rather than accepting this label drift.

## Frozen deterministic policy

For each record:

1. Normalize every released annotator answer with ProActive's frozen VizWiz
   free-form normalizer.
2. Count normalized answers, so punctuation/article variants aggregate.
3. Select the normalized answer with maximum count.
4. If multiple normalized answers remain tied, select the first tied answer in
   released annotation order.
5. Store the selected gold, complete normalized count table, tie size, policy
   identifier, and contract version in the manifest.

The source-order rule is deterministic and auditable. It does not inspect model
outputs. The loader no longer uses unordered set iteration.

## Cache handling

The question, image, and model prompt do not depend on the selected VizWiz gold
answer. Existing raw clean/probe generations therefore remain reusable.
Migration updates the manifest answer metadata, recomputes clean correctness
from the saved raw clean answer, and recomputes gold-dependent teacher labels.
No VizWiz model inference is required solely because of this repair.

## Paper disclosure

> VizWiz provides multiple human answers per image-question pair. We aggregated
> answers after the frozen free-form normalization and selected the normalized
> modal answer. Remaining count ties were resolved by the first tied answer in
> released annotation order, avoiding hash-dependent unordered-set behavior.

This is ProActive's deterministic single-target adaptation, not VizWiz's full
soft VQA scoring protocol. Before paper tables are frozen, report the number of
tied records from the final manifest and consider a sensitivity check using
soft consensus scoring if it changes conclusions.

## Traceability

- Loader/policy: `src/proactive/data/loaders.py::select_vizwiz_gold`
- Frozen config: `configs/data/vizwiz.yaml`
- Manifest validation: `src/proactive/data/manifests.py`
- Cache migration: `scripts/migrate_hallusion_answer_contract.py`
- Regression tests: `tests/test_vizwiz_answer_contract.py`
- Failure record: `FAILURE_LOG.md`
- Experiment chronology: `PROJECT_LOG.md`
