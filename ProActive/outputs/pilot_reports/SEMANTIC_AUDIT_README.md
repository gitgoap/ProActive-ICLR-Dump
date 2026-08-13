# Semantic-Match Human Audit

## Why this audit exists

ProActive compares the clean answer with answers produced after diagnostic probes. For binary datasets, normalized exact matching is sufficient. VizWiz uses free-form answers, so two differently worded answers may still mean the same thing.

The file `semantic_match_audit.csv` contains 50 non-exact VizWiz answer pairs sampled across the observed cosine-similarity range. Human judgements from this file are used to select the embedding-similarity threshold using **train/validation records only**. The threshold must be frozen before Week 4 teacher-cache generation and must never be selected using calibration or test data.

This is an answer-equivalence audit. The annotator should judge whether the two answers convey essentially the same answer—not whether either answer is factually correct from the image.

## Annotation record

Complete this section when the CSV is labelled.

- **Annotation date:** `08-08-2026`
- **File labelled:** `outputs/pilot_reports/semantic_match_audit.csv`
- **Number of rows:** `50`

## How to label

Open `semantic_match_audit.csv` in Excel, LibreOffice, or another CSV-safe editor. Do not reorder rows or modify columns other than `human_match`.

For every row, compare `clean_answer` and `probe_answer`, then enter exactly one value in `human_match`:

- `1` — the answers have essentially the same meaning.
- `0` — the answers differ, contradict each other, identify different things, or one answer is too vague/unresolved to establish equivalence.

Examples:

| Clean answer | Probe answer | Label | Reason |
|---|---|---:|---|
| `white and pink` | `pink and white` | 1 | Same information. |
| `laptop` | `computer` | 1 | Equivalent at the required answer granularity. |
| `dog` | `cat` | 0 | Different object. |
| `unanswerable` | `coffee maker` | 0 | One answer does not resolve to the other. |
| `windows error screen` | `screen` | 0 | The probe answer loses the information needed to establish equivalence. |

## Important rules

1. Judge semantic equivalence only; do not use dataset or model identity as evidence.
2. Do not look at calibration/test records or downstream performance.
3. Do not change `cosine_similarity`, `current_match`, answers, IDs, or row order.
4. Label all 50 rows. Blank or non-binary labels make calibration fail closed.
5. If genuinely uncertain, use `0`; equivalence must be supported rather than assumed.
6. Save as CSV without changing the header or delimiter.

## Calibration command

After completing the CSV and the annotation record above, sync both files to the server and run bash.

The resulting report records the annotator, UTC calibration timestamp, source CSV hash, label counts, selected threshold, and calibration metrics.

## Expected artifacts

- `semantic_match_audit.csv` — immutable answer pairs plus completed human labels.
- `SEMANTIC_AUDIT_README.md` — instructions and human annotation record.
- `semantic_calibration_report.json` — machine-readable provenance and selected threshold.

Retain all three files with the frozen Week 3 configuration.
