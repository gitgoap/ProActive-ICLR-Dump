# ProActive human annotation

This folder explains the only manual study required for Week 8. The goal is to
check whether ProActive's **behavioural labels** agree with human reading of the
same clean and probed answers. It does not ask annotators to guess the hidden
cause inside a model.

## What Aman must do

1. Recruit **three different people**. One person must not imitate three people.
2. Give each person only their own `annotator_N/` folder produced by
   `scripts/prepare_human_annotation_packets.py`.
3. Ask each person to fill every row independently. Do not let them discuss
   answers until all three CSV files are returned.
4. Merge the three files with `scripts/merge_human_annotations.py`.
5. For rows where all three labels differ, resolve only the
   `adjudicated_label6` cell while still blinded.
6. Run `scripts/analyze_human_audit.py`. Only this last script reads the private
   teacher-label key.

Plan roughly **2–4 hours per annotator**. All three people can work in parallel,
so start this while the held-out datasets download. The task does not require a
GPU and is unrelated to PRE-HAL or IllusionBench.

Do not send `human_audit_private_key.jsonl`, dataset/model names, teacher scores,
or another annotator's answers to an annotator.

See [ANNOTATOR_GUIDE.md](ANNOTATOR_GUIDE.md) for the one-page instructions to
give annotators and [ADJUDICATION_GUIDE.md](ADJUDICATION_GUIDE.md) for the final
disagreement step.

## Commands after the three files return

Put each completed file back at
`outputs/human_annotation_packets/annotator_N/annotations.csv`, then run:

```bash
python scripts/merge_human_annotations.py \
  --packet_dir outputs/human_annotation_packets \
  --output_dir outputs/human_annotation_merged \
  --overwrite
```

If the merge reports a three-way disagreement, fill only the empty
`adjudicated_label6` cells in
`outputs/human_annotation_merged/human_audit_merged_blinded.csv` while still
blinded. Then run:

```bash
python scripts/analyze_human_audit.py \
  --config configs/experiments/week8_evaluation.yaml \
  --merged_csv outputs/human_annotation_merged/human_audit_merged_blinded.csv \
  --private_key outputs/human_audit/human_audit_private_key.jsonl \
  --output_dir outputs/week8_reports/human_audit \
  --overwrite
```

Finally, run the CPU-only Week 8 completion gate:

```bash
python scripts/validate_week8.py \
  --mode full \
  --config configs/experiments/week8_evaluation.yaml \
  --output_dir outputs/week8_reports \
  --overwrite
```

Week 8 is COMPLETE only if this writes
`outputs/week8_reports/week8_full_validation.json` with `is_valid: true` and no
errors. A low human-agreement result must be reported honestly; it must not be
relabelled to force the gate.
