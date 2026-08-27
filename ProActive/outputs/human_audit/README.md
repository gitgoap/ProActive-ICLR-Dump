# ProActive Week 4 human audit

Purpose: validate whether the rule-backed labels are understandable from the
observed model behaviour. This is a behavioural audit, not a causal-source
annotation task.

Use only `human_audit_blinded.csv` and the `images/` directory while annotating.
Do not open `human_audit_private_key.jsonl`; it contains the hidden dataset,
model, teacher scores, and teacher label used only for later analysis.

Three annotators independently fill their own six columns (`ann1_*`, `ann2_*`,
or `ann3_*`). Use 1/0 for the four binary questions and one of
`no-failure`, `visual`, `language-prior`, `alignment`, `mixed`, or `unclear`
for `label6`. Leave `adjudicated_label6` empty until independent annotation is
finished, then resolve disagreements without consulting the private key.

The images are losslessly materialized and renamed by audit ID so their source
dataset is not visible. This packet is for internal research annotation only;
do not redistribute dataset images.
