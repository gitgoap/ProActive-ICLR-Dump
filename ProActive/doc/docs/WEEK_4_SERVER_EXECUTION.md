# Week 4 Server Execution

These commands are staged deliberately. Do not start the full four-GPU cache
until the model revisions and draft scientific settings are approved.

## 1. Existing environment verification and revision evidence

Use the Python environment already present in the tmux pane. No explicit Conda
activation is required. Run the verification lines again in each new pane
because environment variables are local to that shell.

```bash
cd ~/ProActive

which python
python --version
python -c "import torch, transformers, yaml; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"

export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004
mkdir -p outputs/logs/week4 outputs/teacher_core outputs/labels_core outputs/states_v1 outputs/week4_reports
set -o pipefail

python scripts/inspect_model_revisions.py \
  --model_configs configs/models/qwen3_vl_8b.yaml configs/models/gemma4_e4b.yaml configs/models/internvl3_9b.yaml \
  2>&1 | tee outputs/logs/week4/model_revision_inspection.log
```

Send the complete revision-inspection output back before editing the model
YAMLs. The corrected inspector reads repository commits separately from
per-file Hugging Face ETags. The expected result is `RESOLVED` with exactly one
revision candidate per model. A genuinely unresolved or ambiguous path is a
blocker; do not replace it with the current remote `main` hash.

Repeat the verification and `export` lines in every new tmux pane. No
environment activation command is needed, but exports set in one pane do not
propagate to the other panes.

After the pinned and approved configs are synced, run the CPU readiness gate:

```bash
python scripts/validate_week4.py \
  --mode readiness \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --output_dir outputs/week4_reports \
  --overwrite \
  2>&1 | tee outputs/logs/week4/readiness.log
```

It must report `"is_valid": true`, `"approval_status": "APPROVED"`, and an
empty `errors` list.

## 2. No separate dry run required

The local and server readiness gates cover the CPU-only contract checks. Skip a
separate teacher dry run and proceed to the mandatory one-row GPU stage only
after readiness reports no errors.

## 3. Staged GPU checks after revision evidence and approval cards

After the owner-approved config is synced, run a one-row check in a separate
staging directory, then a ten-row check. These are validation artifacts, not
part of the full cache.

Approved staged-card estimate: approximately 3,274 MLLM passes across the
Qwen 1/10/100/full-VSR checks and Gemma one-row check, using the Week 3 measured
rate of 1.1694 seconds/pass. Expected cost is about 1.06 GPU-hours, about 1.1
hours sequential wall time on one A6000 before loading overhead, and under
10 MB of JSONL/log output. The 33.23 GPU-hour full core remains unapproved and
is blocked by `compute_authorization.full_core_approved: false`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --limit 1 \
  --output_dir outputs/week4_staging/limit1 \
  --resume \
  2>&1 | tee outputs/logs/week4/qwen_limit1.log
```

Review this output, then run the corresponding one-row Gemma check:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model gemma4_e4b \
  --device cuda:0 \
  --limit 1 \
  --output_dir outputs/week4_staging/limit1 \
  --resume \
  2>&1 | tee outputs/logs/week4/gemma_limit1.log
```

Only after both one-row outputs are reviewed, run the Qwen ten-row check:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --limit 10 \
  --output_dir outputs/week4_staging/limit10 \
  --resume \
  2>&1 | tee outputs/logs/week4/qwen_limit10.log
```

The next mandatory stages are a 100-row run and one complete model-dataset
validation. Do not skip them even when the 1/10-row stages pass.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --limit 100 \
  --output_dir outputs/week4_staging/limit100 \
  --resume \
  2>&1 | tee outputs/logs/week4/qwen_limit100.log

CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_vsr.jsonl \
  --model qwen3_vl_8b \
  --dataset vsr \
  --device cuda:0 \
  --output_dir outputs/week4_staging/qwen_vsr_complete \
  --resume \
  2>&1 | tee outputs/logs/week4/qwen_vsr_complete.log
```

## 4. Full deterministic-shard launch (approved 2026-08-13)

All staged gates passed and the owner approved the conservative 33.23 GPU-hour
Qwen+Gemma core. Each model has four deterministic shards. Start a shard only
on a physical GPU that `nvidia-smi` shows as free immediately before launch.
`CUDA_VISIBLE_DEVICES=N` exposes physical GPU N as logical `cuda:0`, so every
command below uses `--device cuda:0`.

At approval time only physical GPU 1 was free. The immediate launch is Qwen
shard 0:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_teacher.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b \
  --device cuda:0 \
  --num_shards 4 \
  --shard_id 0 \
  --output_dir outputs/teacher_core \
  --resume \
  2>&1 | tee outputs/logs/week4/full_core/qwen_shard00-of-04.log
```

When other GPUs become free, launch Qwen shard IDs 1, 2, and 3 in separate
panes, changing both `CUDA_VISIBLE_DEVICES` and `--shard_id` and using distinct
log filenames. Never run two processes for the same model/shard output. After
all four Qwen shards validate, run the same four shard IDs for
`--model gemma4_e4b`. Interrupted shards resume with the identical command.

## 5. Daily/cache-end integrity check

```bash
python scripts/validate_week4.py \
  --mode teacher_progress \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --teacher_path outputs/teacher_core \
  --output_dir outputs/week4_reports \
  --resume \
  2>&1 | tee outputs/logs/week4/teacher_progress.log
```

## 6. Offline labels and states

Run only after both core model caches are complete.

```bash
python scripts/build_labels.py \
  --config configs/experiments/teacher_core.yaml \
  --teacher_path outputs/teacher_core \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --output_dir outputs/labels_core \
  --resume

python scripts/sample_states.py \
  --config configs/experiments/teacher_core.yaml \
  --teacher_path outputs/teacher_core \
  --labels_path outputs/labels_core \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --output_dir outputs/states_v1 \
  --resume
```

## 7. Human-audit packet

The literal plan requires all three models. Run this after InternVL is present.
For an explicitly interim two-model packet only, add
`--allow_incomplete_model_coverage`; it cannot pass the final gate.

InternVL must use the isolated `proactive-internvl` environment documented in
`doc/docs/WEEK_4_INTERNVL_CATCHUP.md`. Do not downgrade the Qwen/Gemma base
environment.

```bash
python scripts/export_human_audit.py \
  --config configs/experiments/teacher_core.yaml \
  --teacher_path outputs/teacher_core \
  --labels_path outputs/labels_core \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --output_dir outputs/human_audit \
  --resume
```

## 8. Full completion gate

```bash
python scripts/validate_week4.py \
  --mode full \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --teacher_path outputs/teacher_core \
  --labels_path outputs/labels_core \
  --states_path outputs/states_v1 \
  --audit_dir outputs/human_audit \
  --output_dir outputs/week4_reports \
  --resume \
  2>&1 | tee outputs/logs/week4/full_validation.log
```

Do not delete or merge shard files manually. Resume the exact same command when
a pane stops; the script validates every existing row before appending only
missing model-instance keys.

## 9. Qwen 53-row fail-closed recovery (added 2026-08-16)

The first Qwen full pass saved 7,238/7,291 valid rows and left 53 mandatory
grounding failures. This is not a smoke test and it is not a second full run:
`--resume` validates the 7,238 existing rows and regenerates only the missing
rows. Finish any currently running Gemma job first, then verify that the chosen
physical GPU is free.

Only these two runtime files are required on the server before recovery:

```text
src/proactive/prompts/templates.py
scripts/run_teacher.py
```

Do **not** overwrite the three server YAMLs or the combined manifest during
this recovery sync. The Windows copies use CRLF line endings, while the
existing teacher rows record the server LF byte hashes. The scientific content
is identical, but resume correctly treats any byte-hash change as provenance
drift. Before recovery, the server must print these exact hashes:

```bash
sha256sum \
  outputs/manifests/manifest_combined.jsonl \
  configs/probes/frozen_week3_config.yaml \
  configs/experiments/teacher_core.yaml \
  configs/models/qwen3_vl_8b.yaml
```

```text
b945f3f03d25024a9d693c069575c13dd7da8094a664b08bc16614cfbaf40de3  outputs/manifests/manifest_combined.jsonl
5cfdbcde50b28f7645ad4d73046d9997ad393dde3b3c197a3c5d630b5f6d5271  configs/probes/frozen_week3_config.yaml
9820cf2c566ddad75c218842cb410813ca8ab9c8a7fa958d2053bf12958f5481  configs/experiments/teacher_core.yaml
662e4dfc560a55cdb97e8f85217f3c33afd17f2aee23148d83ed25d44c1c2434  configs/models/qwen3_vl_8b.yaml
```

Stop and report the output if any hash differs; do not use `--overwrite`.

Also sync the following when keeping the server clone aligned with the local
repository, but they are not required to execute the GPU recovery:

```text
tests/test_grounding_parsing.py
tests/test_teacher_failure_ledger.py
doc/docs/WEEK_4_SERVER_EXECUTION.md
```

In one tmux pane, run the following. Physical GPU 1 is shown as the example;
change only `CUDA_VISIBLE_DEVICES` if a different physical GPU is actually
free. Because that one GPU is exposed to the process, `--device` remains
`cuda:0`.

```bash
cd ~/ProActive

which python
python --version

export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004

mkdir -p outputs/logs/week4/full_core outputs/teacher_core outputs/week4_reports
set -o pipefail

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 python scripts/run_teacher.py \
    --config configs/experiments/teacher_core.yaml \
    --manifest_path outputs/manifests/manifest_combined.jsonl \
    --model qwen3_vl_8b \
    --device cuda:0 \
    --num_shards 4 \
    --shard_id "$SHARD" \
    --output_dir outputs/teacher_core \
    --resume \
    2>&1 | tee "outputs/logs/week4/full_core/qwen_recovery_shard0${SHARD}-of-04.log"
  echo "Qwen recovery shard ${SHARD} exit code: ${PIPESTATUS[0]}"
done
```

The loop deliberately continues if one shard still has a fail-closed row so
all four failure ledgers are produced for inspection. After it ends, run:

```bash
python scripts/validate_week4.py \
  --mode teacher_progress \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --teacher_path outputs/teacher_core \
  --output_dir outputs/week4_reports \
  --overwrite \
  2>&1 | tee outputs/logs/week4/qwen_recovery_validation.log

wc -l outputs/teacher_core/teacher_qwen3_vl_8b_all_all_shard*-of-04.jsonl
find outputs/teacher_core -maxdepth 1 \
  -name 'teacher_qwen3_vl_8b_all_all_shard*-of-04.failures.jsonl' \
  -print -exec wc -l {} \;
```

Expected Qwen total is exactly 7,291 teacher rows. Zero-row failure files are
good, and it is also valid for no failure sidecar to exist when every pending
row succeeds. If any failure file is nonempty, sync the existing
`*.failures.jsonl` files and the recovery logs for review; do not proceed to
label generation yet.

## 10. Uniform two-model grounding refresh after truncation diagnosis

The parser-only retry showed that most unresolved Qwen outputs were truncated
at 256 tokens. Do not retry the original cache again. The refresh creates a
separate cache and gives every Qwen and Gemma instance exactly one 512-token
grounding pass, so the larger budget is not selected only for failures.

Runtime files required after the previous recovery sync:

```text
src/proactive/prompts/templates.py
scripts/refresh_grounding_cache.py
```

Do not sync YAMLs, manifests, or overwrite `outputs/teacher_core`. Run Qwen on
physical GPU 1 and Gemma on physical GPU 0 in separate tmux panes. Both use
logical `cuda:0` after `CUDA_VISIBLE_DEVICES` isolation.

First run the two CPU-only source-coverage checks. They are not smoke tests and
perform no inference:

```bash
cd ~/ProActive

export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004

mkdir -p outputs/logs/week4/grounding512 outputs/teacher_core_grounding512

for MODEL in qwen3_vl_8b gemma4_e4b; do
  for SHARD in 0 1 2 3; do
    python scripts/refresh_grounding_cache.py \
      --config configs/experiments/teacher_core.yaml \
      --manifest_path outputs/manifests/manifest_combined.jsonl \
      --model "$MODEL" --shard_id "$SHARD" --num_shards 4 \
      --input_dir outputs/teacher_core \
      --output_dir outputs/teacher_core_grounding512 \
      --max_new_tokens 512 --device cpu --resume --dry_run || exit 1
  done
done
```

Each must report a complete source shard and no coverage mismatch. Then run
this in the GPU 1/Qwen pane:

```bash
cd ~/ProActive
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004
mkdir -p outputs/logs/week4/grounding512 outputs/teacher_core_grounding512
set -o pipefail

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 python scripts/refresh_grounding_cache.py \
    --config configs/experiments/teacher_core.yaml \
    --manifest_path outputs/manifests/manifest_combined.jsonl \
    --model qwen3_vl_8b --shard_id "$SHARD" --num_shards 4 \
    --input_dir outputs/teacher_core \
    --output_dir outputs/teacher_core_grounding512 \
    --max_new_tokens 512 --device cuda:0 --resume \
    2>&1 | tee "outputs/logs/week4/grounding512/qwen_shard0${SHARD}.log"
  echo "Qwen grounding refresh shard ${SHARD}: ${PIPESTATUS[0]}"
done
```

Run this simultaneously in the GPU 0/Gemma pane:

```bash
cd ~/ProActive
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004
mkdir -p outputs/logs/week4/grounding512 outputs/teacher_core_grounding512
set -o pipefail

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 python scripts/refresh_grounding_cache.py \
    --config configs/experiments/teacher_core.yaml \
    --manifest_path outputs/manifests/manifest_combined.jsonl \
    --model gemma4_e4b --shard_id "$SHARD" --num_shards 4 \
    --input_dir outputs/teacher_core \
    --output_dir outputs/teacher_core_grounding512 \
    --max_new_tokens 512 --device cuda:0 --resume \
    2>&1 | tee "outputs/logs/week4/grounding512/gemma_shard0${SHARD}.log"
  echo "Gemma grounding refresh shard ${SHARD}: ${PIPESTATUS[0]}"
done
```

Expected refreshed shard sizes are Qwen `1822, 1821, 1801, 1847` and Gemma
`1836, 1787, 1826, 1842`. Validate only after both loops finish:

```bash
python scripts/validate_week4.py \
  --mode teacher_progress \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --teacher_path outputs/teacher_core_grounding512 \
  --output_dir outputs/week4_reports/grounding512 \
  --overwrite \
  2>&1 | tee outputs/logs/week4/grounding512/validation.log
```

## 11. HallusionBench and VizWiz answer-contract v1 recovery (supersedes §10)

The official source audit found 14 open-ended table questions inside the 951
image-paired HallusionBench rows. Separately, the first rebuild exposed 207
VizWiz `gold_answer` changes caused by the old use of an unordered Python set
when multiple answers tied for the majority. Do not exclude either group and do
not build labels from the old manifest/cache. The corrected VizWiz policy first
normalizes released answers, selects the maximum count, and resolves any
remaining tie by released annotation order. This procedure preserves every
instance, relabels reused VizWiz inference on CPU, reruns only
prompt-invalidated or already failed base rows, and then applies one uniform
grounding budget to all rows. The 512-token cache remains useful diagnostic
evidence but is not the final Week 4 teacher source after the manifest changes.

Sync these runtime/config files before running anything:

```text
configs/data/hallusionbench.yaml
configs/data/hallusionbench_open_ended_references.json
configs/data/vizwiz.yaml
src/proactive/data/hallusion_contract.py
src/proactive/data/loaders.py
src/proactive/data/manifests.py
src/proactive/features/normalization.py
src/proactive/features/semantic.py
src/proactive/prompts/templates.py
src/proactive/probes/probe_runner.py
src/proactive/teacher/cache_builder.py
src/proactive/audits/week4_validation.py
scripts/build_manifests.py
scripts/migrate_hallusion_answer_contract.py
scripts/run_teacher.py
scripts/refresh_grounding_cache.py
scripts/validate_week4.py
```

The matching tests and the two audit notes
`doc/docs/HALLUSIONBENCH_14_OPEN_ENDED_RECORDS.md` and
`doc/docs/VIZWIZ_DETERMINISTIC_GOLD_SELECTION.md` should also be synced for repository
traceability, although the tests are not imported by the runtime commands.

Do **not** upload the Windows copy of
`configs/probes/frozen_week3_config.yaml`. Its CRLF line endings change the
byte-level provenance hash even when the scientific YAML content is identical.
Existing teacher rows require the original server LF hash
`5cfdbcde50b28f7645ad4d73046d9997ad393dde3b3c197a3c5d630b5f6d5271`.
If it was uploaded accidentally, use the exact-hash-guarded LF restoration
below before migration; never bypass or rewrite the provenance field.

### 11.1 Preserve provenance, rebuild, and migrate on CPU

Run the provenance backup only if it does not already exist. The expected old
combined hash is the hash embedded in the existing teacher rows. If the backup
already exists, preserve it: do not replace it with the currently rebuilt
manifest.

```bash
cd ~/ProActive

which python
python --version

export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004

mkdir -p outputs/logs/week4/hallusion_contract \
  outputs/week4_reports \
  outputs/teacher_core_contract_v1

cp -n outputs/manifests/manifest_combined.jsonl \
  outputs/manifests/manifest_combined_pre_hallusion_contract.jsonl

sha256sum outputs/manifests/manifest_combined_pre_hallusion_contract.jsonl
```

The preserved backup hash must be
`b945f3f03d25024a9d693c069575c13dd7da8094a664b08bc16614cfbaf40de3`.
Stop if it differs. Then rebuild all active manifests (the builder now refuses
to write a partial combined manifest if any active dataset fails):

```bash
set -o pipefail

python scripts/build_manifests.py \
  --config_dir configs/data \
  --data_root "$PROACTIVE_DATA_ROOT" \
  --output_dir outputs/manifests \
  --datasets hallusionbench pope vizwiz vsr \
  --overwrite --seed 42 \
  2>&1 | tee outputs/logs/week4/hallusion_contract/rebuild_manifests.log

wc -l \
  outputs/manifests/manifest_hallusionbench.jsonl \
  outputs/manifests/manifest_combined.jsonl

python scripts/migrate_hallusion_answer_contract.py \
  --old_manifest outputs/manifests/manifest_combined_pre_hallusion_contract.jsonl \
  --new_manifest outputs/manifests/manifest_combined.jsonl \
  --config configs/experiments/teacher_core.yaml \
  --input_dir outputs/teacher_core \
  --output_dir outputs/teacher_core_contract_v1 \
  --report outputs/week4_reports/hallusion_answer_contract_migration.json \
  --overwrite \
  2>&1 | tee outputs/logs/week4/hallusion_contract/migration.log
```

Expected manifest counts are 951 and 7,291. The new hashes will intentionally
differ from the old hash because HallusionBench and VizWiz contract metadata
changed. The migration report must say `is_valid: true`, retain 7,291 selected
rows/model, and invalidate exactly 14 open-ended rows/model. VizWiz rows are
reused and their gold-dependent correctness is recomputed from stored raw
answers; no VizWiz model inference is repeated. The report records the exact
number of resulting VizWiz correctness changes. For the currently synced
original cache, expected pending counts are 67 Qwen and 88 Gemma; these include
prior fail-closed rows.

The accepted report must use migration `schema_version: 2`. Version 1 was
superseded after the first resume dry-run found a stale VizWiz grounding
normalization (`cannot be determined` versus canonical `unanswerable`). Version
2 reparses saved grounding outputs, refreshes parser-dependent probe features,
uses the pinned CPU semantic matcher only for changed non-exact pairs, and
validates every migrated valid row against the current resume parser before
signing the report.

Validate every migrated resume boundary without loading a model:

```bash
for MODEL in qwen3_vl_8b gemma4_e4b; do
  for SHARD in 0 1 2 3; do
    python scripts/run_teacher.py \
      --config configs/experiments/teacher_core.yaml \
      --manifest_path outputs/manifests/manifest_combined.jsonl \
      --model "$MODEL" --device cpu \
      --num_shards 4 --shard_id "$SHARD" \
      --output_dir outputs/teacher_core_contract_v1 \
      --resume --dry_run || exit 1
  done
done
```

### 11.2 Regenerate only pending base rows on two GPUs

After confirming physical GPUs 0 and 1 are free, run Qwen in one tmux pane and
Gemma in another. Each process sees one physical GPU, so `--device cuda:0` is
correct in both panes.

Qwen / physical GPU 0:

```bash
cd ~/ProActive
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004
mkdir -p outputs/logs/week4/hallusion_contract
set -o pipefail

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 python scripts/run_teacher.py \
    --config configs/experiments/teacher_core.yaml \
    --manifest_path outputs/manifests/manifest_combined.jsonl \
    --model qwen3_vl_8b --device cuda:0 \
    --num_shards 4 --shard_id "$SHARD" \
    --output_dir outputs/teacher_core_contract_v1 --resume \
    2>&1 | tee "outputs/logs/week4/hallusion_contract/qwen_base_shard0${SHARD}.log"
  echo "Qwen corrected-base shard ${SHARD}: ${PIPESTATUS[0]}"
done
```

Gemma / physical GPU 1:

```bash
cd ~/ProActive
export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data
export PROACTIVE_SEMANTIC_MODEL_PATH=/home/models/all-MiniLM-L6-v2
export PROACTIVE_SEMANTIC_MODEL_REVISION=e4ce9877abf3edee10b0257f22713854020a4004
mkdir -p outputs/logs/week4/hallusion_contract
set -o pipefail

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 python scripts/run_teacher.py \
    --config configs/experiments/teacher_core.yaml \
    --manifest_path outputs/manifests/manifest_combined.jsonl \
    --model gemma4_e4b --device cuda:0 \
    --num_shards 4 --shard_id "$SHARD" \
    --output_dir outputs/teacher_core_contract_v1 --resume \
    2>&1 | tee "outputs/logs/week4/hallusion_contract/gemma_base_shard0${SHARD}.log"
  echo "Gemma corrected-base shard ${SHARD}: ${PIPESTATUS[0]}"
done
```

An exit code 1 for an individual base shard is not permission to delete a row;
it means the fail-closed record was retained for the uniform grounding pass.
After both loops, require complete valid-plus-failure source coverage:

```bash
mkdir -p outputs/teacher_core_contract_v1_grounding1024

for MODEL in qwen3_vl_8b gemma4_e4b; do
  for SHARD in 0 1 2 3; do
    python scripts/refresh_grounding_cache.py \
      --config configs/experiments/teacher_core.yaml \
      --manifest_path outputs/manifests/manifest_combined.jsonl \
      --model "$MODEL" --shard_id "$SHARD" --num_shards 4 \
      --input_dir outputs/teacher_core_contract_v1 \
      --output_dir outputs/teacher_core_contract_v1_grounding1024 \
      --max_new_tokens 1024 --device cpu --resume --dry_run || exit 1
  done
done
```

### 11.3 Uniform grounding refresh and validation

Run the same Qwen/GPU-0 and Gemma/GPU-1 loop pattern in parallel, replacing the
base command with the following model-specific refresh command. Every row gets
the same 1024-token maximum; generation may stop earlier normally.

Qwen command inside the shard loop:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/refresh_grounding_cache.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model qwen3_vl_8b --shard_id "$SHARD" --num_shards 4 \
  --input_dir outputs/teacher_core_contract_v1 \
  --output_dir outputs/teacher_core_contract_v1_grounding1024 \
  --max_new_tokens 1024 --device cuda:0 --resume \
  2>&1 | tee "outputs/logs/week4/hallusion_contract/qwen_grounding1024_shard0${SHARD}.log"
```

Gemma command inside the shard loop:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/refresh_grounding_cache.py \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --model gemma4_e4b --shard_id "$SHARD" --num_shards 4 \
  --input_dir outputs/teacher_core_contract_v1 \
  --output_dir outputs/teacher_core_contract_v1_grounding1024 \
  --max_new_tokens 1024 --device cuda:0 --resume \
  2>&1 | tee "outputs/logs/week4/hallusion_contract/gemma_grounding1024_shard0${SHARD}.log"
```

After both four-shard loops finish:

```bash
python scripts/validate_week4.py \
  --mode teacher_progress \
  --config configs/experiments/teacher_core.yaml \
  --manifest_path outputs/manifests/manifest_combined.jsonl \
  --teacher_path outputs/teacher_core_contract_v1_grounding1024 \
  --output_dir outputs/week4_reports/hallusion_contract \
  --overwrite \
  2>&1 | tee outputs/logs/week4/hallusion_contract/final_teacher_validation.log
```

Proceed to labels only if validation reports exactly 14,582 valid teacher rows,
zero errors, and no unresolved refresh failures. Use
`outputs/teacher_core_contract_v1_grounding1024` as `--teacher_path` for all
subsequent label/state/audit commands.
