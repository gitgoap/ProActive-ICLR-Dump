# Week 8 held-out data, execution, and human-audit guide

**Status:** IMPLEMENTED, NOT VALIDATED  
**Last updated:** 2026-09-08  
**Purpose:** Run reviewer-facing transfer and stress tests without changing the
frozen Week 7 model, policy, or calibration.

## What Week 8 is testing

Week 8 asks four practical questions:

1. Does ProActive still work when one core model family is withheld from
   training (leave-one-model-out, or LOMO)?
2. What happens on two genuinely new visual-hallucination datasets?
3. Which components of the method are actually responsible for the result?
4. Do the automatic behavioural labels make sense to independent people?

The Week 7 stack is immutable. PRE-HAL and IllusionBench are **stress tests**,
not new tuning data. Their rows must never be used to select hyperparameters,
fit APS thresholds, or change the stopping policy. Coverage on them is reported
as empirical shift coverage, not as a new formal conformal guarantee.

## Official release audit

### PRE-HAL

- Official release: <https://huggingface.co/datasets/TerryHWong/PRE-HAL>
- Pinned revision: `3aac833638f7808dea7eb0145af4d8c6f028714b`
- Declared license: CC-BY-4.0
- Size needed by ProActive: about 3.7 GB; 10,000 CSV rows and 6,469 unique
  images.
- `dataset.csv` fields: `index`, `question`, `A`, `B`, `C`, `D`, `answer`,
  `image`, `hallucination_type`.
- `answer` is the correct option letter A–D. A reference such as
  `/mmbench/241.png` resolves to `images/mmbench/241.png`.
- The release also contains `model_weights/`. ProActive does not use or
  download those weights.

Expected server layout:

```text
/home/aman/MMUQ/data/PREHAL/
├── README.md
├── dataset.csv
└── images/
    └── mmbench/
        └── *.png
```

The loader trims the known trailing whitespace in `hallucination_type`, checks
all referenced images, converts each question to an explicit A–D prompt, and
draws a deterministic 600-row sample stratified by hallucination type.

### IllusionBench

- Official release: <https://huggingface.co/datasets/MingZhangSJTU/IllusionBench>
- Pinned revision: `8ee046d8acfb968c1a6a564f231396bc0c4039ae`
- Download needed by ProActive: about 0.93 GB.
- The official Hugging Face card does not declare a license. Keep the snapshot
  private for research and do not redistribute it until the license is
  confirmed with the authors.
- `Image_properties.json` contains one object per annotated image. Each object
  has `image_property` (`image_name`, `Difficult Level`, `Category`,
  `Description`) and `qa_data` (`Question`, `Question Type`, `Correct Answer`).
- Questions are either true/false or roman-numeral multiple choice. The loader
  turns valid multiple-choice rows into explicit A–F prompts.
- The ZIP contains 1,008 real PNG images after ignoring macOS metadata files.

Expected server layout:

```text
/home/aman/MMUQ/data/IllusionBench/
├── README.md
├── Image_properties.json
├── IllusionDataset.zip
└── images/
    └── *.png
```

The official release is internally imperfect: 44 annotated image names are
missing from the archive, 11 archive images are not referenced, and some
multiple-choice strings are malformed or conflict with their supplied answer.
The filter is applied uniformly **before seeing any model output**. Every
rejected row and reason is written to `illusionbench_exclusions.jsonl`; this is
release-integrity filtering, not selective removal of hard model examples.
The eligible rows are deterministically sampled to 600 using category,
question type, and difficulty strata grouped by source image.

## Download once; do not stream inference

Use a pinned local snapshot on the server. This is simpler and safer than
loading images over the network during multi-hour GPU jobs:

- a network interruption cannot stop inference;
- every retry uses exactly the same bytes;
- hashes and exclusion counts make the experiment auditable;
- tmux jobs can resume without redownloading examples;
- the required disk footprint is only about 4.6 GB.

The setup script downloads only PRE-HAL CSV/README/images and the three needed
IllusionBench files. It deliberately skips PRE-HAL model weights. Inference
must read only the verified local snapshot.

## Phase A — run immediately after syncing the code

This phase uses CPU, disk, and network only. It does not need or occupy a GPU.

```bash
cd ~/ProActive

source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

export PROACTIVE_DATA_ROOT=/home/aman/MMUQ/data

mkdir -p outputs/logs/week8 outputs/week8_data
set -o pipefail

PYTHONPATH=src python -m pytest -q \
  tests/test_week8_heldout.py \
  tests/test_week8_statistics.py \
  tests/test_week8_ablation_models.py \
  tests/test_week8_lomo_and_human.py \
  tests/test_week8_integrity.py \
  2>&1 | tee outputs/logs/week8/week8_focused_tests.log

FOCUSED_RC=${PIPESTATUS[0]}
echo "Week 8 focused tests exit code: $FOCUSED_RC"

if [ "$FOCUSED_RC" -eq 0 ]; then
  PYTHONPATH=src python -m pytest -q \
    2>&1 | tee outputs/logs/week8/week8_full_pytest.log
  FULL_TEST_RC=${PIPESTATUS[0]}
  echo "Full pytest exit code: $FULL_TEST_RC"
  if [ "$FULL_TEST_RC" -ne 0 ]; then
    echo "Full test suite failed; dataset setup skipped."
    exit 1
  fi
else
  echo "Focused tests failed; dataset setup skipped."
  exit 1
fi

python scripts/setup_heldout_datasets.py \
  --data_root "$PROACTIVE_DATA_ROOT" \
  --download \
  --extract \
  --verify \
  --output_report outputs/week8_data/dataset_setup_report.json \
  --overwrite \
  2>&1 | tee outputs/logs/week8/dataset_setup.log

SETUP_RC=${PIPESTATUS[0]}
echo "Held-out dataset setup exit code: $SETUP_RC"

if [ "$SETUP_RC" -eq 0 ]; then
  python scripts/build_heldout_manifests.py \
    --config_dir configs/data \
    --data_root "$PROACTIVE_DATA_ROOT" \
    --output_dir outputs/week8_data/manifests \
    --datasets prehal illusionbench \
    --overwrite \
    2>&1 | tee outputs/logs/week8/build_heldout_manifests.log
  MANIFEST_RC=${PIPESTATUS[0]}
  echo "Held-out manifest build exit code: $MANIFEST_RC"
  if [ "$MANIFEST_RC" -ne 0 ]; then
    echo "Held-out manifest build failed; Week 8 validation skipped."
    exit 1
  fi
else
  echo "Dataset setup failed; manifest build skipped."
  exit 1
fi

python scripts/validate_week8.py \
  --mode implementation \
  --config configs/experiments/week8_evaluation.yaml \
  --output_dir outputs/week8_reports \
  --overwrite \
  2>&1 | tee outputs/logs/week8/implementation_validation.log
```

Expected time is dominated by download speed: approximately 15–60 minutes.
Manifest construction should take only a few minutes. If any hash, count, or
schema differs, stop and sync the logs/report; do not bypass the check.

## Human annotation — start in parallel now

This step is CPU-only and independent of the new datasets. The source audit
already contains 180 blinded examples and images.

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl

python scripts/prepare_human_annotation_packets.py \
  --audit_dir outputs/human_audit \
  --guide_dir human_annotation \
  --output_dir outputs/human_annotation_packets \
  --overwrite
```

Give `annotator_1/`, `annotator_2/`, and `annotator_3/` to three different
people. Each person completes their own 180-row `annotations.csv` independently
using only the included README and images. Do not give them
`human_audit_private_key.jsonl` or another person's file. Budget about 2–4
hours per annotator; the three people can work at the same time. See
`human_annotation/README.md` for the short owner workflow.

## Phase B — mandatory cheap pilots before the held-out full run

After Phase A output is synced and checked, run only a 1-row and then a 10-row
teacher stage for each model. A 100-row smoke is deliberately omitted to save
time. These two short gates are necessary because both datasets introduce a
new multiple-choice answer contract. They catch prompt/parser mistakes before
roughly 16,800 expensive forward passes are launched.

The full held-out run is not yet authorized in
`configs/experiments/week8_evaluation.yaml`. Change that file only after the
Phase A and 1/10-row evidence has been reviewed and Aman explicitly approves
the measured full scope.

## Phase C — complete held-out shift experiment

Once approved:

1. Run Qwen and Gemma concurrently on two genuinely available GPUs, one model
   per GPU, with `--resume` and separate output/log directories.
2. Expect 1,200 questions per model and about 8,400 clean/probe generations per
   model. The combined total is about 16,800 generations.
3. Conservative scheduling estimate: **6–12 uninterrupted wall-clock hours on
   two A6000 GPUs**, plus recovery time for any fail-closed malformed output.
   Reserve a full day rather than promising a 2–3 hour finish.
4. Build labels, partial states, the state manifest, and the shift vector
   manifest on CPU.
5. Apply the immutable Week 7 diagnostic, policy, and APS thresholds with
   `eval_frontier.py --phase shift`. Never fit a target-domain threshold.
6. Produce grouped bootstrap intervals, dataset/model slices, and qualitative
   cases with `analyze_week8.py`.

Exact Phase B/C commands are issued only after the synced dataset-setup report
confirms the official hashes and the GPU scope is approved. This prevents a
multi-hour run against an unverified or structurally changed release.

## Other Week 8 work

- LOMO: build two leakage-safe folds (hold out Qwen once and Gemma once), train
  predeclared Deep Sets/clean/scalar/policy stacks on the remaining model, fit
  APS on source calibration only, then evaluate the held-out model test rows.
- Ablations: reuse existing Week 5–7 evidence where possible; retrain only the
  predeclared feature, budget, signature, and independent-source variants.
- Latency: 10 CUDA warm-ups plus 100 synchronized measurements of the full
  diagnostic encoder and VOI controller; cached MLLM pass latencies are kept
  separate and normalized by clean-pass latency.
- Statistics: 2,000 grouped bootstrap resamples by source image/base group.
- Optional leave-one-dataset-out and GQA-Relation remain deferred unless all
  mandatory evidence and the paper are secure.

## Outputs that prove completion

```text
outputs/week8_data/dataset_setup_report.json
outputs/week8_data/manifests/heldout_manifest_bundle.json
outputs/week8_reports/shift/frontier_shift.csv
outputs/week8_reports/lomo/*/lomo.csv
outputs/week8_reports/ablations.csv
outputs/week8_reports/latency.csv
outputs/week8_reports/confidence_intervals.csv
outputs/week8_reports/qualitative_examples.json
outputs/week8_reports/human_audit/human_audit_summary.json
outputs/week8_reports/week8_full_validation.json
```

Week 8 is COMPLETE only when `validate_week8.py --mode full` passes. Code
existence or a successful dataset download is not completion.
