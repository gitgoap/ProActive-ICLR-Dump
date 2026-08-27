# Week 4 InternVL catch-up shard

**Status:** 1/10/100-ROW AND COMPLETE-VSR GPU STAGES PASSED; CATCH-UP VALIDATED

InternVL3-9B is the third-model catch-up path permitted by the Week 4
completion gate. It must use the same grouped manifest, frozen Week 3 probe
configuration, immutable model revision, seed 42, and four deterministic
  shards as Qwen and Gemma. The corrected server smoke has passed; larger
  catch-up stages still require their own validation before full generation.

The model files are present on `bumblebee` at
`/home/models/InternVL3-9B`. Download availability is confirmed; this status
must not be confused with a successful adapter/teacher GPU validation.

Server evidence resolved immutable revision
`5f618513e35a9b85922341b8057feddfc8880e50`, now pinned in
`configs/models/internvl3_9b.yaml`. Use the same `scripts/run_teacher.py` commands documented in
`doc/docs/WEEK_4_SERVER_EXECUTION.md`, substituting `--model internvl3_9b`.

Record start/end times, GPU ID, command, output files, row counts, and SHA-256
checksums in `RUN_REGISTRY.md` after execution.

## Isolated runtime required

The first server smoke exposed two environment facts: the base environment is
Python 3.13 with Transformers 5.5.4, while the pinned InternVL3 checkpoint uses
the older custom `InternVLChatModel` interface. Transformers 5.5.4 failed while
loading the checkpoint with a missing `all_tied_weights_keys` attribute. The
base environment must not be downgraded because it produced the accepted Qwen
and Gemma artifacts.

InternVL therefore runs in the separate `proactive-internvl` environment. Its
dependencies are pinned in `requirements-internvl.txt`; the official custom
runtime pin is Transformers 4.37.2. The ProActive adapter uses InternVL's native
dynamic 448-pixel image tiling, image-context token injection, generation
scores, and teacher-forced scoring. It does not treat `AutoTokenizer` as a
generic multimodal processor.

Create the environment once. Pinning `pip`, `certifi`, CA certificates, and
OpenSSL is deliberate: the first unpinned creation attempt selected pip 26.1.2,
failed while resolving its bundled CA path, and left no registered environment.
The following recipe is the server-validated replacement:

```bash
cd ~/ProActive

source ~/miniconda3/etc/profile.d/conda.sh
conda create -n proactive-internvl \
  python=3.11 pip=25.2 certifi ca-certificates openssl -y
conda activate proactive-internvl

export PIP_CERT="$CONDA_PREFIX/lib/python3.11/site-packages/certifi/cacert.pem"

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements-internvl.txt
python -m pip install -e . --no-deps
python -m pip check

python - <<'PY'
import torch, transformers, accelerate, einops
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("einops:", einops.__version__)
PY
```

Expected Transformers version: exactly `4.37.2`. FlashAttention is optional;
the smoke uses the deterministic eager-attention fallback.

## Validated server evidence

On 2026-08-23 the persistent environment resolved to
`/home/aman/miniconda3/envs/proactive-internvl/bin/python` with Python 3.11.15,
PyTorch 2.6.0+cu124, Torchvision 0.21.0+cu124, Transformers 4.37.2,
Accelerate 0.30.1, and einops 0.6.1. `pip check` reported no broken
requirements. The focused adapter suite passed `11/11`, followed by the full
repository suite at `225/225` (one non-blocking Hugging Face deprecation
warning). After the grounding-recovery and offline-sidecar regressions were
added, the expanded server suite passed `236/236` on 2026-08-24. These CPU
results validate the isolated software contract but do not
replace the required real-GPU smoke evidence.

## Corrected GPU smoke evidence

The corrected one-row run completed on physical A6000 GPU 0 on 2026-08-23.
It produced one valid VizWiz teacher row, six applicable probe observations
(relation correctly not applicable), a valid tagged grounding answer, and zero
failures in 101.6 seconds end to end. The row pins model revision
`5f618513e35a9b85922341b8057feddfc8880e50`, combined-manifest SHA-256
`05fd0dbc554c5fb85664dd87e99f8eebdd995320e6b34e2fffe856867d3d0859`,
and seed 42. Output SHA-256:
`fd813be00a5b37fa6fb75586afdbefb4b58869882ed99198f67cf8c0dcf8fca0`.

The subsequent 10-row stage completed in 170.0 seconds with 10 unique valid
rows, 60 legal probe observations, no invalid applicable probes, and zero
failures. It covered POPE and VizWiz plus train/calibration/test splits and
produced four distinct six-way labels. Output SHA-256:
`635b8095415e669da6c725798c6ab4983a602a65b1d2c843e90cca59a3d661ef`.

On 2026-08-24 the 100-row stage produced 100/100 valid teacher rows, 602
legal probes, two relation-applicable rows, all four datasets, and zero
failures in 1,979.0 seconds. Output SHA-256:
`c1a447aebad945ac270fea11f8ad7d500b9e7e14db867aed292a44fc9f1369bd`.

The complete-VSR stage produced 340/340 valid rows, 2,150 legal probes, all
110 relation-applicable rows, and zero failures in 5,682.5 seconds. Output
SHA-256:
`929b563aa9302e943918e797b8204040f7e8363fc581c929dcb6bf836987ba2b`.

The generic Week 3 schema CLI reported the 20 calibration/test rows in the
100-row cache and the 67 calibration/test rows in VSR as invalid because that
pilot validator intentionally allows only train/validation. This is a
validator-scope mismatch, not an inference failure. Independent Week 4 audits
confirmed valid rows, exact legal probe sets, no duplicates, matching manifest
fields, and complete provenance. The caches must not be regenerated for this
false-positive report.
