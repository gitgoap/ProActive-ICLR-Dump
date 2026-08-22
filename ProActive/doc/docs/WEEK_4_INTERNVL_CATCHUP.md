# Week 4 InternVL catch-up shard

**Status:** DOWNLOADED AND REVISION PINNED; GPU SMOKE BLOCKED ON ISOLATED-RUNTIME VALIDATION

InternVL3-9B is the third-model catch-up path permitted by the Week 4
completion gate. It must use the same grouped manifest, frozen Week 3 probe
configuration, immutable model revision, seed 42, and four deterministic
shards as Qwen and Gemma. The cache is not part of the two-model core approval
until its server smoke test passes.

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

Create the environment once:

```bash
cd ~/ProActive

conda create -n proactive-internvl python=3.11 -y
conda activate proactive-internvl

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements-internvl.txt
python -m pip install -e . --no-deps

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
