# ProActive Server Runbook

This runbook outlines how to execute scripts on the remote compute server and return the necessary logs/outputs to the local workspace for analysis by Antigravity.

## Use and verify the existing server environment

Use the Python environment already available in the server shell. Do not run a
separate Conda activation step for the current ProActive workflow. If the
prompt already contains `(base)`, leave it as it is; that may be server-side
auto-activation and does not require another `conda activate` command.

Run this verification once in every new SSH session, tmux window, or tmux pane
before its first ProActive Python command:

```bash
cd ~/ProActive

which python
python --version
python -c "import torch, transformers, yaml; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
```

Stop if an import fails or if `which python` returns an unexpected interpreter.
Do not install packages or switch environments merely to continue a run;
return the verification output for review first.

## Preflight Protocol
Before running large inference or training workloads, run the preflight script on your server to confirm its hardware and software environment.

```bash
chmod +x scripts/server_preflight.sh
./scripts/server_preflight.sh
```

**Action:** Copy the entire output of this script and paste it back into your Antigravity chat.

## Running General GPU Workloads
Every script in this repository will accept standard arguments for output paths and device allocation. 

**Important Arguments:**
* `--device`: specify the GPU ID(s), e.g. `cuda:0` or `0`
* `--limit`: to run a small pilot (e.g. `--limit 100`) before running a full pass
* `--resume`: ALWAYS use this flag for large generation runs so you don't overwrite completed shards

Example command for pilot generation on GPU 0:
```bash
python scripts/run_teacher.py --config configs/experiments/teacher_pilot.yaml --device cuda:0 --limit 100 --out outputs/teacher_pilot.jsonl
```

## How to Return Logs
When the agent asks you to run a command, it will provide the full bash string.
1. Log into your server.
2. Run the command within the server's repository clone or appropriate environment.
3. Once the command completes or fails, copy the terminal output.
4. Paste the terminal output to the agent. 

If the output is massive, just paste the tail end containing the summary/failure, or provide the path to the log file if it's synced locally.

## Week 4 Qwen failed-row recovery

After syncing the recovery files documented in
`doc/docs/WEEK_4_SERVER_EXECUTION.md`, use the original four shard paths with
`--resume`. Resume validates all existing rows against the current grounding
parser and generates only missing rows. Each shard also maintains an atomic
`teacher_*.failures.jsonl` sidecar for unresolved rows; never edit, merge, or
delete either file manually.

After truncation diagnosis, use `scripts/refresh_grounding_cache.py` to build
`outputs/teacher_core_grounding512`. It runs one uniform 512-token grounding
pass for every row and never edits `outputs/teacher_core`. Qwen and Gemma may
run concurrently only in separate physical GPUs and separate tmux panes.

## InternVL3 isolated environment exception

The existing `(base)` rule remains correct for Qwen and Gemma. It does not
apply to InternVL3: server evidence on 2026-08-22 showed Transformers 5.5.4 in
base, while the pinned InternVL custom checkpoint requires Transformers 4.x
and failed before inference. Never downgrade base.

Create `proactive-internvl` once using the commands in
`doc/docs/WEEK_4_INTERNVL_CATCHUP.md`. Before every InternVL command, activate
and verify it explicitly:

```bash
conda activate proactive-internvl
which python
python --version
python -c "import transformers; print(transformers.__version__)"
```

The expected Transformers version is `4.37.2`. After the InternVL command,
`conda deactivate` returns to base. Qwen/Gemma commands must continue using
their original environment.
