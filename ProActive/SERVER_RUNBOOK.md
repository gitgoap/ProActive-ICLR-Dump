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
