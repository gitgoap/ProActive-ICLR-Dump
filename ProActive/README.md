# ProActive

ProActive studies whether a multimodal large language model can be diagnosed reliably without running every possible diagnostic probe. The system begins with one clean model response, decides which additional probe is most useful, observes the probe result, updates its diagnostic state, and either acquires another probe or stops. The final output is not a single overconfident diagnosis. It is a **calibrated set of plausible behavioural failure states**.

## Key Contributions

1. **Active diagnostic measurement:** learn which behavioural probe to acquire next under a limited forward-pass budget.
2. **Unordered evidence modelling:** represent the acquired probe outcomes as a set rather than as an artificial sequence.
3. **Calibrated diagnostic sets:** return a set with target marginal coverage after the complete acquisition policy has been frozen.
4. **Robust evaluation:** compare cost–diagnostic frontiers, permutation stability, leave-one-model-out transfer, held-out dataset shift, and strong fixed or oracle baselines.

Start with `PROACTIVE_RESEARCH_HANDOFF_AND_RESULTS_INDEX.md` for the current
paper story and direct links to every primary result family. See
`PROJECT_STATUS.md` for the live gate and `SERVER_RUNBOOK.md` for server
execution instructions.

## Server Python environment

The repository has two validated server runtimes. Use the existing `(base)`
environment for Qwen/Gemma teacher generation. Use the dedicated
`proactive-internvl` environment for InternVL and for Weeks 5–8 cached-data
training, tests, statistics, and validators. Do not downgrade `(base)`.

Before the first ProActive Python command in a new shell or tmux pane, verify
the existing environment:

```bash
cd ~/ProActive
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proactive-internvl
which python
python --version
python -c "import torch, transformers, yaml; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
```

For Qwen/Gemma teacher commands, replace the activation line with
`conda activate base`. If an import fails, stop and report the output instead
of installing packages or changing a validated environment during an
experiment. The complete environment table is in `SERVER_RUNBOOK.md`.
