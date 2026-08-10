# ProActive

ProActive studies whether a multimodal large language model can be diagnosed reliably without running every possible diagnostic probe. The system begins with one clean model response, decides which additional probe is most useful, observes the probe result, updates its diagnostic state, and either acquires another probe or stops. The final output is not a single overconfident diagnosis. It is a **calibrated set of plausible behavioural failure states**.

## Key Contributions
1. **Active diagnostic measurement:** learn which behavioural probe to acquire next under a limited forward-pass budget.
2. **Unordered evidence modelling:** represent the acquired probe outcomes as a set rather than as an artificial sequence.
3. **Calibrated diagnostic sets:** return a set with target marginal coverage after the complete acquisition policy has been frozen.
4. **Robust evaluation:** compare cost–diagnostic frontiers, permutation stability, leave-one-model-out transfer, held-out dataset shift, and strong fixed or oracle baselines.

See `PROJECT_STATUS.md` for current progress and `SERVER_RUNBOOK.md` for server execution instructions.

## Server Python environment

Use the Python environment already available after logging into the server or
opening a tmux pane. No explicit Conda activation is required for the current
ProActive workflow. A prompt such as `(base)` may indicate that the server
auto-activated Conda, but the project instructions do not depend on manually
running `conda activate`.

Before the first ProActive Python command in a new shell or tmux pane, verify
the existing environment:

```bash
cd ~/ProActive
which python
python --version
python -c "import torch, transformers, yaml; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
```

If an import fails, stop and report the output instead of installing packages
or switching environments during an experiment.
