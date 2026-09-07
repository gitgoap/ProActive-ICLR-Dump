# ProActive ICLR Deferred Work and External Setup

**Last updated:** 2026-09-07  
**Purpose:** preserve a paper-facing record of work deferred under the ICLR
deadline, work blocked on external inputs, and conditional work that correctly
did not enter the main path. This file is a recovery checklist, not permission
to alter frozen train/validation/test decisions.

## Current critical path

Weeks 1–7 are complete. The immediate critical path is:

1. Start the three-person human audit immediately because people, not compute,
   determine its latency.
2. Generate cached Week 8 robustness, ablation, latency, uncertainty, and
   paper-facing results.
3. Decide whether PRE-HAL and IllusionBench shift evidence justifies new loader
   and teacher-cache work; if yes, obtain and checksum both official releases
   now while cached analyses proceed.
4. Freeze paper tables, figures, claims, and the reproducibility package.

Weeks 6–7 use cached features and small networks. They do not require another
Qwen, Gemma, or InternVL download and do not repeat the Week 4 teacher passes.

## Deferred or externally blocked work

| Priority | Item | Current state | What is needed | Why it was deferred / when to resume |
|---|---|---|---|---|
| P0, parallel human work | Three-person human audit | The blinded 180-row packet and all 180 images exist; direct inspection on 2026-09-07 confirms `0/540` independent row-blocks complete | Three different people each annotate all 180 rows in exactly one block (`ann1_*`, `ann2_*`, or `ann3_*`); then adjudicate disagreements without opening the private key; only afterward compare with hidden rule-backed labels | No GPU is needed. Start immediately in parallel because recruiting and annotation latency cannot be recovered with compute |
| P1, transfer evidence | Complete InternVL3-9B teacher cache over all 7,291 core instances | InternVL environment/model are validated; 1/10/100-row all-dataset stages and complete 340-row VSR passed, but a complete four-dataset cache was not run | Use `/home/models/InternVL3-9B` in `proactive-internvl`; obtain explicit high-cost approval; run deterministic resumable shards; rebuild compatible labels/states without changing Qwen/Gemma | Deferred to protect the core schedule. Needed for a complete three-model leave-one-model-out claim; otherwise describe InternVL as staged/catch-up evidence only |
| P1, relation coverage | GQA-Relation slice | Config and loader contract exist; construction status is `pending`; the referenced construction script is not present | Download GQA images plus `sceneGraphs/val_sceneGraphs.json`; implement/audit `scripts/build_gqa_relation.py`; construct up to 1,000 reversible-relation examples; manually inspect at least 100 pairs; then build a frozen manifest and teacher cache | Resume before the relation-heavy/transfer appendix if time permits. Do not fabricate a slice or reuse VSR labels as GQA |
| P1, held-out shift | PRE-HAL | `configs/data/prehal.yaml` is a placeholder; no registered loader or shift-evaluation script currently exists | Obtain the official dataset under its license; place it at `${PROACTIVE_DATA_ROOT}/PREHAL`; document schema/images/answers; implement loader, tests, frozen held-out manifest, and shift evaluation | Run only after the main stack is frozen. It is held-out evaluation data and must never influence thresholds or model selection |
| P1, held-out shift | IllusionBench | `configs/data/illusionbench.yaml` is a placeholder; no registered loader or shift-evaluation script currently exists | Obtain the official dataset under its license; place it at `${PROACTIVE_DATA_ROOT}/IllusionBench`; document schema/images/answers; implement loader, tests, frozen held-out manifest, and shift evaluation | Run only after the main stack is frozen. It is held-out evaluation data and must never influence thresholds or model selection |
| P2, triggered appendix | RAPS efficiency ablation | The predeclared gate triggered: mean 90%-APS set size `3.2087`, singleton rate `0.2075`, and coverage remained near target. Gate recording exists, but RAPS fitting/evaluation code does not | Implement regularized APS with frozen validation-only hyperparameters; test it; fit before test unlock; report as an appendix comparison while retaining APS as the main method | Triggered but deliberately kept outside the critical path. It must not replace APS because of test performance |
| P2, sensitivity | HallusionBench 937-row binary-only analysis | Main answer-contract analysis retains all 951 rows; the 937 binary rows are identifiable | CPU-only secondary evaluation excluding the 14 genuinely open-ended rows; report beside, not instead of, the 951-row primary result | Deferred because it does not change training or the primary population |
| P2, sensitivity | VizWiz soft VQA scoring | Deterministic single-target source-order tie breaking is frozen; annotation counts were retained | Implement CPU-only soft-VQA rescoring and compare conclusions with the frozen single-target result | Deferred as a robustness appendix; it must not be used to retune the model |
| P3, optional transfer | Additional MLLM families (for example Molmo or LLaVA-OneVision) | Not downloaded or integrated | Download only after selecting a specific checkpoint and license; implement/pin adapter and revision; stage 1/10/100 before any full cache | Optional transfer evidence, not a replacement chosen after viewing core results |

## Work that was intentionally not run and is not technical debt

- **Set Transformer:** the predeclared trigger did not fire. Deep Sets was
  within `0.000116` of the best GRU source-bit Macro-F1, led on six-way
  Macro-F1, and had exactly zero permutation drift. Not training Set
  Transformer is the preregistered outcome, not a deadline shortcut.
- **Repeated smoke tests:** completed stages and hash-valid resume checks are
  not repeated unless code/config/model provenance changes.
- **New MLLM downloads for Weeks 6–7:** none are required. These weeks use the
  frozen Deep Sets checkpoint and cached teacher/state artifacts.

## External setup checklist

### Human inputs

- Recruit three independent annotators.
- Give them only `outputs/human_audit/human_audit_blinded.csv` and
  `outputs/human_audit/images/`.
- Do not expose `human_audit_private_key.jsonl` during independent annotation
  or adjudication.
- One researcher must not imitate three annotators.

### Dataset inputs

Expected server locations after legally obtaining the official releases:

```text
/home/aman/MMUQ/data/GQA/sceneGraphs/val_sceneGraphs.json
/home/aman/MMUQ/data/GQA/images/
/home/aman/MMUQ/data/PREHAL/
/home/aman/MMUQ/data/IllusionBench/
```

Before copying data, record the official release URL, version/commit, license,
download date, archive checksum, extracted-file checksum, and any filtering.
Do not inspect held-out model results until the main stack is frozen.

### Model/runtime inputs

- No new model setup is needed for the Week 6–7 critical path.
- Qwen and Gemma remain in the accepted base runtime.
- InternVL3-9B already exists at `/home/models/InternVL3-9B` and must run only
  in the `proactive-internvl` Python 3.11 / Transformers 4.37.2 environment.
- The semantic model remains `/home/models/all-MiniLM-L6-v2` with pinned
  revision `e4ce9877abf3edee10b0257f22713854020a4004`.

## How to use two free GPUs during Week 6

The dependency order matters:

1. A bounded VOI construction must pass first.
2. The complete VOI corpus is then built by one process; the current script is
   not safe for two processes writing the same output directory.
3. After the complete VOI manifest exists, policy jobs across seeds, cost
   multipliers, and encoder comparisons can be distributed across both GPUs.
4. The second GPU may independently run the complete InternVL catch-up only
   after explicit high-cost approval; that job is not required for Week 6.

Do not duplicate identical VOI builds merely to keep a GPU occupied. Idle GPU
time is cheaper than invalid, racing, or unnecessary experiments.

## Paper disclosure rule

For every unfinished item, report one of:

- completed with artifact/hash;
- deferred appendix/sensitivity analysis;
- unavailable because an external dataset or annotator input was not ready;
- correctly omitted because its predeclared trigger did not fire.

Never describe a missing held-out dataset, incomplete InternVL cache, or empty
human audit as a completed experiment.
