# Week 3 Requirements & Audit Verification Matrix (Plan §13, §14, §25.5)

Final verification evidence: `158` local CPU tests passed; 16 server pilot-cache files contain 10,400 valid records; the strict full-week artifact gate passed on 2026-08-09.

| Plan Requirement | Code Location | Unit Test | Integration Test | Output Artifact | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Blank image probe** ($b_L$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Blur probe** ($b_V$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Crop probe** ($b_V$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Brightness probe** ($b_V$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Noise probe** ($b_V$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Grounding isolated scoring** | `src/proactive/prompts/templates.py` | `tests/test_grounding_parsing.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Relation instance applicability** | `src/proactive/probes/relation_swap.py` | `tests/test_relation_swap.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Relation outcome status & invariance** | `src/proactive/probes/relation_swap.py` | `tests/test_relation_swap.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Semantic matching (SentenceTransformer)** | `src/proactive/features/semantic.py` | `tests/test_semantic_match.py` | `tests/test_week3_integration.py` | `configs/probes/frozen_week3_config.yaml` | **COMPLETE** |
| **Teacher label derivation** | `src/proactive/teacher/label_computation.py` | `tests/test_teacher_labels.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Stratified train/val-only pilot sampling** | `scripts/run_pilot_cache.py` | `tests/test_week3_pipeline_safety.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **COMPLETE** |
| **Strict JSONL schema and duplicate validator** | `src/proactive/audits/schema_validator.py` | `tests/test_week3_pipeline_safety.py` | `tests/test_week3_integration.py` | `outputs/pilot_reports/schema_validation_report.json` | **COMPLETE** |
| **Efficient three-level severity grid** | `src/proactive/teacher/cache_builder.py` | `tests/test_week3_pipeline_safety.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*_severity_pilot.jsonl` | **COMPLETE** |
| **Severity selection with hard safety gates** | `src/proactive/audits/pilot_analysis.py` | `tests/test_week3_pipeline_safety.py` | `tests/test_week3_integration.py` | `configs/probes/frozen_week3_config.yaml` | **COMPLETE** |
| **Human-labelled semantic threshold calibration** | `scripts/calibrate_semantic_threshold.py` | `tests/test_week3_pipeline_safety.py` | `tests/test_week3_integration.py` | `outputs/pilot_reports/semantic_calibration_report.json` | **COMPLETE** |
| **Distribution plots (4 required + source bit)** | `scripts/analyze_pilot.py` | `tests/test_week3_confounds_and_schema.py` | `tests/test_week3_integration.py` | `outputs/pilot_reports/plots/*.png` | **COMPLETE** |
| **Image transform sample inspection** | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_reports/inspection/*` | **COMPLETE** |
| **Confound audit & shortcut mitigation** | `src/proactive/audits/confound_audit.py` | `tests/test_week3_confounds_and_schema.py` | `tests/test_week3_integration.py` | `outputs/pilot_reports/pilot_analysis_summary.json` | **COMPLETE** |
| **HallusionBench image-paired-only loading** | `src/proactive/data/loaders.py` | `tests/test_week3_pipeline_safety.py` | `tests/test_week3_integration.py` | `outputs/manifests/manifest_hallusionbench.jsonl` | **COMPLETE** |
| **Completion gate** | `scripts/check_week_completion.py` | `tests/test_week3_pipeline_safety.py` | `tests/test_week3_integration.py` | `configs/probes/frozen_week3_config.yaml` | **COMPLETE** |

## Frozen Week 3 values

- Visual severities: blur `8`, crop `0.65`, brightness `0.15`, noise `25`.
- Semantic threshold: `0.50`, calibrated from 50 train/validation VizWiz pairs.
- Label thresholds remain those recorded in `configs/probes/frozen_week3_config.yaml`.
