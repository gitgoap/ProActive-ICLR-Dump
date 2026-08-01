# Week 3 Requirements & Audit Verification Matrix (Plan §13, §14, §25.5)

| Plan Requirement | Code Location | Unit Test | Integration Test | Output Artifact | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Blank image probe** ($b_L$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Blur probe** ($b_V$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Crop probe** ($b_V$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Brightness probe** ($b_V$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Noise probe** ($b_V$) | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Grounding isolated scoring** | `src/proactive/prompts/templates.py` | `tests/test_grounding_parsing.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Relation instance applicability** | `src/proactive/probes/relation_swap.py` | `tests/test_relation_swap.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Relation outcome status & invariance** | `src/proactive/probes/relation_swap.py` | `tests/test_relation_swap.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Semantic matching (SentenceTransformer)** | `src/proactive/features/semantic.py` | `tests/test_semantic_match.py` | `tests/test_week3_integration.py` | `configs/probes/candidate_week3_config.yaml` | **VERIFIED_CODE_AND_TESTS** |
| **Teacher label derivation** | `src/proactive/teacher/label_computation.py` | `tests/test_teacher_labels.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Stratified train/val-only pilot sampling** | `scripts/run_pilot_cache.py` | `tests/test_week3_confounds_and_schema.py` | `tests/test_week3_integration.py` | `outputs/pilot_cache/*.jsonl` | **VERIFIED_CODE_AND_TESTS** |
| **Strict JSONL schema validator** | `src/proactive/audits/schema_validator.py` | `tests/test_week3_confounds_and_schema.py` | `tests/test_week3_integration.py` | `outputs/pilot_reports/schema_validation_report.json` | **VERIFIED_CODE_AND_TESTS** |
| **Severity grid pilot & automated selection** | `src/proactive/audits/pilot_analysis.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `configs/probes/candidate_week3_config.yaml` | **VERIFIED_CODE_AND_TESTS** |
| **Distribution plots (4 required + source bit)** | `scripts/analyze_pilot.py` | `tests/test_week3_confounds_and_schema.py` | `tests/test_week3_integration.py` | `outputs/pilot_reports/plots/*.png` | **VERIFIED_CODE_AND_TESTS** |
| **Image transform sample inspection** | `src/proactive/probes/image_transforms.py` | `tests/test_image_transforms.py` | `tests/test_week3_integration.py` | `outputs/pilot_reports/inspection/*` | **VERIFIED_CODE_AND_TESTS** |
| **Confound audit & shortcut mitigation** | `src/proactive/audits/confound_audit.py` | `tests/test_week3_confounds_and_schema.py` | `tests/test_week3_integration.py` | `outputs/pilot_reports/pilot_analysis_summary.json` | **VERIFIED_CODE_AND_TESTS** |
| **Completion gate pre-flight checker** | `scripts/check_week_completion.py` | `tests/test_week3_integration.py` | `tests/test_week3_integration.py` | `scripts/check_week_completion.py` | **VERIFIED_CODE_AND_TESTS** |
