# Run Registry

This log records major server runs, their configurations, resource usage, and statuses. Exact per-dataset commands are preserved in `doc/docs/WEEK_3_SERVER_EXECUTION.md`; the four matrix rows below each correspond to four independent one-GPU dataset commands.

| Run ID | Exact Command Record | Config | Commit / Code State | Model Rev | Dataset Hash | Seed | GPUs Used | Est GPU-hours | Act GPU-hours | Status | Output Path | Key Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W3-QWEN-CANONICAL | `doc/docs/WEEK_3_SERVER_EXECUTION.md` §3 | `configs/probes/frozen_week3_config.yaml`; `configs/models/qwen3_vl_8b.yaml` | Base `c3f68e68874e` + Week 3 hardening sync | `main` (must be pinned before Week 4) | See manifest table below | 42 | One GPU/job; up to four concurrent; exact physical IDs not retained | Pilot estimate recorded in execution plan | At least `0.4733`; earlier VizWiz/VSR segments not present in synced logs | COMPLETE | `outputs/pilot_cache/qwen3_vl_8b_{dataset}.jsonl` | Four datasets × 100 unique canonical rows; zero final schema/duplicate errors |
| W3-QWEN-SEVERITY | `doc/docs/WEEK_3_SERVER_EXECUTION.md` §§4–5 | Same frozen/model configs | Base `c3f68e68874e` + Week 3 hardening sync | `main` | See manifest table below | 42 | One GPU/job; up to four concurrent; exact physical IDs not retained | Pilot estimate recorded in execution plan | At least `0.3470`; earlier POPE/VizWiz/VSR segments not present in synced logs | COMPLETE | `outputs/pilot_cache/qwen3_vl_8b_{dataset}_severity_pilot.jsonl` | Four datasets × 1,200 unique severity rows; zero final schema/duplicate errors |
| W3-GEMMA-CANONICAL | `doc/docs/WEEK_3_SERVER_EXECUTION.md` §6 | `configs/probes/frozen_week3_config.yaml`; `configs/models/gemma4_e4b.yaml` | Base `c3f68e68874e` + Week 3 hardening sync | `main` (must be pinned before Week 4) | See manifest table below | 42 | One GPU/job; up to four concurrent; exact physical IDs not retained | `0.8–2.0` for Gemma matrix was provisionally budgeted with severity included | At least `0.5038`; earlier POPE/HallusionBench segments not present in synced logs | COMPLETE | `outputs/pilot_cache/gemma4_e4b_{dataset}.jsonl` | Four datasets × 100 unique canonical rows; zero final schema/duplicate errors |
| W3-GEMMA-SEVERITY | `doc/docs/WEEK_3_SERVER_EXECUTION.md` §6 | Same frozen/model configs | Base `c3f68e68874e` + Week 3 hardening sync | `main` | See manifest table below | 42 | One GPU/job; up to four concurrent; exact physical IDs not retained | Included in the provisional Gemma matrix estimate | `1.5892` from complete retained severity logs | COMPLETE | `outputs/pilot_cache/gemma4_e4b_{dataset}_severity_pilot.jsonl` | Four datasets × 1,200 unique severity rows; all logged runs finished with zero failures |
| W3-VALIDATE-FREEZE | `python scripts/validate_teacher_schema.py outputs/pilot_cache ...`; `python scripts/analyze_pilot.py ... --freeze --confirm_freeze ...`; `python scripts/check_week_completion.py --mode full_week` | `configs/probes/frozen_week3_config.yaml` | Same Week 3 sync | N/A | Directory-wide validation | 42 | CPU | Negligible | Negligible | COMPLETE | `outputs/pilot_reports/`; `configs/probes/frozen_week3_config.yaml` | 10,400 valid rows, 0 duplicates, 5 plots, 250 inspections, semantic threshold `0.50`; full gate passed |

## Manifest SHA-256 values

| Dataset | Rows | SHA-256 |
| :--- | ---: | :--- |
| HallusionBench | 951 | `62c4a71fb534b2dbcfa7f9410bfc7440682da7aa44cfecacca782d1e03bb86d3` |
| POPE | 3,000 | `7e8e9c891a030e2239d70cf119478829ef1595ba4bee66b5ed8a0ec5da801111` |
| VizWiz | 3,000 | `c665ce24cc83497d5e74c0b4d4a6f2bb7fc32de14c21b4a757d3318469b794b8` |
| VSR | 340 | `cc2f88a283975a60820f79ba3036b478e0b85c9ebd6d2c28c1225587df42f1ce` |

The retained Week 3 logs account for at least `2.9133` GPU-hours. This is a lower bound, not the total: resume/no-op logs replaced several earlier start-to-finish logs. The missing exact GPU allocation and complete timing are an audit gap that must not recur for Week 4.
