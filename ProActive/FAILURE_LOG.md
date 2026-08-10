# Failure Log

Record any failures (crashes, errors, unexpected logic faults) encountered during generation, training, or evaluation, along with their suspected causes and fixes.

| Command | Error | Suspected Cause | Fix | Are Outputs Trustworthy? | Rerun Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| - | - | - | - | - | - |
| Earlier POPE smoke run followed by the 100-example pilot | 31 duplicate canonical rows | `append_jsonl` used append mode and the same seed resampled the smoke examples | Added fail-closed existing-output checks, duplicate-aware resume keys, durable writes, and regenerated Qwen POPE with `--overwrite` | No; the duplicated file was rejected | Regenerated; final directory validation reports zero duplicates |
| Multiline `run_pilot_cache.py` invocation | `unrecognized arguments` followed by `--limit: command not found` | A backslash had trailing whitespace (`\ `), so Bash did not continue the command | Removed trailing whitespace and reran with clean line continuations | No pilot output was produced by the failed command | Reran successfully |
| `pytest -q` on bumblebee base environment | `pytest: command not found` | Pytest was not installed in the active server environment | Used the already validated local CPU suite and retained server execution/schema evidence; no system package installation was required | GPU artifacts unaffected | No GPU rerun; final local suite passed 158 tests |
| Local full-suite run inside the desktop filesystem sandbox | Six temp-dependent tests failed/errored with Windows `PermissionError`; 159 tests had already passed | The sandbox denied pytest/PIL access to the system and first workspace temporary directories | Re-ran the identical suite with a dedicated external test directory and bytecode writes disabled | Yes; no research artifacts were produced or changed by the failed attempts | Final rerun passed all 168 tests in 2.75s; temporary directories were removed |
