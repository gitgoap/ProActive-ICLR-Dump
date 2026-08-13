# Week 4 Requirements — Full teacher cache, labels, and partial states

**Plan source:** v3.5 Plan §25.6
**Implementation status:** IMPLEMENTED, NOT VALIDATED
**GPU status:** NOT STARTED
**Completion status:** NOT COMPLETE

Week 4 builds the offline substrate used by every later diagnostic learner,
policy, conformal calibration, and evaluation. Code being present is not the
completion gate: the server artifacts, review, checksums, audit packet, and
leakage/balance reports must also exist and pass.

## Requirement traceability

| Requirement | Code location | Unit/adversarial test | Integration check | Required artifact | Current status |
|---|---|---|---|---|---|
| Stable four-GPU sharding | `src/proactive/teacher/offline.py::stable_shard_id`, `scripts/run_teacher.py` | `test_stable_sharding_is_order_independent_and_total` | `validate_week4.py --mode teacher_progress` | `outputs/teacher_core/teacher_*_shard*.jsonl` | IMPLEMENTED, NOT VALIDATED |
| All legal canonical probes (6, or 7 for relation rows) | `scripts/run_teacher.py`, `src/proactive/teacher/cache_builder.py` | `test_relation_applicable_record_requires_exactly_seven_legal_probes` | teacher progress/full validator | teacher cache | IMPLEMENTED, NOT VALIDATED |
| Resume without append duplicates or overwrite | `scripts/run_teacher.py::_validate_existing_rows` | duplicate and deterministic-shard adversarial tests | resume one server smoke shard twice | unchanged teacher shard hash/row count | IMPLEMENTED, NOT VALIDATED |
| Daily row-count and checksum validation | `scripts/validate_week4.py --mode teacher_progress` | Week 4 validator tests | server progress command | `outputs/week4_reports/teacher_manifest.json` | IMPLEMENTED, NOT VALIDATED |
| Continuous signatures, source bits, six-way labels | `src/proactive/teacher/offline.py::build_label_record`, `scripts/build_labels.py` | `test_labels_are_recomputed_deterministically_and_fail_closed` | Week 4 end-to-end validator | `outputs/labels_core/*.jsonl` | IMPLEMENTED, NOT VALIDATED |
| Class/bit balance by dataset and model | `src/proactive/audits/week4_validation.py` | end-to-end validator | full validation | two distribution CSVs | IMPLEMENTED, NOT VALIDATED |
| Grouped partial-state subsets | `src/proactive/teacher/offline.py::sample_partial_subsets`, `scripts/sample_states.py` | mandatory-source test | end-to-end validator | `outputs/states_v1/*.jsonl` | IMPLEMENTED, NOT VALIDATED |
| No unacquired evidence | `build_state_records`, recursive leakage validator | `test_partial_states_never_serialize_unacquired_evidence` | full validation | leakage results in full report | IMPLEMENTED, NOT VALIDATED |
| 180-example blinded human audit | `src/proactive/audits/human_audit.py`, `scripts/export_human_audit.py` | `test_human_audit_sampling_is_deterministic_balanced_and_covered`; packet validator | `validate_week4.py --mode full` | blinded CSV, private key, 180 renamed images, README, manifest | IMPLEMENTED, NOT VALIDATED |
| No split/group leakage | `week4_validation.py` | split-hash adversary test | full validation against grouped manifest | full validation report | IMPLEMENTED, NOT VALIDATED |
| Teacher/label/state manifests with checksums | `scripts/validate_week4.py` | end-to-end validator | full validation | three JSON manifests | IMPLEMENTED, NOT VALIDATED |
| Exact model/semantic revisions | `scripts/inspect_model_revisions.py`, `scripts/run_teacher.py`, `semantic.py` | unpinned runs fail closed; metadata parser regression tests | server revision inspection passed; staged model smoke remains | pinned model YAMLs and cache provenance | IMPLEMENTED, NOT VALIDATED |

## Partial-state scope

The Week 4 serializer creates the subsets that are scientifically possible
before a policy exists: empty, all legal singletons, every prefix of four fixed
baselines, and 16 deterministic random draws at sizes 1–4. The eight
policy-rollout subsets and oracle-next-action subsets from Plan §16.2 are marked
as deferred in every state and must be added after the first policy checkpoint
and oracle baseline exist. They are not fabricated in Week 4.

## Owner decisions and compute authorization

The owner approved the following on 2026-08-10:

1. `max_sixway_fraction = 0.80`;
2. at least 5 positive and 5 negative source-bit rows per dataset/model slice;
3. 60 naturally sampled plus 120 targeted human-audit rows;
4. interim two-model work is allowed, but the final audit requires InternVL;
5. the mandatory 1/10/100/full-VSR staged GPU checks.

`configs/experiments/teacher_core.yaml` is scientifically approved and the
readiness gate passes. The projected 33.23 GPU-hour full Qwen+Gemma core is not
yet approved. `scripts/run_teacher.py` therefore permits only limits up to 100
or a complete VSR run and fails closed on the combined full core.

## Completion gate

- Qwen and Gemma contain exactly one valid teacher row for every grouped
  manifest instance.
- InternVL is complete or the catch-up shard is documented and scheduled.
- Every teacher row has exactly every legal probe and no duplicates.
- Labels recompute exactly from the frozen Week 3 configuration.
- Every teacher/label pair has all mandatory pre-policy partial subsets.
- The full leakage validator reports zero errors.
- Label/bit balance passes the approved operational gates.
- The complete, non-interim 180-example audit packet is exported.
- Teacher, label, and state manifests record row counts and SHA-256 checksums.
- Local CPU tests and staged server commands pass; server logs are retained.
