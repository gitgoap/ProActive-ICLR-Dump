# Project Status

**Current Phase:** Week 1 — Freeze the specification and build the skeleton

**Completed Work:**
- Scaffolded repository structure.
- Generated living documentation (`PROJECT_STATUS.md`, `DECISIONS.md`, `RUN_REGISTRY.md`, `FAILURE_LOG.md`, `SERVER_RUNBOOK.md`).
- Drafted `server_preflight.sh` to check remote server environment.

**Current Blocker:**
- Waiting for server preflight logs from the user to ascertain GPU topologies and available package versions.

**Running or Awaiting Server Jobs:**
- Awaiting `scripts/server_preflight.sh` run by the user.

**Next Three Tasks:**
1. Define the `EvidenceState` dataclass.
2. Build grouped split manifests builder (`build_manifests.py`).
3. Create model adapters with a shared API.

**Deviations from the Plan:**
- None so far.
