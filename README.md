# roboVLA

Fine-tuning **OpenVLA-7B** (a 7.5B vision–language–action model) for a high-school advanced-robotics capstone: up to **12 individual students**, each driving their own mecanum-base rover, training and serving QLoRA adapters through a **web-based training console** on the campus LAN.

## Docs

| Doc | What it is |
|---|---|
| [`docs/training-console-design.md`](docs/training-console-design.md) | **The spec.** v0.5.1, implementation-ready: fleet config, full API schema, admin controls, TDD/DRY contract, test matrix, build phases, repo layout. A future AI coder starts here. |
| [`docs/model-card-review.md`](docs/model-card-review.md) | Fact-check of the OpenVLA-7B model card / repo against the project plan (verified 2026-10-20): action un-normalization, ~100-demo dataset floor, ~1.5–2.5 Hz closed-loop latency rebase, official sanity checks. |
| [`docs/lesson-plan.md`](docs/lesson-plan.md) | 24-class / 85-min active-learning plan with milestone gates M1–M8. ⚠️ Being revised from the 4-team model to 12 individual students / 1 robot each (see console design §12). |
| [`docs/setup-and-materials.md`](docs/setup-and-materials.md) | BOM, pinned software, episode data spec, pre-class checklist. ⚠️ Same 12-student revision pending. |

## Status (2026-10-21)

- **Design:** v0.5.1 complete and frozen for implementation (§8 admin controls, §9 API schema, §10 portability/scalability, §11 TDD/DRY, Appendices A–C).
- **Phase 0 core (done, test-first per §11):** `robovla/core` (errors, states, spec = single JobSpec contract + preset catalog, config = fleet.yaml schema + diff, budgets, norm_stats, split, qa, ticket, audit, scheduler) + shared `robovla/episode.py`. 314 unit tests, 99% line coverage (target ≥95%), no I/O in core, no node names/IPs in code.
- **Phase 0 seams + console (done, against `FakeRunner` on CPU):** SQLite storage seam (WAL, sessions, shown-once tokens with O(1) lookup, idempotency 24 h, audit), file seam (zip-slip-safe chunked uploads, 224² prep, agent bundles), runner seam (protocol + `FakeRunner` scripted agent + `HttpRunner` — the single real implementation), service orchestrator (FIFO/run-now/no-preemption scheduler over measured VRAM, budget guard at dispatch, leases, SPOF job rebuild), full FastAPI surface §9 (auth + one RBAC dep, datasets, runs, tickets, SSE with Last-Event-ID resume, inference slots + the robot path, agent bundle pull, admin §9.7 all audited) and the `ui/` surfaces §7 (vanilla JS + HTMX, no build step — the class reads this code). **387 tests green in ~13 s** on CPU (`uv run pytest -m "not hardware"`; unit < 1 s, integration < 5 min budgets), all files ≤ 500 lines.
- **Fleet (verified):** pai1 (RTX 5090 32 GB, hosts the 27B assistant via `ninfer-serve`), campus 5090 Ti + 4070 Ti (wiring pending), home Strix Halo 8060S dev runner. GTX 1070 excluded.
- **Remaining Phase 0:** `robovla/cli.py` (model_download, gpu_selftest via agent, node bootstrap, backup/restore), `agent/` (runner container: server :8601 + vram probe + bundle pull), `images/` (Dockerfile.cuda/rocm), default `fleet.yaml` (Appendix A), `systemd/` units, `robovla-node` host script, 12 student accounts; then the hardware test rows (`GPU_TESTS=1 uv run pytest -m hardware`) and Phase 1 (Blackwell pinning, gpu_selftest).
