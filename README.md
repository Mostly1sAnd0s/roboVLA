# roboVLA

Fine-tuning **OpenVLA-7B** (a 7.5B vision–language–action model) for a high-school advanced-robotics capstone: up to **12 individual students**, each driving their own mecanum-base rover, training and serving QLoRA adapters through a **web-based training console** on the campus LAN.

## Docs

| Doc | What it is |
|---|---|
| [`docs/training-console-design.md`](docs/training-console-design.md) | **The spec.** v0.3, implementation-ready: fleet config, full API schema, admin controls, TDD/DRY contract, test matrix, build phases, repo layout. A future AI coder starts here (Phase 0 = `robovla/core` test-first). |
| [`docs/model-card-review.md`](docs/model-card-review.md) | Fact-check of the OpenVLA-7B model card / repo against the project plan (verified 2026-10-20): action un-normalization, ~100-demo dataset floor, ~1.5–2.5 Hz closed-loop latency rebase, official sanity checks. |
| [`docs/lesson-plan.md`](docs/lesson-plan.md) | 24-class / 85-min active-learning plan with milestone gates M1–M8. ⚠️ Being revised from the 4-team model to 12 individual students / 1 robot each (see console design §12). |
| [`docs/setup-and-materials.md`](docs/setup-and-materials.md) | BOM, pinned software, episode data spec, pre-class checklist. ⚠️ Same 12-student revision pending. |

## Status (2026-10-20)

- **Design:** v0.5.1 complete and frozen for implementation (§8 admin controls, §9 API schema, §10 portability/scalability, §11 TDD/DRY, Appendices A–C).
- **Implementation (2026-10-21):** Phase 0 core **done, test-first per §11**: `robovla/core` (errors, states, spec = single JobSpec contract + preset catalog, config = fleet.yaml schema + diff, budgets, norm_stats, split, qa, ticket, audit, scheduler) + shared `robovla/episode.py`. 314 unit tests, 99% line coverage (target ≥95%), suite < 1 s on CPU (`uv run pytest -m "not hardware"`), all files ≤ 500 lines, no I/O in core, no node names/IPs in code. Next: seams (`storage`/`fs`/`runner` + FastAPI app + SSE + UI) against `FakeRunner`.
- **Fleet (verified):** pai1 (RTX 5090 32 GB, hosts the 27B assistant via `ninfer-serve`), campus 5090 Ti + 4070 Ti (wiring pending), home Strix Halo 8060S dev runner. GTX 1070 excluded.
- **Implementation (rest of Phase 0):** console scaffold (auth, users, nodes, upload + prep/QA, tickets, queue, SSE, UI, LAN bind) against local agent + FakeRunner; admin surface §9.7; systemd units; 12 accounts — pending.
