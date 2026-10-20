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

- **Design:** v0.3 complete and frozen for implementation (§8 admin controls, §9 API schema, §10 portability/scalability, §11 TDD/DRY, Appendices A–C).
- **Fleet (verified):** pai1 (RTX 5090 32 GB, hosts the 27B assistant via `ninfer-serve`), campus 5090 Ti + 4070 Ti (wiring pending), home Strix Halo 8060S dev runner. GTX 1070 excluded.
- **Implementation:** not started. Phase 0 (CPU-side, assistant stays loaded) awaits green light: uv venv @ 3.12 + torch cu128 provisioning, console scaffold, auth, dataset upload/QA, scheduler + FakeRunner, SSE, 12 student accounts.
