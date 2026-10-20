# roboVLA Training Console — Design Specification (v0.4)

A web-based, OpenVLA-specific training control plane for a class of up to 12 individual students (currently 9), each with their own robot. Hosted on the campus LAN. Drafted from a live inspection of `pai1` on 2026-10-20; v0.3 finalized 2026-10-20 as the implementation-ready spec.

**v0.4 (2026-10-20):** Launch fleet = **pai1 only** (single trainer, 2 inference slots); the 5090 Ti / 4070 Ti / home 8060S are verified inventory and disabled config entries that join later purely via the admin node page — no code, no redeploy. **Node selection at submit**: student picks (teacher-gated `student_selectable`), teacher assigns/pins, or auto first-fit.

**v0.3 (2026-10-20) — implementation-ready.** Added: §8 Admin controls (complete taxonomy; every admin action is a gated REST endpoint) · §9 Full API schema (REST + SSE + security model) · §10 Portability & scalability (fleet-as-config, runner abstraction, deployment, scaling path, operational durability) · §11 TDD & DRY (pure core, single JobSpec contract, binding DRY rules, test matrix) · Appendices A–C (default `fleet.yaml`, JobSpec JSON, repo layout for the implementing coder).

**v0.2 (2026-10-20):** LAN-only access (no public tunnel) · multi-runner fleet (pai1 + 5090 Ti + 4070 Ti + home dev box) · "run now" = no preemption · 12 robots, 1 per student · standalone desktop capture app · demo-day slot relay · GTX 1070 / Strix Halo assessment.

## 1. The fleet (verified 2026-10-20)

| Node | GPU (VRAM) | Role | Training | Inference slots | Notes |
|---|---|---|---|---|---|
| **pai1** (primary console host) | RTX 5090, 32 GB | Primary trainer, 2 infer slots, hosts the assistant 27B | QLoRA batch 4–8 (bf16 also fits) | 2× 8-bit (~10 GB each) | Verified: driver 615.71/CUDA 13.4; `ninfer-serve` (27B NVFP4) holds 29.5 GB, plain process via `~/start-ninfer.sh` (no systemd/docker) → clean stop/start; 44 GB RAM free, 1.7 TB disk; system Python 3.14 → uv venv @ 3.12. **Blackwell sm_120: official OpenVLA pins (torch 2.2/cu121, flash-attn 2.5.5) won't work** → re-pin: torch 2.8/2.9 cu128, bnb ≥ 0.46, SDPA attention, transformers pinned after selftest. |
| **Campus node 2** (5090 Ti, on hand) | 32 GB | Second trainer, 2 infer slots | QLoRA batch 4–8 | 2× 8-bit | Same stack as pai1. |
| **Campus node 3** (4070 Ti, on hand) | 16 GB | Overflow trainer, 1 infer slot | QLoRA **8-bit, batch 1 only** (~2× slower); bf16 training doesn't fit (15 GB weights) | 1× 8-bit | VRAM budget guard is mandatory on this node. |
| **Home dev box** (Strix Halo, 128 GB unified) | 8060S iGPU (~7.4 TF FP32-class, ~15 TF bf16, 256 GB/s) | Development + integration runner (off-campus) | **bf16 LoRA, no quantization** — 20–25 GB total fits the 128 GB pool, which sidesteps bnb-on-ROCm entirely; ~3–4 h/epoch → overnight runs | bf16 inference testing (~1–3 s/action) | ROCm gfx1151 (RDNA 3.5); `HSA_OVERRIDE_GFX_VERSION=11.0.0` fallback; verify with `home_selftest` (30-min checklist). |
| ~~GTX 1070 (8 GB) on home box~~ | — | **Excluded** | — | — | 8 GB < floor for 7.5B even 4-bit-QLoRA (borderline OOM); no tensor cores; Pascal sm_61 deprecated (CUDA 12.8) / removed (CUDA 13). |

**Launch fleet (current): pai1 only** — one trainer, 2 inference slots. Rows 2–4 (5090 Ti, 4070 Ti, home 8060S) are verified inventory and planned fleet entries: `enabled: false` in `fleet.yaml` until they come online, and they join purely via the admin node registry (§8.2: add → probe → enable) — no code change, no redeploy. With a single trainer at launch, full runs are batched to the morning + overnight release windows (lesson-plan rhythm).

**Campus inference concurrency: 2 slots at launch** (pai1) → 5 slots once node 2/3 join → see §7.7 for the 12-robot demo relay.

## 2. What the system is

One Python service per campus node (the **console**; production instance on pai1) that is the single control plane for the campus GPUs. 12 student accounts, LAN-only access (students use classroom devices; no home/remote access required — per teacher decision). Students:

1. **Upload training data** (episode folders from their robot) or use the **standalone capture app** (§5.3) to collect; see a per-episode QA report before any queue time is spent;
2. **Configure a QLoRA fine-tune** (OpenVLA-specific form, safe defaults — the fragile stack is never exposed);
3. **Submit to a queue** and watch **live progress** (loss/accuracy sparklines, log tail, VRAM, ETA, queue position);
4. Get **automatic held-out validation** per checkpoint, **artifacts** (adapter, merged export), and a **class board** comparing all 12;
5. **Claim an inference slot** — the console serves their fine-tuned model to their robot's Pi (the console *is* the model server; no laptops-as-servers).

## 3. GPU tenancy & scheduling

Per-runner tenancy (same table as v0.1, per node): assistant 27B ~29.5 GB (pai1 only) · QLoRA 7.5B ~12–18 GB (1–2 concurrent on 32 GB nodes with budget guard) · bf16 inference ~15–19 GB (1) · 8-bit inference ~9–11 GB (2 on 32 GB, 1 on 16 GB).

**Scheduling rules (per teacher decision):**
- **No preemption.** "Run now" (teacher or student) is admitted to the front of the line *after* whatever is currently running on the selected runner finishes. The scheduler picks the first runner whose free VRAM fits the job; if all runners are busy, "run now" waits for the earliest completion (the student sees "waiting for a slot, ETA X").
- **VRAM budget guard:** a job starts only if measured free VRAM ≥ its budget (per-runner, per-job-type presets). A 2nd concurrent train slot on a 32 GB node is allowed only with the reduced-batch preset + guard (default: 1 train slot/node).
- **pai1-specific:** any training/inference job requires GPU release → `stop ninfer.service` (systemd wrapper around `start-ninfer.sh`, one-time setup) → poll for ≥20 GB free → run → restart. Configurable **release windows** (e.g., 07:00–08:30, overnight) + teacher "release now". The assistant being offline during jobs is a visible, honest tradeoff the class sees on the board.

**Admin levers (teacher panel):** per runner — enable/disable, train slot count (1–2), batch preset (1 / 4 / 8 with matching grad-accum), VRAM budget, priority, SSH target; global — ninfer release windows, job caps, per-student limits (1 running / 2 queued), queue position override (bump/kill). Full taxonomy in §8.

## 4. Architecture

```
 12 student browsers (classroom LAN devices; HTTP, self-signed TLS optional)
        │
        ▼
 robovla console — FastAPI + SQLite + SSE, bound to campus LAN IP :8501 (pai1 = production)
  ├─ /api             REST: auth, datasets, runs, artifacts, slots, admin (§9)
  ├─ /stream/{job_id} SSE: metrics JSONL tail, log tail, nvidia-smi poll
  ├─ scheduler        SQLite-backed FIFO; per-runner slots; no preemption
  │    └─ runner interface (JobSpec JSON, location-agnostic; Appendix B)
  │         ├─ local runner: subprocess under the console (default)
  │         └─ ssh runner: ssh + nohup; same venv recipe & path layout;
  │                        datasets pulled via rsync over LAN; metrics/logs synced back
  ├─ gpu manager      VRAM probe, ninfer stop/start (pai1), zombie-VRAM watchdog
  └─ dataset service  chunked upload → unzip → validate → resize 224² → norm stats
        │
        ▼
 /home/adam/robovla/{venv, openvla, datasets, runs, artifacts}   (same layout per node)
        │
        ▼
 campus GPUs (launch = pai1 only): pai1 5090 (1 train slot / 2 infer)
 · node2 / node3 joined later via the admin page (§8.2) — config only
```

**Why FastAPI + SSE + vanilla JS/HTMX (not Streamlit/Gradio):** 12 concurrent long-lived live dashboards, real per-user auth, no per-interaction rerun model, ~1.5–2k LOC, and students can read the whole codebase. LAN-only: no tunnel, no public exposure; auth + rate limits still apply (students are students).

## 5. Data plane

**5.1 Upload (MVP).** Per-student, versioned datasets (`s07/task-stop-at-target/v2`) on pai1. Chunked zip upload of the episode folders (their Pi pipeline: `frames/`, `actions.json`, `episode.json`); 100 episodes ≈ 0.5–1 GB; 2 GB cap, zip-slip-safe, jpeg/json-only. SSH runners pull the prepared dataset via rsync over LAN (trivial at ~1 GB).

**5.2 Server-side dataset prep (CPU, ~minutes):** per-episode validation (frame==action count, 10 Hz timing, idle >2 s flag, sign/jump anomalies), 224×224 resize via the official processor's preprocessing (train/inference augmentation parity), `norm_stats.json` (per-dim min/max over 7 dims), held-out split (15%, seeded per student+version). Output: "ready" dataset + QA report page — the gate that stops bad data from eating queue time.

**5.3 Standalone capture app (separate deliverable, built in Phase 3 with students).** A desktop app students run on their own laptops on the classroom LAN — per teacher decision, this is a standalone project, not a console feature:
- **Stack: PySide6 v1** (pure Python, students can read it, pip-installable; single-file via PyInstaller for student laptops). Tauri polish is a v2 option.
- **Features:** robot discovery (static IP list from the class sheet + optional mDNS), live 10 Hz view (reuses their `frame_sender`), keyboard + gamepad teleop (pygame) → Pi (their `twist` path), episode logging to the standard spec, built-in pre-upload QA (sync check, idle flags), one-click upload to the console, "replay check" (open-loop replay of the last episode on the robot).
- Works in class or at home if a robot comes home. Same episode spec as §5.1, so capture and upload feed one pipeline. **Shared code:** the capture app vendors `episode.py` / `norm_stats.py` / JobSpec presets from the console repo verbatim (§11) so capture-side QA and console QA can never drift.

## 6. Training engine (OpenVLA-specific)

**Decision (unchanged from v0.1): a thin in-house `robovla_train.py` (~400–600 LOC) on the openvla repo's model code + PEFT QLoRA**, emitting structured JSONL metrics (first-class observability → SSE), validation hooks, per-epoch checkpoints, crash resume. The official `finetune.py` is *not* abandoned: a `crosscheck` job runs it on a 50-episode subset and students compare curves (Class 9–10 exercise). The recipe is identical either way: 8-bit, rank 32, α 32, lr 5e-4, image aug on.

**Per-runner presets** (admin-configurable):

| Runner | Quantization | Batch | Notes |
|---|---|---|---|
| pai1 5090 / 5090 Ti (32 GB) | 8-bit | 4–8 | ~45 min–2 h/run |
| 4070 Ti (16 GB) | 8-bit | 1 (+grad accum) | ~2× slower; budget guard |
| Home 8060S (128 GB unified) | **bf16, no quant** | 1–2 | overnight; no bnb dependency (ROCm) |

**Job types:** `smoke` (1 batch, ~2–5 min, mandatory before full runs) · `train` (cap 3 h) · `validate` (held-out eval, auto after checkpoints) · `merge` (`merge_and_unload()` + 10-image identity check; 15 GB export on demand, 14-day purge) · `infer` (serve adapter, token auth) · `crosscheck` (official script on subset) · `gpu_selftest` / `home_selftest` (stack verification per node: load, fwd/bwd, `predict_action`, timings).

**Stack re-pin (Blackwell):** uv venv @ Python 3.12, torch 2.8/2.9 cu128, transformers pinned after selftest, bnb ≥ 0.46, SDPA (flash-attn only if it builds clean for sm_120), timm/pillow/numpy. Pinned in the console repo. **Escape hatch:** the runner interface is location-agnostic — if a node fails, the same JobSpec dispatches to any other node (or a cloud A100).

## 7. UI surfaces

1. **Dashboard** — queue position + ETA, my datasets, my runs, per-node GPU state (who holds it, VRAM), class activity feed.
2. **Datasets** — upload wizard → prep job → QA report, versioning, norm stats display, held-out info, "ready" badge.
3. **New run** — dataset (ready only), task prompt template, **training node** (default *auto* = first-fit scheduler; or a specific node when the teacher has marked it student-selectable, §8.2; the teacher can assign/pin any node via requeue), rank/α/lr/epochs/aug/seed in a collapsible *advanced* section (values drawn from the single preset catalog in core), **required experiment ticket** (hypothesis → config → prediction; outcome at completion), submit → smoke → train.
4. **Run detail (SSE)** — train + held-out val sparklines, step/total/ETA, grad norm, LR, VRAM + util, checkpoint events, raw log (collapsible), kill (teacher).
5. **Class board** — all 12 side-by-side (best held-out accuracy, run counts, dataset stats, tickets); printable; the final-week post-mortem material.
6. **Artifacts** — adapter download, merged export (on demand), run config JSON.
7. **Inference slots** — "serve my adapter" (8-bit), slot queue, live status, **Pi connection info** (endpoint + per-student token).
   **7.7 Demo-day relay (12 robots, 2 slots at launch; 5 after node2/3 join):** students are scheduled into ~4-min arena runs; runs pipeline with presentations — student *i* presents their run's result while student *i+1's* robot executes on a slot. 9–12 students × (4-min run + 3-min talk + 2-min Q&A) ≈ 55–90 min total; robot slot demand is only ~4 min/student (≤ 48 robot-minutes), which 2 slots absorb comfortably inside the presentation timeline. The slot board is visible to the class. For practice weeks, the same relay gives every student 30–60 s of autonomy back-to-back — "everyone's robot is always moving."
8. **Teacher panel** — thin forms over the admin API (§9.7): runner registry (enable, slots, batch preset, VRAM budget, priority, SSH target), queue ops, ninfer release windows + "release now" (assistant-offline warning), student accounts, disk usage, raw logs. Complete taxonomy in §8; every panel control is an audited endpoint.

## 8. Admin controls

Complete taxonomy of everything the teacher can do. **All admin actions are exposed as gated REST endpoints (§9.7), rate-limited (default 60 req/min), and written to the append-only audit log** (actor, timestamp, action, target, before→after JSON). The UI (§7.8) is a thin form over these endpoints — there are no admin capabilities that are not in the API.

### 8.1 Accounts

| Control | Behavior |
|---|---|
| Create / import students | single create or CSV roster import (id `s01`–`s12`, display name, email optional); created inactive until activated |
| Activate / deactivate | deactivated: no login, no token use; queued jobs requeue to fitting runners, running jobs complete; no new submissions |
| Roles | `teacher` (everything) · `student` (own data, class board, inference claims) · `observer` (read-only board + GPU state; demo-day visitors) |
| Reset password | out-of-band: teacher sets a one-time code/link (paper), no email dependency in the classroom |
| Per-student limits (defaults) | max running (1) · max queued (2) · max dataset bytes (2 GB) · max run minutes (180) · run-now allowed (yes — front of line after current completion, §3) |
| Pi tokens | generate / rotate / revoke per student; scoped to that student's datasets, adapters, and inference claims; value shown once, argon2-hashed at rest |
| Sessions | list active sessions per user; revoke any session (immediate) |
| Audit log | all admin actions + auth events; append-only; admin-readable; JSON export |

### 8.2 Nodes / endpoints

The node registry **is** the fleet config (§10, `fleet.yaml`) — the admin UI edits config, never machines.

| Control | Behavior |
|---|---|
| Add node | form: id, label, runner (`local` \| `ssh target`), LAN IP, GPU VRAM, train slots (1–2), infer slots (1–2), quant profile (`bnb8` \| `bf16`), batch preset (1 \| 4 \| 8 + matching grad-accum), VRAM budget per job type, priority (int, default 0), student-selectable (default **off**), enabled (default **off**) → **dry-run join probe** (ssh reachability, `nvidia-smi`, disk free ≥ 50 GB, model dir present) → node stays disabled until teacher enables it |
| Edit node | any field; changes apply at the next scheduling tick; active jobs unaffected (a budget shrinking below a running job completes that job; no new admissions on that node until refit) |
| Enable / disable | disable: queued jobs requeue to other fitting runners; running jobs fail-fast with checkpoint-resume hint; inference slots drain with client notification (Pi watchdog stops the robot after lease expiry) |
| Remove node | as disable, then removed from config; its jobs/artifacts remain on disk, flagged |
| Probe | on-demand re-run of the join probe (network changes, after maintenance) |
| GPU reset / VRAM purge | `nvidia-smi --gpu-reset` (confirmation required, no running jobs) · zombie purge: post-job probe lists orphan PIDs holding VRAM → teacher confirms → kill (audit-logged) |
| Service ops | restart `robovla-console` on that node (ssh nodes), tail service log (last 500 lines) |
| Endpoint visibility | each node's LAN IP + robot port published on the class board for Pi setup; robots authenticate with per-student tokens, so no per-robot endpoint config is needed |
| Student-selectable | teacher toggles per node whether students can name that node at run submission (§9.3); off ⇒ auto first-fit or teacher-assign only; `enabled: false` nodes are hidden from students entirely |

### 8.3 Jobs & queue

| Control | Behavior |
|---|---|
| Queue view (all roles) | all nodes, FIFO, position, ETA (queue wait + estimated run time), node fit; students read-only |
| Bump (teacher) | move a job to the front of the queue on the nodes it fits — effective at the next completion boundary (no preemption, §3) |
| Kill | students: own queued jobs; teacher: any (queued → `killed`; running → SIGTERM, 10 s grace, SIGKILL → state `killed`, VRAM probe, checkpoint artifacts preserved) |
| Reschedule (teacher) | pin a queued job to a specific node, or clear a pin |
| Manual completion (teacher) | mark a job `done` for work done outside the console (e.g., teacher ran `finetune.py` by hand); note required; inference only enabled if an artifact manifest exists |
| Retry from checkpoint (teacher) | resubmit a failed/killed training job from its last checkpoint (dataset rsynced if the node differs) |
| Annotate (teacher) | free-text note on a job, visible on the class board |
| Caps | global max concurrent (default = sum of node train slots) · per-node (slots) · per-student (§8.1); scheduler enforces before admission |
| Ninfer release windows (pai1) | schedule-like windows (defaults 07:00–08:30 + 22:00–06:00, teacher-editable): inside a window the scheduler may stop `ninfer.service` for jobs; outside, jobs wait. **Release now** = manual stop (class board shows assistant-offline banner); **Restore now** = start. Window state visible to the class — honest tradeoff, §3 |

### 8.4 Datasets & artifacts

| Control | Behavior |
|---|---|
| Dataset inventory | per student/version: episode count, size, QA status, which runs used it |
| Reject (teacher) | block a dataset from new runs (reason shown on the student's page); un-reject after a corrected re-upload |
| Purge (teacher) | two-step confirmation; deletes episodes + QA + stats; runs that used it keep their checkpoints/adapters (run config is self-contained) |
| Retention | default: keep all student data; optional per-student max-versions (LRU purge *proposal* list — teacher approves) · global disk cap per node with 80 %/90 % alerts |
| Artifact inventory | per run: adapter (always) · merged export (on demand, 15 GB, 14-day auto-purge — teacher-adjustable) · run config JSON |
| Force purge / export | teacher may purge early; **backup bundle** zips datasets + adapters + DB snapshot for off-site archive |

### 8.5 System

| Control | Behavior |
|---|---|
| Health | `GET /api/health` (LAN-public): service uptime, per-node state (online, GPU free/busy, queue depth) — consumable by a classroom display |
| Logs | console service log (per node) · per-job training logs (run detail) · audit log (§8.1) |
| Backup | nightly (teacher-adjustable time) of DB + datasets + adapters to a configurable destination (local dir or NAS mount); status view (last run, size, checksums); restore = documented procedure, covered by the test matrix |
| Config | view effective config (rendered YAML + per-node profiles); **hot reload**: validate → apply; active jobs unaffected, queued jobs revalidated against new budgets |
| Maintenance mode | console read-only: no new submissions/uploads; running jobs and live slots continue; for stack upgrades |
| VRAM watchdog | post-job probe compares running PIDs to expected; orphans → alert banner + one-click purge (§8.2) |

## 9. API schema

Base URL: `http://<console-lan-ip>:8501` (console); robot inference on each node's robot port (default 8502). Conventions: JSON bodies; errors `{"error":{"code","message"}}` (§9.6); list endpoints take `?cursor=&limit=` (default 50); `Idempotency-Key` header on `POST /api/runs` (client UUID, 24 h dedup → returns the existing job); artifacts are ETag-cached; timestamps RFC 3339 UTC.

### 9.1 Auth (cookie sessions, 12 h sliding)
- `POST /api/auth/login` `{username, password}` → sets httpOnly cookie
- `POST /api/auth/logout`
- `GET /api/auth/me` → identity, role, limits, own token metadata
- `POST /api/auth/token` `{action: generate|rotate|revoke}` (own Pi token; value shown once on generate/rotate)

### 9.2 Datasets (student: own · teacher: all)
- `POST /api/datasets/upload` (chunked multipart; `{dataset_id: "s07/task-stop-at-target/v2", total_chunks, chunk_index, sha256}`)
- `POST /api/datasets/upload/finalize` `{dataset_id}` → starts server-side prep (§5.2); 409 if chunks incomplete
- `GET /api/datasets` · `GET /api/datasets/{id}` (status: `uploading|preparing|ready|rejected|failed`)
- `GET /api/datasets/{id}/qa` (per-episode QA report + summary)
- `GET /api/datasets/{id}/norm-stats`
- `POST /api/datasets/{id}/revalidate` (re-run prep after corrections; new version)
- `GET /api/datasets/{id}/episodes/{ep}/frames/{i}.jpg` (browser preview)

### 9.3 Runs
- `POST /api/runs` `{dataset_id, job_type: smoke|train|validate|merge|infer|crosscheck, node_id?: "auto"|<node-id>, config: {rank, alpha, lr, epochs, batch, grad_accum, aug, seed, prompt_template}, run_now?: bool}` → `{job_id, position, eta_s}`
  - Node selection: `"auto"` (default) ⇒ scheduler first-fit across enabled nodes; a specific node is accepted for students only if enabled **and** `student_selectable: true` (§8.2) — teacher may name any enabled node; pinned jobs wait on that node only
  - Validation: dataset `ready`; per-student caps; job-type budget fit — if no node fits now, response carries `waiting: "slot"` + earliest ETA
- `GET /api/runs` (own) · `GET /api/runs/{job_id}` (state, config, node, progress, artifacts)
- `GET /api/runs/queue` (all nodes, all students; position + ETA; feeds the class board)
- `POST /api/runs/{job_id}/kill` (own queued; teacher: any)
- `POST /api/runs/{job_id}/requeue` `{node_id?}` (teacher) · `POST /api/runs/{job_id}/bump` (teacher)
- `POST /api/runs/{job_id}/complete` `{note}` (teacher)
- `GET /api/runs/{job_id}/logs?offset=` (tail)
- `GET /api/runs/{job_id}/artifacts` (adapter path, merged status, run config JSON)
- `GET /api/runs/{job_id}/ticket` · `PUT /api/runs/{job_id}/ticket` `{hypothesis, config_summary, prediction}` — **required before submit**; outcome field completed at run end

### 9.4 SSE streams
- `GET /stream/{job_id}` — events: `state` (`queued|gpu_releasing|running|finishing|done|failed|killed` + reason) · `metrics` (JSONL: `step, total, loss, lr, grad_norm, vram_mb, vram_util_pct, eta_s`) · `val` (held-out accuracy per checkpoint) · `log` (tail lines) · `artifacts` (checkpoint/adapter/merged events). Heartbeat every 15 s; `Last-Event-ID` resume.
- `GET /stream/class` — class-board feed (queue, node state, slot board, activity), ≤1 msg/s, all roles.

### 9.5 Inference (robot path)
- `POST /api/infer/slots/claim` `{run_id}` (adapter must exist) → `{slot, node_id, endpoint: "http://<node-lan-ip>:8502", token: <own, scoped>, lease_s}`
- `GET /api/infer/slots` (all: holder, adapter, node, load)
- `POST /api/infer/slots/{slot}/release`
- **Robot endpoint (node robot port, `Authorization: Bearer <student token>`):**
  `POST /api/robot/predict` `{image: base64 jpg, prompt}` → `{action_norm[7], action_robot[7] (un-normalized with the dataset's min/max stats), latency_ms}`
  - Rate limit ~5 req/s; slot lease TTL (default 300 s idle → slot auto-released; the Pi watchdog already stops the robot on stale actions); target p95 < 400 ms incl. LAN on 8-bit 32 GB nodes (~150–300 ms inference per the latency rebase in `model-card-review.md`).

### 9.6 Error codes (stable — Pi client and tests depend on them)
`auth` · `forbidden` · `not_found` · `conflict` · `dataset_not_ready` · `budget_unfit` · `limit_exceeded` · `rate_limited` · `node_offline` · `internal`

### 9.7 Admin API (role = teacher · audit-logged · 60 req/min) — the §8 mapping
- **Users:** `GET /api/admin/users` · `POST /api/admin/users` · `POST /api/admin/users/import` (CSV) · `PATCH /api/admin/users/{u}` (active, role, limits) · `POST /api/admin/users/{u}/reset-password` · `GET /api/admin/users/{u}/sessions` · `DELETE /api/admin/users/{u}/sessions/{sid}` · `POST /api/admin/users/{u}/tokens/{action}`
- **Nodes:** `GET /api/admin/nodes` · `POST /api/admin/nodes` · `PATCH /api/admin/nodes/{id}` · `DELETE /api/admin/nodes/{id}` · `POST /api/admin/nodes/{id}/probe` · `POST /api/admin/nodes/{id}/purge-vram` · `POST /api/admin/nodes/{id}/gpu-reset` · `GET /api/admin/nodes/{id}/logs`
- **Jobs:** `GET /api/admin/jobs` · `POST /api/admin/jobs/{id}/bump` · `POST /api/admin/jobs/{id}/kill` · `POST /api/admin/jobs/{id}/requeue` · `POST /api/admin/jobs/{id}/complete` · `POST /api/admin/jobs/{id}/note`
- **Ninfer (pai1):** `GET /api/admin/ninfer` · `POST /api/admin/ninfer/release` · `POST /api/admin/ninfer/restore` · `GET /api/admin/ninfer/windows` · `POST /api/admin/ninfer/windows`
- **Datasets / artifacts:** `DELETE /api/admin/datasets/{id}` · `POST /api/admin/datasets/{id}/reject` · `GET /api/admin/artifacts` · `DELETE /api/admin/artifacts/{id}` · `POST /api/admin/backup-bundle`
- **System:** `GET /api/health` (public) · `GET /api/admin/audit?since=` · `GET /api/admin/config` · `POST /api/admin/config/reload` · `POST /api/admin/maintenance` `{on|off}` · `POST /api/admin/backups/trigger` · `GET /api/admin/backups`

### 9.8 Security model
- Accounts: 12 students + teacher (optional observers); argon2id; httpOnly session cookies; passwords never logged.
- RBAC: student → own datasets/runs/tokens, read queue + class board + slots, claim/release own slot · teacher → everything (§9.7) · observer → board + node state only. Enforced by **one** FastAPI dependency — never per-route.
- Tokens: per-student Pi tokens, scoped to their datasets/adapters/slots, shown once, hashed at rest, revocable.
- Network: console binds the campus LAN IP; no public exposure, no tunnel; optional self-signed TLS via reverse proxy; students use classroom devices.
- Uploads: 2 GB cap, chunked, zip-slip-safe, path-sanitized, jpeg/json only.
- **No shell, ever:** validated JobSpec JSON → fixed argv.
- Rate limits: login 5/min/IP · uploads 2 concurrent/student · submissions 10/min/student · admin 60/min.
- Audit: all §8 actions + auth events.
- (Flagged, out of scope: `infergate` :8317 and `flygym` :8765 on pai1 bind 0.0.0.0 → recommend localhost-only.)

## 10. Portability & scalability

**Core principle: hardware lives in config, not code.** The console has no knowledge of "pai1" or "5090" as a machine — a node is an entry in `fleet.yaml` with a hardware profile. Moving the class to a new room, a new school, or a cloud box = a new `fleet.yaml`, same image, same code. (Appendix A: full default config.)

- **Fleet = one YAML file** (`fleet.yaml`, validated against one schema in core): console block (bind, ports, TLS, session TTL, limits, retention) + `nodes[]` (id, label, runner `{type: local|ssh, target, workdir, venv}`, gpu `{vram_gb, quant: bnb8|bf16, train_slots, infer_slots, budgets: {train_gb, infer_gb}}`, endpoints `{lan_ip, robot_port}`, presets, features `{ninfer: {...}}`) + per-student limit defaults. The shipped default describes the launch fleet (pai1 enabled; node2/node3/home-8060s as disabled inventory entries); a new deployment ships with a minimal one-node config.
- **Runner = the only hardware boundary.** One interface, two implementations: `LocalRunner` (subprocess in the console host's venv) and `SshRunner` (`ssh` + `nohup`; datasets/metrics over `rsync` LAN). Same JobSpec in, same metrics/events out. A third implementation (ssh into a rented cloud A100/H100) is an escape hatch that requires no code change anywhere else.
- **Per-node hardware profiles** drive all presets (VRAM budgets, quantization, batch, slots) — a new 24 GB node joins as a config line + probe; nothing is recompiled or re-pinned globally.
- **Deploy anywhere:** Ubuntu 22.04/24.04 + any supported accelerator (NVIDIA CUDA or ROCm). Two install paths: Docker Compose (console + optional nginx for TLS) or bare `uv venv && uv sync` + shipped systemd units (`robovla-console.service`, `ninfer.service` where the assistant 27B runs). Provisioning jobs (`model_download`, `gpu_selftest`) run per node; the console itself is ~1.5–2k LOC.
- **State is portable:** SQLite (WAL) + the data dir (`datasets/ runs/ artifacts/`) = the entire system state. Rebuild or move = clone repo + copy data dir; nightly backup (§8.5) covers the rest.

**Scaling path:**

| Dimension | What changes |
|---|---|
| More GPUs | one config entry (LAN ssh or cloud) + join probe. Scheduler, UI, tests unchanged. |
| More students | accounts are rows; per-student caps + morning/overnight windows (launch, single trainer) and added nodes (scaling) keep the queue sane; board scales. |
| Multi-room / multi-school | same console image + different `fleet.yaml`; optional class-code tenancy if two classes share one console. |
| Bigger / newer VLA models (OFT, FAST, PRIME) | job spec carries model id + processor; same upload/QA/queue/SSE pipeline. |
| Console SPOF | rebuild from repo + restore backup < 1 h; episode data also lives on the 12 PIs. |

**Operational durability (carried from v0.2 §9):**
- systemd: `robovla-console.service`, `ninfer.service` (wraps `start-ninfer.sh`).
- Job state machine: `queued → gpu_releasing → running → finishing → done|failed|killed` — a pure transition table in core, unit-tested.
- Runner failure: dispatch failure or mid-run node death → job fails fast, auto-reschedules on the next fitting runner (never preempts a *running* job), student notified with ETA; crash resume from checkpoints (dataset rsynced to the new node).
- Post-job VRAM probe → zombie alert + teacher purge. Retention: 20 run logs/student; merged exports 14 days; nightly backup of datasets + adapters + DB.

## 11. TDD & DRY

**Process (binding for all implementation work):**
1. Every feature starts with a **failing test** (red) → minimal code (green) → refactor. No module ships without tests.
2. GitHub CI on push: the full CPU suite (< 5 min). Hardware-gated tests (`gpu_selftest`, `home_selftest`, robot) run manually / at node join behind `GPU_TESTS=1`.
3. **Definition of done:** CI green · no new TODOs · §9 updated if the API surface changed · every new admin action has audit coverage + a test.

**Architecture for testability:**
- **Pure core** (`robovla/core/`; no I/O, no network, no wall clock — time is injected): scheduler (FIFO, run-now = front-of-line after current completion, no preemption, per-node slot + VRAM-budget fit, kill/reschedule state machine) · VRAM budget calculator · JobSpec validation (one JSON schema) · dataset QA rules · norm stats + un-normalization · held-out split (seeded, deterministic) · experiment-ticket schema · audit reducer · config schema. Target: ≥ 95 % line coverage; suite < 60 s on CPU.
- **I/O only at seams:** `storage` (SQLite) · `fs` (datasets/artifacts) · `runner` (local/ssh) · `proc` (subprocess/ssh) · `http` (FastAPI app) · `clock`. Tests use fakes at every seam — `FakeRunner` emits scripted metric/event streams so the **entire SSE path runs on CPU**.
- **One contract:** JobSpec JSON (Appendix B) — the only artifact shared by UI, scheduler, runners, docs, and the crosscheck job. One schema, one validator, one preset catalog (the QLoRA default recipe is defined once: 8-bit, rank 32, α 32, lr 5e-4, aug on; the "advanced" form reads from that catalog).
- **One config:** `fleet.yaml` — one schema, one loader, hot reload revalidates. No machine-specific string in code.
- **One episode validator:** `episode.py` (format, frame/action counts, 10 Hz timing, idle > 2 s, sign/jump checks) + `norm_stats.py` + the preset catalog are **shared code vendored verbatim into the standalone capture app** — capture-side QA and console QA cannot drift.

**DRY checklist (rules for the implementing coder):**
- No inline SQL outside `storage.py` · no JSON schema defined in two places · one error type (§9.6) · one logging setup · auth/RBAC through the single FastAPI dependency (never per-route) · no node names/IPs in code (all in config) · capture app and console share `episode.py` / `norm_stats.py` / presets verbatim.

**Test matrix:**

| Layer | Coverage | Gate |
|---|---|---|
| Unit — core | scheduler ordering (incl. run-now, no-preemption, budget fit: 16 GB node → batch-1 preset, 2-slot admission on 32 GB) · kill/reschedule transition table · QA rules (frame/action mismatch, 10 Hz timing, idle > 2 s, sign flip, jump anomaly) · norm stats + un-normalization round-trip · held-out split determinism · JobSpec + config schema validation | every PR · CPU · < 60 s |
| Integration — service (CPU, FakeRunner) | chunked upload (incl. zip-slip payload rejected) → prep → QA gate → queue admission → SSE (state/metrics/val events, heartbeat, resume) → artifact manifest → robot `predict` with mocked model + token auth · RBAC (student 403 on others' runs and all `/api/admin`) · idempotency dedup · rate limits · audit-log entry for every admin action | every PR · CPU · < 5 min |
| Simulated class E2E (CPU) | scripted 12-student scenario: 12 uploads → 12 submissions (launch config = pai1 only; extended config = +node2/node3 to exercise multi-endpoint first-fit, node pinning, and student-selectable gating) → queue drains in expected order → kill/bump/requeue mid-drain → demo-day 2-slot relay schedule (§7.7) → backup + restore round-trip | nightly + per release |
| Hardware-gated (GPU) | `gpu_selftest` per node (torch/CUDA visible, load 7.5 B, 8-bit fwd/bwd, bf16 fwd/bwd where it fits, `predict_action` on held-out images, 10-step QLoRA) · `home_selftest` (ROCm gfx1151: `torch.cuda.is_available`, bf16 GEMM benchmark, load + predict, 10-step bf16 fwd/bwd — no bnb) | manual · node join + stack changes |
| Robot (Pi) | token auth · LAN predict round-trip · stale-action watchdog stop · slot lease expiry | per class milestone (M5–M7) |

## 12. Build plan

| Phase | When | Who | Scope | Exit |
|---|---|---|---|---|
| **0** | Now → ~1 wk | instructor + AI coder (assistant) | Home-box ROCm verify (`home_selftest`); provisioning (uv venv, torch cu128, openvla clone, 15 GB model download); core (scheduler/budgets/QA/spec/config) test-first per §11; console scaffold (auth, users, nodes, upload + prep/QA, queue, SSE, UI, LAN bind) against FakeRunner; admin surface §9.7; systemd units; 12 accounts | 12 students log in on the campus LAN, upload, watch a simulated run; teacher panel works end-to-end |
| **1** | First GPU window | AI coder; teacher triggers | `gpu_selftest` on pai1, Blackwell stack pin, `smoke`, 50-episode `crosscheck` | real training verified (class gate M5) |
| **2** | By Class 9 | AI coder | `robovla_train.py` integration, validation harness, merge/export, inference slots + Pi token auth, class board, teacher panel complete; 5090 Ti + 4070 Ti join via the admin node registry when they come online (deferred — not needed at launch) | console production-ready for the class |
| **3** | Extension week (Class 17) | **students** + AI coder | standalone capture app (PySide6; vendors `episode.py`/`norm_stats.py`), Foxglove live-feed panel, QA-flag + board-view contributions | students own part of the instrument |

**Pedagogical integration:** the console *replaces* the "build the TUI" class activity. Classes 9–10: the console is live → students operate it, read `robovla_train.py` (12 sections, one per student), run the crosscheck, own the experiment tickets. With 12 individual students and 1 robot each, the console is the shared lab instrument — like the eval harness.

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Blackwell stack pinning (torch/tf/bnb/flash-attn) | Early selftest; SDPA fallback; pinned in-repo; runner interface dispatches to any node if pai1 fails |
| Runner heterogeneity (32 GB / 16 GB / iGPU) | Per-runner batch presets + VRAM budget guard; per-node selftest; 4070 Ti batch-1-only |
| Home box ROCm (gfx1151) friction | `home_selftest` first; bf16-only (no bnb); fallback: CPU torch for engine verification; class never depends on it |
| No-preemption + 12 students → queue pressure (single trainer at launch) | morning + overnight windows carry full runs; fast free smoke runs in class; run-now = front-of-line after current completion; pressure relieved by adding nodes via the admin page (config-only, §8.2) |
| Demo-day 12-robot concurrency | 5 inference slots + pipelined relay (§7.7) — all robots cycling continuously |
| ssh dispatch failure | Fail-fast, auto-reschedule on next runner, checkpoint resume |
| Zombie VRAM | Post-job probe + teacher purge |
| Single-host (console) SPOF | Console rebuilds from its repo in a day; episode data also on the 12 PIs |
| 12 robots in one room (practice weeks) | 4 sub-arenas × 3 students, zone rotation (see lesson plan) |
| Config drift (fleet.yaml vs. reality) | Single schema + CI validation + join probes + hot reload revalidates queued jobs |
| AI coder implementing from this doc | §11 TDD contract + test matrix are the acceptance criteria; every admin action = endpoint + audit + test |

## 14. Open questions

1. **Wiring (later — not needed for launch):** when the 5090 Ti and 4070 Ti machines come online, are they on the campus LAN (wired 1 GbE preferred)? Each needs a LAN IP for the SSH runner + Pi inference endpoints, then joins via the admin node registry (§8.2).
2. **ninfer release windows:** confirm mornings (e.g., 07:00–08:30) + overnight as defaults?
3. **Demo day date + audience** (sizing the relay schedule; judges/audience seating in the sub-arenas).
4. **Student laptop inputs:** keyboards only, or do some students have USB gamepads (capture app input design)?

## Appendix A — `fleet.yaml` (default, current fleet)

```yaml
version: 1
console:
  bind: 0.0.0.0            # on pai1; students reach via LAN IP; LAN-only by placement
  port: 8501
  tls: false               # optional self-signed via reverse proxy
  session_ttl_h: 12
  limits:
    student:
      max_running: 1
      max_queued: 2
      max_dataset_bytes: 2147483648   # 2 GB
      max_run_min: 180
      run_now: true
    admin_rate: "60/min"
  retention:
    run_logs_per_student: 20
    merged_export_days: 14
    backup: {enabled: true, at: "02:30", dest: /home/adam/robovla-backups}

nodes:
  - id: pai1
    label: "pai1 (RTX 5090, primary)"
    runner: {type: local, workdir: /home/adam/robovla, venv: .venv}
    gpu: {vram_gb: 32, quant: bnb8, train_slots: 1, infer_slots: 2,
          budgets: {train_gb: 18, infer_gb: 11}}
    endpoints: {lan_ip: "<PAI1_LAN_IP>", robot_port: 8502}
    student_selectable: true     # launch node; the only node students can name at submit
    features:
      ninfer:
        service: ninfer
        free_vram_gb: 20
        windows: ["07:00-08:30", "22:00-06:00"]
    enabled: true

  - id: node2
    label: "campus RTX 5090 Ti"
    runner: {type: ssh, target: "adam@<NODE2_LAN_IP>", workdir: /home/adam/robovla, venv: .venv}
    gpu: {vram_gb: 32, quant: bnb8, train_slots: 1, infer_slots: 2,
          budgets: {train_gb: 18, infer_gb: 11}}
    endpoints: {lan_ip: "<NODE2_LAN_IP>", robot_port: 8502}
    enabled: false          # until join probe passes

  - id: node3
    label: "campus RTX 4070 Ti (batch-1 only)"
    runner: {type: ssh, target: "adam@<NODE3_LAN_IP>", workdir: /home/adam/robovla, venv: .venv}
    gpu: {vram_gb: 16, quant: bnb8, train_slots: 1, infer_slots: 1,
          budgets: {train_gb: 14, infer_gb: 11}}
    presets: {batch: 1, grad_accum: 4}
    endpoints: {lan_ip: "<NODE3_LAN_IP>", robot_port: 8502}
    enabled: false          # until join probe passes

  - id: home-8060s
    label: "home Strix Halo 8060S (ROCm, overnight dev)"
    runner: {type: ssh, target: "adam@<HOME_IP>", workdir: /home/adam/robovla, venv: .venv}
    gpu: {vram_gb: 128, quant: bf16, train_slots: 1, infer_slots: 0,
          budgets: {train_gb: 30, infer_gb: 20}}
    presets: {batch: 2}
    enabled: false          # off-campus; teacher enables for overnight integration work
```

Notes: **pai1 is the only enabled node at launch**; node2/node3/home-8060s are verified inventory, disabled until they come online — adding them later is an admin-page operation (add node → probe → enable, §8.2), no code or redeploy. `train_slots: 1` everywhere is the §3 default; the teacher raises a 32 GB node to 2 via the admin lever (reduced-batch preset + budget guard apply). `student_selectable` (default false) controls whether students can name a node at submission (§9.3). Placeholder IPs are filled in at wiring time (§14 Q1) — code never hardcodes them.

## Appendix B — JobSpec JSON (the single contract)

The only artifact shared by UI, scheduler, runners, docs, and the crosscheck job. Validated by one schema in core; the runner derives a **fixed argv** from it (no shell).

```json
{
  "job_id": "j_20261020_0007",
  "student": "s07",
  "type": "train",
  "model": "openvla/openvla-7b",
  "dataset": "s07/task-stop-at-target/v2",
  "node": {"id": "pai1", "runner": "local"},
  "quant": "bnb8",
  "lora": {"rank": 32, "alpha": 32},
  "optim": {"lr": 0.0005, "epochs": 3, "batch": 4, "grad_accum": 1, "max_seq_len": 32},
  "image_aug": true,
  "seed": 42,
  "prompt_template": "In: What action should the robot take to {task}?\nOut:",
  "limits": {"max_min": 180, "budget_gb": 18},
  "resume_from": null,
  "idempotency_key": "uuid-from-client"
}
```

`type` values: `smoke | train | validate | merge | infer | crosscheck | gpu_selftest | home_selftest`. `type: crosscheck` swaps the engine for the official `finetune.py` on a 50-episode subset (student-visible curve comparison). `resume_from` carries the checkpoint id for crash recovery (§10).

## Appendix C — repo layout (for the implementing AI coder)

```
roboVLA/
├── docs/                        # this spec + lesson plan + materials + model-card review
├── src/robovla/
│   ├── core/                    # PURE (no I/O; time injected) — unit-tested to ≥95 %
│   │   ├── scheduler.py         # FIFO, run-now, no preemption, slot+VRAM fit, transition table
│   │   ├── budgets.py           # VRAM budget calculator
│   │   ├── spec.py              # JobSpec JSON schema + validator + preset catalog
│   │   ├── config.py            # fleet.yaml schema + loader + diff/reload rules
│   │   ├── qa.py                # dataset QA rules
│   │   ├── norm_stats.py        # min/max stats + un-normalization
│   │   ├── split.py             # seeded held-out split
│   │   ├── ticket.py            # experiment-ticket schema
│   │   └── audit.py             # audit reducer
│   ├── episode.py               # shared episode validator (vendored into capture app)
│   ├── storage.py               # SQLite (WAL): users, sessions, tokens, datasets, jobs, artifacts
│   ├── fs.py                    # datasets/ runs/ artifacts/ layout, zip-slip-safe unzip
│   ├── proc.py                  # subprocess/ssh exec — fixed argv, never shell=True
│   ├── runner/
│   │   ├── interface.py         # Runner protocol: submit/poll/kill/tail/fetch_artifacts
│   │   ├── local.py             # LocalRunner (subprocess)
│   │   ├── ssh.py               # SshRunner (ssh + nohup, rsync)
│   │   └── fake.py              # FakeRunner (scripted metric/event streams; CPU tests)
│   ├── gpu.py                   # VRAM probe, ninfer stop/start, zombie watchdog
│   ├── http/                    # FastAPI app
│   │   ├── app.py               # router wiring, rate limits, error shape (§9.6)
│   │   ├── auth.py              # sessions, tokens, the ONE RBAC dependency
│   │   ├── datasets.py          # §9.2
│   │   ├── runs.py              # §9.3
│   │   ├── stream.py            # §9.4 (SSE)
│   │   ├── infer.py             # §9.5
│   │   ├── admin.py             # §9.7 (all audited)
│   │   └── health.py            # /api/health
│   └── cli.py                   # model_download, gpu_selftest, home_selftest, backup, restore
├── capture/                     # Phase 3: standalone PySide6 capture app (vendors core shared modules)
├── ui/                          # vanilla JS + HTMX (readable by students)
├── tests/
│   ├── unit/                    # core — every PR, < 60 s
│   ├── integration/             # service + FakeRunner — every PR, < 5 min
│   ├── e2e_class/               # scripted 12-student scenario — nightly
│   └── hardware/                # gpu_selftest / home_selftest / robot — GPU_TESTS=1
├── fleet.yaml                   # the fleet (§10 / Appendix A)
├── systemd/robovla-console.service
├── systemd/ninfer.service
└── pyproject.toml               # uv-managed; torch cu128 pins finalized after gpu_selftest (Phase 1)
```

Commands: `uv run pytest -m "not hardware"` (CI) · `GPU_TESTS=1 uv run pytest -m hardware` · `uv run robovla cli gpu_selftest --node pai1`.
