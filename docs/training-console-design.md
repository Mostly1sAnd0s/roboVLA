# roboVLA Training Console — Design Specification (v0.5)

A web-based, OpenVLA-specific training control plane for a class of up to 12 individual students (currently 9), each with their own robot. Hosted on the campus LAN. Drafted from a live inspection of `pai1` on 2026-10-20; v0.3 finalized 2026-10-20 as the implementation-ready spec.

**v0.5 (2026-10-20):** **GPU tenancy is manual** — the console does not manage pre-existing models (the 27B assistant); it shows measured free VRAM + holding processes per node and gates jobs on the VRAM budget only (ninfer stop/start, release windows, and the `gpu_releasing` state are all removed). **Runner = a per-node podman container**: the pinned stack lives in one image per accelerator family (cuda/rocm); the console talks to a small agent API over LAN HTTP — no ssh, no per-node venvs, no rsync (the agent pulls the dataset bundle). **Robots talk to the console only** (`/api/robot/predict`; the console verifies student token + slot lease and proxies to the node agent) — a stable endpoint even when slots move nodes. Other decisions: experiment tickets are a standalone resource created before run submission · `max_seq_len` fixed at 287 (official OpenVLA context; consistent with the model card's 256-bin × 7-token action space) · node `priority` = first-fit order (higher first) · held-out split seeded per student+task (stable across versions) · `smoke` is a verification tool, not a submission gate · all ports configurable with defaults (console 8501, agent 8601) · the home 8060S is not a cluster node (roboVLA is LAN-only) — a self-contained dev lab.

**v0.4 (2026-10-20):** Launch fleet = **pai1 only** (single trainer, 2 inference slots); the 5090 Ti / 4070 Ti / home 8060S are verified inventory and disabled config entries that join later purely via the admin node page — no code, no redeploy. **Node selection at submit**: student picks (teacher-gated `student_selectable`), teacher assigns/pins, or auto first-fit.

**v0.3 (2026-10-20) — implementation-ready.** Added: §8 Admin controls (complete taxonomy; every admin action is a gated REST endpoint) · §9 Full API schema (REST + SSE + security model) · §10 Portability & scalability (fleet-as-config, runner abstraction, deployment, scaling path, operational durability) · §11 TDD & DRY (pure core, single JobSpec contract, binding DRY rules, test matrix) · Appendices A–C (default `fleet.yaml`, JobSpec JSON, repo layout for the implementing coder).

**v0.2 (2026-10-20):** LAN-only access (no public tunnel) · multi-runner fleet (pai1 + 5090 Ti + 4070 Ti + home dev box) · "run now" = no preemption · 12 robots, 1 per student · standalone desktop capture app · demo-day slot relay · GTX 1070 / Strix Halo assessment.

## 1. The fleet (verified 2026-10-20)

| Node | GPU (VRAM) | Role | Training | Inference slots | Notes |
|---|---|---|---|---|---|
| **pai1** (primary console host) | RTX 5090, 32 GB | Primary trainer, 2 infer slots, hosts the assistant 27B | QLoRA batch 4–8 (bf16 also fits) | 2× 8-bit (~10 GB each) | Verified: driver 615.71/CUDA 13.4; the 27B course assistant (`ninfer-serve`, NVFP4) holds 29.5 GB — **not managed by the console**; teacher stops/starts it via `~/start-ninfer.sh` (plain process, no systemd/docker → clean stop/start verified) and the board shows free VRAM while it's loaded (§3); 44 GB RAM free, 1.7 TB disk; system Python 3.14 → uv venv @ 3.12. **Blackwell sm_120: official OpenVLA pins (torch 2.2/cu121, flash-attn 2.5.5) won't work** → re-pin in the runner image: torch 2.8/2.9 cu128, bnb ≥ 0.46, SDPA attention, transformers pinned after selftest. |
| **Campus node 2** (5090 Ti, on hand) | 32 GB | Second trainer, 2 infer slots | QLoRA batch 4–8 | 2× 8-bit | Same stack as pai1. |
| **Campus node 3** (4070 Ti, on hand) | 16 GB | Overflow trainer, 1 infer slot | QLoRA **8-bit, batch 1 only** (~2× slower); bf16 training doesn't fit (15 GB weights) | 1× 8-bit | VRAM budget guard is mandatory on this node. |
| **Home dev box** (Strix Halo, 128 GB unified) | 8060S iGPU (~7.4 TF FP32-class, ~15 TF bf16, 256 GB/s) | **Off-campus dev lab — not a cluster node** (roboVLA is LAN-only) | **bf16 LoRA, no quantization** — 20–25 GB total fits the 128 GB pool, which sidesteps bnb-on-ROCm entirely; ~3–4 h/epoch → overnight runs | bf16 inference testing (~1–3 s/action) | ROCm gfx1151 (RDNA 3.5); `HSA_OVERRIDE_GFX_VERSION=11.0.0` fallback; runs the same runner image **self-contained on the home LAN** (console + agent + model, never exposed); verify with `home_selftest` (30-min checklist). |
| ~~GTX 1070 (8 GB) on home box~~ | — | **Excluded** | — | — | 8 GB < floor for 7.5B even 4-bit-QLoRA (borderline OOM); no tensor cores; Pascal sm_61 deprecated (CUDA 12.8) / removed (CUDA 13). |

**Launch fleet (current): pai1 only** — one trainer, 2 inference slots. Rows 2–3 (5090 Ti, 4070 Ti) are verified inventory and planned fleet entries: `enabled: false` in `fleet.yaml` until they come online, and they join purely via the admin node registry (§8.2: install agent → add → probe → enable) — no code change, no redeploy. The home 8060S is not a fleet entry (off-LAN); it is a self-contained dev lab. With a single trainer at launch, the teacher batches full runs to the morning + overnight windows (lesson-plan rhythm); the console itself gates only on measured free VRAM (§3).

**Campus inference concurrency: 2 slots at launch** (pai1) → 5 slots once node 2/3 join → see §7.7 for the 12-robot demo relay.

## 2. What the system is

One console per class (FastAPI + SQLite + SSE; production instance on pai1) that is the single control plane for the campus GPUs; each GPU node runs a standard **runner container** (podman) hosting the training/serving stack (§4, §10). 12 student accounts, LAN-only access (students use classroom devices; no home/remote access required — per teacher decision). Students:

1. **Upload training data** (episode folders from their robot) or use the **standalone capture app** (§5.3) to collect; see a per-episode QA report before any queue time is spent;
2. **Configure a QLoRA fine-tune** (OpenVLA-specific form, safe defaults — the fragile stack is never exposed);
3. **Submit to a queue** and watch **live progress** (loss/accuracy sparklines, log tail, VRAM, ETA, queue position);
4. Get **automatic held-out validation** per checkpoint, **artifacts** (adapter, merged export), and a **class board** comparing all 12;
5. **Claim an inference slot** — the console serves their fine-tuned model to their robot's Pi (the console *is* the model-server endpoint; the node's runner container hosts the weights — no laptops-as-servers).

## 3. GPU tenancy & scheduling

**Tenancy is manual.** The console does not load, stop, or unload pre-existing models (e.g., the 27B assistant on pai1, ~29.5 GB). It shows **measured free VRAM per node and the processes holding it** (admin panel + dashboard, from the node agent's probe); the teacher unloads non-roboVLA models before training/inference windows. Tenancy reference (pai1, assistant unloaded): QLoRA 7.5B ~12–18 GB (1–2 concurrent on 32 GB nodes with budget guard) · 8-bit inference ~9–11 GB (2 on 32 GB, 1 on 16 GB).

**Scheduling rules (per teacher decision):**
- **No preemption.** "Run now" (teacher or student) is admitted to the front of the line *after* whatever is currently running on the selected node finishes. The scheduler picks the first node whose free VRAM fits the job; if all nodes are busy, "run now" waits for the earliest completion (the student sees "waiting for a slot, ETA X").
- **First-fit with priority:** first-fit order is **`priority` (higher first; ties → `fleet.yaml` order)** across enabled nodes, and a job is admitted only where its **VRAM budget fits for the job's chosen config** (the student's batch determines the estimate; §6 presets).
- **VRAM budget guard:** a job starts only if measured free VRAM ≥ its budget (per-node, per-job-type presets). A 2nd concurrent train slot on a 32 GB node is allowed only with the reduced-batch preset + guard (default: 1 train slot/node).

**Admin levers (teacher panel):** per node — enable/disable, image tag, train slot count (1–2), allowed batch (1 / 4 / 8 with matching grad-accum), VRAM budget, priority, agent endpoint/token; global — job caps, per-student limits (1 running — an active slot counts — / 2 queued), queue position override (bump/kill). Full taxonomy in §8.

## 4. Architecture

```
 12 student browsers (classroom LAN devices; HTTP, self-signed TLS optional)
        │
        ▼
 robovla console — FastAPI + SQLite + SSE, bound to campus LAN IP :8501 (pai1 = production)
  ├─ /api             REST: auth, datasets, runs, tickets, artifacts, slots, admin (§9)
  ├─ /api/robot       the ONLY robot-facing endpoint: token + lease check → proxy to node agent (§9.5)
  ├─ /stream/{job_id} SSE: metrics JSONL tail, log tail, VRAM poll
  ├─ scheduler        SQLite-backed FIFO; per-node slots; no preemption
  │    └─ runner client (JobSpec JSON → agent HTTP, location-agnostic; Appendix B)
  ├─ gpu manager      per-node free-VRAM view + holding processes (via agent); orphan watchdog
  └─ dataset service  chunked upload → unzip → validate → resize 224² → norm stats
        │  LAN HTTP (per-node agent token; default agent port 8601)
        ▼
 per-node runner container (podman, GPU passthrough, pinned stack image)
  ├─ agent API        :8601 — jobs/serve/vram/health; launches fixed-argv processes (no shell)
  ├─ training jobs    robovla_train.py / official finetune.py (crosscheck) / merge / selftest
  └─ inference serve  8-bit adapter loaded for a claimed slot (proxied via the console)
        │
        ▼
 /home/adam/robovla/{datasets, runs, artifacts, models}   (host bind mount, same layout per node)
        ▲
 12 robot Pis ──(POST /api/robot/predict, student token)──▶ console ──▶ node agent /predict
```

**Why FastAPI + SSE + vanilla JS/HTMX (not Streamlit/Gradio):** 12 concurrent long-lived live dashboards, real per-user auth, no per-interaction rerun model, ~1.5–2k LOC, and students can read the whole codebase. Node hosts need only the runner container (image pull + systemd unit); the console host adds the console venv + its unit. LAN-only: no tunnel, no public exposure; auth + rate limits still apply (students are students).

## 5. Data plane

**5.1 Upload (MVP).** Per-student, versioned datasets (`s07/task-stop-at-target/v2`) on pai1. Chunked zip upload of the episode folders (their Pi pipeline: `frames/`, `actions.json`, `episode.json`); 100 episodes ≈ 0.5–1 GB; 2 GB cap, zip-slip-safe, jpeg/json-only. At dispatch, the node's agent pulls the prepared dataset bundle from the console over LAN HTTP (~1 GB — seconds-to-minutes at GbE; trivial).

**5.2 Server-side dataset prep (CPU, ~minutes):** per-episode validation (frame==action count, 10 Hz timing, idle >2 s flag, sign/jump anomalies), 224×224 resize via the official processor's preprocessing (train/inference augmentation parity), `norm_stats.json` (per-dim min/max over 7 dims), held-out split (15%, **seeded per student + task — stable across versions of the same content; recomputed deterministically if the episode set changes**, so v1→v2 comparisons run on the same held-out episodes and re-uploads cannot reshuffle the holdout). Output: "ready" dataset + QA report page — the gate that stops bad data from eating queue time.

**5.3 Standalone capture app (separate deliverable, built in Phase 3 with students).** A desktop app students run on their own laptops on the classroom LAN — per teacher decision, this is a standalone project, not a console feature:
- **Stack: PySide6 v1** (pure Python, students can read it, pip-installable; single-file via PyInstaller for student laptops). Tauri polish is a v2 option.
- **Features:** robot discovery (static IP list from the class sheet + optional mDNS), live 10 Hz view (reuses their `frame_sender`), keyboard + gamepad teleop (pygame) → Pi (their `twist` path), episode logging to the standard spec, built-in pre-upload QA (sync check, idle flags), one-click upload to the console, "replay check" (open-loop replay of the last episode on the robot).
- Collection is **fully offline** (episodes log to local disk; the app works anywhere, e.g., at home if a robot comes home). **Upload requires the classroom LAN** — datasets created at home are uploaded when the student is back in class (LAN-only console, §9.8). Same episode spec as §5.1, so capture and upload feed one pipeline. **Shared code:** the capture app vendors `episode.py` / `norm_stats.py` / JobSpec presets from the console repo verbatim (§11) so capture-side QA and console QA can never drift.

## 6. Training engine (OpenVLA-specific)

**Decision (unchanged from v0.1): a thin in-house `robovla_train.py` (~400–600 LOC) on the openvla repo's model code + PEFT QLoRA**, emitting structured JSONL metrics (first-class observability → SSE), validation hooks, per-epoch checkpoints, crash resume. The official `finetune.py` is *not* abandoned: a `crosscheck` job runs it on a 50-episode subset and students compare curves (Class 9–10 exercise). The recipe is identical either way: 8-bit, rank 32, α 32, lr 5e-4, image aug on.

**Per-runner presets** (admin-configurable):

| Runner | Quantization | Batch | Notes |
|---|---|---|---|
| pai1 5090 / 5090 Ti (32 GB) | 8-bit | 4–8 | ~45 min–2 h/run |
| 4070 Ti (16 GB) | 8-bit | 1 (+grad accum) | ~2× slower; budget guard |
| Home 8060S (128 GB unified) | **bf16, no quant** | 1–2 | overnight, dev-lab only (off-campus); no bnb dependency (ROCm) |

**`max_seq_len` is fixed at 287** (the official OpenVLA context: 256 visual tokens + prompt/action text — consistent with the model card's 256-bin × 7-token action space) and lives in the preset catalog, not the student form.

**Job types:** `smoke` (1 batch, ~2–5 min — stack/config verification at node join and as an optional student sanity check; **not a submission gate**: stack correctness is covered by `gpu_selftest` + the test matrix, §11) · `train` (cap 3 h) · `validate` (held-out eval, auto after checkpoints) · `merge` (`merge_and_unload()` + 10-image identity check; 15 GB export on demand, 14-day purge) · `infer` (serve adapter, token auth; a claimed slot *is* an infer job, §9.5) · `crosscheck` (official script on a **deterministic 50-episode subset**, seeded per student + dataset version — same episodes every re-run, so students compare their console engine against the official engine on identical data) · `gpu_selftest` / `home_selftest` (stack verification per node: load, fwd/bwd, `predict_action`, timings).

**Stack (pinned in the runner image):** uv venv @ Python 3.12, torch 2.8/2.9 cu128, transformers pinned after selftest, bnb ≥ 0.46, SDPA (flash-attn only if it builds clean for sm_120), timm/pillow/numpy — baked into `robovla-runner-cuda`; the **image tag is pinned per node in `fleet.yaml`** (new pin = new tag + selftest at node join). **Escape hatch:** the runner protocol is location-agnostic — any host (or a rented cloud A100 running the same image, reachable over a private link the teacher sets up) accepts the same JobSpec.

## 7. UI surfaces

1. **Dashboard** — queue position + ETA, my datasets, my runs, per-node GPU state (free VRAM + who/what holds it), class activity feed.
2. **Datasets** — upload wizard → prep job → QA report, versioning, norm stats display, held-out info, "ready" badge.
3. **New run** — dataset (ready only), task prompt template, **training node** (default *auto* = first-fit in priority order, admitting only where the chosen config's budget fits, §3; a specific node only when the teacher has marked it student-selectable, §8.2; the teacher can assign/pin any node via requeue), rank/α/lr/epochs/aug/seed + batch (options filtered by node fit; auto ⇒ options that fit at least one enabled node) in a collapsible *advanced* section (values drawn from the single preset catalog in core; `max_seq_len` fixed at 287, not exposed), **required experiment ticket** (standalone resource created before submit: hypothesis → config → prediction; outcome at completion; §9.3), submit → optional smoke → train.
4. **Run detail (SSE)** — train + held-out val sparklines, step/total/ETA, grad norm, LR, VRAM + util, checkpoint events, raw log (collapsible), kill (teacher).
5. **Class board** — all 12 side-by-side (best held-out accuracy, run counts, dataset stats, tickets); printable; the final-week post-mortem material.
6. **Artifacts** — adapter download, merged export (on demand), run config JSON.
7. **Inference slots** — "serve my adapter" (8-bit), slot queue (claims with no free slot queue like any job, §9.5), live status, **Pi connection info** (the stable console endpoint + per-student token — no per-node robot config, §9.5).
   **7.7 Demo-day relay (12 robots, 2 slots at launch; 5 after node2/3 join):** students are scheduled into ~4-min arena runs; runs pipeline with presentations — student *i* presents their run's result while student *i+1's* robot executes on a slot. 9–12 students pipelined ≈ **55–70 min total** (4-min run overlapped with the 3-min talk + 1-min Q&A; sizing per `setup-and-materials.md`); robot slot demand is only ~4 min/student (≤ 48 robot-minutes), which 2 slots absorb comfortably inside the presentation timeline. The slot board is visible to the class. For practice weeks, the same relay gives every student 30–60 s of autonomy back-to-back — "everyone's robot is always moving."
8. **Teacher panel** — thin forms over the admin API (§9.7): runner registry (enable, image tag, slots, allowed batch, VRAM budget, priority, agent endpoint/token), queue ops, student accounts, disk usage, raw logs. Node host lifecycle (container restart, image update, GPU reset) is shown as copy-paste host commands, not panel actions (§8.2). Complete taxonomy in §8; every panel control is an audited endpoint.

## 8. Admin controls

Complete taxonomy of everything the teacher can do. **All admin actions are exposed as gated REST endpoints (§9.7), rate-limited (default 60 req/min), and written to the append-only audit log** (actor, timestamp, action, target, before→after JSON). The UI (§7.8) is a thin form over these endpoints — there are no admin capabilities that are not in the API.

### 8.1 Accounts

| Control | Behavior |
|---|---|
| Create / import students | single create or CSV roster import (id `s01`–`s12`, display name, email optional); created inactive until activated |
| Activate / deactivate | deactivated: no login, no token use; queued jobs requeue to fitting runners, running jobs complete; no new submissions |
| Roles | `teacher` (everything) · `student` (own data, class board, inference claims) · `observer` (read-only board + GPU state; demo-day visitors) |
| Reset password | out-of-band: teacher sets a one-time code/link (paper), no email dependency in the classroom |
| Per-student limits (defaults) | max running (1 — **an active inference slot counts as running**, it holds the GPU) · max queued (2) · max dataset bytes (2 GB) · max run minutes (180) · run-now allowed (yes — front of line after current completion, §3) |
| Pi tokens | generate / rotate / revoke per student; scoped to that student's datasets, adapters, and inference claims; value shown once, argon2-hashed at rest |
| Sessions | list active sessions per user; revoke any session (immediate) |
| Audit log | all admin actions + auth events; append-only; admin-readable; JSON export |

### 8.2 Nodes / endpoints

The node registry **is** the fleet config (§10, `fleet.yaml`) — the admin UI edits config, never machines. A node = a host running the standard runner container (agent API, §4); the console talks to the agent, not the host.

| Control | Behavior |
|---|---|
| Add node | form: id, label, image tag (`robovla-runner-cuda` \| `-rocm`), LAN IP + agent port (default 8601), agent token (shown once; console stores hash), GPU VRAM, train slots (1–2), infer slots (1–2), quant profile (`bnb8` \| `bf16`), allowed batch (1 \| 4 \| 8 + matching grad-accum), VRAM budget per job type, priority (int, default 0; first-fit order, higher first), student-selectable (default **off**), enabled (default **off**) → **dry-run join probe** (agent `/health`: image tag matches, GPU visible with expected VRAM, disk free ≥ 50 GB, model dir present) → node stays disabled until teacher enables it |
| Edit node | any field; changes apply at the next scheduling tick; active jobs unaffected (a budget shrinking below a running job completes that job; no new admissions on that node until refit) |
| Enable / disable | disable: queued jobs requeue to other fitting runners; running jobs fail-fast with checkpoint-resume hint; inference slots drain with client notification (Pi watchdog stops the robot after lease expiry) |
| Remove node | as disable, then removed from config; its jobs/artifacts remain on disk, flagged |
| Probe | on-demand re-run of the join probe (network changes, after maintenance, after image update) |
| VRAM purge (agent-scoped) | post-job probe lists orphan processes the agent spawned but didn't reap → teacher confirms → agent kills (audit-logged). **Host-level** processes (e.g., the 27B assistant) are shown on the board with their VRAM but are the teacher's to manage (§3) — the console does not touch host processes. |
| Node host lifecycle (host-side, not API) | container start/restart, image pull/update, `nvidia-smi --gpu-reset` (confirmation required, no running jobs) = documented one-liners the teacher runs on the host (`robovla-node` script in the repo); the admin panel shows agent version/health, an "update available" flag, and the exact command. The console cannot restart a remote node by design (it talks to the agent, not the host). |
| Service log | tail the node's agent log (last 500 lines) via `GET /api/admin/nodes/{id}/logs`; the console's own service log is local to its host |
| Endpoint visibility | robots talk to the **console** (one stable endpoint + per-student token, §9.5) — no per-robot node config; each node's LAN IP + agent port is published on the admin page for Pi debugging |
| Student-selectable | teacher toggles per node whether students can name that node at run submission (§9.3); off ⇒ auto first-fit or teacher-assign only; `enabled: false` nodes are hidden from students entirely |

### 8.3 Jobs & queue

| Control | Behavior |
|---|---|
| Queue view (all roles) | all nodes, FIFO, position, ETA (queue wait + estimated run time), node fit; students read-only |
| Bump (teacher) | move a job to the front of the queue on the nodes it fits — effective at the next completion boundary (no preemption, §3) |
| Kill | students: own queued jobs; teacher: any (queued → `killed`; running → SIGTERM to the job process via the agent, 10 s grace, SIGKILL → state `killed`, VRAM probe, checkpoint artifacts preserved) |
| Reschedule (teacher) | pin a queued job to a specific node, or clear a pin |
| Manual completion (teacher) | mark a job `done` for work done outside the console (e.g., teacher ran `finetune.py` by hand); note required; inference only enabled if an artifact manifest exists |
| Retry from checkpoint (teacher) | resubmit a failed/killed training job from its last checkpoint (dataset bundle re-pulled if the node differs) |
| Annotate (teacher) | free-text note on a job, visible on the class board |
| Caps | global max concurrent (default = sum of node train slots) · per-node (slots) · per-student (§8.1); scheduler enforces before admission |
### 8.4 Datasets & artifacts

| Control | Behavior |
|---|---|
| Dataset inventory | per student/version: episode count, size, QA status, which runs used it |
| Reject (teacher) | block a dataset from new runs (reason shown on the student's page); un-reject after a corrected re-upload |
| Purge (teacher) | two-step confirmation; deletes episodes + QA + stats; runs that used it keep their checkpoints/adapters (run config is self-contained) |
| Retention | default: keep all student data; optional per-student max-versions (LRU purge *proposal* list — teacher approves) · global disk cap per node with 80 %/90 % alerts |
| Artifact inventory | per run: adapter (always) · merged export (on demand, 15 GB, 14-day auto-purge — teacher-adjustable) · run config JSON |
| Force purge / export | teacher may purge early; **backup bundle** zips datasets + adapters (node-host artifacts collected via the agent) + DB snapshot for off-site archive |

### 8.5 System

| Control | Behavior |
|---|---|
| Health | `GET /api/health` (LAN-public): service uptime, per-node state (online, GPU free/busy, free VRAM, queue depth) — consumable by a classroom display |
| Logs | console service log (per node) · per-job training logs (run detail) · audit log (§8.1) |
| Backup | nightly (teacher-adjustable time) of DB + datasets + adapters to a configurable destination (local dir or NAS mount); status view (last run, size, checksums); restore = documented procedure, covered by the test matrix |
| Config | view effective config (rendered YAML + per-node profiles); **hot reload**: validate → apply; active jobs unaffected, queued jobs revalidated against new budgets |
| Maintenance mode | console read-only: no new submissions/uploads; running jobs and live slots continue; for stack upgrades |
| VRAM watchdog | post-job probe (via agent) compares running processes to expected; agent-scoped orphans → alert banner + one-click purge (§8.2); host-level VRAM holders are shown on the board (teacher-managed, §3) |

## 9. API schema

Base URL: `http://<console-lan-ip>:8501` (console — the only endpoint robots and students talk to); node agents on their LAN port (default 8601, console-only). All ports configurable per node in `fleet.yaml`. Conventions: JSON bodies; errors `{"error":{"code","message"}}` (§9.6); list endpoints take `?cursor=&limit=` (default 50); `Idempotency-Key` header on `POST /api/runs` (client UUID, 24 h dedup → returns the existing job); artifacts are ETag-cached; timestamps RFC 3339 UTC.

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

### 9.3 Runs & tickets
- **Tickets (standalone resource — created before submission):**
  - `POST /api/tickets` `{hypothesis, config_summary, prediction}` → `{ticket_id}` (student-owned, editable until linked)
  - `GET /api/tickets/{id}` · `PUT /api/tickets/{id}`
  - `GET /api/runs/{job_id}/ticket` — linked ticket + outcome field (completed at run end)
- `POST /api/runs` `{dataset_id, job_type: smoke|train|validate|merge|infer|crosscheck, ticket_id (required for train/smoke: must be own, not yet linked), node_id?: "auto"|<node-id>, config: {rank, alpha, lr, epochs, batch, grad_accum, aug, seed, prompt_template}, run_now?: bool}` → `{job_id, position, eta_s}`
  - Node selection: `"auto"` (default) ⇒ first-fit across enabled nodes in **priority order (higher first; ties → `fleet.yaml` order)**, admitting only where the **chosen config's VRAM budget fits**; a specific node is accepted for students only if enabled **and** `student_selectable: true` (§8.2) — teacher may name any enabled node; pinned jobs wait on that node only
  - Validation: dataset `ready`; ticket present + unlinked; per-student caps — if no node fits now, response carries `waiting: "slot"` + earliest ETA (reason: busy vs budget)
- `GET /api/runs` (own) · `GET /api/runs/{job_id}` (state, config, node, progress, artifacts)
- `GET /api/runs/queue` (all nodes, all students; position + ETA; feeds the class board)
- `POST /api/runs/{job_id}/kill` (own queued; teacher: any)
- `POST /api/runs/{job_id}/requeue` `{node_id?}` (teacher) · `POST /api/runs/{job_id}/bump` (teacher)
- `POST /api/runs/{job_id}/complete` `{note}` (teacher)
- `GET /api/runs/{job_id}/logs?offset=` (tail, via agent)
- `GET /api/runs/{job_id}/artifacts` (adapter path, merged status, run config JSON)

### 9.4 SSE streams
- `GET /stream/{job_id}` — events: `state` (`queued|running|finishing|done|failed|killed` + reason) · `metrics` (JSONL: `step, total, loss, lr, grad_norm, vram_mb, vram_util_pct, eta_s`) · `val` (held-out accuracy per checkpoint) · `log` (tail lines) · `artifacts` (checkpoint/adapter/merged events). Heartbeat every 15 s; `Last-Event-ID` resume.
- `GET /stream/class` — class-board feed (queue, node state, slot board, activity), ≤1 msg/s, all roles.

### 9.5 Inference (robot path)
- `POST /api/infer/slots/claim` `{run_id}` (adapter must exist) → creates an **infer job** (first-fit in priority order: infer slot free + free VRAM ≥ infer budget + `student_selectable`/pin rules as §9.3). Response: slot available → 200 `{slot, node_id, endpoint: "http://<console-lan-ip>:8501/api/robot/predict", lease_s}`; none available → 202 `{queued: true, position, eta_s}` (the claim queues like any job; no preemption).
- `GET /api/infer/slots` (all: holder, adapter, node, load)
- `POST /api/infer/slots/{slot}/release` (= kill the infer job; agent unloads the adapter, VRAM probe)
- **Robot endpoint (console port 8501, `Authorization: Bearer <student token>`):**
  `POST /api/robot/predict` `{image: base64 jpg, prompt}` → console verifies token + active slot lease, proxies to the holding node's agent `/predict`, returns `{action_norm[7], action_robot[7] (un-normalized with the dataset's min/max stats), latency_ms}`
  - Robots talk **only to the console** — one stable endpoint for the life of the course, no per-node robot config, and a loopback hop at launch (slots live on pai1). Node agent ports are console-only.
  - Rate limit ~5 req/s per student; slot lease TTL (default 300 s idle → slot auto-released; the Pi watchdog already stops the robot on stale actions); target p95 < 400 ms incl. LAN on 8-bit 32 GB nodes (~150–300 ms inference per the latency rebase in `model-card-review.md`).

### 9.6 Error codes (stable — Pi client and tests depend on them)
`auth` · `forbidden` · `not_found` · `conflict` · `dataset_not_ready` · `budget_unfit` · `limit_exceeded` · `rate_limited` · `node_offline` · `internal`

### 9.7 Admin API (role = teacher · audit-logged · 60 req/min) — the §8 mapping
- **Users:** `GET /api/admin/users` · `POST /api/admin/users` · `POST /api/admin/users/import` (CSV) · `PATCH /api/admin/users/{u}` (active, role, limits) · `POST /api/admin/users/{u}/reset-password` · `GET /api/admin/users/{u}/sessions` · `DELETE /api/admin/users/{u}/sessions/{sid}` · `POST /api/admin/users/{u}/tokens/{action}`
- **Nodes:** `GET /api/admin/nodes` · `POST /api/admin/nodes` · `PATCH /api/admin/nodes/{id}` · `DELETE /api/admin/nodes/{id}` · `POST /api/admin/nodes/{id}/probe` · `POST /api/admin/nodes/{id}/purge-vram` (agent-scoped) · `GET /api/admin/nodes/{id}/logs`
  - Node host lifecycle (container restart, image update, GPU reset) is deliberately **not** in the API — host-side teacher one-liners (§8.2, §10).
- **Jobs:** `GET /api/admin/jobs` · `POST /api/admin/jobs/{id}/bump` · `POST /api/admin/jobs/{id}/kill` · `POST /api/admin/jobs/{id}/requeue` · `POST /api/admin/jobs/{id}/complete` · `POST /api/admin/jobs/{id}/note`
- **Datasets / artifacts:** `DELETE /api/admin/datasets/{id}` · `POST /api/admin/datasets/{id}/reject` · `GET /api/admin/artifacts` · `DELETE /api/admin/artifacts/{id}` · `POST /api/admin/backup-bundle`
- **System:** `GET /api/health` (public) · `GET /api/admin/audit?since=` · `GET /api/admin/config` · `POST /api/admin/config/reload` · `POST /api/admin/maintenance` `{on|off}` · `POST /api/admin/backups/trigger` · `GET /api/admin/backups`

### 9.8 Security model
- Accounts: 12 students + teacher (optional observers); argon2id; httpOnly session cookies; passwords never logged.
- RBAC: student → own datasets/runs/tokens, read queue + class board + slots, claim/release own slot · teacher → everything (§9.7) · observer → board + node state only. Enforced by **one** FastAPI dependency — never per-route.
- Tokens: per-student Pi tokens, scoped to their datasets/adapters/slots, shown once, hashed at rest, revocable.
- Agent tokens: per-node pre-shared (console stores a hash; the agent holds it in its env); all console↔agent traffic authenticated; rotation = admin action + host-side restart.
- Network: console binds the campus LAN IP; node agents bind their LAN port (console-only traffic); no public exposure, no tunnel; optional self-signed TLS via reverse proxy; students use classroom devices; **robots talk to the console only**.
- Uploads: 2 GB cap, chunked, zip-slip-safe, path-sanitized, jpeg/json only.
- **No shell, ever:** validated JobSpec JSON → fixed argv (inside the agent).
- Rate limits: login 5/min/IP · uploads 2 concurrent/student · submissions 10/min/student · admin 60/min.
- Audit: all §8 actions + auth events.
- (Flagged, out of scope: `infergate` :8317 and `flygym` :8765 on pai1 bind 0.0.0.0 → recommend localhost-only.)

## 10. Portability & scalability

**Core principle: hardware lives in config, not code; the stack lives in the image, not the machine.** The console has no knowledge of "pai1" or "5090" as machines — a node is an entry in `fleet.yaml` with a hardware profile, and a node's software is a **standard runner container** (one image per accelerator family). Moving the class to a new room, a new school, or a cloud box = a new `fleet.yaml`, same images, same code. (Appendix A: full default config.)

- **Fleet = one YAML file** (`fleet.yaml`, validated against one schema in core): console block (bind, port, TLS, session TTL, limits, retention) + `nodes[]` (id, label, `image` tag, `workdir` host bind mount, `agent {lan_ip, port, token_ref}`, gpu `{vram_gb, quant: bnb8|bf16, train_slots, infer_slots, budgets: {train_gb, infer_gb}}`, `presets {batch: [...]}`, `priority`, `student_selectable`, `enabled`) + per-student limit defaults. The shipped default describes the launch fleet (pai1 enabled; node2/node3 as disabled inventory entries); a new deployment ships with a minimal one-node config.
- **Runner = a per-node podman container** — the only hardware boundary. The image (`robovla-runner-cuda` / `robovla-runner-rocm`) bakes in the pinned stack (§6); the container runs a small **agent API** on its LAN port (default 8601) and launches job processes inside the container with a **fixed argv** (no shell, ever). The console's runner client is a **single implementation** (`HttpRunner`) — the local/ssh distinction is gone: pai1 runs its own container locally, so every node, including the console host, is uniform. Datasets cross as a tar bundle the agent pulls from the console over LAN HTTP at dispatch. Why containers: the Blackwell/ROCm stack-pinning and per-node version-drift risks move into one image build (tag pinned per node; the join probe verifies the tag); adding machines = image pull + agent start + one config line + probe; zombie VRAM is structurally mitigated (the agent reaps its job processes; stopping the container frees the GPU). Ephemeral per-job containers are a v2 option.
- **Node host lifecycle stays host-side.** The console talks to the agent, not the host — it commands jobs, models, and probes; container start/restart, image pull/update, and `nvidia-smi --gpu-reset` are documented teacher one-liners on the host (the `robovla-node` script in the repo). The admin panel shows agent version/health, an "update available" flag, and the exact command.
- **Per-node hardware profiles** drive all presets (VRAM budgets, quantization, batch, slots) — a new 24 GB node joins as an image + config line + probe; nothing is recompiled or re-pinned globally.
- **Deploy anywhere:** Ubuntu 22.04/24.04 + any supported accelerator (NVIDIA CUDA or ROCm). Console = bare `uv venv && uv sync` + shipped systemd unit (`robovla-console.service`) on its host (no GPU needed); runner = podman container under a shipped systemd unit (quadlet, `robovla-runner.service`) with GPU passthrough. Provisioning jobs (`model_download` into the model volume, `gpu_selftest`) run at node join, via the agent. The console itself is ~1.5–2k LOC; the agent is a few hundred.
- **State is portable:** the console's SQLite (WAL) + data dir (`datasets/` + DB) and each node host's data dir (`runs/ artifacts/ models/` under `workdir`) = the entire system state. Rebuild or move = clone repo + restore backup; nightly backup (§8.5) covers the rest (node-host artifacts are collected via the agent).

**Scaling path:**

| Dimension | What changes |
|---|---|
| More GPUs | image + agent installed on the host (one-liner) + one config entry + join probe. Scheduler, UI, tests unchanged. |
| More students | accounts are rows; per-student caps + morning/overnight windows (launch, single trainer) and added nodes (scaling) keep the queue sane; board scales. |
| Multi-room / multi-school | same console image + different `fleet.yaml`; optional class-code tenancy if two classes share one console. |
| Bigger / newer VLA models (OFT, FAST, PRIME) | job spec carries model id + processor; same upload/QA/queue/SSE pipeline; new model = new image tag. |
| Rented cloud box (escape hatch) | same image on a rented host, reachable over a private link the teacher sets up (VPN/tunnel) — the protocol is plain LAN-style HTTP, so nothing else changes. |
| Console SPOF | rebuild from repo + restore backup < 1 h; episode data also lives on the 12 PIs. |

**Operational durability (carried from v0.2 §9):**
- systemd: `robovla-console.service` on the console host; `robovla-runner.service` (podman quadlet) on each node host.
- Job state machine: `queued → running → finishing → done|failed|killed` — a pure transition table in core, unit-tested.
- Runner failure: dispatch failure or mid-run node death (agent unreachable) → job fails fast, auto-reschedules on the next fitting runner (never preempts a *running* job), student notified with ETA; crash resume from checkpoints (dataset bundle re-pulled to the new node).
- Post-job VRAM probe (via agent) → orphan alert + agent-scoped purge. Retention: 20 run logs/student; merged exports 14 days; nightly backup of datasets + adapters + DB.

## 11. TDD & DRY

**Process (binding for all implementation work):**
1. Every feature starts with a **failing test** (red) → minimal code (green) → refactor. No module ships without tests.
2. GitHub CI on push: the full CPU suite (< 5 min). Hardware-gated tests (`gpu_selftest`, `home_selftest`, robot) run manually / at node join behind `GPU_TESTS=1`.
3. **Definition of done:** CI green · no new TODOs · §9 updated if the API surface changed · every new admin action has audit coverage + a test.

**Architecture for testability:**
- **Pure core** (`robovla/core/`; no I/O, no network, no wall clock — time is injected): scheduler (FIFO, run-now = front-of-line after current completion, no preemption, per-node slot + VRAM-budget fit, kill/reschedule state machine) · VRAM budget calculator · JobSpec validation (one JSON schema) · dataset QA rules · norm stats + un-normalization · held-out split (seeded, deterministic) · experiment-ticket schema · audit reducer · config schema. Target: ≥ 95 % line coverage; suite < 60 s on CPU.
- **I/O only at seams:** `storage` (SQLite) · `fs` (datasets/artifacts) · `runner` (agent HTTP client) · `http` (FastAPI app) · `clock`. Tests use fakes at every seam — `FakeRunner` speaks the agent protocol with scripted metric/event streams so the **entire SSE path runs on CPU**.
- **One contract:** JobSpec JSON (Appendix B) — the only artifact shared by UI, scheduler, runners, docs, and the crosscheck job. One schema, one validator, one preset catalog (the QLoRA default recipe is defined once: 8-bit, rank 32, α 32, lr 5e-4, aug on, max_seq_len 287; the "advanced" form reads from that catalog).
- **One config:** `fleet.yaml` — one schema, one loader, hot reload revalidates. No machine-specific string in code.
- **One episode validator:** `episode.py` (format, frame/action counts, 10 Hz timing, idle > 2 s, sign/jump checks) + `norm_stats.py` + the preset catalog are **shared code vendored verbatim into the standalone capture app and the runner agent** — capture-side QA, console QA, and agent-side QA cannot drift.

**DRY checklist (rules for the implementing coder):**
- No inline SQL outside `storage.py` · no JSON schema defined in two places · one error type (§9.6) · one logging setup · auth/RBAC through the single FastAPI dependency (never per-route) · no node names/IPs in code (all in config) · capture app and agent share `core`/`fs`/`episode`/`proc` code with the console verbatim (vendored at image build) · **no shell, ever** (validated JobSpec JSON → fixed argv, inside the agent).

**Test matrix:**

| Layer | Coverage | Gate |
|---|---|---|
| Unit — core | scheduler ordering (incl. run-now, no-preemption, priority + config-aware first-fit: student batch vs node budget, 16 GB node → batch-1 preset, 2-slot admission on 32 GB) · kill/reschedule transition table · ticket-before-run (400 without ticket, 409 if already linked) · QA rules (frame/action mismatch, 10 Hz timing, idle > 2 s, sign flip, jump anomaly) · norm stats + un-normalization round-trip · held-out split determinism (stable across versions of unchanged content) · JobSpec + config schema validation | every PR · CPU · < 60 s |
| Integration — service (CPU, FakeRunner) | chunked upload (incl. zip-slip payload rejected) → prep → QA gate → ticket → queue admission → SSE (state/metrics/val events, heartbeat, resume) → artifact manifest → robot `predict` with mocked agent + token auth · RBAC (student 403 on others' runs and all `/api/admin`) · idempotency dedup · rate limits · slot claim 202-queue when full · dataset bundle pull (fake agent) · audit-log entry for every admin action | every PR · CPU · < 5 min |
| Simulated class E2E (CPU) | scripted 12-student scenario: 12 uploads → 12 tickets → 12 submissions (launch config = pai1 only; extended config = +node2/node3 to exercise multi-endpoint first-fit by priority, node pinning, and student-selectable gating) → queue drains in expected order → kill/bump/requeue mid-drain → demo-day 2-slot relay schedule (§7.7) → backup + restore round-trip | nightly + per release |
| Hardware-gated (GPU) | `gpu_selftest` per node (via the agent, inside the container: torch/CUDA visible, load 7.5 B, 8-bit fwd/bwd, bf16 fwd/bwd where it fits, `predict_action` on held-out images, 10-step QLoRA) · `home_selftest` (self-contained dev lab, ROCm gfx1151: `torch.cuda.is_available`, bf16 GEMM benchmark, load + predict, 10-step bf16 fwd/bwd — no bnb) | manual · node join + stack changes |
| Robot (Pi) | token auth · LAN predict round-trip (via console) · stale-action watchdog stop · slot lease expiry | per class milestone (M5–M7) |

## 12. Build plan

| Phase | When | Who | Scope | Exit |
|---|---|---|---|---|
| **0** | Now → ~1 wk | instructor + AI coder (assistant) | Home-box ROCm verify (`home_selftest`, self-contained dev lab); **build runner image (cuda) + agent**; provisioning (model download); core (scheduler/budgets/QA/spec/config) test-first per §11; console scaffold (auth, users, nodes, upload + prep/QA, tickets, queue, SSE, UI, LAN bind) against local agent + FakeRunner; admin surface §9.7; systemd units; 12 accounts | 12 students log in on the campus LAN, upload, watch a simulated run; teacher panel works end-to-end |
| **1** | First GPU window | AI coder; teacher triggers | `gpu_selftest` on pai1 (via the agent, in-container), Blackwell stack pin in the image, `smoke`, 50-episode `crosscheck` | real training verified (class gate M5) |
| **2** | By Class 9 | AI coder | `robovla_train.py` integration, validation harness, merge/export, inference slots + Pi token auth, class board, teacher panel complete; 5090 Ti + 4070 Ti join: agent installed on each host + admin add → probe → enable when they come online (deferred — not needed at launch) | console production-ready for the class |
| **3** | Extension week (Class 17) | **students** + AI coder | standalone capture app (PySide6; vendors `episode.py`/`norm_stats.py`), Foxglove live-feed panel, QA-flag + board-view contributions | students own part of the instrument |

**Pedagogical integration:** the console *replaces* the "build the TUI" class activity. Classes 9–10: the console is live → students operate it, read `robovla_train.py` (12 sections, one per student), run the crosscheck, own the experiment tickets. With 12 individual students and 1 robot each, the console is the shared lab instrument — like the eval harness.

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Blackwell stack pinning (torch/tf/bnb/flash-attn) | Pinned in the runner image (tag in `fleet.yaml`); early selftest at node join; SDPA fallback; escape hatch = any host running the same image (§6, §10) |
| Runner heterogeneity (32 GB / 16 GB / iGPU) | Per-node batch presets + VRAM budget guard; per-node selftest; 4070 Ti batch-1-only |
| Image drift across nodes | Image tag pinned per node in `fleet.yaml`; join probe verifies tag + selftest pass; updates are explicit host-side ops with "update available" flag (§10) |
| Home box ROCm (gfx1151) friction | `home_selftest` first (self-contained dev lab); bf16-only (no bnb); fallback: CPU torch for engine verification; class never depends on it (off-LAN, not a cluster node) |
| No-preemption + 12 students → queue pressure (single trainer at launch) | morning + overnight windows carry full runs (teacher batches; console gates on free VRAM, §3); fast free smoke runs in class; run-now = front-of-line after current completion; pressure relieved by adding nodes via the admin page (config-only, §8.2) |
| Demo-day 12-robot concurrency | 5 inference slots + pipelined relay (§7.7) — all robots cycling continuously |
| Agent dispatch failure / node death | Fail-fast, auto-reschedule on next runner, checkpoint resume (§10) |
| Zombie VRAM | Agent reaps its job processes; container stop frees the GPU; post-job probe + agent-scoped purge (§8.2) |
| Single-host (console) SPOF | Rebuild from repo + restore backup < 1 h; episode data also on the 12 PIs |
| 12 robots in one room (practice weeks) | 4 sub-arenas × 3 students, zone rotation (see lesson plan) |
| Config drift (fleet.yaml vs. reality) | Single schema + CI validation + join probes + hot reload revalidates queued jobs |
| AI coder implementing from this doc | §11 TDD contract + test matrix are the acceptance criteria; every admin action = endpoint + audit + test |

## 14. Open questions

1. **Wiring (later — not needed for launch):** when the 5090 Ti and 4070 Ti machines come online, are they on the campus LAN (wired 1 GbE preferred)? Each needs a LAN IP for its agent endpoint (port 8601), then joins via the admin node registry (§8.2).
2. **Demo day date + audience** (sizing the relay schedule; judges/audience seating in the sub-arenas).
3. **Student laptop inputs:** keyboards only, or do some students have USB gamepads (capture app input design)?

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
      max_running: 1       # an active inference slot counts as running
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
    image: robovla-runner-cuda:2026.10
    workdir: /home/adam/robovla        # host bind mount → container's standard data layout
    agent: {lan_ip: "<PAI1_LAN_IP>", port: 8601, token_ref: pai1-agent}
    gpu: {vram_gb: 32, quant: bnb8, train_slots: 1, infer_slots: 2,
          budgets: {train_gb: 18, infer_gb: 11}}
    presets: {batch: [1, 4, 8]}        # allowed student batch options (grad-accum auto-matched)
    priority: 0
    student_selectable: true           # launch node; the only node students can name at submit
    enabled: true

  - id: node2
    label: "campus RTX 5090 Ti"
    image: robovla-runner-cuda:2026.10
    workdir: /home/adam/robovla
    agent: {lan_ip: "<NODE2_LAN_IP>", port: 8601, token_ref: node2-agent}
    gpu: {vram_gb: 32, quant: bnb8, train_slots: 1, infer_slots: 2,
          budgets: {train_gb: 18, infer_gb: 11}}
    presets: {batch: [1, 4, 8]}
    priority: 0
    student_selectable: false
    enabled: false          # until join probe passes

  - id: node3
    label: "campus RTX 4070 Ti (batch-1 only)"
    image: robovla-runner-cuda:2026.10
    workdir: /home/adam/robovla
    agent: {lan_ip: "<NODE3_LAN_IP>", port: 8601, token_ref: node3-agent}
    gpu: {vram_gb: 16, quant: bnb8, train_slots: 1, infer_slots: 1,
          budgets: {train_gb: 14, infer_gb: 11}}
    presets: {batch: [1]}       # grad-accum auto-matched to the recipe's effective batch
    priority: 0
    student_selectable: false
    enabled: false          # until join probe passes

# Home Strix Halo (8060S, ROCm): off-campus → not a cluster node (roboVLA is LAN-only).
# Runs the same robovla-runner-rocm image as a self-contained dev lab on the home LAN
# (console + agent + model, never exposed) for Phase 0 integration / home_selftest.
```

Notes: **pai1 is the only enabled node at launch**; node2/node3 are verified inventory, disabled until they come online — adding them later is an admin-page operation (install agent → add node → probe → enable, §8.2), no code or redeploy. The home 8060S is intentionally absent: off-LAN, not a cluster node — it runs the same image as a self-contained dev lab (§1). The 27B assistant on pai1 (~29.5 GB) is **not** console-managed (§3): the teacher unloads it before training/inference windows; the board shows free VRAM. `train_slots: 1` everywhere is the §3 default; the teacher raises a 32 GB node to 2 via the admin lever (reduced-batch preset + budget guard apply). `priority` = first-fit order (higher first; ties → file order). `student_selectable` (default false) controls whether students can name a node at submission (§9.3). `token_ref` = per-node agent token (shown once at node add; console stores a hash, the agent holds it in env). Placeholder IPs are filled in at wiring time (§14 Q1) — code never hardcodes them.

## Appendix B — JobSpec JSON (the single contract)

The only artifact shared by UI, scheduler, runners, docs, and the crosscheck job. Validated by one schema in core; the agent derives a **fixed argv** from it (no shell).

```json
{
  "job_id": "j_20261020_0007",
  "student": "s07",
  "type": "train",
  "model": "openvla/openvla-7b",
  "dataset": "s07/task-stop-at-target/v2",
  "node": {"id": "pai1", "agent": "<PAI1_LAN_IP>:8601"},
  "quant": "bnb8",
  "lora": {"rank": 32, "alpha": 32},
  "optim": {"lr": 0.0005, "epochs": 3, "batch": 4, "grad_accum": 1, "max_seq_len": 287},
  "image_aug": true,
  "seed": 42,
  "prompt_template": "In: What action should the robot take to {task}?\nOut:",
  "limits": {"max_min": 180, "budget_gb": 18},
  "resume_from": null,
  "idempotency_key": "uuid-from-client"
}
```

`type` values: `smoke | train | validate | merge | infer | crosscheck | gpu_selftest | home_selftest`. `type: crosscheck` swaps the engine for the official `finetune.py` on a 50-episode subset (student-visible curve comparison). `max_seq_len` is fixed at 287 in the preset catalog (official OpenVLA context: 256 visual tokens + prompt/action text — consistent with the model card's 256-bin × 7-token action space); not exposed to students. `resume_from` carries the checkpoint id for crash recovery (§10).

## Appendix C — repo layout (for the implementing AI coder)

```
roboVLA/
├── docs/                        # this spec + lesson plan + materials + model-card review
├── agent/                       # runner agent — runs inside the per-node podman container
│   ├── server.py                # HTTP API :8601 — jobs/serve/vram/health; launches fixed-argv processes
│   ├── vram.py                  # nvidia-smi / rocm-smi probe (free VRAM + holding processes)
│   └── pull.py                  # dataset bundle pull from the console (LAN HTTP)
├── images/
│   ├── Dockerfile.cuda          # uv @ 3.12, torch cu128, bnb ≥ 0.46, openvla (pinned)
│   └── Dockerfile.rocm          # ROCm gfx1151 build (dev lab)
├── src/robovla/
│   ├── core/                    # PURE (no I/O; time injected) — unit-tested to ≥95 %
│   │   ├── scheduler.py         # FIFO, run-now, no preemption, priority + config-aware fit, transition table
│   │   ├── budgets.py           # VRAM budget calculator
│   │   ├── spec.py              # JobSpec JSON schema + validator + preset catalog
│   │   ├── config.py            # fleet.yaml schema + loader + diff/reload rules
│   │   ├── qa.py                # dataset QA rules
│   │   ├── norm_stats.py        # min/max stats + un-normalization
│   │   ├── split.py             # seeded held-out split (stable per student+task)
│   │   ├── ticket.py            # experiment-ticket schema
│   │   └── audit.py             # audit reducer
│   ├── episode.py               # shared episode validator (vendored into capture app + agent)
│   ├── storage.py               # SQLite (WAL): users, sessions, tokens, datasets, tickets, jobs, artifacts
│   ├── fs.py                    # datasets/ runs/ artifacts/ layout, zip-slip-safe unzip (shared with the agent)
│   ├── proc.py                  # fixed-argv exec — used by the agent; never shell=True
│   ├── runner/
│   │   ├── interface.py         # Runner protocol: submit/poll/kill/tail/fetch_artifacts
│   │   ├── http.py              # HttpRunner — the only real implementation (console → agent HTTP)
│   │   └── fake.py              # FakeRunner (agent-protocol fakes; CPU tests)
│   ├── gpu.py                   # per-node VRAM view + orphan alert (via the agent)
│   ├── http/                    # FastAPI app
│   │   ├── app.py               # router wiring, rate limits, error shape (§9.6)
│   │   ├── auth.py              # sessions, tokens, the ONE RBAC dependency
│   │   ├── datasets.py          # §9.2
│   │   ├── runs.py              # §9.3
│   │   ├── tickets.py           # §9.3 ticket resource
│   │   ├── stream.py            # §9.4 (SSE)
│   │   ├── infer.py             # §9.5 (claims + /api/robot/predict proxy)
│   │   ├── admin.py             # §9.7 (all audited)
│   │   └── health.py            # /api/health
│   └── cli.py                   # model_download, gpu_selftest (via agent), node bootstrap, backup, restore
├── capture/                     # Phase 3: standalone PySide6 capture app (vendors core shared modules)
├── ui/                          # vanilla JS + HTMX (readable by students)
├── tests/
│   ├── unit/                    # core — every PR, < 60 s
│   ├── integration/             # service + FakeRunner — every PR, < 5 min
│   ├── e2e_class/               # scripted 12-student scenario — nightly
│   └── hardware/                # gpu_selftest / home_selftest / robot — GPU_TESTS=1
├── fleet.yaml                   # the fleet (§10 / Appendix A)
├── systemd/robovla-console.service
├── systemd/robovla-runner.service   # per node: podman quadlet wrapping the runner container
└── pyproject.toml               # uv-managed; image pins finalized after gpu_selftest (Phase 1)
```

Commands: `uv run pytest -m "not hardware"` (CI) · `GPU_TESTS=1 uv run pytest -m hardware` · `uv run robovla cli gpu_selftest --node pai1`.
