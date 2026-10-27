# roboVLA Training Console - Design Specification (v0.5.1)

A web-based, OpenVLA-specific training control plane for a class of up to 12 individual students (currently 9). Each student has their own robot. The console is hosted on the campus LAN. It was drafted from a live inspection of `pai1` on 2026-10-20. v0.3 was finalized 2026-10-20 as the implementation-ready spec.

**v0.5.1 (2026-10-21):** post-review clarifications (no architecture change). The ticket link **releases when the linked run reaches a terminal state** (`done|failed|killed`). The ticket is retained for the record. It can be re-linked, or a new ticket can be created (section 9.3). The `smoke` job emits a `predict_action` probe on one held-out dataset image. It publishes to the job detail. This is the console-side evidence for lesson-plan gate M5. No slot lease is needed (section 6, test matrix section 11). The agent to console bundle pull authenticates with the **same per-node pre-shared agent token**. The console verifies it against the argon2 hash in the console DB. This is the only non-cookie console route (section 9.9). Node workdir `datasets/` = the pulled bundle. It is ephemeral. It is re-pullable at dispatch. It is excluded from backup (section 10). `bind: 0.0.0.0` is annotated as pai1-specific (single-LAN-interface condition. Bind the LAN IP elsewhere) (section 9.8, Appendix A). VRAM budgets are defined at the node's **max allowed batch**. They scale down with smaller batches (section 3). The section 8.5 log row is disambiguated (agent log per node vs. console log local).

**v0.5 (2026-10-20):** **GPU tenancy is manual.** The console does not manage pre-existing models (the 27B assistant). It shows measured free VRAM + holding processes per node. It gates jobs on the VRAM budget only (ninfer stop/start, release windows, and the `gpu_releasing` state are all removed). **Runner = a per-node podman container**: the pinned stack lives in one image per accelerator family (cuda/rocm). The console talks to a small agent API over LAN HTTP. No ssh. No per-node venvs. No rsync (the agent pulls the dataset bundle). **Robots talk to the console only** (`/api/robot/predict`). The console verifies student token + slot lease. It proxies to the node agent. A stable endpoint even when slots move nodes. Other decisions: experiment tickets are a standalone resource created before run submission. `max_seq_len` fixed at 287 (official OpenVLA context. Consistent with the model card's 256-bin x 7-token action space). Node `priority` = first-fit order (higher first). Held-out split seeded per student+task (stable across versions). `smoke` is a verification tool, not a submission gate. All ports configurable with defaults (console 8501, agent 8601). The home 8060S is not a cluster node (roboVLA is LAN-only). It is a self-contained dev lab.

**v0.4 (2026-10-20):** Launch fleet = **pai1 only** (single trainer, 2 inference slots). The 5090 Ti / 4070 Ti / home 8060S are verified inventory and disabled config entries. They join later purely via the admin node page. No code, no redeploy. **Node selection at submit**: student picks (teacher-gated `student_selectable`). Teacher assigns/pins. Or auto first-fit.

**v0.3 (2026-10-20): implementation-ready.** Added: section 8 Admin controls (complete taxonomy. Every admin action is a gated REST endpoint). Section 9 Full API schema (REST + SSE + security model). Section 10 Portability and scalability (fleet-as-config, runner abstraction, deployment, scaling path, operational durability). Section 11 TDD and DRY (pure core, single JobSpec contract, binding DRY rules, test matrix). Appendices A-C (default `fleet.yaml`, JobSpec JSON, repo layout for the implementing coder).

**v0.2 (2026-10-20):** LAN-only access (no public tunnel). Multi-runner fleet (pai1 + 5090 Ti + 4070 Ti + home dev box). "Run now" = no preemption. 12 robots, 1 per student. Standalone desktop capture app. Demo-day slot relay. GTX 1070 / Strix Halo assessment.

## 1. The fleet (verified 2026-10-20)

| Node | GPU (VRAM) | Role | Training | Inference slots | Notes |
|---|---|---|---|---|---|
| **pai1** (primary console host) | RTX 5090, 32 GB | Primary trainer, 2 infer slots, hosts the assistant 27B | QLoRA batch 4-8 (bf16 also fits) | 2x 8-bit (about 10 GB each) | Verified: driver 615.71/CUDA 13.4. The 27B course assistant (`ninfer-serve`, NVFP4) holds 29.5 GB. **The console does not manage it**. The teacher stops/starts it via `~/start-ninfer.sh` (plain process, no systemd/docker. Clean stop/start verified). The board shows free VRAM while it is loaded (section 3). 44 GB RAM free. 1.7 TB disk. System Python 3.14. uv venv @ 3.12. **Blackwell sm_120: the official OpenVLA pins (torch 2.2/cu121, flash-attn 2.5.5) do not work**. Re-pin in the runner image: torch 2.8/2.9 cu128, bnb >= 0.46, SDPA attention, transformers pinned after selftest. |
| **Campus node 2** (5090 Ti, on hand) | 32 GB | Second trainer, 2 infer slots | QLoRA batch 4-8 | 2x 8-bit | Same stack as pai1. |
| **Campus node 3** (4070 Ti, on hand) | 16 GB | Overflow trainer, 1 infer slot | QLoRA **8-bit, batch 1 only** (about 2x slower). bf16 training does not fit (15 GB weights) | 1x 8-bit | The VRAM budget guard is mandatory on this node. |
| **Home dev box** (Strix Halo, 128 GB unified) | 8060S iGPU (about 7.4 TF FP32-class, about 15 TF bf16, 256 GB/s) | **Off-campus dev lab. Not a cluster node** (roboVLA is LAN-only) | **bf16 LoRA, no quantization**. 20-25 GB total fits the 128 GB pool. This avoids bnb-on-ROCm entirely. 3-4 h per epoch. Overnight runs | bf16 inference testing (1-3 s per action) | ROCm gfx1151 (RDNA 3.5). `HSA_OVERRIDE_GFX_VERSION=11.0.0` fallback. Runs the same runner image **self-contained on the home LAN** (console + agent + model, never exposed). Verify with `home_selftest` (30-min checklist). |
| ~~GTX 1070 (8 GB) on home box~~ | - | **Excluded** | - | - | 8 GB is below the floor for 7.5B even at 4-bit QLoRA (borderline OOM). No tensor cores. Pascal sm_61 is deprecated (CUDA 12.8) / removed (CUDA 13). |

**Launch fleet (current): pai1 only** - one trainer, 2 inference slots. Rows 2-3 (5090 Ti, 4070 Ti) are verified inventory and planned fleet entries. They have `enabled: false` in `fleet.yaml` until they come online. They join purely via the admin node registry (section 8.2: install agent -> add -> probe -> enable). No code change, no redeploy. The home 8060S is not a fleet entry (off-LAN). It is a self-contained dev lab. With a single trainer at launch, the teacher batches full runs to the morning + overnight windows (lesson-plan rhythm). The console itself gates only on measured free VRAM (section 3).

**Campus inference concurrency: 2 slots at launch** (pai1). It becomes 5 slots once node 2/3 join. See section 7.7 for the 12-robot demo relay.

## 2. What the system is

One console per class (FastAPI + SQLite + SSE. Production instance on pai1). It is the single control plane for the campus GPUs. Each GPU node runs a standard **runner container** (podman). It hosts the training/serving stack (section 4, section 10). 12 student accounts. LAN-only access (students use classroom devices. No home/remote access is needed. Per teacher decision). Students:

1. **Upload training data** (episode folders from their robot). Or use the **standalone capture app** (section 5.3) to collect. They see a per-episode QA report before any queue time is spent.
2. **Configure a QLoRA fine-tune** (OpenVLA-specific form, safe defaults. The fragile stack is never exposed).
3. **Submit to a queue**. Watch **live progress** (loss/accuracy sparklines, log tail, VRAM, ETA, queue position).
4. Get **automatic held-out validation** per checkpoint. Get **artifacts** (adapter, merged export). Get a **class board** comparing all 12.
5. **Claim an inference slot**. The console serves their fine-tuned model to their robot's Pi. The console *is* the model-server endpoint. The node's runner container hosts the weights. No laptops-as-servers.

## 3. GPU tenancy and scheduling

**Tenancy is manual.** The console does not load, stop, or unload pre-existing models (example: the 27B assistant on pai1, about 29.5 GB). It shows **measured free VRAM per node and the processes holding it** (admin panel + dashboard, from the node agent's probe). The teacher unloads non-roboVLA models before training/inference windows. Tenancy reference (pai1, assistant unloaded): QLoRA 7.5B is about 12-18 GB (1-2 concurrent on 32 GB nodes with budget guard). 8-bit inference is about 9-11 GB (2 on 32 GB, 1 on 16 GB).

**Scheduling rules (per teacher decision):**
- **No preemption.** "Run now" (teacher or student) is admitted to the front of the line *after* the current job on the selected node finishes. The scheduler picks the first node whose free VRAM fits the job. If all nodes are busy, "run now" waits for the earliest completion (the student sees "waiting for a slot, ETA X").
- **First-fit with priority:** first-fit order is **`priority` (higher first. Ties go to `fleet.yaml` order)** across enabled nodes. A job is admitted only where its **VRAM budget fits the job's chosen config** (the student's batch determines the estimate. Section 6 presets).
- **VRAM budget guard:** a job starts only if measured free VRAM >= its budget (per-node, per-job-type presets). A 2nd concurrent train slot on a 32 GB node is allowed only with the reduced-batch preset + guard (default: 1 train slot per node). `train_gb`/`infer_gb` are the budgets at the node's **max allowed batch**. Smaller batches scale the estimate down (core/budgets.py). So a batch-1 job on a 32 GB node needs less than `train_gb`. Two concurrent train slots fit only under reduced-batch presets.

**Admin levers (teacher panel):** per node: enable/disable, image tag, train slot count (1-2), allowed batch (1 / 4 / 8 with matching grad-accum), VRAM budget, priority, agent endpoint/token. Global: job caps, per-student limits (1 running - an active slot counts - / 2 queued), queue position override (bump/kill). Full taxonomy in section 8.

## 4. Architecture

```
 12 student browsers (classroom LAN devices; HTTP, self-signed TLS optional)
        |
        v
 robovla console - FastAPI + SQLite + SSE, bound to campus LAN IP :8501 (pai1 = production)
  +- /api             REST: auth, datasets, runs, tickets, artifacts, slots, admin (section 9)
  +- /api/robot       the ONLY robot-facing endpoint: token + lease check -> proxy to node agent (section 9.5)
  +- /stream/{job_id} SSE: metrics JSONL tail, log tail, VRAM poll
  +- scheduler        SQLite-backed FIFO; per-node slots; no preemption
  |    +- runner client (JobSpec JSON -> agent HTTP, location-agnostic; Appendix B)
  +- gpu manager      per-node free-VRAM view + holding processes (via agent); orphan watchdog
  `- dataset service  chunked upload -> unzip -> validate -> resize 224x224 -> norm stats
        |  LAN HTTP (per-node agent token; default agent port 8601)
        v
 per-node runner container (podman, GPU passthrough, pinned stack image)
  +- agent API        :8601 - jobs/serve/vram/health; launches fixed-argv processes (no shell)
  +- training jobs    robovla_train.py / official finetune.py (crosscheck) / merge / selftest
  `- inference serve  8-bit adapter loaded for a claimed slot (proxied via the console)
        |
        v
 /home/adam/robovla/{datasets, runs, artifacts, models}   (host bind mount, same layout per node)
        ^
 12 robot Pis --(POST /api/robot/predict, student token)--> console --> node agent /predict
```

**Why FastAPI + SSE + vanilla JS/HTMX (not Streamlit/Gradio):** 12 concurrent long-lived live dashboards. Real per-user auth. No per-interaction rerun model. About 1.5-2k LOC. Students can read the whole codebase. Node hosts need only the runner container (image pull + systemd unit). The console host adds the console venv + its unit. LAN-only: no tunnel, no public exposure. Auth + rate limits still apply (students are students).

## 5. Data plane

**5.1 Upload (MVP).** Per-student, versioned datasets (`s07/task-stop-at-target/v2`) on pai1. Chunked zip upload of the episode folders (their Pi pipeline: `frames/`, `actions.json`, `episode.json`). 100 episodes = about 0.5-1 GB. 2 GB cap. zip-slip-safe. jpeg/json only. At dispatch, the node's agent pulls the prepared dataset bundle from the console over LAN HTTP (about 1 GB. Seconds to minutes at GbE. Trivial).

**5.2 Server-side dataset prep (CPU, about minutes):** per-episode validation (frame==action count, 10 Hz timing, idle > 2 s flag, sign/jump anomalies). 224x224 resize via the official processor's preprocessing (train/inference augmentation parity). `norm_stats.json` (per-dim min/max over 7 dims). Held-out split (15%, **seeded per student + task. Stable across versions of the same content. Recomputed deterministically if the episode set changes**). So v1 to v2 comparisons run on the same held-out episodes. And re-uploads cannot reshuffle the holdout. Output: "ready" dataset + QA report page. This is the gate that stops bad data from eating queue time.

**5.3 Standalone capture app (separate deliverable, built in Phase 3 with students).** A desktop app students run on their own laptops on the classroom LAN. Per teacher decision, this is a standalone project, not a console feature:
- **Stack: PySide6 v1** (pure Python. Students can read it. pip-installable. Single-file via PyInstaller for student laptops). Tauri polish is a v2 option.
- **Features:** robot discovery (static IP list from the class sheet + optional mDNS). Live 10 Hz view (reuses their `frame_sender`). Keyboard + gamepad teleop (pygame) -> Pi (their `twist` path). Episode logging to the standard spec. Built-in pre-upload QA (sync check, idle flags). One-click upload to the console. "Replay check" (open-loop replay of the last episode on the robot).
- Collection is **fully offline** (episodes log to local disk. The app works anywhere. Example: at home if a robot comes home). **Upload requires the classroom LAN**. Datasets created at home upload when the student is back in class (LAN-only console, section 9.8). Same episode spec as section 5.1. So capture and upload feed one pipeline. **Shared code:** the capture app vendors `episode.py` / `norm_stats.py` / JobSpec presets from the console repo verbatim (section 11). So capture-side QA and console QA can never drift.

## 6. Training engine (OpenVLA-specific)

**Decision (unchanged from v0.1): a thin in-house `robovla_train.py` (about 400-600 LOC) on the openvla repo's model code + PEFT QLoRA.** It emits structured JSONL metrics (first-class observability -> SSE), validation hooks, per-epoch checkpoints, crash resume. The official `finetune.py` is *not* abandoned. A `crosscheck` job runs it on a 50-episode subset. Students compare curves (Class 9-10 exercise). The recipe is identical either way: 8-bit, rank 32, alpha 32, lr 5e-4, image aug on.

**Per-runner presets** (admin-configurable):

| Runner | Quantization | Batch | Notes |
|---|---|---|---|
| pai1 5090 / 5090 Ti (32 GB) | 8-bit | 4-8 | about 45 min-2 h per run |
| 4070 Ti (16 GB) | 8-bit | 1 (+grad accum) | about 2x slower. Budget guard |
| Home 8060S (128 GB unified) | **bf16, no quant** | 1-2 | overnight, dev-lab only (off-campus). No bnb dependency (ROCm) |

**`max_seq_len` is fixed at 287** (the official OpenVLA context: 256 visual tokens + prompt/action text. Consistent with the model card's 256-bin x 7-token action space). It lives in the preset catalog, not the student form.

**Job types:** `smoke` (1 batch, about 2-5 min. Stack/config verification at node join. Optional student sanity check. The job also runs a `predict_action` probe on one held-out dataset image. It publishes the 7-dim action to the job detail. This is console-side evidence for lesson-plan gate M5. No slot lease needed. **Not a submission gate**: stack correctness is covered by `gpu_selftest` + the test matrix, section 11). `train` (cap 3 h). `validate` (held-out eval, auto after checkpoints). `merge` (`merge_and_unload()` + 10-image identity check. 15 GB export on demand. 14-day purge). `infer` (serve adapter, token auth. A claimed slot *is* an infer job, section 9.5). `crosscheck` (official script on a **deterministic 50-episode subset**, seeded per student + dataset version. Same episodes every re-run. So students compare their console engine against the official engine on identical data). `gpu_selftest` / `home_selftest` (stack verification per node: load, fwd/bwd, `predict_action`, timings).

**Stack (pinned in the runner image):** uv venv @ Python 3.12. torch 2.8/2.9 cu128. transformers pinned after selftest. bnb >= 0.46. SDPA (flash-attn only if it builds clean for sm_120). timm/pillow/numpy. The image `robovla-runner-cuda` holds them. The **image tag is pinned per node in `fleet.yaml`** (new pin = new tag + selftest at node join). **Escape hatch:** the runner protocol is location-agnostic. Any host accepts the same JobSpec. So does a rented cloud A100 running the same image (reachable over a private link the teacher sets up).

## 7. UI surfaces

1. **Dashboard**: queue position + ETA. My datasets. My runs. Per-node GPU state (free VRAM + who/what holds it). Class activity feed.
2. **Datasets**: upload wizard -> prep job -> QA report. Versioning. Norm stats display. Held-out info. "Ready" badge.
3. **New run**: dataset (ready only). Task prompt template. **Training node** (default *auto* = first-fit in priority order. Admits only where the chosen config's budget fits, section 3. A specific node only when the teacher has marked it student-selectable, section 8.2. The teacher can assign/pin any node via requeue). Rank/alpha/lr/epochs/aug/seed + batch (options filtered by node fit. auto = options that fit at least one enabled node). In a collapsible *advanced* section (values drawn from the single preset catalog in core. `max_seq_len` fixed at 287, not exposed). **Required experiment ticket** (standalone resource created before submit: hypothesis -> config -> prediction. Outcome at completion. Section 9.3). Submit -> optional smoke -> train.
4. **Run detail (SSE)**: train + held-out val sparklines. Step/total/ETA. Grad norm. LR. VRAM + util. Checkpoint events. Raw log (collapsible). Kill (teacher).
5. **Class board**: all 12 side-by-side (best held-out accuracy, run counts, dataset stats, tickets). Printable. The final-week post-mortem material.
6. **Artifacts**: adapter download. Merged export (on demand). Run config JSON.
7. **Inference slots**: "serve my adapter" (8-bit). Slot queue (claims with no free slot queue like any job, section 9.5). Live status. **Pi connection info** (the stable console endpoint + per-student token. No per-node robot config, section 9.5).
   **7.7 Demo-day relay (12 robots, 2 slots at launch. 5 after node2/3 join):** students are scheduled into about 4-min arena runs. Runs pipeline with presentations. Student *i* presents their run's result while student *i+1's* robot executes on a slot. 9-12 students pipelined = about **55-70 min total** (4-min run overlapped with the 3-min talk + 1-min Q&A. Sizing per `setup-and-materials.md`). Robot slot demand is only about 4 min per student (<= 48 robot-minutes). Two slots absorb that comfortably inside the presentation timeline. The slot board is visible to the class. For practice weeks, the same relay gives every student 30-60 s of autonomy back-to-back. "Everyone's robot is always moving."
8. **Teacher panel**: thin forms over the admin API (section 9.7): runner registry (enable, image tag, slots, allowed batch, VRAM budget, priority, agent endpoint/token). Queue ops. Student accounts. Disk usage. Raw logs. Node host lifecycle (container restart, image update, GPU reset) is shown as copy-paste host commands, not panel actions (section 8.2). Complete taxonomy in section 8. Every panel control is an audited endpoint.

## 8. Admin controls

Complete taxonomy of everything the teacher can do. **All admin actions are exposed as gated REST endpoints (section 9.7). They are rate-limited (default 60 req/min). They are written to the append-only audit log** (actor, timestamp, action, target, before to after JSON). The UI (section 7.8) is a thin form over these endpoints. There are no admin capabilities that are not in the API.

### 8.1 Accounts

| Control | Behavior |
|---|---|
| Create / import students | single create or CSV roster import (id `s01`-`s12`, display name, email optional). Created inactive until activated |
| Activate / deactivate | deactivated: no login, no token use. Queued jobs requeue to fitting runners. Running jobs complete. No new submissions |
| Roles | `teacher` (everything). `student` (own data, class board, inference claims). `observer` (read-only board + GPU state. Demo-day visitors) |
| Reset password | out-of-band: teacher sets a one-time code/link (paper). No email dependency in the classroom |
| Per-student limits (defaults) | max running (1. **An active inference slot counts as running**, it holds the GPU). Max queued (2). Max dataset bytes (2 GB). Max run minutes (180). Run-now allowed (yes. Front of line after current completion, section 3) |
| Pi tokens | generate / rotate / revoke per student. Scoped to that student's datasets, adapters, and inference claims. Value shown once. Argon2-hashed at rest |
| Sessions | list active sessions per user. Revoke any session (immediate) |
| Audit log | all admin actions + auth events. Append-only. Admin-readable. JSON export |

### 8.2 Nodes / endpoints

The node registry **is** the fleet config (section 10, `fleet.yaml`). The admin UI edits config, never machines. A node = a host running the standard runner container (agent API, section 4). The console talks to the agent, not the host.

| Control | Behavior |
|---|---|
| Add node | form: id, label, image tag (`robovla-runner-cuda` \| `-rocm`), LAN IP + agent port (default 8601), agent token (shown once. Console host env var + agent container env. Hash in DB for audit), GPU VRAM, train slots (1-2), infer slots (1-2), quant profile (`bnb8` \| `bf16`), allowed batch (1 \| 4 \| 8 + matching grad-accum), VRAM budget per job type, priority (int, default 0. First-fit order, higher first), student-selectable (default **off**), enabled (default **off**). Then a **dry-run join probe** (agent `/health`: image tag matches, GPU visible with expected VRAM, disk free >= 50 GB, model dir present). The node stays disabled until the teacher enables it |
| Edit node | any field. Changes apply at the next scheduling tick. Active jobs are unaffected. A budget shrinking below a running job completes that job. No new admissions on that node until refit |
| Enable / disable | disable: queued jobs requeue to other fitting runners. Running jobs fail-fast with checkpoint-resume hint. Inference slots drain with client notification (the Pi watchdog stops the robot after lease expiry) |
| Remove node | as disable, then removed from config. Its jobs/artifacts remain on disk, flagged |
| Probe | on-demand re-run of the join probe (network changes, after maintenance, after image update) |
| VRAM purge (agent-scoped) | post-job probe lists orphan processes the agent spawned but did not reap. The teacher confirms. The agent kills them (audit-logged). **Host-level** processes (example: the 27B assistant) are shown on the board with their VRAM. They are the teacher's to manage (section 3). The console does not touch host processes |
| Node host lifecycle (host-side, not API) | container start/restart. Image pull/update. `nvidia-smi --gpu-reset` (confirmation required, no running jobs). These are documented one-liners the teacher runs on the host (the `robovla-node` script in the repo). The admin panel shows agent version/health, an "update available" flag, and the exact command. The console cannot restart a remote node by design (it talks to the agent, not the host) |
| Service log | tail the node's agent log (last 500 lines) via `GET /api/admin/nodes/{id}/logs`. The console's own service log is local to its host |
| Endpoint visibility | robots talk to the **console** (one stable endpoint + per-student token, section 9.5). No per-robot node config. Each node's LAN IP + agent port is published on the admin page for Pi debugging |
| Student-selectable | teacher toggles per node whether students can name that node at run submission (section 9.3). Off = auto first-fit or teacher-assign only. `enabled: false` nodes are hidden from students entirely |

### 8.3 Jobs and queue

| Control | Behavior |
|---|---|
| Queue view (all roles) | all nodes. FIFO. Position. ETA (queue wait + estimated run time). Node fit. Students read-only |
| Bump (teacher) | move a job to the front of the queue on the nodes it fits. Effective at the next completion boundary (no preemption, section 3) |
| Kill | students: own queued jobs. Teacher: any. Queued: `killed`. Running: SIGTERM to the job process via the agent. 10 s grace. SIGKILL. State `killed`. VRAM probe. Checkpoint artifacts preserved |
| Reschedule (teacher) | pin a queued job to a specific node, or clear a pin |
| Manual completion (teacher) | mark a job `done` for work done outside the console (example: the teacher ran `finetune.py` by hand). Note required. Inference is only enabled if an artifact manifest exists |
| Retry from checkpoint (teacher) | resubmit a failed/killed training job from its last checkpoint. The dataset bundle is re-pulled if the node differs |
| Annotate (teacher) | free-text note on a job. Visible on the class board |
| Caps | global max concurrent (default = sum of node train slots). Per-node (slots). Per-student (section 8.1). The scheduler enforces before admission |

### 8.4 Datasets and artifacts

| Control | Behavior |
|---|---|
| Dataset inventory | per student/version: episode count, size, QA status, which runs used it |
| Reject (teacher) | block a dataset from new runs (reason shown on the student's page). Un-reject after a corrected re-upload |
| Purge (teacher) | two-step confirmation. Deletes episodes + QA + stats. Runs that used it keep their checkpoints/adapters (run config is self-contained) |
| Retention | default: keep all student data. Optional per-student max-versions (LRU purge *proposal* list. Teacher approves). Global disk cap per node with 80 %/90 % alerts |
| Artifact inventory | per run: adapter (always). Merged export (on demand, 15 GB, 14-day auto-purge. Teacher-adjustable). Run config JSON |
| Force purge / export | teacher may purge early. **Backup bundle** zips datasets + adapters (node-host artifacts collected via the agent) + DB snapshot. For off-site archive |

### 8.5 System

| Control | Behavior |
|---|---|
| Health | `GET /api/health` (LAN-public): service uptime. Per-node state (online, GPU free/busy, free VRAM, queue depth). Consumable by a classroom display |
| Logs | agent service log per node (last 500 lines, `GET /api/admin/nodes/{id}/logs`, section 8.2). Console service log (local to the console host). Per-job training logs (run detail). Audit log (section 8.1) |
| Backup | nightly (teacher-adjustable time) of DB + datasets + adapters. To a configurable destination (local dir or NAS mount). Status view (last run, size, checksums). Restore = documented procedure. Covered by the test matrix |
| Config | view effective config (rendered YAML + per-node profiles). **Hot reload**: validate -> apply. Active jobs unaffected. Queued jobs revalidated against new budgets |
| Maintenance mode | console read-only. No new submissions/uploads. Running jobs and live slots continue. For stack upgrades |
| VRAM watchdog | post-job probe (via agent) compares running processes to expected. Agent-scoped orphans: alert banner + one-click purge (section 8.2). Host-level VRAM holders are shown on the board (teacher-managed, section 3) |

## 9. API schema

Base URL: `http://<console-lan-ip>:8501` (console. The only endpoint robots and students talk to). Node agents on their LAN port (default 8601, console-only). All ports configurable per node in `fleet.yaml`. Conventions: JSON bodies. Errors `{"error":{"code","message"}}` (section 9.6). List endpoints take `?cursor=&limit=` (default 50). `Idempotency-Key` header on `POST /api/runs` (client UUID. 24 h dedup. Returns the existing job). Artifacts are ETag-cached. Timestamps RFC 3339 UTC.

### 9.1 Auth (cookie sessions, 12 h sliding)
- `POST /api/auth/login` `{username, password}`. Sets httpOnly cookie
- `POST /api/auth/logout`
- `GET /api/auth/me`. Identity, role, limits, own token metadata
- `POST /api/auth/token` `{action: generate|rotate|revoke}` (own Pi token. Value shown once on generate/rotate)

### 9.2 Datasets (student: own. Teacher: all)
- `POST /api/datasets/upload` (chunked multipart. `{dataset_id: "s07/task-stop-at-target/v2", total_chunks, chunk_index, sha256}`)
- `POST /api/datasets/upload/finalize` `{dataset_id}`. Starts server-side prep (section 5.2). 409 if chunks incomplete
- `GET /api/datasets`. `GET /api/datasets/{id}` (status: `uploading|preparing|ready|rejected|failed`)
- `GET /api/datasets/{id}/qa` (per-episode QA report + summary)
- `GET /api/datasets/{id}/norm-stats`
- `POST /api/datasets/{id}/revalidate` (re-run prep after corrections. New version. Held-out split stays stable for unchanged content, section 5.2)
- `GET /api/datasets/{id}/episodes/{ep}/frames/{i}.jpg` (browser preview)

### 9.3 Runs and tickets
- **Tickets (standalone resource. Created before submission):**
  - `POST /api/tickets` `{hypothesis, config_summary, prediction}` -> `{ticket_id}` (student-owned. Editable until linked. **The link releases when the linked run reaches a terminal state** - `done|failed|killed`. The ticket is retained for the record. It can be re-linked, or a new ticket can be created)
  - `GET /api/tickets/{id}`. `PUT /api/tickets/{id}`
  - `GET /api/runs/{job_id}/ticket`. Linked ticket + outcome field (completed at run end)
- `POST /api/runs` `{dataset_id, job_type: smoke|train|validate|merge|infer|crosscheck, ticket_id (required for train - the experiment run. smoke/merge/crosscheck exempt - must be own, not yet linked), node_id?: "auto"|<node-id>, config: {rank, alpha, lr, epochs, batch, grad_accum, aug, seed, prompt_template}, run_now?: bool}` -> `{job_id, position, eta_s}`
  - Node selection: `"auto"` (default) = first-fit across enabled nodes in **priority order (higher first. Ties go to `fleet.yaml` order)**. Admits only where the **chosen config's VRAM budget fits**. A specific node is accepted for students only if enabled **and** `student_selectable: true` (section 8.2). The teacher may name any enabled node. Pinned jobs wait on that node only
  - Validation: dataset `ready`. Ticket present + unlinked (train). Per-student caps. If no node fits now, the response carries `waiting: "slot"` + earliest ETA (reason: busy vs budget)
- `GET /api/runs` (own). `GET /api/runs/{job_id}` (state, config, node, progress, artifacts)
- `GET /api/runs/queue` (all nodes, all students. Position + ETA. Feeds the class board)
- `POST /api/runs/{job_id}/kill` (own queued. Teacher: any)
- `POST /api/runs/{job_id}/requeue` `{node_id?}` (teacher). `POST /api/runs/{job_id}/bump` (teacher)
- `POST /api/runs/{job_id}/complete` `{note}` (teacher)
- `GET /api/runs/{job_id}/logs?offset=` (tail, via agent)
- `GET /api/runs/{job_id}/artifacts` (adapter path, merged status, run config JSON)

### 9.4 SSE streams
- `GET /stream/{job_id}`. Events: `state` (`queued|running|finishing|done|failed|killed` + reason). `metrics` (JSONL: `step, total, loss, lr, grad_norm, vram_mb, vram_util_pct, eta_s`). `val` (held-out accuracy per checkpoint). `log` (tail lines). `artifacts` (checkpoint/adapter/merged events). Heartbeat every 15 s. `Last-Event-ID` resume.
- `GET /stream/class`. Class-board feed (queue, node state, slot board, activity). <= 1 msg/s. All roles.

### 9.5 Inference (robot path)
- `POST /api/infer/slots/claim` `{run_id}` (adapter must exist). Creates an **infer job** (auto first-fit in priority order: infer slot free + free VRAM >= infer budget. The claim takes no node param. The teacher pins a specific node via requeue, section 9.3). Response: slot available -> 200 `{slot, node_id, endpoint: "http://<console-lan-ip>:8501/api/robot/predict", lease_s}`. None available -> 202 `{queued: true, position, eta_s}` (the claim queues like any job. No preemption).
- `GET /api/infer/slots` (all: holder, adapter, node, load)
- `POST /api/infer/slots/{slot}/release` (= kill the infer job. Agent unloads the adapter. VRAM probe)
- **Robot endpoint (console port 8501, `Authorization: Bearer <student token>`):**
  `POST /api/robot/predict` `{image: base64 jpg, prompt}`. The console verifies token + active slot lease. It proxies to the holding node's agent `/predict`. It returns `{action_norm[7], action_robot[7] (un-normalized with the dataset's min/max stats), latency_ms}`
  - Robots talk **only to the console**. One stable endpoint for the life of the course. No per-node robot config. A loopback hop at launch (slots live on pai1). Node agent ports are console-only.
  - Rate limit about 5 req/s per student. Slot lease TTL (default 300 s idle. The slot auto-releases. The Pi watchdog already stops the robot on stale actions). Target p95 < 400 ms including LAN on 8-bit 32 GB nodes (about 150-300 ms inference per the latency rebase in `model-card-review.md`).

### 9.6 Error codes (stable. The Pi client and tests depend on them)
`auth` / `forbidden` / `not_found` / `conflict` / `dataset_not_ready` / `budget_unfit` / `limit_exceeded` / `rate_limited` / `node_offline` / `internal`

### 9.7 Admin API (role = teacher. Audit-logged. 60 req/min) - the section 8 mapping
- **Users:** `GET /api/admin/users`. `POST /api/admin/users`. `POST /api/admin/users/import` (CSV). `PATCH /api/admin/users/{u}` (active, role, limits). `POST /api/admin/users/{u}/reset-password`. `GET /api/admin/users/{u}/sessions`. `DELETE /api/admin/users/{u}/sessions/{sid}`. `POST /api/admin/users/{u}/tokens/{action}`
- **Nodes:** `GET /api/admin/nodes`. `POST /api/admin/nodes`. `PATCH /api/admin/nodes/{id}`. `DELETE /api/admin/nodes/{id}`. `POST /api/admin/nodes/{id}/probe`. `POST /api/admin/nodes/{id}/purge-vram` (agent-scoped). `GET /api/admin/nodes/{id}/logs`
  - Node host lifecycle (container restart, image update, GPU reset) is deliberately **not** in the API. It is host-side teacher one-liners (section 8.2, section 10).
- **Jobs:** `GET /api/admin/jobs`. `POST /api/admin/jobs/{id}/bump`. `POST /api/admin/jobs/{id}/kill`. `POST /api/admin/jobs/{id}/requeue`. `POST /api/admin/jobs/{id}/complete`. `POST /api/admin/jobs/{id}/note`
- **Datasets / artifacts:** `DELETE /api/admin/datasets/{id}`. `POST /api/admin/datasets/{id}/reject`. `GET /api/admin/artifacts`. `DELETE /api/admin/artifacts/{id}`. `POST /api/admin/backup-bundle`
- **System:** `GET /api/health` (public). `GET /api/admin/audit?since=`. `GET /api/admin/config`. `POST /api/admin/config/reload`. `POST /api/admin/maintenance` `{on|off}`. `POST /api/admin/backups/trigger`. `GET /api/admin/backups`

### 9.8 Security model
- Accounts: 12 students + teacher (optional observers). argon2id. httpOnly session cookies. Passwords are never logged.
- RBAC: student -> own datasets/runs/tokens. Read queue + class board + slots. Claim/release own slot. Teacher -> everything (section 9.7). Observer -> board + node state only. Enforced by **one** FastAPI dependency. Never per-route.
- Tokens: per-student Pi tokens. Scoped to their datasets/adapters/slots. Shown once. Hashed at rest. Revocable.
- Agent tokens: per-node pre-shared. Plaintext in the console host env (referenced by `token_ref`. Never in `fleet.yaml` or logs). And in the agent container env. Argon2 hash in the console DB for audit/rotation. All console-to-agent traffic is authenticated. Rotation = admin action + host-side restart.
- Network: the console binds the campus LAN IP (Appendix A ships `bind: 0.0.0.0` for pai1. This is safe only because that host sits on a single LAN interface. Bind the explicit LAN IP elsewhere). Node agents bind their LAN port (console-only traffic). No public exposure. No tunnel. Optional self-signed TLS via reverse proxy. Students use classroom devices. **Robots talk to the console only.**
- Uploads: 2 GB cap. Chunked. zip-slip-safe. Path-sanitized. jpeg/json only.
- **No shell, ever:** validated JobSpec JSON -> fixed argv (inside the agent).
- Rate limits: login 5/min/IP. Uploads 2 concurrent per student. Submissions 10/min/student. Admin 60/min.
- Audit: all section 8 actions + auth events.
- (Flagged, out of scope: `infergate` :8317 and `flygym` :8765 on pai1 bind 0.0.0.0. Recommend localhost-only.)

### 9.9 Agent to console endpoints (agent-token auth, both directions)
- Agent-side API (on the agent's port, default 8601, console-only traffic): `jobs/serve/vram/health` per section 4. `Authorization: Bearer <agent token>`.
- `GET /api/agent/datasets/{id}/bundle`. The one **console-side** route the agent calls. A tar of the prepared dataset (sha256 in the response. The agent pulls at dispatch, section 5.1). The agent presents the **same per-node pre-shared agent token**. The console verifies it against the argon2 hash in its DB (section 9.8). This is the only non-cookie console route. The agent holds no session.
- The bundle pull is the only agent to console call. Everything else is the console polling the agent (section 4).

## 10. Portability and scalability

**Core principle: hardware lives in config, not code. The stack lives in the image, not the machine.** The console has no knowledge of "pai1" or "5090" as machines. A node is an entry in `fleet.yaml` with a hardware profile. A node's software is a **standard runner container** (one image per accelerator family). Moving the class to a new room, a new school, or a cloud box = a new `fleet.yaml`, same images, same code. (Appendix A: full default config.)

- **Fleet = one YAML file** (`fleet.yaml`, validated against one schema in core): console block (bind, port, TLS, session TTL, limits, retention) + `nodes[]` (id, label, `image` tag, `workdir` host bind mount, `agent {lan_ip, port, token_ref}`, gpu `{vram_gb, quant: bnb8|bf16, train_slots, infer_slots, budgets: {train_gb, infer_gb}}`, `presets {batch: [...]}`, `priority`, `student_selectable`, `enabled`) + per-student limit defaults. The shipped default describes the launch fleet (pai1 enabled. node2/node3 as disabled inventory entries). A new deployment ships with a minimal one-node config.
- **Runner = a per-node podman container**. The only hardware boundary. The image (`robovla-runner-cuda` / `robovla-runner-rocm`) holds the pinned stack (section 6). The container runs a small **agent API** on its LAN port (default 8601). It launches job processes inside the container with a **fixed argv** (no shell, ever). The console's runner client is a **single implementation** (`HttpRunner`). The local/ssh distinction is gone. pai1 runs its own container locally. So every node, including the console host, is uniform. Datasets cross as a tar bundle the agent pulls from the console over LAN HTTP at dispatch. Why containers: the Blackwell/ROCm stack-pinning and per-node version-change risks move into one image build (tag pinned per node. The join probe verifies the tag). Adding machines = image pull + agent start + one config line + probe. Zombie VRAM is structurally mitigated (the agent reaps its job processes. Stopping the container frees the GPU). Ephemeral per-job containers are a v2 option.
- **Node host lifecycle stays host-side.** The console talks to the agent, not the host. It commands jobs, models, and probes. Container start/restart, image pull/update, and `nvidia-smi --gpu-reset` are documented teacher one-liners on the host (the `robovla-node` script in the repo). The admin panel shows agent version/health, an "update available" flag, and the exact command.
- **Per-node hardware profiles** drive all presets (VRAM budgets, quantization, batch, slots). A new 24 GB node joins as an image + config line + probe. Nothing is recompiled or re-pinned globally.
- **Deploy anywhere:** Ubuntu 22.04/24.04 + any supported accelerator (NVIDIA CUDA or ROCm). Console = bare `uv venv && uv sync` + shipped systemd unit (`robovla-console.service`) on its host (no GPU needed). Runner = podman container under a shipped systemd unit (quadlet, `robovla-runner.service`) with GPU passthrough. Provisioning jobs (`model_download` into the model volume, `gpu_selftest`) run at node join, via the agent. The console itself is about 1.5-2k LOC. The agent is a few hundred.
- **State is portable:** the console's data dir (`data_dir`: SQLite (WAL) + uploaded/prepared datasets) and each node host's data dir (`runs/ artifacts/ models/` under `workdir`. Plus `datasets/` holding the pulled bundle - ephemeral, re-pullable at dispatch, excluded from backup) = the entire system state. On the console host these are **separate directories** (pai1: `robovla-console/` vs `robovla/`). The pai1 agent pulls its dataset bundle from the console exactly like any other node (local hop. No special case). Rebuild or move = clone repo + restore backup. Nightly backup (section 8.5) covers the rest (node-host artifacts are collected via the agent).

**Scaling path:**

| Dimension | What changes |
|---|---|
| More GPUs | image + agent installed on the host (one-liner) + one config entry + join probe. Scheduler, UI, tests unchanged. |
| More students | accounts are rows. Per-student caps + morning/overnight windows (launch, single trainer) and added nodes (scaling) keep the queue sane. The board scales. |
| Multi-room / multi-school | same console image + different `fleet.yaml`. Optional class-code tenancy if two classes share one console. |
| Bigger / newer VLA models (OFT, FAST, PRIME) | job spec carries model id + processor. Same upload/QA/queue/SSE pipeline. New model = new image tag. |
| Rented cloud box (escape hatch) | same image on a rented host. Reachable over a private link the teacher sets up (VPN/tunnel). The protocol is plain LAN-style HTTP. So nothing else changes. |
| Console SPOF | rebuild from repo + restore backup < 1 h. **Running jobs continue** (owned by the agent container, not the console). Episode data also lives on the 12 PIs. |

**Operational durability (carried from v0.2 section 9):**
- systemd: `robovla-console.service` on the console host. `robovla-runner.service` (podman quadlet) on each node host.
- Job state machine: `queued -> running -> finishing -> done|failed|killed`. A pure transition table in core. Unit-tested.
- Runner failure: dispatch failure or mid-run node death (agent unreachable). The job fails fast. It auto-reschedules on the next fitting runner (never preempts a *running* job). The student is notified with ETA. Crash resume from checkpoints (dataset bundle re-pulled to the new node).
- Post-job VRAM probe (via agent). Orphan alert + agent-scoped purge. Retention: 20 run logs per student. Merged exports 14 days. Nightly backup of datasets + adapters + DB.

## 11. TDD and DRY

**Process (binding for all implementation work):**
1. Every feature starts with a **failing test** (red). Then minimal code (green). Then refactor. No module ships without tests.
2. GitHub CI on push: the full CPU suite (< 5 min). Hardware-gated tests (`gpu_selftest`, `home_selftest`, robot) run manually / at node join behind `GPU_TESTS=1`.
3. **Definition of done:** CI green. No new TODOs. Section 9 updated if the API surface changed. Every new admin action has audit coverage + a test.

**Architecture for testability:**
- **Pure core** (`robovla/core/`. No I/O, no network, no wall clock. Time is injected): scheduler (FIFO, run-now = front-of-line after current completion, no preemption, per-node slot + VRAM-budget fit, kill/reschedule state machine). VRAM budget calculator. JobSpec validation (one JSON schema). Dataset QA rules. Norm stats + un-normalization. Held-out split (seeded, deterministic). Experiment-ticket schema. Audit reducer. Config schema. Target: >= 95 % line coverage. Suite < 60 s on CPU.
- **I/O only at seams:** `storage` (SQLite). `fs` (datasets/artifacts). `runner` (agent HTTP client). `http` (FastAPI app). `clock`. Tests use fakes at every seam. `FakeRunner` speaks the agent protocol with scripted metric/event streams. So the **entire SSE path runs on CPU**.
- **One contract:** JobSpec JSON (Appendix B). The only artifact shared by UI, scheduler, runners, docs, and the crosscheck job. One schema. One validator. One preset catalog. The QLoRA default recipe is defined once: 8-bit, rank 32, alpha 32, lr 5e-4, aug on, max_seq_len 287. The "advanced" form reads from that catalog.
- **One config:** `fleet.yaml`. One schema. One loader. Hot reload revalidates. No machine-specific string in code.
- **One episode validator:** `episode.py` (format, frame/action counts, 10 Hz timing, idle > 2 s, sign/jump checks) + `norm_stats.py` + the preset catalog are **shared code vendored verbatim into the standalone capture app and the runner agent**. Capture-side QA, console QA, and agent-side QA cannot drift.

**DRY checklist (rules for the implementing coder):**
- No inline SQL outside `storage.py`. No JSON schema defined in two places. One error type (section 9.6). One logging setup. Auth/RBAC through the single FastAPI dependency (never per-route). No node names/IPs in code (all in config). The capture app and agent share `core`/`fs`/`episode`/`proc` code with the console verbatim (vendored at image build). **No shell, ever** (validated JobSpec JSON -> fixed argv, inside the agent).

**Test matrix:**

| Layer | Coverage | Gate |
|---|---|---|
| Unit - core | scheduler ordering (incl. run-now, no-preemption, priority + config-aware first-fit: student batch vs node budget, 16 GB node = batch-1 preset, 2-slot admission on 32 GB). Kill/reschedule transition table. Ticket-before-run (train: 400 without ticket, 409 if already linked). QA rules (frame/action mismatch, 10 Hz timing, idle > 2 s, sign flip, jump anomaly). Norm stats + un-normalization round-trip. Held-out split determinism (stable across versions of unchanged content). JobSpec + config schema validation | every PR. CPU. < 60 s |
| Integration - service (CPU, FakeRunner) | chunked upload (incl. zip-slip payload rejected). Prep. QA gate. Ticket. Queue admission. SSE (state/metrics/val events, heartbeat, resume). Artifact manifest. Smoke predict-action probe (scripted FakeRunner). Robot `predict` with mocked agent + token auth. RBAC (student 403 on others' runs and all `/api/admin`). Idempotency dedup. Rate limits. Slot claim 202-queue when full. Dataset bundle pull (fake agent, agent-token auth on the console route). Ticket link released after terminal state (re-link / new ticket). Audit-log entry for every admin action | every PR. CPU. < 5 min |
| Simulated class E2E (CPU) | scripted 12-student scenario: 12 uploads. 12 tickets. 12 submissions (launch config = pai1 only. Extended config = +node2/node3 to exercise multi-endpoint first-fit by priority, node pinning, and student-selectable gating). Queue drains in expected order. Kill/bump/requeue mid-drain. Demo-day 2-slot relay schedule (section 7.7). Backup + restore round-trip | nightly + per release |
| Hardware-gated (GPU) | `gpu_selftest` per node (via the agent, inside the container: torch/CUDA visible, load 7.5 B, 8-bit fwd/bwd, bf16 fwd/bwd where it fits, `predict_action` on held-out images, 10-step QLoRA). `home_selftest` (self-contained dev lab, ROCm gfx1151: `torch.cuda.is_available`, bf16 GEMM benchmark, load + predict, 10-step bf16 fwd/bwd. No bnb) | manual. Node join + stack changes |
| Robot (Pi) | token auth. LAN predict round-trip (via console). Stale-action watchdog stop. Slot lease expiry | per class milestone (M5-M7) |

## 12. Build plan

| Phase | When | Who | Scope | Exit |
|---|---|---|---|---|
| **0** | Now to about 1 wk | instructor + AI coder (assistant) | Home-box ROCm verify (`home_selftest`, self-contained dev lab). **Build runner image (cuda) + agent**. Provisioning (model download). Core (scheduler/budgets/QA/spec/config) test-first per section 11. Console scaffold (auth, users, nodes, upload + prep/QA, tickets, queue, SSE, UI, LAN bind) against local agent + FakeRunner. Admin surface section 9.7. systemd units. 12 accounts | 12 students log in on the campus LAN. They upload. They watch a simulated run. Teacher panel works end-to-end |
| **1** | First GPU window | AI coder. Teacher triggers | `gpu_selftest` on pai1 (via the agent, in-container). Blackwell stack pin in the image. `smoke`. 50-episode `crosscheck` | real training verified (class gate M5) |
| **2** | By Class 9 | AI coder | `robovla_train.py` integration. Validation harness. Merge/export. Inference slots + Pi token auth. Class board. Teacher panel complete. 5090 Ti + 4070 Ti join: agent installed on each host. Admin add -> probe -> enable when they come online (deferred. Not needed at launch) | console production-ready for the class |
| **3** | Extension week (Class 17) | **students** + AI coder | standalone capture app (PySide6. Vendors `episode.py`/`norm_stats.py`). Foxglove live-feed panel. QA-flag + board-view contributions | students own part of the instrument |

**Pedagogical integration:** the console *replaces* the "build the TUI" class activity. Classes 9-10: the console is live. Students operate it. They read `robovla_train.py` (12 sections, one per student). They run the crosscheck. They own the experiment tickets. With 12 individual students and 1 robot each, the console is the shared lab instrument. Like the eval harness.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Blackwell stack pinning (torch/tf/bnb/flash-attn) | Pinned in the runner image (tag in `fleet.yaml`). Early selftest at node join. SDPA fallback. Escape hatch = any host running the same image (section 6, section 10) |
| Runner heterogeneity (32 GB / 16 GB / iGPU) | Per-node batch presets + VRAM budget guard. Per-node selftest. 4070 Ti batch-1-only |
| Image change across nodes | Image tag pinned per node in `fleet.yaml`. Join probe verifies tag + selftest pass. Updates are explicit host-side ops with an "update available" flag (section 10) |
| Home box ROCm (gfx1151) friction | `home_selftest` first (self-contained dev lab). bf16-only (no bnb). Fallback: CPU torch for engine verification. The class never depends on it (off-LAN, not a cluster node) |
| No-preemption + 12 students. Queue pressure (single trainer at launch) | morning + overnight windows carry full runs (teacher batches. Console gates on free VRAM, section 3). Fast free smoke runs in class. Run-now = front-of-line after current completion. Pressure relieved by adding nodes via the admin page (config-only, section 8.2) |
| Demo-day 12-robot concurrency | 5 inference slots + pipelined relay (section 7.7). All robots cycle continuously |
| Agent dispatch failure / node death | Fail-fast. Auto-reschedule on next runner. Checkpoint resume (section 10) |
| Zombie VRAM | Agent reaps its job processes. Container stop frees the GPU. Post-job probe + agent-scoped purge (section 8.2) |
| Single-host (console) SPOF | Rebuild from repo + restore backup < 1 h. Running jobs continue (agent-owned, section 10). Episode data also on the 12 PIs |
| 12 robots in one room (practice weeks) | 4 sub-arenas x 3 students. Zone rotation (see lesson plan) |
| Config change (fleet.yaml vs. reality) | Single schema + CI validation + join probes + hot reload revalidates queued jobs |
| AI coder implementing from this doc | Section 11 TDD contract + test matrix are the acceptance criteria. Every admin action = endpoint + audit + test |

## 14. Open questions

1. **Wiring (later. Not needed for launch):** when the 5090 Ti and 4070 Ti machines come online, are they on the campus LAN (wired 1 GbE preferred)? Each needs a LAN IP for its agent endpoint (port 8601). Then it joins via the admin node registry (section 8.2).
2. **Demo day date + audience** (sizing the relay schedule. Judges/audience seating in the sub-arenas).
3. **Student laptop inputs:** keyboards only, or do some students have USB gamepads (capture app input design)?

## Appendix A - `fleet.yaml` (default, current fleet)

```yaml
version: 1
console:
  bind: 0.0.0.0            # pai1-only default (single LAN interface; section 9.8) - elsewhere bind the explicit LAN IP
  port: 8501
  tls: false               # optional self-signed via reverse proxy
  data_dir: /home/adam/robovla-console   # console state (DB + uploaded datasets), separate from node workdirs
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
    workdir: /home/adam/robovla        # host bind mount -> container's standard data layout
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

# Home Strix Halo (8060S, ROCm): off-campus. Not a cluster node (roboVLA is LAN-only).
# Runs the same robovla-runner-rocm image as a self-contained dev lab on the home LAN
# (console + agent + model, never exposed) for Phase 0 integration / home_selftest.
```

Notes: **pai1 is the only enabled node at launch**. node2/node3 are verified inventory. They are disabled until they come online. Adding them later is an admin-page operation (install agent -> add node -> probe -> enable, section 8.2). No code or redeploy. The home 8060S is intentionally absent (off-LAN, not a cluster node). It runs the same image as a self-contained dev lab (section 1). The 27B assistant on pai1 (about 29.5 GB) is **not** console-managed (section 3). The teacher unloads it before training/inference windows. The board shows free VRAM. `train_slots: 1` everywhere is the section 3 default. The teacher raises a 32 GB node to 2 via the admin lever (reduced-batch preset + budget guard apply). `priority` = first-fit order (higher first. Ties go to file order). `student_selectable` (default false) controls whether students can name a node at submission (section 9.3). `token_ref` = env var name on the console host holding the per-node agent token (shown once at node add. The same plaintext is in the agent container env. Argon2 hash in the console DB for audit/rotation. Plaintext never in `fleet.yaml` or logs). Placeholder IPs are filled in at wiring time (section 14 Q1). Code never hardcodes them.

## Appendix B - JobSpec JSON (the single contract)

The only artifact shared by UI, scheduler, runners, docs, and the crosscheck job. Validated by one schema in core. The agent derives a **fixed argv** from it (no shell).

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

`type` values: `smoke | train | validate | merge | infer | crosscheck | gpu_selftest | home_selftest`. `type: crosscheck` swaps the engine for the official `finetune.py` on a 50-episode subset (student-visible curve comparison). `max_seq_len` is fixed at 287 in the preset catalog (official OpenVLA context: 256 visual tokens + prompt/action text. Consistent with the model card's 256-bin x 7-token action space). It is not exposed to students. `resume_from` carries the checkpoint id for crash recovery (section 10).

## Appendix C - repo layout (for the implementing AI coder)

```
roboVLA/
+-- docs/                        # this spec + lesson plan + materials + model-card review
+-- agent/                       # runner agent - runs inside the per-node podman container
|   +-- server.py                # HTTP API :8601 - jobs/serve/vram/health; launches fixed-argv processes
|   +-- vram.py                  # nvidia-smi / rocm-smi probe (free VRAM + holding processes)
|   `-- pull.py                  # dataset bundle pull from the console (LAN HTTP)
+-- images/
|   +-- Dockerfile.cuda          # uv @ 3.12, torch cu128, bnb >= 0.46, openvla (pinned)
|   `-- Dockerfile.rocm          # ROCm gfx1151 build (dev lab)
+-- src/robovla/
|   +-- core/                    # PURE (no I/O; time injected) - unit-tested to >= 95 %
|   |   +-- scheduler.py         # FIFO, run-now, no preemption, priority + config-aware fit, transition table
|   |   +-- budgets.py           # VRAM budget calculator
|   |   +-- spec.py              # JobSpec JSON schema + validator + preset catalog
|   |   +-- config.py            # fleet.yaml schema + loader + diff/reload rules
|   |   +-- qa.py                # dataset QA rules
|   |   +-- norm_stats.py        # min/max stats + un-normalization
|   |   +-- split.py             # seeded held-out split (stable per student+task)
|   |   +-- ticket.py            # experiment-ticket schema
|   |   `-- audit.py             # audit reducer
|   +-- episode.py               # shared episode validator (vendored into capture app + agent)
|   +-- storage.py               # SQLite (WAL): users, sessions, tokens, datasets, tickets, jobs, artifacts
|   +-- fs.py                    # datasets/ runs/ artifacts/ layout, zip-slip-safe unzip (shared with the agent)
|   +-- proc.py                  # fixed-argv exec - used by the agent; never shell=True
|   +-- runner/
|   |   +-- interface.py         # Runner protocol: submit/poll/kill/tail/fetch_artifacts
|   |   +-- http.py              # HttpRunner - the only real implementation (console -> agent HTTP)
|   |   `-- fake.py              # FakeRunner (agent-protocol fakes; CPU tests)
|   +-- gpu.py                   # per-node VRAM view + orphan alert (via the agent)
|   +-- http/                    # FastAPI app
|   |   +-- app.py               # router wiring, rate limits, error shape (section 9.6)
|   |   +-- auth.py              # sessions, tokens, the ONE RBAC dependency
|   |   +-- datasets.py          # section 9.2
|   |   +-- runs.py              # section 9.3
|   |   +-- tickets.py           # section 9.3 ticket resource
|   |   +-- stream.py            # section 9.4 (SSE)
|   |   +-- infer.py             # section 9.5 (claims + /api/robot/predict proxy)
|   |   +-- admin.py             # section 9.7 (all audited)
|   |   `-- health.py            # /api/health
|   `-- cli.py                   # model_download, gpu_selftest (via agent), node bootstrap, backup, restore
+-- capture/                     # Phase 3: standalone PySide6 capture app (vendors core shared modules)
+-- ui/                          # vanilla JS + HTMX (readable by students)
+-- tests/
|   +-- unit/                    # core - every PR, < 60 s
|   +-- integration/             # service + FakeRunner - every PR, < 5 min
|   +-- e2e_class/               # scripted 12-student scenario - nightly
|   `-- hardware/                # gpu_selftest / home_selftest / robot - GPU_TESTS=1
+-- robovla-node                 # host-side script: install/update/restart the agent container (section 8.2, section 10)
+-- fleet.yaml                   # the fleet (section 10 / Appendix A)
+-- systemd/robovla-console.service
+-- systemd/robovla-runner.service   # per node: podman quadlet wrapping the runner container
`-- pyproject.toml               # uv-managed; image pins finalized after gpu_selftest (Phase 1)
```

Commands: `uv run pytest -m "not hardware"` (CI). `GPU_TESTS=1 uv run pytest -m hardware`. `uv run robovla cli gpu_selftest --node pai1`.
