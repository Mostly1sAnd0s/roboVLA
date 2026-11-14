# End-to-end recipe: robot → episodes → LoRA → live drive

*The full roboVLA loop, one command at a time, as actually run on the
2026-11-08 testbed (pi3 + a GTX 1070 console host). Two readers:
**advanced users** follow the steps top to bottom. An **AI coder**
building the UI equivalents should read every "UI:" note — each marks
what is already a clean API call, what the UI already does, and what
does not exist yet (do not invent around those; they are decisions).*

Conventions used below:

| token | meaning |
|---|---|
| `CONSOLE` | the console host's LAN IP (testbed: `192.168.68.71`) |
| `<task>` | dataset task slug, e.g. `drive-to-target` |
| `<user>` | student account id, e.g. `s01` |
| pi3 | the robot (SSH `adam@192.168.68.80` on the testbed) |

The one environment rule that has bitten twice: **every ROS command on
the robot runs with `ROS_DOMAIN_ID=99`.** Without it everything
"succeeds" into an empty room — the robot never hears you.

---

## 1. Bring up the console (the GPU host)

The console is a plain FastAPI process; nodes are podman agents. Both
were started once and live across sessions, so first check what is up:

```bash
curl -s http://CONSOLE:8501/api/health | python3 -m json.tool | head
```

If that fails, start the console (repo root, from `uv sync`-ed checkout):

```bash
cd ~/Documents/AI/roboVLA
PASCAL1_AGENT_TOKEN="<agent token>" \
  setsid nohup uv run robovla serve --fleet ~/robovla-dev/fleet.yaml \
  > ~/robovla-dev/console.log 2>&1 < /dev/null & disown
```

The agent container on a GPU node (testbed `pascal1`, bind mounts are
pilot-only — see README; the shipped podman quadlet unit is
`systemd/robovla-runner.service`, per-node values via its env file):

```bash
podman start robovla-pascal1            # if defined
# full launch: see README "Live console pilot" for the exact flags
```

Verify the node joined: `/api/health` shows `pascal1` `online: true`
with `free_gb` close to 8. A node offline = agent container down or its
`ROBOVLA_AGENT_TOKEN`/console URL mismatch.

**UI:** the browser console is at `http://CONSOLE:8501/ui/` (root `/`
redirects). Node health renders on the Nodes area; accounts are created
by a teacher on `accounts.html`.

## 2. Bring up the robot (pi3)

Canonical bringup — bridge + odom + sensors + TF, ONE stack per robot:

```bash
ssh adam@ROBOT_IP
setsid ~/rosbot_ws/tools/robot_up.sh > /tmp/robot.log 2>&1 < /dev/null & disown
```

Verify (each should report a steady rate):

```bash
source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=99
source ~/rosbot_ws/install/setup.bash
ros2 topic hz /scan          # LiDAR ~10 Hz
ros2 topic hz /serial_raw    # base telemetry ~40 Hz
```

Camera sanity (the whole capture path — raw read, GBRG debayer,
gray-world white balance, JPEG):

```bash
python3 -m rosbot_vla.camera --out /tmp/cam.jpg --gain 100
```

It prints median/blown/black and a verdict; point it at a **printed
colored target** and confirm the colors are the colors. (History: three
decode bugs — wrong mosaic, padded-buffer VGA, tail-instead-of-head
frame — were invisible to every numeric check except a colored object on
camera. Keep the target in the loop.)

Teleop check: run your keyboard/gamepad node (it must also carry domain
99) and drive a decimeter. Stop it before the next section — recording
wants `/cmd_vel` to belong to your teleop and *nothing else*.

**UI:** nothing here belongs in the browser — this is the physical
layer. The one UI-facing hook that exists: the Robot view page shows
live frames once any `vla_drive` runs, so it doubles as camera sanity.

## 3. Prepare an episode session

Decide four things before recording anything:

1. **Task + prompt.** The prompt stamped on every frame
   (`"drive to the red target"`) is what the policy will answer to at
   drive time. One task per dataset.
2. **Dataset id:** `<user>/<task>/vN` — increment `vN` per batch. QA
   enforces episode ids `ep_NNNN` (the recorder assigns these).
3. **Lighting.** The camera runs FIXED manual gain (auto-exposure is a
   no-op on this kernel; in VGA mode exposure is inert too — gain is the
   only lever). Pick a gain for the room (`--gain 100` suits daylight,
   200–400 for dim), and **record and drive at the same value** — the
   policy only knows the pixels it was trained on.
4. **Your account.** Student login on the console + a Pi token
   (`slots.html` → *My Pi token* → generate, shown once; the node reads
   it from `token_file:=/home/adam/pi_token.txt`).

Target counts: 10 episodes proves the pipeline; ~100 is where behavior
gets solid (QA warns under 100 — guidance, not a block).

## 4. Record episodes

On the robot, with teleop in another window (domain 99 everywhere):

```bash
cd ~/rosbot_ws && . /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=99
. install/setup.bash
ros2 run rosbot_vla record_episode --ros-args \
  -p duration_s:=15 -p out_dir:=/home/adam/vla_episodes
# add: -p prompt:="drive to the red target" -p gain:=200
```

Drive like the robot is trying to reach the goal: vary start distance
(0.5–2 m), put the target left/center/right between takes, and **end
every episode stopped with ~1 s of stillness**. Between keypresses the
teleop goes silent; the recorder timestamps silence as zero (the
staleness gate), so releases train stops — drive the way you want driven.

Ctrl-C (or the duration cap) finalizes. The node prints the exact
upload line; it zips in python (no `zip` binary needed):

```bash
cd ~/rosbot_ws/src/rosbot_vla
echo '<password>' | python3 -m rosbot_vla.upload \
  --episodes ~/vla_episodes/ep_0001 ~/vla_episodes/ep_0002 \
  --dataset <user>/<task>/v6 --user <user> --password-stdin
```

The reply is the QA verdict: `ready`, or the exact rule that failed.
Warnings worth knowing: `too_few_episodes` (<100, expected early) and
`idle_pause` (>2 s of dead air in a take — re-take or accept, it only
teaches pausing).

**UI:** upload exists ONLY as this CLI today. The API it speaks is
already the UI-ready shape: `POST /api/datasets/upload` with multipart
fields `dataset_id, total_chunks, chunk_index, sha256, file`, then
`POST /api/datasets/upload/finalize` `{dataset_id, total_chunks,
sha256}` — the QA verdict renders on `datasets.html` already; the
missing piece is just the file picker + chunk loop in JS.

## 5. Train — new LoRA

**A "new LoRA" is the normal case:** every train run starts fresh from
the base model (`lerobot/smolvla_base` on the Pascal node — the node's
`model:` pin in fleet.yaml decides it). Nothing to configure.

UI path (`new-run.html`): pick the ready dataset, write the ticket
(hypothesis + prediction — the scientific-record gate), submit, watch
loss on the run page. Terminal equivalent:

```bash
# ticket
curl -s -b "$COOKIES" -X POST http://CONSOLE:8501/api/tickets \
  -H 'Content-Type: application/json' \
  -d '{"hypothesis":"loss < 1.0 and the policy drives toward the disc",
       "config_summary":"smolvla fp32 batch1 epochs2",
       "prediction":"adapter loss < 1.0"}'
# run (node_id "auto" = scheduler places it on family-eligible nodes)
curl -s -b "$COOKIES" -X POST http://CONSOLE:8501/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"<user>/<task>/v6","job_type":"train",
       "ticket_id":"t_...","node_id":"auto",
       "config":{"batch":1,"epochs":2}}'
```

What the scheduler does with it: family-aware admission (a smolvla job
never lands on an openvla node), VRAM budget check, dispatch (the agent
pulls the dataset bundle, the wrapper builds fixed argv). On a done run
an **adapter artifact** (a full servable policy dir + `serving_meta.json`
with the dataset's normalization stats) is what later claims slots
against.

Monitor: run page SSE, or `GET /api/runs/{job_id}` +
`/api/runs/{job_id}/logs`. Expect ~4 steps/s, ~1.7 GB VRAM, 700-frame ×
2 epochs ≈ 7 minutes on the 1070.

**UI:** fully covered today.

## 6. Train — continuing an existing LoRA (read the honest state)

The spec carries `resume_from`, and it is implemented **for the openvla
family only**, for *crash resume within the same job*: the engine loads
`runs/<job_id>/ckpt/<name>` and continues (engine_jobs.py). The
SmolVLA wrapper does not wire resume — **on the Pascal node every run
starts from base, always.** Cross-job continuation (train A, then "run
B = A + 50 new demos") is not implemented for either family.

So today, "continue training" honestly means one of:

- **retrain from base on the merged dataset** (upload the new episodes
  as `v7`, retrain ~7 min) — the recommended path at this dataset
  scale, and the one that avoids the cataract of stacked LoRAs; or
- on an openvla node, rely on same-job crash-resume checkpointing.

**UI (AI coder):** do NOT add a "continue LoRA" button until the
smolvla wrapper implements it; a button that silently retrains from
base would lie. When the feature is asked for seriously, the work is:
`--policy.path`-style adapter preloading in `smolvla_finetune.py`
argv + spec plumbing (validate `resume_from` names a done run of the
same family/model owned by the student) + an "extend run X" affordance
on the run page. Design discussion first — it changes the artifact
story (adapter lineage).

## 7. Serve the policy to the robot

1. **Claim a slot** — `slots.html` → pick a *done* run → Claim (or
   `POST /api/infer/slots/claim {run_id}`). A claim IS a job: it queues
   behind the GPU fairly, gets its own id, and holds an infer budget.
   The agent loads the adapter and warms it in the background (first
   predict never pays the cold load).
2. **Open Robot view** (`robot.html`): per-student live frame + action
   arrow + sparkline, and the two live overrides — **prompt** and
   **sensor gain/exposure** — which ride back to the robot on its next
   telemetry tick (owner/teacher only, audit-logged). Release the slot
   from `slots.html` when done, or let the 300 s idle lease do it.
3. **Dry-run first. Always.** On the robot:

   ```bash
   ros2 run rosbot_vla vla_drive --ros-args -p dry_run:=true -p rate_hz:=2.0
   ```

   Real predicts, real Robot-view stream, zero `/cmd_vel` publishes.
   Confirm the arrows make sense for the scene in front of the camera.
4. **Live drive** (safety drill done, floor clear, human holds pickup
   rights, teleop reachable):

   ```bash
   ros2 run rosbot_vla vla_drive --ros-args -p dry_run:=false
   ```

   Safety envelope lives in `CommandPolicy`, not the model: velocity
   clamps (`max_vx` 0.15 m/s default), acceleration ramp, stale-command
   zeros. Ctrl-C publishes a hard zero. All four stop doors: release
   button, kill the slot job, lease expiry, Ctrl-C — none of them are
   the prompt. *"stop" does not work as a prompt unless stop was
   demonstrated; language is a suggestion, not a control plane.*

**UI:** claim/release/prompt/camera are wired end-to-end today. The gap
on this path is a **UI e-stop** (console→node, bypassing the model);
until it exists, stopping is physical — say so in the UI.

## 8. When something doesn't move — the five-minute triage

| symptom | first check | class of cause |
|---|---|---|
| robot ignores everything | `echo $ROS_DOMAIN_ID` in YOUR shell | domain ≠ 99 = publishing into an empty room |
| predict 403 `no active inference slot` | claim a slot; leases die at 300 s idle | serving lifecycle |
| predict 503/timeout once, then fine | already fixed (serve-warms); older images: retry | cold load used to ride the first step |
| run stuck queued, blocker `budget` | `nvidia-smi` — something off-books holds VRAM (lanes! leaked slots pre-fix) | GPU tenancy |
| black / magenta / wandering camera | `python3 -m rosbot_vla.camera` at the printed target; raise `--gain` | decode is tested — light is not |
| QA rejected an upload | read the issue: `ep_NNNN` ids, 10 Hz action steps | dataset contract |
| all frames pure black after rapid captures | reboot the Pi | CSI wedge (rapid one-shot cycling); the nodes hold ONE stream and are safe |

## 9. UI weave map (AI coder, the short version)

| terminal step | API today | UI today | gap |
|---|---|---|---|
| record | none (robot-local node) | — | remote start/stop via telemetry channel? open question |
| upload | `POST /api/datasets/{id}/upload` | CLI only | file picker |
| ticket + train | `/api/tickets`, `/api/runs` | `new-run.html` | none |
| continue LoRA | `resume_from` (openvla crash-resume only) | — | wrapper support + honest UX, see §6 |
| serve | `/api/infer/slots/*` | claim + release buttons | none |
| watch robot | `/api/robot/telemetry` + `/api/infer/robot-view` | Robot view page | — |
| prompt / gain | `slots/{id}/prompt`, `/camera` | page controls | none |
| drive | robot-side `vla_drive` | — | e-stop button (needs node-side subscribe), teleop-over-console are OPEN, design-first features |
| triage | `/api/health`, run logs | nodes + run pages | none |

Rules for the weave: every robot→console call stays a robot-initiated
HTTP call (the Pi runs no server, listens on no port); every
console→robot action rides the telemetry reply (one round trip, no new
channels); safety controls bypass the model and the prompt, always.
