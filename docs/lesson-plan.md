# Capstone Lesson Plan: "Teaching a 7B Robot"
## Fine-tuning OpenVLA-7B on a Mecanum Rover — 12 Weeks × 85-Minute Classes

**Companion docs:** `model-card-review.md` (fact-check of the original proposal) · `setup-and-materials.md` (BOM, network, pre-class-1 checklist)

---

## 0. Course frame

- **Duration:** 12 weeks, **2 classes/week = 24 sessions × 85 min** (assumes a standard 2-period/week lab elective. See §3 for 1×/week and 3×/week adaptations.)
- **Teams:** 4 teams × 4 students. Each team gets one mecanum rover (Pi 3 B + camera + 4 encoder wheels) and a shared slice of one GPU workstation (RTX 3090/4090 24 GB) or a cloud A100 slot.
- **The artifact:** a fine-tuned OpenVLA-7B LoRA checkpoint + a closed-loop autonomous run of the team's task, documented in a capstone report and defended in a live demo day.
- **Class template (85 min):**
  - `0:00–0:08` **Instructor presentation** (5–10 min, hard stop — the "mini-lecture" the plan is built on)
  - `0:08–1:25` **Project work** in timed blocks, with a **checkpoint demo every ~25 min** (one team, 90 seconds, live)
  - `1:22–1:25` **Wrap-up:** exit ticket (1 win / 1 blocker / 1 question) — collected and used to tune the next class.
- **Standing safety rules (every class, posted on the wall):** (1) physical kill switch within arm's reach; (2) cones define the boundary; (3) instructor go/no-go before any robot move; (4) autonomy runs only with the fail-safe stop (stale action > 500 ms → zero velocity) enabled.
- **Rotating roles** (swap every class so everyone drives, codes, and presents):
  - **Pilot** — teleoperates / runs the robot
  - **Data Engineer** — owns the logger, dataset QA, stats
  - **Systems Engineer** — owns network/GPU code, training runs
  - **Scribe** — owns the team folder, exit tickets, demo, report sections
- **Milestone gates** — students must pass a gate to unlock the next phase (gates are the *official* OpenVLA sanity checks where noted):

| Gate | Unlocked at | What "pass" looks like |
|---|---|---|
| **M1** Stream | Class 2 | 10 Hz JPEG stream Pi→workstation, measured one-way latency < 150 ms |
| **M2** Command chain | Class 4 | Scripted 3-DoF trajectory (square/circle) executed by robot with no teleop |
| **M3** Demo logger | Class 6 | Valid episodes (frames + actions + prompt, 10 Hz) **+ official sanity check 1: replay demo open-loop and succeed** |
| **M4** Dataset | Class 8 | ≥ 100 clean episodes, viewer flags anomalies, normalization stats JSON saved |
| **M5** Model load | Class 9 | `predict_action` smoke test on a still image returns a 7-dim action |
| **M6** Inference server | Class 11 | Server live **+ official sanity check 2: training images reproduce training accuracy** |
| **M7** Autonomy | Class 12 | First supervised closed-loop run completes |
| **M8** Demo day | Class 22 | Timed arena run in front of audience (fallback: teleop/baseline) |

---

## 1. Phase 1 — Architecture & Fleet Setup (Week 1)

### Class 1 — "Mission briefing: from hard-coded robot to AI robot"
**Goal:** teams have a live robot on the network and understand why the AI must live off-board.
- `0:00–0:08` **Presentation:** what a VLA is (image + sentence → action); 2 min of OpenVLA project videos; the numbers that justify the split architecture — *7.5B params × 2 bytes ≈ 15 GB vs. the Pi's 1 GB RAM* (students do this math on whiteboards); course roadmap and gates.
- `0:08–0:30` **Hardware audit (hands-on):** each team confirms: Pi boots (flash fresh Pi OS + ROS 2 Humble armv7l if needed), camera streams, wheel drivers respond, encoders spin. Teams with a clean board start the network.
- `0:30–0:35` *Checkpoint 1:* team with cleanest board posts "robot on Wi-Fi + static IP" to the class board.
- `0:35–1:10` **Bring the robot to life:** teams write `drive.py` — publish `geometry_msgs/Twist` from the keyboard → robot drives a figure-8 inside cones. Systems Engineer sets static IPs on the isolated 5 GHz SSID; baseline network test: `ping` + `iperf3` to the workstation, numbers recorded in the team folder.
- `1:10–1:15` *Checkpoint 2:* 90-second "hello robot" live drive by one team.
- `1:15–1:22` **Latency mini-lab:** all teams record ping p50/p99; quick class discussion: where can 100 ms go to die?
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** robot drives via keyboard over ROS; network baseline numbers in team folder; team roles assigned.
- **Engagement design:** whiteboard VRAM math, board race to first static IP, live 90-sec drives, role cards issued.

### Class 2 — "Wiring the brains: the Pi ↔ GPU communication layer"
**Goal:** M1 — a measured 10 Hz image stream to the workstation and a remote command path back.
- `0:00–0:08` **Presentation:** what crosses the wire (640×480 JPEG q80 ≈ 60–120 KB @ 10 Hz ≈ 1 Mbps — trivial; so why latency still matters); two transports (plain socket vs. the official `deploy.py` REST pattern — why we start simple); the concept of the "action server."
- `0:08–0:35` **Build the stream:** Pi-side `frame_sender.py` (V4L2/PiCamera → JPEG @ 10 Hz, timestamped, sequence-numbered); workstation-side `frame_receiver.py` (decodes, displays, records capture→arrival timestamps).
- `0:35–0:40` *Checkpoint 1:* first frame on the workstation screen; latency numbers posted.
- `0:40–1:10` **Build the command path:** workstation `twist_sender.py` → Pi node → robot. Teams drive the robot **from the workstation** (not the Pi keyboard) using `drive.py` values. Add the **stale-action watchdog**: no command in 500 ms → zero velocity (this is the fail-safe used for the rest of the course).
- `1:10–1:15` *Checkpoint 2:* one team drives across the room from the workstation with a deliberate 1-s pause to demo the watchdog stopping the robot.
- `1:15–1:22` **M1 test:** measure one-way latency (timestamp method), target < 150 ms; teams with > 150 ms diagnose (channel, encoding size, queueing).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M1):** stream + command path + watchdog, latency logged.
- **Engagement design:** "watchdog showcase" demos, latency leaderboard on the board, diagnosis challenges for slow teams.

---

## 2. Phase 2 — Mecanum Action Space & Kinematics (Week 2)

### Class 3 — "Mecanum math: from 3 commands to 4 wheels"
**Goal:** working IK node, unit-tested, with each axis verified on the robot.
- `0:00–0:10` **Presentation:** holonomic motion; derive the 4-wheel IK together (wheel velocity from `v_x, v_y, ω_z`, wheel radius + track geometry); why OpenVLA's 7-DoF *delta end-effector pose* means nothing to a mobile base → our **3-DoF velocity action space** `(v_x, v_y, ω_z)` is the project's key remap.
- `0:10–0:35` **Code the IK:** `mecanum_ik.py` / ROS node: Twist → 4 wheel setpoints. Hand-compute the 3 canonical cases (straight, strafe, spin) as unit tests before running anything.
- `0:35–0:40` *Checkpoint 1:* unit tests pass (teams show green test output).
- `0:40–1:10` **Axis-by-axis verification:** with the base on a bench/low speed, each team verifies: forward → all wheels equal same sign; right strafe → correct pattern; spin → alternating signs. Then on the floor: drive a clean square using only `v_x`/`v_y`/`ω_z` one axis at a time.
- `1:10–1:15` *Checkpoint 2:* "Kinematics bingo" — team completes the 3-axis card and demos the cleanest square.
- `1:15–1:22` **Tuning:** teams set wheel-radius/track-width constants from measured geometry; log any sign-flip bugs (the classic).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** `ik_node` with passing unit tests + recorded axis tests.
- **Engagement design:** derive-then-code (whiteboard first), kinematics bingo cards, sign-flip bug "war stories" at checkpoint.

### Class 4 — "Action space mapping: 7-DoF → 3-DoF, with stats"
**Goal:** M2 — a full command chain from a 3-Dof velocity command to the wheels; normalization design decided.
- `0:00–0:10` **Presentation:** the model card's action space (delta pose, 256 bins, *normalized* — must un-normalize with per-dataset stats). Design decision, discussed with tradeoffs: **keep 7-dim output** (map demos to `(vx, vy, ωz, 0,0,0,0)`, zero the arm dims, read first 3 dims at inference) vs. modifying the repo for a true 3-dim head (advanced track). Introduce `min/max` stats as the team's future deliverable.
- `0:10–0:35` **Build `action_adapter.py`** (both ends): server-side 7-dim → 3-dim with per-dim scaling; Pi-side 3-dim → IK → motors. Normalized velocity convention: `[-1, 1]` mapped to ≤ 0.3 m/s / ≤ 1.5 rad/s in autonomy.
- `0:35–0:40` *Checkpoint 1:* team commits the shared convention doc (units, ranges, signs) to the team folder.
- `0:40–1:10` **Command chain test harness:** a workstation script publishes a scripted trajectory (square → circle → S-curve) through the chain; robot executes. Teams tune scaling so the robot matches a 0.3 m/s target speed (verify with encoder odometry).
- `1:10–1:15` *Checkpoint 2:* 90-sec scripted run by one team — "the chain works, no teleop."
- `1:15–1:22` **M2 test + stretch:** drive the robot through a cone gate using only the command chain.
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M2):** scripted-trajectory run logged; `action_adapter` + convention doc.
- **Engagement design:** encode-odometry speed verification (engineering, not vibes), stretch gate challenge.

---

## 3. Phase 3 — Teleoperation & Dataset (Weeks 3–4)

### Class 5 — "Teleop stations & the demo logger"
**Goal:** teleop + logger built and dry-run validated.
- `0:00–0:10` **Presentation:** why demonstrations *are* the dataset (imitation learning, in one paragraph); the **four official data rules** projected on screen: ① 5–10 Hz, ② no pauses/idle actions, ③ diverse start poses, ④ consistent strategy; the Pilot/Engineer role split during collection.
- `0:10–0:35` **Build `teleop_node`:** gamepad/keyboard → smoothed 3-DoF velocity (rate-limited), with the hardware kill switch wired in as an immediate-zero.
- `0:35–0:40` *Checkpoint 1:* smoothest teleop demo (class votes on the buttery-smooth one).
- `0:40–1:10` **Build `data_logger`:** one folder per episode: `frames/` (JPEG @ exactly 10 Hz, timestamped), `actions.json` (t, vx, vy, ωz @ 10 Hz), `episode.json` (id, task, prompt, start pose, success flag, collector). Dry run: 5 short episodes.
- `1:10–1:15` *Checkpoint 2:* data-integrity audit — open a random episode, confirm N frames = N actions.
- `1:15–1:22` **Fix-and-retest:** teams with frame-rate drift fix it (capture thread, JPEG quality, buffer).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** teleop + logger, 5 valid dry-run episodes.
- **Engagement design:** "buttery teleop" vote, data-integrity audit as a game (find the mismatched episode).

### Class 6 — "First demo drives: task definition & collection v1"
**Goal:** M3 — first 20–30 real episodes + **official sanity check 1 (replay) passed**.
- `0:00–0:10` **Presentation:** the course tasks — primary task per team (e.g., *"drive forward and stop at the blue target"*, *"drive sideways through the gate and stop"*, stretch: a second task). Collection protocol: 3 start poses, slow and continuous, success/failure logged; why the prompt string matters (it's the model's instruction input).
- `0:10–0:40` **Collection round 1:** teams drive 10–15 episodes; Scribe logs each (start pose, success, notes); Data Engineer monitors logger live.
- `0:40–0:45` *Checkpoint 1:* episode tally posted on the board.
- `0:45–1:12` **Replay sanity check (GATE M3):** teams write `replay_episode.py` — open-loop playback of a recorded action sequence from the same start pose. Pass = robot completes the task without the logger on. Failing teams diagnose (sign flips, scaling, start-pose drift).
- `1:12–1:17` *Checkpoint 2:* team that passed M3 replays a demo live for the class.
- `1:17–1:22` **Collection round 2** (top teams) or targeted re-collection (teams still failing replay).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M3):** 20–30 episodes; replay pass logged.
- **Engagement design:** replay is the "aha" moment — the robot does what the human did; failing-replay debugging is the real exercise.

### Class 7 — "Data collection sprint"
**Goal:** 60–80 total clean episodes; diversity checklist in place.
- `0:00–0:05` **Presentation:** target numbers (100 clean demos is the official floor; aim 110+ so curation still leaves ≥ 100); the diversity checklist (start poses, lighting, robot heading, target position); what "good" teleop looks like vs. the four rules.
- `0:05–1:10` **Sprint:** 65 min of structured collection; roles rotate at `0:40` (fresh Pilot); instructor spot-checks raw episode folders live; Scribes maintain the running tally.
- `1:10–1:15` *Checkpoint 1:* tally posted; class sees the distribution across start poses.
- `1:15–1:22` **Quality pass:** teams delete/redo obvious failures (crashes, stalls, idle stretches) — with a rule: *only delete with a logged reason*.
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** 60–80 episodes, diversity checklist filled, deletion log.
- **Engagement design:** sprint format with a public tally, rotation keeps all 4 students driving/coding.

### Class 8 — "Data quality clinic: visualization & stats"
**Goal:** M4 — curated dataset + normalization stats JSON.
- `0:00–0:08` **Presentation:** bad data → confused model (one concrete example: an idle stretch teaches the model to stall); why we compute per-dim min/max stats from *our* data (the un-normalization story from the model card, now operational).
- `0:08–0:40` **Build `dataset_viewer.py`:** load an episode → plot `vx/vy/ωz` over time, strip of 8 frames, and **anomaly flags**: idle > 2 s (rule ②), jump > 0.5×max (sign/scale bug), frame count mismatch.
- `0:40–0:45` *Checkpoint 1:* each team shows their ugliest flagged episode.
- `0:45–1:15` **Curate + compute stats:** teams purge flagged episodes (target ≥ 100 remaining) and compute/save `norm_stats.json` (per-dim min/max over the whole task) — the exact numbers the training pipeline will use.
- `1:15–1:20` *Checkpoint 2 / M4 test:* viewer runs clean on the final set; stats JSON committed.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M4):** ≥ 100 clean episodes + `norm_stats.json` + one "bad episode" writeup.
- **Engagement design:** "ugliest episode" showcase, curation as data science (they own the numbers training will use).

---

## 4. Phase 4 — Fine-Tuning with LoRA (Week 5)

### Class 9 — "GPU lab: LoRA demystified + environment"
**Goal:** M5 — model loads and `predict_action` works on a still image.
- `0:00–0:10` **Presentation:** what LoRA actually does (low-rank adapters on attention/MLP weights — ~1% of parameters trained); QLoRA: 8-bit base (~4 GB) + adapters, why 24 GB VRAM works; the official `finetune.py` flags they'll touch (`lora_rank 32`, `learning_rate 5e-4`, `batch_size`/`grad_accumulation`); the dataset path decision: **custom PyTorch Dataset wrapper** (primary) vs. RLDS (advanced track — and the two notorious tfds/dlimp gotchas).
- `0:10–0:35` **Environment verification** (pre-imaged GPU box): activate conda env (Python 3.10, torch 2.2, transformers 4.40.1, flash-attn 2.5.5), clone `openvla`, confirm `import` works. Teams without a queue slot run the same steps on their own laptops (CPU smoke test).
- `0:35–0:40` *Checkpoint 1:* green import + repo tree screenshot in team folder.
- `0:40–1:12` **Smoke test (GATE M5):** load `openvla/openvla-7b` (bf16, `trust_remote_code=True`), run `predict_action` on a still image with `unnorm_key="bridge_orig"`, `do_sample=False` → verify a 7-dim action returns. Teams time it and note VRAM used.
- `1:12–1:17` *Checkpoint 2:* M5 screenshot posted (action vector + latency + VRAM).
- `1:17–1:22` **Write the training data loader:** wrapper that reads team episode folders → `(image, prompt, 7-dim action)` tensors using `norm_stats.json`.
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M5):** smoke-test log; working data loader on a sample batch.
- **Engagement design:** "who loads the model fastest" board; everyone touches the 7.5B model today (big-moment class).

### Class 10 — "Training day: first LoRA fine-tune"
**Goal:** first real training run launched and monitored; inference server scaffolded.
- `0:00–0:10` **Presentation:** what to watch (loss, action token accuracy); overfitting with only ~100 episodes (memorization risk — why we watch *training* loss and later test on novel poses); checkpoints & `save_steps`; expected wall time (1–3 h QLoRA on 24 GB; ~15–30 min A100 80 GB).
- `0:10–0:25` **Launch:** teams configure `finetune.py` (rank 32, lr 5e-4, batch 1–2 + grad-accum 8–16 for 24 GB QLoRA, their loader, their `unnorm_key`), sanity-check 1 batch, then **launch**.
- `0:25–0:30` *Checkpoint 1:* all teams' runs visible (wandb or local logs); loss on the board.
- `0:30–1:10` **While it trains — build the inference server scaffold** (this is the Phase 5 entry ticket): load base + LoRA adapter → `predict_action` → un-normalize with `norm_stats.json` → return 3-DoF action. Test on one training image (numbers should look sane vs. the recorded action).
- `1:10–1:15` *Checkpoint 2:* team with the lowest first-epoch loss posts its curve; class predicts which checkpoint will be best.
- `1:15–1:22` **Paper sprint:** each team reads one assigned section of the OpenVLA paper (data mixture / LoRA setup / results) and posts a 3-bullet summary.
- `1:22–1:25` **Wrap-up + exit ticket.** *(Runs that don't finish by class end save checkpoints; results are collected in Class 11.)*
- **Deliverable:** training run #1 in progress; inference server scaffold.
- **Engagement design:** loss-curve leaderboard, checkpoint prediction game, paper sprint (every student reads, every student posts bullets).

---

## 5. Phase 5 — Deployment & Closed-Loop (Week 6)

### Class 11 — "Serving the model: inference server + sanity check 2"
**Goal:** M6 — live server passes the **official sanity check 2**.
- `0:00–0:10` **Presentation:** results of run #1 (class's curves side by side); the `deploy.py` REST pattern vs. our socket server; the **closed-loop timing budget** — camera 40 ms + network 30 ms + inference 200–300 ms + network 30 ms ≈ **0.5–0.7 s per action (~2 Hz)**; therefore the Pi runs a 50–100 Hz motor loop that *holds the last commanded velocity* and slews to the new one (this is how a 2 Hz "brain" drives a smooth robot).
- `0:10–0:40` **Finish the server:** image + prompt in → 3-DoF action out (JSON over socket/REST); measure and log end-to-end latency; add request queueing + a "model busy" behavior.
- `0:40–0:45` *Checkpoint 1:* latency numbers posted per team.
- `0:45–1:12` **Sanity check 2 (GATE M6):** run 10 held-out training-set images through the *live* server exactly as the robot would, compare predicted actions to the recorded ones (token accuracy / L1); must reproduce training-level numbers. Failing teams debug the pipeline (crop! scaling! sign!). Meanwhile **Pi-side `autonomous_node`:** capture → request → hold/slew latest velocity → IK → motors, with max-velocity caps.
- `1:12–1:17` *Checkpoint 2 / M6 test:* M6 pass logged.
- `1:17–1:22` **Safety interlock:** HuskyLens target-confirm gate (autonomy only engages after target lock) or LiDAR boundary check — wire in whichever the team has; kill-switch dry run.
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M6):** server + sanity-check-2 log; autonomous node with caps + interlock.
- **Engagement design:** "crop! scaling! sign!" debugging is the classic VLA gotcha — teams that fail check 2 learn it the hard way (instructor pre-arms this).

### Class 12 — "First autonomous run (supervised)"
**Goal:** M7 — first closed-loop run, safely.
- `0:00–0:08` **Presentation:** end-to-end loop recap (one whiteboard diagram, no new theory); the safety protocol in force for autonomy; what a *good* first run looks like (steady, slow, recoverable — not fast).
- `0:08–0:15` **Safety brief:** kill-switch check, boundary cones set, walkie protocol (one human per team within 2 m), instructor go/no-go.
- `0:15–1:05` **First runs:** 5–10 trials per team in a small bounded area; peer safety observer between runs; **every failure goes in the incident log** (timestamp, image of the frame at failure, what the model commanded, hypothesis).
- `1:05–1:10` *Checkpoint 1:* M7 — teams that completed a clean run log it.
- `1:10–1:20` **Hallucination review:** each team shows the *worst* failure image; class debates why (target occluded? start pose never seen in data? latency?); teams pick their #1 failure mode.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M7):** first-run log + incident log opened.
- **Engagement design:** "hallucination review" is the most talked-about 10 minutes of the course — frame failures as the model's *behavior*, not just bugs.

---

## 6. Phase 6 — Evaluation & Iteration (Weeks 7–8)

### Class 13 — "Building the evaluation protocol"
**Goal:** an automated eval harness + a rule-based baseline.
- `0:00–0:10` **Presentation:** what makes this a *science* project: fixed protocol (5 start poses × 10 trials), metrics (success rate, time-to-target, path deviation), and a **baseline** — a hard-coded HuskyLens color-tracking controller (pre-built starter code provided) so students can answer: *does the 7B model actually beat the 30-line rule?*
- `0:10–0:45` **Build the harness:** `eval_run.py` — scripted trials from the 5 poses, CSV logging (success, time, path), timestamped video capture, auto-tag of the frame at failure.
- `0:45–0:50` *Checkpoint 1:* harness runs one trial end-to-end unattended.
- `0:50–1:20` **Baseline trials:** teams run the rule-based controller through the full protocol (50 trials) and post results.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** eval harness + baseline results table.
- **Engagement design:** the "can we beat the 30-line rule?" question drives everything from here; unattended trials = the robot works while students analyze.

### Class 14 — "OpenVLA trials & failure analysis"
**Goal:** first structured A/B results + a written experiment plan v2.
- `0:00–0:05` **Presentation:** reading a results table (success rate ± spread, don't cherry-pick poses); the failure taxonomy: wrong target / stuck-in-loop / stalled / latency-induced overshoot / data gap (pose never demoed).
- `0:05–0:50` **OpenVLA trials:** full protocol with the team's checkpoint (GPU queue managed — teams run while other teams analyze baseline data).
- `0:50–0:55` *Checkpoint 1:* first A/B numbers posted (OpenVLA vs. baseline, per pose).
- `0:55–1:20` **Failure analysis + experiment design:** teams tag every failure, find the dominant mode, and write a **one-page experiment plan v2** (the allowed levers: targeted data for failure poses; hyperparameters — rank 16 vs 32, 1 vs 3 epochs, lr; prompt changes; preprocessing).
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** A/B results table; experiment plan v2 (instructor pre-approves levers at checkpoint).
- **Engagement design:** students choose their own ablation (real experimental design); "dominant failure mode" debate.

### Class 15 — "Retraining: data v2 & hyperparameters"
**Goal:** training run #2 launched with the v2 plan.
- `0:00–0:08` **Presentation:** read the official *VLA Performance Troubleshooting* best-practices together (coverage, consistent strategy, 5–10 Hz, no pauses) — now they have data to apply it to; how to read a *worse* loss curve without panic (it might be correct — check the test set, not the train set).
- `0:08–0:40` **Targeted collection:** teams drive 20–30 episodes *specifically at their failure poses/conditions* (the data-coverage rule in action).
- `0:40–0:45` *Checkpoint 1:* v2 dataset diff posted (what's new vs. v1).
- `0:45–1:15` **Train run #2** (and, if GPU time allows, the one ablation chosen in Class 14). While training: prepare v2 eval scripts + refresh the incident log.
- `1:15–1:20` *Checkpoint 2:* v2 loss curves vs. v1 on the board.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** training run #2 in progress; targeted data logged.
- **Engagement design:** "we chose the fix" — iteration round 2 is where the capstone stops being a tutorial and becomes a project.

### Class 16 — "v2 trials & safety layer"
**Goal:** v1 vs. v2 vs. baseline, final numbers; safety interlock integrated.
- `0:00–0:08` **Presentation:** the safety layer's role in the *capstone story*: HuskyLens as a handoff gate (confirm target → release VLA) and/or LiDAR as an emergency boundary stop; why a VLA system needs a dumb, fast, trustworthy layer.
- `0:08–0:50` **v2 trials:** full protocol with checkpoint #2 (best of runs, per team's judgment).
- `0:50–0:55` *Checkpoint 1:* 3-way table posted: v1 vs. v2 vs. baseline, per pose.
- `0:55–1:20` **Integrate the interlock** into the autonomy stack (gate logic + emergency-stop test with the kill switch and a LiDAR/cone breach).
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** final 3-way results; interlock live in the stack.
- **Engagement design:** the 3-way table is the single number every team argues about; interlock "arm-wrestling" test (who trips the e-stop first).

---

## 7. Phase 7 — Extensions & Capstone Build (Week 9)

### Class 17 — "Extension lab (choose one)"
**Goal:** each team takes one stretch direction.
- `0:00–0:10` **Presentation:** the extension menu (pick one, 65 min to prototype):
  - **(a) Multi-task:** second task + prompt switching (same checkpoint, different instruction — does it work? why/why not?)
  - **(b) Generalization stress test:** novel target color/size, or clutter — where does it break?
  - **(c) Arena design & build** for demo day (start planning now)
  - **(d) "Where the field went":** OFT / FAST study — read the 2025 follow-up releases, benchmark the latency claims, write a 1-page "how OFT would change our course"
  - **(e) Simulation:** replay the dataset in a lightweight 2D sim; visualize policy behavior without touching the robot
- `0:10–1:15` **Build/test** the chosen extension; instructor circulates; checkpoints at `0:40` and `1:00` (90-sec progress demos).
- `1:15–1:22` **Extension demos:** 90 seconds each, live.
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** working (or honestly documented) prototype + 1-page note.
- **Engagement design:** choice = ownership; the (d) track feeds the "what's next" discussion at demo day.

### Class 18 — "Capstone arena design & build"
**Goal:** each team's demo-day arena is specified and built.
- `0:00–0:05` **Presentation:** demo-day constraints (audience, time limit per team, scoring rubric, and the **novelty challenge**: the arena must include at least one start pose or condition the model has *never seen* — this is the generalization test the whole course has been building toward).
- `0:05–1:10` **Design & build:** arena layout (cones, target, start zones incl. the novelty pose), task spec written in the prompt format, timing plan.
- `1:10–1:15` *Checkpoint 1:* instructor rubric walkthrough (autonomy performance 40% / scientific iteration 30% / presentation 30%).
- `1:15–1:22` **First dry run** in the new arena (supervised).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** built arena + task spec + novelty-pose definition.
- **Engagement design:** teams are building *their* stage; the novelty pose is a deliberate bet students get to make.

---

## 8. Phase 8 — Rehearsal, Documentation, Demo Day (Weeks 10–12)

### Class 19 — "Rehearsal 1 + capstone report"
**Goal:** full timed rehearsal; report 70% done.
- `0:00–0:05` **Presentation:** the rubric, again, concretely; what the report needs (architecture diagram, dataset stats + loss curves, 3-way eval table, failure analysis + fixes, what we'd change); "the process is the grade" — a failed run with great analysis out-scores a lucky run with none.
- `0:05–0:50` **Full rehearsal runs:** timed, in the arena, with the novelty pose, interlock live.
- `0:50–0:55` *Checkpoint 1:* rehearsal results posted (team's first "official" score).
- `0:55–1:22` **Report writing:** Scribes lead; each student owns a section; figures pulled straight from the team folder (loss curves, eval CSVs, incident log).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** rehearsal score; report 70%.

### Class 20 — "Rehearsal 2 + peer review"
**Goal:** external critique absorbed; final tweaks made.
- `0:00–0:05` **Presentation:** peer-review protocol (score with the rubric; give 3 *actionable* improvement notes; be specific: "your robot stalls at pose 3 because your data never shows it turning left" — not "make it better").
- `0:05–1:10` **Cross-team runs:** each team runs its arena *for* another team, who scores and notes.
- `1:10–1:15` *Checkpoint 1:* each team reads its 3 notes and decides which to act on before leaving.
- `1:15–1:22` **Act on notes:** one focused fix per team (data gap → 10 quick episodes; stall → tweak interlock; timing → shorten demo segment).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** peer-review scores; one implemented fix.
- **Engagement design:** scoring *someone else's* robot is the sharpest feedback in the course; students audit each other's data.

### Class 21 — "Final tuning & contingency plan"
**Goal:** locked model, locked arena, and a plan B.
- `0:00–0:05` **Presentation:** "what if the model fails in front of an audience?" — the graceful-fallback story (teleop or baseline takeover, framed honestly in the demo); model selection: best checkpoint by eval numbers, not by a lucky run.
- `0:05–1:05` **Final rehearsal + lock:** run the exact demo sequence 2–3×; then **lock** (checkpoint tagged, arena fixed, no more "quick tweaks" — the classic last-minute-change disaster); demo-day artifacts: live camera feed on screen, latency HUD, printed charts for the stand.
- `1:05–1:10` *Checkpoint 1:* each team verbalizes its contingency (instructor confirms).
- `1:10–1:22` **Report final + rehearsal of the 8-min presentation** (each student speaks at least once; Scribe runs the clicker).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** locked checkpoint + arena; rehearsal notes; report final draft.

### Class 22 — "DEMO DAY" 🎓 (audience: another class / families / staff)
**Goal:** M8 — the show.
- `0:00–0:10` **Welcome:** instructor sets the stage (2 min of what OpenVLA is; the 12-week arc in one slide).
- `0:10–0:70` **Team demos (4 × ~15 min):** timed arena run with the novelty pose → 8-min presentation (architecture, data, loss curves, 3-way results, failures & fixes) → 3 min Q&A (students answer; instructor only steps in on safety or license questions).
- `0:70–1:20` **Open Q&A / audience questions** (seeded: "what did the model cost in compute?", "what would you change?", "can the Pi ever run this itself?").
- `1:20–1:25` **Close:** thanks to audience; teams strike; students collect audience scorecards.
- **Deliverable (M8):** completed demo; audience scorecards collected.
- **Engagement design:** students run the entire show — emcees, demo ops, Q&A. The instructor's only job is safety sign-off.

### Class 23 — "Post-mortem: what the numbers say"
**Goal:** close the loop with data; finish the report.
- `0:00–0:10` **Presentation:** the class-wide results — every team's 3-way table side by side; where the models genuinely beat the 30-line rule and where they didn't; "where the field is going" (OFT: ~73 ms/action and 25–50× faster inference; FAST; action chunking; safety research) — the course's last lesson in intellectual honesty.
- `0:10–0:45` **Report finalization + individual reflection:** 1-page "what I learned about AI + robotics" (each student).
- `0:45–1:15` **Cross-team analysis:** pooled failure taxonomy across all teams (which failure mode was universal? which data rule mattered most?); teams finalize appendix (code index, dataset card, compute log).
- `1:15–1:22` *Checkpoint 1:* final report upload; instructor grades gate status (all 8 gates).
- `1:22–1:25` **Wrap-up + exit ticket (last one).**
- **Deliverable:** final capstone report + reflections.

### Class 24 — "Final presentations, awards & the next 15 GB"
**Goal:** the capstone defended in the room; the course closed.
- `0:00–0:52` **Presentations (4 × 8 min + 4 min Q&A):** the formal capstone defense — students present to each other + instructor (and any remaining audience).
- `0:52–1:10` **The big closing question (structured debate):** *"Could we run this without the cloud GPU?"* — the Pi 3 B has 1 GB; the model needs ~15 GB even 8-bit. Students work in teams on whiteboards: what hardware gets us closer? (NPUs, quantization, distillation, smaller VLAs, edge-cloud hybrids). No wrong answers; the point is doing the math like engineers.
- `1:10–1:18` **Awards** (student-voted + instructor): best generalization, best data pipeline, best failure analysis, smoothest teleop, best Q&A answer, most improved robot (v1 → v2).
- `1:18–1:25` **Close:** where to go next (OFT repo, LIBERO benchmark, Open X-Embodiment data, the OpenVLA paper's follow-ups); final reflection cards collected.
- **Deliverable:** course complete.

---

## 9. Standing engagement toolkit (used every class)

1. **Checkpoint demos every ~25 min** — one team, 90 seconds, live. Nobody hides for 85 minutes.
2. **The board** (whiteboard or static site): latency leaderboard, episode tally, loss curves, M1–M8 gate status. Updated by Scribes, read by everyone.
3. **Role rotation** weekly — everyone drives, codes, and presents by the end of the course.
4. **Exit tickets** (2 min, every class): 1 win / 1 blocker / 1 question. Instructor reviews before the next class; blockers get a call-out in the next presentation block.
5. **Instructor circulates, doesn't lecture**: after the 5–10 min presentation, the instructor is a roving consultant; the wall rules + gates do the steering.
6. **Framing failures as data**: every incident goes in the incident log with a frame, a command, and a hypothesis. "Hallucination reviews" (Classes 12, 14) are the most engaged 10 minutes of the course.
7. **Contingency class (Class 21)** and built-in slack (Classes 17–18) absorb GPU queueing, sickness, and hardware breaks without blowing up the demo-day date.

## 10. Grading rubric (capstone)

| Component | Weight | What's scored |
|---|---|---|
| Autonomous performance | 40% | Success rate on protocol (incl. novelty pose), time, path quality; v1→v2 improvement |
| Scientific iteration | 30% | Eval protocol rigor, failure analysis quality, experiment plan v2 execution, honest reporting (failed runs + good analysis score) |
| Presentation & communication | 30% | Demo day (8-min talk, Q&A, live run), report quality, individual reflection |
| Gates M1–M8 | Pass/fail prerequisite | All 8 gates must be evidenced in the team folder (M8 has a documented fallback: teleop/baseline takeover counts with reduced autonomy score) |

## 11. Adapting the schedule

- **3 classes/week (36 sessions):** add a third work block to Classes 1–8 (more collection/iteration), insert a dedicated simulation class (extension (e) promoted to core), and give 2 full extra rehearsal sessions in Weeks 10–11.
- **1 class/week (12 sessions):** each "class" above becomes a two-week unit; split its work blocks across the two 85-min meetings (meeting 1: build; meeting 2: test + gate). The 4-week data phase becomes 8 weeks — use the extra time for a second task.
- **Cloud GPU only:** training runs move to the morning before class or overnight; Classes 10/15 restructure around "queue at start, analyze while running." Budget: ~2 A100-hours per team per round → ~15–25 A100-hours for the whole course.
- **No LiDAR/HuskyLens:** use a second camera as the handoff gate (OpenCV color target on the Pi) — same pedagogical role, cheaper parts.
