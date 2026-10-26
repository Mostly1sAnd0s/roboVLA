# Capstone Lesson Plan: "Teaching a 7B Robot"
## Fine-tuning OpenVLA-7B on a Mecanum Rover — 12 Weeks × 85-Minute Classes

**v1.3 (2026-10-26):** companion pointer → the frozen spec v0.5.1 (post-review clarifications only — no content drift from v1.2; the README ⚠️ flagging the 12-student revision is closed, it landed in v1.1).

**v1.2 (2026-10-21):** console spec refs bumped to v0.5 · gate M5 cites the smoke job's predict-action probe (design §6) · Class 11 slot-rotation arithmetic corrected.

**v1.1 (2026-10-20):** 12 individual students / 1 robot each (no teams) · training-console workflow replaces the TUI build (design doc v0.5) · launch fleet = **pai1 only** (1 trainer, 2 inference slots; more GPUs join later via the console admin page — nothing here depends on them).

**Companion docs:** `training-console-design.md` v0.5.1 (frozen; the console spec — the class *runs on* it) · `model-card-review.md` (fact-check of the original proposal) · `setup-and-materials.md` (BOM, network, pre-class-1 checklist)

---

## 0. Course frame

- **Duration:** 12 weeks, **2 classes/week = 24 sessions × 85 min** (see §11 for 1×/week and 3×/week adaptations).
- **Students:** up to **12 individuals** (currently 9), **1 robot each** (mecanum rover: Pi 3 B + camera + 4 encoder wheels). No teams — every student owns their robot, their dataset, their training runs, their inference slot, and their demo. Collaboration = peer review, rotating class support roles, the class board, and demo day.
- **The fleet (launch = pai1 only):** one console host + RTX 5090 (32 GB) that also runs the course assistant 27B. Full QLoRA runs (~1–2 h) go in the **overnight batch** (submit before end of class; results on the morning class board) or the **morning release window** (07:00–08:30). In-class GPU time = smoke runs (2–5 min), held-out validation, crosscheck, and the **2 inference slots**. No preemption; "run now" = front-of-line after the current job. Per-student limits: 1 running / 2 queued. More GPUs join later via the admin node page — the queue math in §4 is the only thing that changes.
- **The console is the shared lab instrument** (like the eval harness): students log in from classroom devices on the LAN — upload → QA report → experiment ticket → queue → live progress → held-out validation → artifacts → inference slots → class board. It replaces the "build your own TUI" activity; the *pedagogy* is operating and reading the instrument (Classes 9–10), not building it.
- **The artifact (per student):** a fine-tuned OpenVLA-7B LoRA checkpoint + a documented closed-loop autonomous run + a capstone report + a live demo-day run.
- **Class template (85 min):**
  - `0:00–0:08` **Presentation** (5–10 min, hard stop — the "mini-lecture" the plan is built on)
  - `0:08–1:25` **Project work** in timed blocks, with a **checkpoint demo every ~25 min** (one or two students, 90 seconds, live)
  - `1:22–1:25` **Wrap-up:** exit ticket (1 win / 1 blocker / 1 question) — collected and used to tune the next class.
- **Standing safety rules (every class, posted on the wall):** (1) physical kill switch within arm's reach; (2) cones define the boundary; (3) instructor go/no-go before any robot move; (4) autonomy runs only with the fail-safe stop (stale action > 500 ms → zero velocity) enabled.
- **Rotating class support roles** (2 students/week rotate; everyone pilots their own robot):
  - **Queue Wrangler** — watches the console queue + board, calls out ETAs, manages overnight submission order, flags stuck jobs
  - **Data QA Spot-Checker** — spot-checks raw episode folders against the console QA report before upload
  - **Safety Officer** — runs the safety brief, manages cones/kill-switch checks, owns the class incident log
  - **Pi Tech** — first responder for Pi/network issues, static IPs, spares
- **Milestone gates** — each student must pass a gate to unlock the next phase (gates are the *official* OpenVLA sanity checks where noted):

| Gate | Unlocked at | What "pass" looks like |
|---|---|---|
| **M1** Stream | Class 2 | 10 Hz JPEG stream Pi→laptop, measured one-way latency < 150 ms |
| **M2** Command chain | Class 4 | Scripted 3-DoF trajectory (square/circle) executed by robot with no teleop |
| **M3** Demo logger | Class 6 | Valid episodes (frames + actions + prompt, 10 Hz) **+ official sanity check 1: replay demo open-loop and succeed** |
| **M4** Dataset | Class 8 | Console dataset **"ready"**: ≥ 100 clean episodes survive curation, QA report read, `norm_stats.json` displayed by the console |
| **M5** Console + training verified | Class 9 | Student's smoke run completes **and** its predict-action probe returns a 7-dim action on a held-out image (shown in the job detail; design §6) |
| **M6** Inference slot | Class 11 | Student's adapter served on a slot **+ official sanity check 2: training images reproduce training accuracy** through the live slot |
| **M7** Autonomy | Class 12 | First supervised closed-loop run completes |
| **M8** Demo day | Class 22 | Timed arena run in front of audience (fallback: teleop/baseline) |

---

## 1. Phase 1 — Architecture & Fleet Setup (Week 1)

### Class 1 — "Mission briefing: from hard-coded robot to AI robot"
**Goal:** every student has a live robot on the network, a working console account, and an understanding of why the AI must live off-board.
- `0:00–0:08` **Presentation:** what a VLA is (image + sentence → action); 2 min of OpenVLA project videos; the numbers that justify the split architecture — *7.5B params × 2 bytes ≈ 15 GB vs. the Pi's 1 GB RAM* (students do this math on whiteboards); course roadmap and gates; **introduce the console** (students log in with their account, see the class board, the GPU state, and who is "holding" the GPU — the assistant).
- `0:08–0:30` **Hardware audit (hands-on):** each student confirms: Pi boots (fresh Pi OS + ROS 2 Humble armv7l if needed), camera streams, wheel drivers respond, encoders spin. Clean boards start the network (static IPs on the isolated 5 GHz SSID).
- `0:30–0:35` *Checkpoint 1:* first three "robot on Wi-Fi + static IP + console logged in" posted to the board (race, not team).
- `0:35–1:10` **Bring the robot to life:** students write `drive.py` — publish `geometry_msgs/Twist` from the keyboard → figure-8 inside cones. Baseline network test: `ping` + `iperf3` laptop↔Pi, numbers recorded in the student folder.
- `1:10–1:15` *Checkpoint 2:* 90-second "hello robot" live drives (two volunteers, back to back).
- `1:15–1:22` **Latency mini-lab:** everyone records ping p50/p99; quick class discussion: where can 100 ms go to die?
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** robot drives via keyboard over ROS; network baseline numbers in the student folder; console account working; class roles assigned.
- **Engagement design:** whiteboard VRAM math, board race to first static IP, live 90-sec drives, role cards issued.

### Class 2 — "Wiring the brains: the Pi ↔ laptop communication layer"
**Goal:** M1 — a measured 10 Hz image stream and a remote command path back.
- `0:00–0:08` **Presentation:** what crosses the wire (640×480 JPEG q80 ≈ 60–120 KB @ 10 Hz ≈ 1 Mbps — trivial; so why latency still matters); two transports (plain socket vs. the official `deploy.py` REST pattern — why we start simple); the concept of the "action server" (preview: it will live on pai1, not the laptop).
- `0:08–0:35` **Build the stream:** Pi-side `frame_sender.py` (V4L2/PiCamera → JPEG @ 10 Hz, timestamped, sequence-numbered); laptop-side `frame_receiver.py` (decodes, displays, records capture→arrival timestamps).
- `0:35–0:40` *Checkpoint 1:* first frame on the laptop screen; latency numbers posted.
- `0:40–1:10` **Build the command path:** laptop `twist_sender.py` → Pi node → robot. Students drive the robot **from the laptop** (not the Pi keyboard) using `drive.py` values. Add the **stale-action watchdog**: no command in 500 ms → zero velocity (the fail-safe used for the rest of the course).
- `1:10–1:15` *Checkpoint 2:* one student drives across the room from the laptop with a deliberate 1-s pause to demo the watchdog stopping the robot.
- `1:15–1:22` **M1 test:** measure one-way latency (timestamp method), target < 150 ms; students above 150 ms diagnose (channel, encoding size, queueing).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M1):** stream + command path + watchdog, latency logged.
- **Engagement design:** "watchdog showcase" demos, latency leaderboard on the board, diagnosis challenges.

---

## 2. Phase 2 — Mecanum Action Space & Kinematics (Week 2)

### Class 3 — "Mecanum math: from 3 commands to 4 wheels"
**Goal:** working IK node, unit-tested, with each axis verified on the student's own robot.
- `0:00–0:10` **Presentation:** holonomic motion; derive the 4-wheel IK together (wheel velocity from `v_x, v_y, ω_z`, wheel radius + track geometry); why OpenVLA's 7-DoF *delta end-effector pose* means nothing to a mobile base → our **3-DoF velocity action space** `(v_x, v_y, ω_z)` is the project's key remap.
- `0:10–0:35` **Code the IK:** `mecanum_ik.py` / ROS node: Twist → 4 wheel setpoints. Hand-compute the 3 canonical cases (straight, strafe, spin) as unit tests before running anything.
- `0:35–0:40` *Checkpoint 1:* unit tests pass (students show green test output — board race to first green).
- `0:40–1:10` **Axis-by-axis verification:** with the base on a bench/low speed, each student verifies: forward → all wheels equal same sign; right strafe → correct pattern; spin → alternating signs. Then on the floor: drive a clean square using only `v_x`/`v_y`/`ω_z` one axis at a time.
- `1:10–1:15` *Checkpoint 2:* "Kinematics bingo" — students complete the 3-axis card; the two cleanest squares demo.
- `1:15–1:22` **Tuning:** students set wheel-radius/track-width constants from measured geometry; log any sign-flip bugs (the classic).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** `ik_node` with passing unit tests + recorded axis tests.
- **Engagement design:** derive-then-code (whiteboard first), kinematics bingo cards, sign-flip bug "war stories" at checkpoint.

### Class 4 — "Action space mapping: 7-DoF → 3-DoF, with stats"
**Goal:** M2 — a full command chain from a 3-DoF velocity command to the wheels; normalization convention locked.
- `0:00–0:10` **Presentation:** the model card's action space (delta pose, 256 bins, *normalized* — must un-normalize with per-dataset stats). Design decision, discussed with tradeoffs: **keep 7-dim output** (map demos to `(vx, vy, ωz, 0,0,0,0)`, zero the arm dims, read first 3 dims at inference) vs. modifying the repo for a true 3-dim head (advanced track). Introduce `min/max` stats as the student's future deliverable — the console will compute them from their uploaded data.
- `0:10–0:35` **Build `action_adapter.py`** (both ends): server-side 7-dim → 3-dim with per-dim scaling; Pi-side 3-dim → IK → motors. Normalized velocity convention: `[-1, 1]` mapped to ≤ 0.3 m/s / ≤ 1.5 rad/s in autonomy.
- `0:35–0:40` *Checkpoint 1:* students commit the shared convention doc (units, ranges, signs) to their student folder (the whole class uses the same convention — post yours, diff against the class standard).
- `0:40–1:10` **Command chain test harness:** a laptop script publishes a scripted trajectory (square → circle → S-curve) through the chain; robot executes. Students tune scaling so the robot matches a 0.3 m/s target speed (verify with encoder odometry).
- `1:10–1:15` *Checkpoint 2:* 90-sec scripted run — "the chain works, no teleop" (two volunteers).
- `1:15–1:22` **M2 test + stretch:** drive the robot through a cone gate using only the command chain.
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M2):** scripted-trajectory run logged; `action_adapter` + convention doc.
- **Engagement design:** encoder-odometry speed verification (engineering, not vibes), stretch gate challenge.

---

## 3. Phase 3 — Teleoperation & Dataset (Weeks 3–4)

### Class 5 — "Teleop & the demo logger"
**Goal:** teleop + logger built and dry-run validated.
- `0:00–0:10` **Presentation:** why demonstrations *are* the dataset (imitation learning, in one paragraph); the **four official data rules** projected on screen: ① 5–10 Hz, ② no pauses/idle actions, ③ diverse start poses, ④ consistent strategy.
- `0:10–0:35` **Build `teleop_node`:** gamepad/keyboard → smoothed 3-DoF velocity (rate-limited), with the hardware kill switch wired in as an immediate-zero. (Keyboard is day one; gamepad support lands with the capture app, Class 17.)
- `0:35–0:40` *Checkpoint 1:* smoothest teleop demo (class votes on the buttery-smooth one).
- `0:40–1:10` **Build `data_logger`:** one folder per episode: `frames/` (JPEG @ exactly 10 Hz, timestamped), `actions.json` (t, vx, vy, ωz @ 10 Hz), `episode.json` (id, task, prompt, start pose, success flag, collector). Dry run: 5 short episodes.
- `1:10–1:15` *Checkpoint 2:* data-integrity audit — open a random episode, confirm N frames = N actions.
- `1:15–1:22` **Fix-and-retest:** students with frame-rate drift fix it (capture thread, JPEG quality, buffer).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** teleop + logger, 5 valid dry-run episodes.
- **Engagement design:** "buttery teleop" vote, data-integrity audit as a game (find the mismatched episode).

### Class 6 — "First demo drives: task definition & collection v1"
**Goal:** M3 — first 20–30 real episodes + **official sanity check 1 (replay) passed**.
- `0:00–0:10` **Presentation:** the course tasks — primary task per student (e.g., *"drive forward and stop at the blue target"*, *"drive sideways through the gate and stop"*; stretch: a second task). Collection protocol: 3 start poses, slow and continuous, success/failure logged; why the prompt string matters (it's the model's instruction input).
- `0:10–0:40` **Collection round 1:** each student drives 10–15 episodes; the Data QA Spot-Checker circulates with a raw-folder checklist; students log each episode (start pose, success, notes).
- `0:40–0:45` *Checkpoint 1:* episode tally posted on the board (per-student counts, live).
- `0:45–1:12` **Replay sanity check (GATE M3):** students write `replay_episode.py` — open-loop playback of a recorded action sequence from the same start pose. Pass = robot completes the task without the logger on. Failing students diagnose (sign flips, scaling, start-pose drift).
- `1:12–1:17` *Checkpoint 2:* students who passed M3 replay a demo live (two at a time — the relay keeps the floor busy).
- `1:17–1:22` **Collection round 2** (students ahead) or targeted re-collection (students still failing replay).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M3):** 20–30 episodes; replay pass logged.
- **Engagement design:** replay is the "aha" moment — the robot does what the human did; failing-replay debugging is the real exercise.

### Class 7 — "Data collection sprint"
**Goal:** 60–80 total clean episodes; diversity checklist in place.
- `0:00–0:05` **Presentation:** target numbers (100 clean demos is the official floor; aim 110+ so curation still leaves ≥ 100); the diversity checklist (start poses, lighting, robot heading, target position); what "good" teleop looks like vs. the four rules.
- `0:05–1:10` **Sprint:** 65 min of structured collection; the Safety Officer enforces zone rotation (4 sub-arenas × 3 students so 12 robots don't collide); instructor spot-checks raw episode folders live; the board shows the running tally.
- `1:10–1:15` *Checkpoint 1:* tally posted; class sees the distribution across start poses.
- `1:15–1:22` **Quality pass:** students delete/redo obvious failures (crashes, stalls, idle stretches) — with a rule: *only delete with a logged reason*.
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** 60–80 episodes, diversity checklist filled, deletion log.
- **Engagement design:** sprint format with a public per-student tally.

### Class 8 — "Data quality clinic: upload, QA, and stats"
**Goal:** M4 — console dataset **"ready"** (≥ 100 episodes, QA report read, norm stats displayed).
- `0:00–0:08` **Presentation:** bad data → confused model (one concrete example: an idle stretch teaches the model to stall); how the console's server-side prep works (per-episode validation, 224×224 via the official processor, `norm_stats.json` per-dim min/max, 15% seeded held-out split) — the un-normalization story from the model card, now operational.
- `0:08–0:40` **Upload + QA clinic:** students zip their episode folders → chunked upload → read the **QA report** (per-episode flags: frame/action mismatch, 10 Hz timing drift, idle > 2 s, sign/jump anomalies) → fix and re-upload as a new version (revalidate) until the dataset is **ready**.
- `0:40–0:45` *Checkpoint 1:* every student shows their ugliest flagged episode (from the QA report).
- `0:45–1:15` **Curate + verify:** purge/re-collect so ≥ 100 episodes survive; read the **norm stats page** (the exact numbers training will use); the Data QA Spot-Checker double-checks 3 students' raw folders against the report (they can never drift).
- `1:15–1:20` *Checkpoint 2 / M4 test:* all datasets carry the "ready" badge; stats page committed to the student folder.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M4):** console dataset **ready** + ≥ 100 episodes + one "bad episode" writeup from the QA report.
- **Engagement design:** "ugliest episode" showcase, curation as data science (they own the numbers training will use).

---

## 4. Phase 4 — Fine-Tuning with LoRA (Week 5) — *the console is live*

### Class 9 — "The console goes live: your data, your first run"
**Goal:** M5 — smoke run passes and each student's first full run is queued for the overnight.
- `0:00–0:10` **Presentation:** the console as the shared lab instrument — upload → QA → ticket → queue → live progress → validation → artifacts → slots → class board. **The queue economy:** 12 students, 1 trainer, no preemption — why the **experiment ticket** (hypothesis → config → prediction, required before submit) is the anti-waste rule, and why the overnight batch is the rhythm (honest queue math on the whiteboard: 12 × ~1.5 h ≈ 2 nights; results land in queue order).
- `0:10–0:30` **Upload + QA clinic (final):** any dataset not yet **ready** gets fixed and re-uploaded (new version); confirm the "ready" badge + norm stats page for all 12.
- `0:30–0:35` *Checkpoint 1:* all 12 "ready" badges on the board.
- `0:35–0:55` **First smoke run (GATE M5):** each student's `smoke` job (1 batch, 2–5 min) — queued, then watched live on their own SSE page (loss, VRAM, ETA, queue position). Pass = run completes **and** its predict-action probe returns a 7-dim action on a held-out image (visible on the live job page — no slot lease needed; design §6). (The stack itself was verified in Phase 1 by `gpu_selftest`; today the class verifies it with their own eyes.)
- `0:55–1:00` *Checkpoint 2:* the first smoke to finish posts its action vector + latency to the board.
- `1:00–1:20` **First full run, overnight:** write the experiment ticket (required), submit with safe defaults (8-bit, rank 32, α 32, lr 5e-4, batch preset — the advanced section stays closed), and the **Queue Wrangler** confirms the overnight submission order. Every student sees their queue position + ETA on the board.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M5):** dataset ready + smoke pass + experiment ticket + overnight submission.
- **Engagement design:** everyone watches their own live progress page today; the queue is visible and real; "whose run finishes first?" prediction game.

### Class 10 — "Training day: overnight results + the crosscheck"
**Goal:** read real training curves; verify the engine against the official script; own the trainer's code; decide run #2.
- `0:00–0:08` **Presentation:** reading a training curve (loss, action-token accuracy, LR schedule); overfitting with only ~100 episodes (memorization risk — watch the *held-out* accuracy, not just train loss); what "done" means (val plateau + ticket prediction vs. outcome).
- `0:08–0:35` **Overnight results review (as they land):** loss + held-out curves from the class board; **fill in the ticket outcome** (did the prediction hold?); students still in the queue do eval-harness scaffolding (Class 13) or targeted data collection.
- `0:35–0:40` *Checkpoint 1:* best held-out accuracy so far posted; the class predicts which queue entry will beat it.
- `0:40–1:00` **Crosscheck exercise:** the instructor triggers a console `crosscheck` job (the official `finetune.py` on a 50-episode subset of a representative dataset); the class compares curve *shape* (in-house engine vs. official script) — the instrument is verified, not a black box.
- `1:00–1:15` **Read the engine:** each student is assigned **1 of 12 sections** of `robovla_train.py` → 3-bullet summary posted to the class board. Every student now knows the trainer's internals.
- `1:15–1:22` **Run #2 decisions:** students who want changes (ticket v2: data fix, epochs, rank) submit for the next overnight; queue discipline enforced (1 running / 2 queued).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** results review + completed ticket outcome + engine section summary (+ optional run #2 submission).
- **Engagement design:** prediction game, crosscheck = "is the instrument honest?", 12-section read = everyone owns a piece of the engine.

---

## 5. Phase 5 — Deployment & Closed-Loop (Week 6)

### Class 11 — "Serving the model: your slot, your robot"
**Goal:** M6 — the student's adapter live on an inference slot; **official sanity check 2** passed.
- `0:00–0:08` **Presentation:** results so far (class board: best held-out accuracy per student); the slot model (2 slots, claim/release, lease, per-student Pi token); the **closed-loop timing budget** — camera 40 ms + LAN 30 ms + inference 150–300 ms ≈ **0.3–0.5 s per action (~2 Hz effective)**; therefore the Pi runs a 50–100 Hz motor loop that *holds the last commanded velocity* and slews to the new one (this is how a 2 Hz "brain" drives a smooth robot).
- `0:08–0:30` **Claim slots + wire the Pi:** `autonomous_node` on the Pi: capture → `POST /api/robot/predict` (student's token) → un-normalize with the dataset's `norm_stats` → 3-DoF → IK → motors, with max-velocity caps; stale-action watchdog (500 ms → zero). **Slot relay:** 2 robots live at a time, everyone else wires/debugs — the relay rotation gives every robot 30–60 s of model-driven motion while the class works ("everyone's robot is always moving").
- `0:30–0:35` *Checkpoint 1:* first robots driving under model control; relay rotation posted on the slot board.
- `0:35–1:05` **Sanity check 2 (GATE M6):** 10 held-out training images through the *live slot* exactly as the robot would send them (image + prompt → action), compared to the recorded actions (token accuracy / L1) — must reproduce training-level numbers. The 12 rotate through the slots two at a time — 10 images ≈ 30–45 s per student, so ~4 min of slot time, comfortably inside the 30-min block. The classic gotchas, pre-armed by the instructor: **crop! scaling! sign!** Meanwhile: Pi-side hold/slew tuning.
- `1:05–1:12` *Checkpoint 2 / M6 test:* M6 pass logged per student (two at a time on the floor; image checks queue through the slot board).
- `1:12–1:20` **Safety interlock:** HuskyLens target-confirm gate (autonomy only engages after target lock) or LiDAR boundary check — wire in whichever the student has; kill-switch dry run.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M6):** slot claim + sanity-check-2 log; autonomous node with caps + interlock.
- **Engagement design:** the relay keeps all 12 robots in motion; "crop! scaling! sign!" debugging is the classic VLA gotcha — students who fail check 2 learn it the hard way.

### Class 12 — "First autonomous run (supervised)"
**Goal:** M7 — first closed-loop run, safely.
- `0:00–0:08` **Presentation:** end-to-end loop recap (one whiteboard diagram, no new theory); the safety protocol in force for autonomy; what a *good* first run looks like (steady, slow, recoverable — not fast).
- `0:08–0:15` **Safety brief:** kill-switch check, boundary cones set, walkie protocol (each student within 2 m of their own robot), instructor + Safety Officer go/no-go.
- `0:15–1:05` **First runs:** slot relay — max 2 robots driving at a time, 5–10 trials per student in a small bounded area; between rotations, students analyze their own incident frames. **Every failure goes in the incident log** (timestamp, image of the frame at failure, what the model commanded, hypothesis).
- `1:05–1:10` *Checkpoint 1:* M7 — students who completed a clean run log it (board).
- `1:10–1:20` **Hallucination review:** each student shows their *worst* failure image; class debates why (target occluded? start pose never seen in data? latency?); students pick their #1 failure mode.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable (M7):** first-run log + incident log opened.
- **Engagement design:** "hallucination review" is the most talked-about 10 minutes of the course — frame failures as the model's *behavior*, not just bugs.

---

## 6. Phase 6 — Evaluation & Iteration (Weeks 7–8)

### Class 13 — "Building the evaluation protocol"
**Goal:** an automated eval harness + a rule-based baseline (per student).
- `0:00–0:10` **Presentation:** what makes this a *science* project: fixed protocol (5 start poses × 10 trials), metrics (success rate, time-to-target, path deviation), and a **baseline** — a hard-coded HuskyLens color-tracking controller (pre-built starter code provided) so students can answer: *does the 7B model actually beat the 30-line rule?*
- `0:10–0:45` **Build the harness:** `eval_run.py` — scripted trials from the 5 poses, CSV logging (success, time, path), timestamped video capture, auto-tag of the frame at failure.
- `0:45–0:50` *Checkpoint 1:* harness runs one trial end-to-end unattended.
- `0:50–1:20` **Baseline trials:** students run the rule-based controller through the protocol (slot relay for model-dependent steps; the baseline needs no slot); results posted per student.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** eval harness + baseline results table.
- **Engagement design:** the "can I beat the 30-line rule?" question drives everything from here; unattended trials = the robot works while students analyze.

### Class 14 — "OpenVLA trials & failure analysis"
**Goal:** first structured A/B results (per student) + a written experiment plan v2.
- `0:00–0:05` **Presentation:** reading a results table (success rate ± spread, don't cherry-pick poses); the failure taxonomy: wrong target / stuck-in-loop / stalled / latency-induced overshoot / data gap (pose never demoed).
- `0:05–0:50` **OpenVLA trials:** full protocol with the student's checkpoint (slot relay — queue managed by the Queue Wrangler; students run while other students analyze baseline data).
- `0:50–0:55` *Checkpoint 1:* first A/B numbers posted (OpenVLA vs. baseline, per pose, on the class board).
- `0:55–1:20` **Failure analysis + experiment design:** students tag every failure, find the dominant mode, and write a **one-page experiment plan v2** (the allowed levers: targeted data for failure poses; hyperparameters — rank 16 vs 32, 1 vs 3 epochs, lr; prompt changes; preprocessing).
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** A/B results table; experiment plan v2 (instructor pre-approves levers at checkpoint).
- **Engagement design:** students choose their own ablation (real experimental design); "dominant failure mode" debate.

### Class 15 — "Retraining: data v2 & hyperparameters"
**Goal:** training run #2 queued for the overnight with the v2 plan.
- `0:00–0:08` **Presentation:** read the official *VLA Performance Troubleshooting* best-practices together (coverage, consistent strategy, 5–10 Hz, no pauses) — now they have data to apply it to; how to read a *worse* loss curve without panic (it might be correct — check the test set, not the train set); queue etiquette for round 2 (tickets first, no "shotgun" runs).
- `0:08–0:40` **Targeted collection:** students drive 20–30 episodes *specifically at their failure poses/conditions* (the data-coverage rule in action); upload as a **v2 version** of their dataset.
- `0:40–0:45` *Checkpoint 1:* v2 dataset diff posted (what's new vs. v1).
- `0:45–1:15` **Run #2, queued for the overnight** (and, if the queue allows, the one ablation chosen in Class 14). While waiting: prepare v2 eval scripts + refresh the incident log.
- `1:15–1:20` *Checkpoint 2:* v2 tickets on the board; queue order confirmed by the Queue Wrangler.
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** run #2 submitted (ticket v2); targeted data logged.
- **Engagement design:** "we chose the fix" — iteration round 2 is where the capstone stops being a tutorial and becomes a project.

### Class 16 — "v2 trials & safety layer"
**Goal:** v1 vs. v2 vs. baseline, final numbers; safety interlock integrated.
- `0:00–0:08` **Presentation:** the safety layer's role in the *capstone story*: HuskyLens as a handoff gate (confirm target → release VLA) and/or LiDAR as an emergency boundary stop; why a VLA system needs a dumb, fast, trustworthy layer.
- `0:08–0:50` **v2 trials:** full protocol with checkpoint #2 (best of runs, per student's judgment) — slot relay, as in Class 14.
- `0:50–0:55` *Checkpoint 1:* 3-way table posted: v1 vs. v2 vs. baseline, per pose (class board, all 12).
- `0:55–1:20` **Integrate the interlock** into the autonomy stack (gate logic + emergency-stop test with the kill switch and a LiDAR/cone breach).
- `1:20–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** final 3-way results; interlock live in the stack.
- **Engagement design:** the 3-way table is the single number every student argues about; interlock "arm-wrestling" test (who trips the e-stop first).

---

## 7. Phase 7 — Extensions & Capstone Build (Week 9)

### Class 17 — "Building the instrument: the capture app (with the instructor)"
**Goal:** the class co-builds the **standalone capture app** (the shared instrument from design doc Phase 3) and each student picks a stretch direction.
- `0:00–0:10` **Presentation:** why the class builds its own instrument (the console is the lab's; the capture app is *theirs* — it runs on their laptops, vendors the console's `episode.py`/`norm_stats.py` so QA can never drift); the PySide6 stack (pure Python, readable, pip-installable); architecture: robot discovery (static IP list) → live 10 Hz view → keyboard/gamepad teleop → episode logging to the standard spec → built-in pre-upload QA → one-click upload to the console → open-loop "replay check".
- `0:10–0:45` **Guided build (instructor + students, on the laptops):** scaffold + robot discovery + live view; **students own pieces**: QA-flag panel, teleop input handling, upload button, board view — everyone commits something to the shared `capture/` repo.
- `0:45–0:50` *Checkpoint 1:* first student captures a real episode through the app and uploads it (QA report appears on the console — the loop is closed).
- `0:50–1:15` **Stretch menu (pick one, 25 min):** (a) multi-task prompt switching · (b) generalization stress test (novel target color/size or clutter — where does it break?) · (c) demo-day arena design start · (d) "where the field went" — OFT/FAST one-pager · (e) 2D simulation replay of your dataset.
- `1:15–1:22` **Stretch demos:** 90 seconds each, live (volunteers).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** working capture-app contributions (merged) + stretch prototype (or honest writeup).
- **Engagement design:** students own part of the instrument; the closed loop (app → console → QA report) is the class's moment of pride.

### Class 18 — "Capstone arena design & build"
**Goal:** each student's demo-day arena is specified and built.
- `0:00–0:05` **Presentation:** demo-day constraints (audience, ~4-min run per student, scoring rubric, and the **novelty challenge**: the arena must include at least one start pose or condition the model has *never seen* — this is the generalization test the whole course has been building toward).
- `0:05–1:10` **Design & build:** arena layout (cones, target, start zones incl. the novelty pose), task spec written in the prompt format, timing plan. 4 sub-arenas × 3 students.
- `1:10–1:15` *Checkpoint 1:* instructor rubric walkthrough (autonomy performance 40% / scientific iteration 30% / presentation 30%).
- `1:15–1:22` **First dry run** in the new arena (supervised; slot relay).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** built arena + task spec + novelty-pose definition.
- **Engagement design:** students are building *their* stage; the novelty pose is a deliberate bet students get to make.

---

## 8. Phase 8 — Rehearsal, Documentation, Demo Day (Weeks 10–12)

### Class 19 — "Rehearsal 1 + capstone report"
**Goal:** full timed rehearsal; report 70% done.
- `0:00–0:05` **Presentation:** the rubric, again, concretely; what the report needs (architecture diagram, dataset stats + loss curves from the console, 3-way eval table, failure analysis + fixes, what I'd change); "the process is the grade" — a failed run with great analysis out-scores a lucky run with none.
- `0:05–0:50` **Full rehearsal runs:** timed, in the arena, with the novelty pose, interlock live (slot relay — 2 at a time; others analyze their own run video).
- `0:50–0:55` *Checkpoint 1:* rehearsal results posted (each student's first "official" score).
- `0:55–1:22` **Report writing:** figures pulled straight from the console (loss curves, eval CSVs, incident log, class-board exports); each student owns their report sections.
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** rehearsal score; report 70%.

### Class 20 — "Rehearsal 2 + peer review"
**Goal:** external critique absorbed; final tweaks made.
- `0:00–0:05` **Presentation:** peer-review protocol (score with the rubric; give 3 *actionable* improvement notes; be specific: "your robot stalls at pose 3 because your data never shows it turning left" — not "make it better").
- `0:05–1:10` **Cross-student runs + data audit:** the 2 slots run a rotation of students' demos while peers score and note (6 pairs); in parallel, students **audit each other's datasets and incident logs** (paired, 10 min swap).
- `1:10–1:15` *Checkpoint 1:* each student reads their 3 notes and decides which to act on before leaving.
- `1:15–1:22` **Act on notes:** one focused fix per student (data gap → 10 quick episodes + v3 overnight if the queue allows; stall → tweak interlock; timing → shorten demo segment).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** peer-review scores; one implemented fix.
- **Engagement design:** scoring *someone else's* robot is the sharpest feedback in the course; students audit each other's data.

### Class 21 — "Final tuning & contingency plan"
**Goal:** locked model, locked arena, and a plan B.
- `0:00–0:05` **Presentation:** "what if the model fails in front of an audience?" — the graceful-fallback story (teleop or baseline takeover, framed honestly in the demo); model selection: best checkpoint by eval numbers, not by a lucky run.
- `0:05–1:05` **Final rehearsal + lock:** run the exact demo sequence 2–3× (slot relay); then **lock** (checkpoint tagged, arena fixed, no more "quick tweaks" — the classic last-minute-change disaster); demo-day artifacts: live camera feed on screen, latency HUD, printed charts for the stand.
- `1:05–1:10` *Checkpoint 1:* each student verbalizes their contingency (instructor confirms).
- `1:10–1:22` **Report final + rehearsal of the 3-min lightning talk** (each student presents their 3 slides: data, curves, result + the one thing I'd change).
- `1:22–1:25` **Wrap-up + exit ticket.**
- **Deliverable:** locked checkpoint + arena; rehearsal notes; report final draft.

### Class 22 — "DEMO DAY" 🎓 (audience: another class / families / staff)
**Goal:** M8 — the show. **12 robots, 2 slots, pipelined.**
- `0:00–0:05` **Welcome:** instructor sets the stage (2 min of what OpenVLA is; the 12-week arc in one slide; the relay explained for the audience).
- `0:05–1:10` **Student demos (~6 min each, pipelined):** each student's robot runs its ~4-min timed arena run (novelty pose included) **while the student gives the 3-min talk** (architecture, data, loss curves, 3-way results, failures & fixes); 1-min Q&A; the *next* robot is pre-staged on the second slot so the floor never goes quiet. 9 students ≈ 55 min · 12 students ≈ 70 min — both fit. The slot board is visible to the class.
- `1:10–1:20` **Open Q&A / audience questions** (seeded: "what did the model cost in compute?", "what would you change?", "can the Pi ever run this itself?").
- `1:20–1:25` **Close:** thanks to audience; robots strike to sub-arenas; students collect audience scorecards.
- **Deliverable (M8):** completed demo; audience scorecards collected.
- **Engagement design:** students run the entire show — emcees, demo ops, Q&A. The instructor's only job is safety sign-off.

### Class 23 — "Post-mortem: what the numbers say"
**Goal:** close the loop with data; finish the report.
- `0:00–0:10` **Presentation:** the class-wide results — all 12 students' 3-way tables side by side on the class board; where the models genuinely beat the 30-line rule and where they didn't; "where the field is going" (OFT: ~73 ms/action and 25–50× faster inference; FAST; action chunking; safety research) — the course's last lesson in intellectual honesty.
- `0:10–0:45` **Report finalization + individual reflection:** 1-page "what I learned about AI + robotics" (each student).
- `0:45–1:15` **Pooled analysis:** class failure taxonomy across all 12 (which failure mode was universal? which data rule mattered most?); students finalize the appendix (code index, dataset card, compute log from the console).
- `1:15–1:22` *Checkpoint 1:* final report upload; instructor grades gate status (all 8 gates, per student).
- `1:22–1:25` **Wrap-up + exit ticket (last one).**
- **Deliverable:** final capstone report + reflection.

### Class 24 — "Final defenses, awards & the next 15 GB"
**Goal:** the capstone defended in the room; the course closed.
- `0:00–0:48` **Lightning defenses (12 × 4 min):** the formal capstone defense — 3-min talk (data, curves, result, one thing I'd change) + 1-min Q&A (peers ask; instructor only on safety or license).
- `0:48–1:08` **The big closing question (structured debate):** *"Could we run this without the campus GPU?"* — the Pi 3 B has 1 GB; the model needs ~15 GB even 8-bit. Students work in groups of 3–4 on whiteboards: what hardware gets us closer? (NPUs, quantization, distillation, smaller VLAs, edge-cloud hybrids, the Strix Halo's 128 GB unified memory as a real example). No wrong answers; the point is doing the math like engineers.
- `1:08–1:16` **Awards** (student-voted + instructor): best generalization, best data pipeline, best failure analysis, smoothest teleop, best Q&A answer, most improved robot (v1 → v2).
- `1:16–1:25` **Close:** where to go next (OFT repo, LIBERO benchmark, Open X-Embodiment data, the OpenVLA paper's follow-ups); final reflection cards collected.
- **Deliverable:** course complete.

---

## 9. Standing engagement toolkit (used every class)

1. **Checkpoint demos every ~25 min** — one or two students, 90 seconds, live. Nobody hides for 85 minutes.
2. **The class board = the console** (queue positions + ETAs, GPU state, slot board, per-student dataset stats, best held-out accuracy, tickets) + whiteboard for quick numbers. The Queue Wrangler and Safety Officer keep it current; everyone reads it.
3. **Rotating class support roles** weekly — everyone serves on the instrument (queue, QA, safety, Pi) by the end of the course, while always piloting their own robot.
4. **Exit tickets** (2 min, every class): 1 win / 1 blocker / 1 question. Instructor reviews before the next class; blockers get a call-out in the next presentation block.
5. **Instructor circulates, doesn't lecture**: after the 5–10 min presentation, the instructor is a roving consultant; the wall rules + gates do the steering.
6. **Framing failures as data**: every incident goes in the per-student incident log with a frame, a command, and a hypothesis. "Hallucination reviews" (Classes 12, 14) are the most engaged 10 minutes of the course.
7. **The inference relay** (Classes 11 onward): 2 slots → every robot gets 30–60 s of model-driven motion in rotation while the class works. No standing around.
8. **Queue transparency**: the queue is the curriculum. ETAs, positions, and the overnight submission order are visible to all 12 — students learn real compute economics (12 × ~1.5 h on one GPU), not just ML.
9. **Contingency class (Class 21)** and built-in slack (Classes 17–18) absorb queue delays, sickness, and hardware breaks without blowing up the demo-day date.

## 10. Grading rubric (individual)

| Component | Weight | What's scored |
|---|---|---|
| Autonomous performance | 40% | Success rate on protocol (incl. novelty pose), time, path quality; v1→v2 improvement |
| Scientific iteration | 30% | Eval protocol rigor, failure analysis quality, experiment tickets (hypothesis → outcome) executed honestly, targeted retraining done for a *reason*; contributions to shared instruments (capture app, board) |
| Presentation & communication | 30% | Demo day (timed run + lightning talk + Q&A), report quality, individual reflection |
| Gates M1–M8 | Pass/fail prerequisite | All 8 gates must be evidenced in the student folder / console (M8 has a documented fallback: teleop/baseline takeover counts with reduced autonomy score) |

## 11. Adapting the schedule

- **3 classes/week (36 sessions):** add a third work block to Classes 1–8 (more collection/iteration), promote a stretch direction to a core class, and give 2 full extra rehearsal sessions in Weeks 10–11.
- **1 class/week (12 sessions):** each "class" above becomes a two-week unit; split its work blocks across the two 85-min meetings (meeting 1: build; meeting 2: test + gate). The 4-week data phase becomes 8 weeks — use the extra time for a second task. The overnight-batch rhythm is *easier* this way (one submission per week, results next meeting).
- **9 students (current):** queue math improves (~9 × 1.5 h ≈ 1.5 nights); demo day ≈ 55 min; sub-arenas drop to 3 × 3.
- **Console host down / pai1 maintenance:** runs resume from checkpoints; the class falls back to CPU-side work (QA curation, eval CSVs, reports, data collection). The design doc's escape hatch — adding a node via the admin page (LAN machine or cloud) — restores the queue without touching student workflow.
- **No LiDAR/HuskyLens:** use a second camera as the handoff gate (OpenCV color target on the Pi) — same pedagogical role, cheaper parts.
