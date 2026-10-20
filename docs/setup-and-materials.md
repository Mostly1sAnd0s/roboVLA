# Setup, Materials & Pre-Class-1 Checklist

Companion to `lesson-plan.md` (v1.1) and `model-card-review.md`; fleet and console detail lives in `training-console-design.md` v0.4 (authoritative).

**v1.1 (2026-10-20):** 12 individual students / 1 robot each · launch fleet = **pai1 only** (console host + RTX 5090 32 GB; the 5090 Ti / 4070 Ti / home Strix Halo join later via the admin node page — nothing here assumes them) · LAN-only console access from classroom devices.

## 1. Hardware bill of materials

### Per student (×12)
| Item | Spec / notes | Est. $ |
|---|---|---|
| Mecanum rover chassis | 4× mecanum wheels, 24V DC gear motors with encoders, motor driver board (dual H-bridge or 4× single), battery + switch — most students likely reuse existing rovers; line item is if buying new | 150–300 |
| **Raspberry Pi 3 B** (or 3 B+) | 32-bit OS (armv7l) for ROS 2 Humble; 1 GB RAM is fine — it only runs the data plane (10 Hz JPEG + IK + watchdog) | used/45 |
| Raspberry Pi Camera Module 2 (IMX219) | 640×480 JPEG @ 10 Hz is trivial for it; V2 works with the modern libcamera stack on Bookworm | 25–30 |
| USB gamepad (optional) | keyboard is day one; gamepad support lands with the capture app (Class 17) — **open question, design §14 Q4** | 15 |
| **HuskyLens** (optional but recommended) | handoff gate / color-target confirmation; USB on the Pi | 60–90 |
| 2D LiDAR (optional) | e.g., RPLIDAR A1/A2-class for emergency boundary stop; advanced students only | 100–180 |
| Cones, blue/green target (printable), masking tape | arena + collection fixtures (shared class stock) | 30 |
| Kill switch | physical momentary switch wired to the motor driver power path (or driver enable pin) — **instructor-approved design, tested before Class 12** | 10 |
| Student laptop | each student's own classroom device: teleop, stream viewer, and the console UI (browser). No local installs before Class 17. | existing |

### Classroom (×1)
| Item | Spec / notes | Est. $ |
|---|---|---|
| **Console host = pai1 (existing)** | RTX 5090 32 GB, 59 GB RAM, 1.7 TB disk. Hosts the console, QLoRA training, 2× 8-bit inference slots, and the 27B course assistant. **No purchase at launch.** (5090 Ti / 4070 Ti / Strix Halo = later, added via the admin node page — no redeploy.) | existing |
| Dedicated Wi-Fi router | **Isolated 5 GHz SSID**, fixed channel (36/149), WPA2; 12 Pis + student laptops only. 12 robots + 12 laptops on one SSID — wired 1 GbE for the console host preferred. The single most important "boring" purchase. | 80–150 |
| Spares (×2 of each) | Pi, camera, motor driver, batteries — 12 robots over 12 weeks makes hardware breaks certain | 300 |
| Display + capture | wall screen for demo day (live feeds + the console slot board); second monitor at the console | 100–200 |
| Whiteboard / class board | the console *is* the class board (queue, GPU state, slots, held-out accuracy, tickets); whiteboard for quick numbers | 50 |

**Rough totals:** reusing the existing 9 rovers → ~**$500–1,000** (router, spares, wall display, optional LiDAR/HuskyLens). Buying all 12 chassis new → ~**$2,500–4,500**.

## 2. Software stack (pinned versions)

**Console (pai1)** — Ubuntu, NVIDIA driver 615.71 (CUDA 13.4)
- **uv venv @ Python 3.12** (system Python is 3.14 — do not use it for the training stack)
- **torch 2.8/2.9 + cu128** — finalized by `gpu_selftest`. **Blackwell sm_120: the repo's old pins (torch 2.2/cu121, flash-attn 2.5.5) do NOT work** → SDPA attention instead of flash-attn unless it builds clean
- `bitsandbytes` ≥ 0.46 (QLoRA 8-bit), `transformers` pinned after selftest, `peft`, `timm`, `pillow`, `numpy`
- Console itself: FastAPI + SQLite + SSE (design §4), ~1.5–2k LOC, LAN-bound :8501
- `openvla` repo cloned at a pinned commit; `openvla/openvla-7b` checkpoint pre-downloaded (~15 GB)
- systemd: `robovla-console.service` + `ninfer.service` (wraps `~/start-ninfer.sh` for clean stop/start); release windows 07:00–08:30 + overnight

**Pi (ROS 2 side)** — Raspberry Pi OS (32-bit, Bookworm)
- ROS 2 **Humble** (armv7l builds; do *not* upgrade — later ROS 2 releases dropped 32-bit ARM)
- Python 3.11 (Pi default) — ROS nodes + `frame_sender.py`, `autonomous_node.py`, `teleop_node.py`, `data_logger.py`, `replay_episode.py`
- OpenCV (libcamera/PiCamera2 stack), `pygame` or `evdev` for gamepad

**Student laptops** — nothing to install for Classes 1–16 (browser + the console). **Class 17:** standalone capture app — **PySide6 + pygame**, PyInstaller single-file build; vendors the console's `episode.py` / `norm_stats.py` so capture-side and console-side QA can never drift (design §11).

## 3. Pre-class-1 prep checklist (instructor, 1 week out)

- [ ] pai1 venv built (uv @ 3.12, torch cu128, bnb ≥ 0.46); `openvla` cloned; **model checkpoint (~15 GB) downloaded to local disk** (demo must work on a bad-network day)
- [ ] `gpu_selftest` passed: load 7.5 B, 8-bit fwd/bwd, `predict_action` on a still image, timings logged — **you must have done M5 yourself before the students** (design Phase 1)
- [ ] Console live on the LAN: auth + **12 student accounts** + teacher, dataset upload/QA, queue, SSE with mock trainer, class board, teacher panel; verified from a classroom device
- [ ] `ninfer.service` wraps `~/start-ninfer.sh` (clean stop/start, ≥ 20 GB free check); release windows configured (07:00–08:30 + overnight)
- [ ] Isolated 5 GHz SSID live; ping Pi↔pai1 < 20 ms; `iperf3` > 100 Mbps; **static IPs assigned for all 12 Pis** (+ laptops if needed)
- [ ] All 12 Pis flashed (Bookworm 32-bit + ROS 2 Humble), cameras streaming, encoders verified, **kill-switch wiring reviewed**
- [ ] Baseline (HuskyLens color-tracking) starter code written and tested by you (it's the Class 13 comparison — students should never meet it for the first time during class)
- [ ] `frame_sender/receiver`, `twist_sender` reference snippets sketched (to hand out as *partial* code at Classes 2/4 — students finish them)
- [ ] Student folders created (one folder per student, mirroring their console account); class-level git repo for shared code incl. the `capture/` app
- [ ] Cones/targets printed; 4 sub-arenas × 3 measured; demo-day wall screen + slot board tested
- [ ] Exit-ticket collection set up (paper slips or a form); audience invite drafted for Class 22 (demo day)

## 4. Code inventory (what students build, when)

| Script | Lives on | Built in class |
|---|---|---|
| `drive.py` (keyboard → Twist) | Pi | 1 |
| `frame_sender.py` / `frame_receiver.py` (timestamped JPEG @ 10 Hz) | Pi / laptop | 2 |
| `twist_sender.py` + **stale-action watchdog** | laptop / Pi | 2 |
| `mecanum_ik.py` + `ik_node` (+ unit tests) | Pi | 3 |
| `action_adapter.py` (7↔3-DoF, scaling, both ends) + convention doc | both | 4 |
| `trajectory_test.py` (scripted square/circle/S) | laptop | 4 |
| `teleop_node.py` (keyboard/gamepad, smoothing, kill switch) | Pi | 5 |
| `data_logger.py` (frames + actions + `episode.json`) | Pi | 5 |
| `replay_episode.py` (open-loop demo replay — **sanity check 1**) | Pi | 6 |
| Dataset prep: QA report, `norm_stats.json`, held-out split | **console** (pre-built; students learn to *read* it) | 8 |
| Smoke + first QLoRA runs (UI + experiment ticket) | **console** | 9–10 |
| Crosscheck run (official `finetune.py`, 50-episode subset) | **console** | 10 |
| `robovla_train.py` read (12 sections, one per student) | console repo | 10 |
| Inference slots + `POST /api/robot/predict` + **sanity check 2** harness | **console** | 11 |
| `autonomous_node.py` (capture → predict → hold/slew → IK → motors) | Pi | 11 |
| Safety interlock (HuskyLens gate / LiDAR boundary) | Pi | 11–16 |
| `eval_run.py` (5 poses × 10 trials, CSV, video, failure frame) + baseline controller | both | 13 |
| **Standalone capture app** (PySide6: discovery, 10 Hz view, teleop, logging, pre-upload QA, one-click upload, replay check) — students own the QA panel, input, upload, board view | laptop | 17 |
| Extension prototypes | various | 17 |

**Suggested split of labor:** instructor provides *partial* implementations of `frame_sender/receiver`, `ik_node` skeleton, `action_adapter` skeleton, and the baseline controller — students complete, test, and extend them. Everything console-side (design doc v0.4) is pre-built by the AI coder; the class's job is to **operate and read** the instrument (Classes 9–10), then **own part of it** (the capture app, Class 17).

## 5. Network & control-loop timing budget (teach this in Class 2, enforce in Class 11)

```
Pi: capture+JPEG 640×480        ~ 30–50 ms
LAN: one-way (5 GHz, isolated)  ~  5–20 ms
Console: decode + 224×224 proc  ~ 10 ms
Console: OpenVLA inference      ~150–300 ms (pai1 32 GB, 8-bit slot)
LAN: action back                ~  5–20 ms
-------------------------------
End-to-end loop                 ~ 0.4–0.7 s  →  ~1.5–2.5 Hz new actions
```

Design consequence (the core control lesson): the Pi motor loop runs 50–100 Hz and **holds the last commanded velocity**, slewing (rate-limited) to each new command. Autonomy speed caps: ≤ 0.3 m/s linear, ≤ 1.5 rad/s. Stale action > 500 ms → zero velocity. This makes a 2 Hz stochastic "brain" drive a smooth, safe robot.

## 6. Data spec (teach in Class 5, audit in Classes 6–8)

- **Episode folder:** `ep_0042/{frames/*.jpg (10 Hz, timestamped), actions.json, episode.json}`
- **actions.json:** `[{t, vx, vy, wz}, ...]` at 10 Hz, velocities **normalized to [-1, 1]** against the caps above (store the raw values in `episode.json` too)
- **episode.json:** `{id, task, prompt, start_pose, success, pilot, notes, created}`
- **Action convention for training:** 7-dim `(vx, vy, wz, 0, 0, 0, 0)`; `norm_stats.json` = per-dim min/max over the task (dims 4–7 from the zeros) — computed server-side by the console during dataset prep
- **Prompt template:** `In: What action should the robot take to {task prompt}?\nOut:` (the model card's format)
- **Rules (official OpenVLA guidance):** 10 Hz; continuous slow motion — no pausing; 3+ start poses; one consistent driving strategy; ≥ 100 successful episodes; delete failures with a logged reason.

## 7. Demo-day concurrent inference plan (2 slots)

- pai1 serves **2× 8-bit inference slots** (robot port 8502; per-student Bearer token issued by the console). The slot board is visible to the class.
- **Practice weeks (Classes 11+):** the slot relay gives every robot 30–60 s of model-driven motion in rotation while the class works — all 12 robots cycling, nobody standing around.
- **Rehearsals (Classes 19–21):** timed runs + talks practiced pipelined, 2 robots at a time.
- **Demo day (Class 22):** ~6 min/student pipelined — 4-min timed arena run (novelty pose) **while the student gives the 3-min talk**, 1-min Q&A, next robot pre-staged on the second slot. 9 students ≈ 55 min; 12 ≈ 70 min — both fit the 85-minute block.
- **When the 5090 Ti / 4070 Ti come online** (admin node page: add → probe → enable): slots scale to 5; this plan is unchanged, with more parallelism and less relay waiting.
