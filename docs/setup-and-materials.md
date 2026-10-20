# Setup, Materials & Pre-Class-1 Checklist

Companion to `lesson-plan.md` and `model-card-review.md`.

## 1. Hardware bill of materials

### Per team (×4)
| Item | Spec / notes | Est. $ |
|---|---|---|
| Mecanum rover chassis | 4× mecanum wheels, 24V DC gear motors with encoders, motor driver board (dual H-bridge or 4× single), battery + switch | 150–300 |
| **Raspberry Pi 3 B** (or 3 B+) | 32-bit OS (armv7l) for ROS 2 Humble; 1 GB RAM is fine — it only runs the data plane (10 Hz JPEG + IK + watchdog) | used/45 |
| Raspberry Pi Camera Module 2 (IMX219) | 640×480 JPEG @ 10 Hz is trivial for it; V2 works with the modern libcamera stack on Bookworm | 25–30 |
| USB gamepad (1) | or a laptop keyboard per teleop station; gamepad → `evdev`/`python-xbox-controller` | 15 |
| **HuskyLens** (optional but recommended) | handoff gate / color-target confirmation; USB on the Pi | 60–90 |
| 2D LiDAR (optional) | e.g., RPLIDAR A1/A2-class for emergency boundary stop; advanced teams only | 100–180 |
| Cones, blue/green target (printable), masking tape | arena + collection fixtures | 30 |
| Kill switch | physical momentary switch wired to the motor driver power path (or driver enable pin) — **instructor-approved design, tested Class 12** | 10 |

### Classroom (×1)
| Item | Spec / notes | Est. $ |
|---|---|---|
| **GPU workstation** | RTX 3090/4090 **24 GB** (QLoRA training + inference server). Pre-image it (§3). | existing/2000 |
| — or — cloud GPU | A100 40/80 GB slots (~2 GPU-h per team per training round; ~15–25 total) | $1–2/h |
| Dedicated Wi-Fi router | **Isolated 5 GHz SSID**, no clients, fixed channel (36/149), WPA2; Pi + workstation + teleop laptop only. This is the single most important "boring" purchase. | 80–150 |
| Spare: Pi, camera, motor driver, gamepad, batteries | hardware breaks are a certainty over 12 weeks | 150 |
| Display + capture | second monitor on the workstation for the live feed; USB capture for demo-day wall screen (or just mirror the workstation) | 100–200 |
| Whiteboard / class board | latency leaderboard, episode tally, loss curves, M1–M8 gates | 50 |

**Total (4 teams, no LiDAR):** roughly **$1,600–2,200** if buying new chassis; less if reusing existing robots.

## 2. Software stack (pinned versions)

**Pi (ROS 2 side)** — Raspberry Pi OS (32-bit, Bookworm)
- ROS 2 **Humble** (armv7l builds; do *not* upgrade — later ROS 2 releases dropped 32-bit ARM)
- Python 3.11 (Pi default) — ROS nodes + `frame_sender.py`, `autonomous_node.py`, `teleop_node.py`, `data_logger.py`, `replay_episode.py`
- OpenCV (libcamera/PiCamera2 stack), `python-can`/pymotors per your driver, `pygame` or `evdev` for gamepad

**GPU workstation (inference/training)** — Ubuntu 22.04, CUDA 12.1
- conda env `openvla`: **Python 3.10, PyTorch 2.2.0 + cu121, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, flash-attn 2.5.5** (the repo's pinned, tested set — do not "update to latest"), `peft`, `bitsandbytes` (QLoRA), `pillow`, `msgpack`/`websockets` (transport), `matplotlib` (curves), optional `wandb` free tier
- `openvla` repo cloned at a pinned commit; `openvla/openvla-7b` checkpoint pre-downloaded (~15 GB)

## 3. Pre-class-1 prep checklist (instructor, 1 week out)

- [ ] GPU box imaged: conda env built, repo cloned, **model checkpoint downloaded and `predict_action` smoke-tested on a still image** (you must have done M5 yourself before the students)
- [ ] 15 GB checkpoint + repo on a local disk (not cloud-only) so the demo works on a bad-network day
- [ ] Isolated 5 GHz SSID live; Pings Pi↔workstation < 20 ms; `iperf3` > 100 Mbps
- [ ] All 4 Pi's flashed (Bookworm 32-bit + ROS 2 Humble), static IPs assigned, cameras streaming
- [ ] Chassis assembled, wheel drivers verified, encoders verified, **kill-switch wiring reviewed**
- [ ] Baseline (HuskyLens color-tracking) starter code written and tested by you (it's the Class 13 comparison — students should never meet it for the first time during class)
- [ ] `frame_sender/receiver`, `twist_sender` reference snippets sketched (to hand out as *partial* code at Classes 2/4 — students finish them)
- [ ] Roles assigned; team folders created (shared drive or git repo, one repo per team + a class-level `docs/` for the board)
- [ ] Cones/targets printed; arena footprint measured
- [ ] Exit-ticket collection set up (paper slips or a form)
- [ ] Audience invite drafted for Class 22 (demo day)

## 4. Code inventory (what teams build, when)

| Script | Lives on | Built in class |
|---|---|---|
| `drive.py` (keyboard → Twist) | Pi | 1 |
| `frame_sender.py` / `frame_receiver.py` (timestamped JPEG @ 10 Hz) | Pi / workstation | 2 |
| `twist_sender.py` + **stale-action watchdog** | workstation / Pi | 2 |
| `mecanum_ik.py` + `ik_node` (+ unit tests) | Pi | 3 |
| `action_adapter.py` (7↔3-DoF, scaling, both ends) + convention doc | both | 4 |
| `trajectory_test.py` (scripted square/circle/S) | workstation | 4 |
| `teleop_node.py` (gamepad, smoothing, kill switch) | Pi | 5 |
| `data_logger.py` (frames + actions + `episode.json`) | Pi | 5 |
| `replay_episode.py` (open-loop demo replay — **sanity check 1**) | Pi | 6 |
| `dataset_viewer.py` (plots, anomaly flags) + `norm_stats.json` | workstation | 8 |
| dataset loader (PyTorch wrapper → 7-dim actions) | workstation | 9–10 |
| `finetune` run (official `finetune.py`, QLoRA config) | workstation | 10, 15 |
| `inference_server.py` (image+prompt → 3-DoF action) + **sanity check 2** harness | workstation | 10–11 |
| `autonomous_node.py` (capture → request → hold/slew → IK → motors) | Pi | 11 |
| safety interlock (HuskyLens gate / LiDAR boundary) | Pi | 11–16 |
| `eval_run.py` (5 poses × 10 trials, CSV, video, failure frame) + baseline controller | both | 13 |
| extension prototypes | various | 17 |

**Suggested split of labor:** instructor provides *partial* implementations of `frame_sender/receiver`, `ik_node` skeleton, `action_adapter` skeleton, and the baseline controller — students complete, test, and extend them. Everything else is built by students from specs (the specs are the class presentations).

## 5. Network & control-loop timing budget (teach this in Class 2, enforce in Class 11)

```
Pi: capture+JPEG 640×480      ~ 30–50 ms
LAN: one-way (5 GHz, isolated)~  5–20 ms
Server: decode + 224×224 proc  ~ 10 ms
Server: OpenVLA inference      ~150–300 ms (24 GB, bf16/8-bit)
LAN: action back               ~  5–20 ms
-------------------------------
End-to-end loop                ~ 0.4–0.7 s  →  ~1.5–2.5 Hz new actions
```
Design consequence (the core control lesson): the Pi motor loop runs 50–100 Hz and **holds the last commanded velocity**, slewing (rate-limited) to each new command. Autonomy speed caps: ≤ 0.3 m/s linear, ≤ 1.5 rad/s. Stale action > 500 ms → zero velocity. This makes a 2 Hz stochastic "brain" drive a smooth, safe robot.

## 6. Data spec (teach in Class 5, audit in Classes 6–8)

- **Episode folder:** `ep_0042/{frames/*.jpg (10 Hz, timestamped), actions.json, episode.json}`
- **actions.json:** `[{t, vx, vy, wz}, ...]` at 10 Hz, velocities **normalized to [-1, 1]** against the caps above (store the raw values in `episode.json` too)
- **episode.json:** `{id, task, prompt, start_pose, success, pilot, notes, created}`
- **Action convention for training:** 7-dim `(vx, vy, wz, 0, 0, 0, 0)`; `norm_stats.json` = per-dim min/max over the task (dims 4–7 from the zeros)
- **Prompt template:** `In: What action should the robot take to {task prompt}?\nOut:` (the model card's format)
- **Rules (official OpenVLA guidance):** 10 Hz; continuous slow motion — no pausing; 3+ start poses; one consistent driving strategy; ≥ 100 successful episodes; delete failures with a logged reason.
