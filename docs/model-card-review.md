# OpenVLA-7B Model Card Review & Proposal Fact-Check

**Reviewed sources:**
- Model card: https://huggingface.co/openvla/openvla-7b (raw `README.md`)
- Official repo: https://github.com/openvla/openvla (README, incl. *Fine-Tuning via LoRA*, *VLA Performance Troubleshooting*, *Evaluating OpenVLA*)
- Paper: Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model* (arXiv:2406.09246)
- Latency cross-checks from third-party benchmarking (AY-Robots OpenVLA arena, Spheron deploy guide)

---

## 1. What the model card & official repo actually say

| Item | Fact |
|---|---|
| **Model** | "OpenVLA 7B" = **7.5B parameters** (the repo says "all 7.5 billion parameters"). Built on the Prismatic `prism-dinosiglip-224px` VLM: fused **DINOv2 ViT-L/14 + SigLIP ViT-So400M/14** vision encoders + **Llama-2 7B** language model. |
| **Pretraining** | **970K real robot episodes** from Open X-Embodiment (Bridge V2, RT-1 X, BC-Z, Fractal, etc. — "Open-X Magic Soup++"). |
| **Input** | One 224×224 camera image + a natural-language instruction (prompt template: `"In: What action should the robot take to {instruction}?\nOut:"`). |
| **Output** | A **7-DoF end-effector *delta* pose** `(x, y, z, roll, pitch, yaw, gripper)`, each dimension **discretized into 256 bins** (7 action tokens per timestep). Outputs are **normalized** and must be **un-normalized with per-robot, per-dataset statistics** before execution. |
| **Zero-shot scope** | Works out-of-the-box only for embodiment/task combos **in the pretraining mix** (e.g., WidowX on Bridge V2). **Explicitly does NOT zero-shot generalize to unseen robots** — the card's own guidance: collect demos and fine-tune. |
| **Fine-tuning** | `vla-scripts/finetune.py` with PEFT **LoRA** (example: `lora_rank 32`, `learning_rate 5e-4`). Official example: single A100 80 GB (batch 16 ≈ 72 GB VRAM). Repo: "You can also fine-tune with a smaller GPU, as long as it has **at least ~27 GB** of memory, by modifying the batch size." QLoRA (8-bit) fits in 16–24 GB. Custom datasets: convert to **RLDS** (recommended path) **or** write a **custom PyTorch Dataset wrapper**. |
| **Data volume** | Official troubleshooting doc: "**OpenVLA typically requires fine-tuning on a small demonstration dataset (~100 demos) from your target domain robot.**" |
| **Data quality rules (official)** | (1) Collect at **5–10 Hz** — the model was *not* trained with action chunking and struggles with high-frequency data. (2) **Avoid pauses/idle actions** — the model can "get stuck" on near-zero actions at inference. (3) **Ensure data coverage** — include the variety of initial conditions you'll test. (4) **Use consistent task strategies** — less multi-modal data is easier to learn. |
| **Official sanity checks** | (1) **Replay** recorded demo actions on the robot (open-loop) and confirm the task succeeds. (2) After fine-tuning, **feed fine-tuning-set images into the inference pipeline** and verify you reproduce training token-accuracy/L1 error. |
| **Inference compute** | Weights ≈ **15 GB** (bf16). Measured: **~150 ms/action on A100**, **~6 Hz on RTX 4090** (~150–300 ms per action). One action per image call — no temporal context, no action chunking. |
| **Image pipeline** | Fine-tuning uses **random 90%-area crop augmentation**; at test time use the **center 90% crop** (`--center_crop True` in their eval scripts). Use the official `processor(prompt, image)` so train/test preprocessing matches. |
| **Deployment** | `vla-scripts/deploy.py` serves the model over a **REST API**, explicitly "removing any requirement for powerful on-device compute" — the official endorsement of the split (Pi ↔ remote GPU) architecture. |
| **Licensing** | Code: **MIT**. Weights: derived from Llama-2 → **Llama Community License** applies (repo README's "Explicit Notes on Model Licensing"). School/educational use is fine; note it in student reports. |
| **2025 successors (curiosity items)** | **OFT** (Optimized Fine-Tuning, Mar 2025): 25–50× faster inference, continuous actions, ~73 ms/action on A100, higher success rates. **FAST** (Jan 2025): learned action tokenizer, up to 15× faster discrete inference. Both are follow-ups to base OpenVLA — good "where the field went" discussion material. |

---

## 2. Proposal scorecard

| # | Proposal claim | Verdict | Note |
|---|---|---|---|
| 1 | "7-billion parameter VLA" | ✅ | Exactly 7.5B. |
| 2 | "Pi 3 B cannot run or fine-tune OpenVLA locally → split architecture" | ✅ **Strongly supported** | 15 GB weights vs. 1 GB Pi RAM. The official `deploy.py` (REST serving, "removes any requirement for powerful on-device compute") is the canonical pattern for exactly this. |
| 3 | "ROS 2 Humble on the Pi" | ✅ | Humble ships **armv7l** builds for 32-bit Raspberry Pi OS (Pi 3 B). ⚠️ Pin to Humble — newer ROS 2 releases dropped 32-bit ARM support. |
| 4 | "Foxglove WebSocket or msgpack socket" | ⚠️ Adjust | Works, but the documented low-friction path is the repo's **`deploy.py` REST pattern** (or a plain Python socket). Use **Foxglove Studio** as a *live dashboard* (camera + Twist topics) — great demo-day engagement tool. |
| 5 | "Remap 7-DoF delta pose → 3-DoF velocity (vx, vy, ωz)" | ✅ Framing correct | **⚠️ Missing the normalization layer.** The model emits *normalized* 256-bin actions; students **must compute per-dimension min/max statistics from their own dataset and un-normalize** (registered as an `unnorm_key` in the repo). Without this step the robot receives meaningless numbers — this is the #1 silent killer of student projects and must be a graded deliverable. Simplest robust design: **keep 7-dim output**, map demos to `(vx, vy, ωz, 0, 0, 0, 0)` (zero the unused dims), and read the first three dims at inference. |
| 6 | "Mecanum IK node on the Pi" | ✅ | Standard 4-wheel IK. Fine as specified. |
| 7 | "Capture at 5–10 Hz" | ✅ | Matches the official troubleshooting guidance **exactly**. |
| 8 | "50–200 episodes per task" | ⚠️ Raise the floor | Official guidance is **~100 demos minimum**. Target **100–150 clean episodes**; treat 50 as "will likely underfit." |
| 9 | "bitsandbytes 8-bit → fits 24 GB VRAM" | ✅ (with caveat) | True for **QLoRA**. Note the repo's bf16-LoRA floor is ~27 GB; on 24 GB you run QLoRA with small batch + gradient accumulation. Expect **~1–3 h per run** on a 24 GB GPU (vs. ~15–30 min on an A100 80 GB). |
| 10 | "Actions returned in ~100–200 ms" | ⚠️ Recalibrate | 100–200 ms is an **A100-class number** (~150 ms). On RTX 3090/4090 expect **~150–300 ms per action** (≈3–6 Hz), and after adding camera capture + network RTT the *effective closed-loop rate is ~1.5–3 Hz*. **The control design must tolerate this**: the Pi runs a 50–100 Hz motor loop that *holds/slews* the latest commanded velocity between model updates, with max-velocity limits and slew-rate limits. This turns a "problem" into one of the best engineering lessons of the course. |
| 11 | Safety overrides (HuskyLens/LiDAR, kill-switch) | ✅ | Keep; add the official **two sanity checks** as graded gates (see below). |
| 12 | "Gazebo simulator first (optional)" | ⚠️ De-prioritize | Gazebo + ROS 2 is heavy on a Pi 3 B and adds a week of environment risk. The two official sanity checks (replay demo; pipeline accuracy check) cover most of the validation value. Keep simulation as a **stretch extension only**. |

---

## 3. Top 5 changes to make before teaching

1. **Add the un-normalization pipeline as a first-class work item** (compute per-dim min/max stats from the team's own data; apply at inference; register an `unnorm_key`). Nothing else works without it.
2. **Raise the dataset floor to ~100 clean episodes** and adopt the four official data rules verbatim: 5–10 Hz, no pauses, diverse start poses, consistent strategy.
3. **Re-baseline control expectations to ~2 Hz** (2–3 Hz with a 4090): design "velocity hold + slew-rate limit + hard limits" into the Pi-side controller from day one, and teach *why* (a VLA is a slow, stochastic supervisor; the motor loop is the fast layer).
4. **Insert the two official sanity checks as milestone gates**:
   - *Gate 1 (before training):* replay a recorded demo open-loop and succeed.
   - *Gate 2 (before robot autonomy):* feed fine-tuning-set images into the live inference pipeline and confirm it reproduces training accuracy.
   These two checks eliminate ~80% of the "my model is bad" debugging that would otherwise consume the course.
5. **Pin the image preprocessing**: capture 640×480 JPEG on the Pi; let the official prismatic `processor` do resize/crop on the server (matching the 90%-crop augmentation, center crop at test time). Never resize ad-hoc.

## 4. Keep as-is (the proposal's good instincts)

- Split architecture with the Pi as hardware bridge — correct and officially sanctioned.
- Dedicated interference-free Wi-Fi — correct (make it an isolated 5 GHz SSID).
- Hardware kill-switch + RC override — correct (make it a standing rule, not a suggestion).
- Task scope ("drive sideways through the cones," "stop at the color target") — correct; VLA demos work best at this scale.
- 5–10 Hz logging — correct.
- Language instructions per task — correct; it's the lever students will use for multi-task experiments.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 24 GB GPU contention (4 teams, sequential training) | Pre-image the GPU box; QLoRA runs are 1–3 h each → schedule runs in the morning/overnight, and while a run is active the team works on the inference server / eval harness (built into the lesson plan). |
| RLDS conversion rabbit hole (tfds version pins, `dlimp` rebuild — both documented as the top two GitHub issues) | Primary path: **custom PyTorch Dataset wrapper** (repo option (b)); RLDS as the advanced track. |
| Wi-Fi dropout mid-run | UDP with sequence numbers; Pi-side "stale action" watchdog → zero velocity after 500 ms of no commands (fail-safe stop). |
| Model commands a wall-crash | Speed caps (≤0.3 m/s autonomy), LiDAR/cone boundary, HuskyLens handoff gate, physical kill switch, and the fail-safe stop above. |
| Class falls behind (GPU queue, sick days) | The plan has built-in slack: classes 17–18 are extension/design, class 21 is a contingency class. Worst case, demo day runs the *best available* checkpoint + teleop fallback (the process is the grade). |
