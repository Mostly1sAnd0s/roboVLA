# HANDOFF — SmolVLA family: Pascal (GTX 1070) variant + fp32 profile

**Transient doc. Delete when the work below lands.** Written 2026-11-08 by the
session that implemented the SmolVLA family (CPU-complete + real-stack CPU
verified). The owner authorized killing the llama lane on this box to free the
1070 for testing.

## 1. State at handoff

- Commits (on top of the docs-only history): `01b2554` (family pilot: routing,
  writer, wrapper, sink, serving seam, Dockerfile.cuda-smolvla, 134 tests) and
  `b98723a` (three real-stack fixes). Working tree clean after this doc.
- CPU suite: **722 passed** (`uv run pytest -m "not hardware and not e2e"`;
  run `uv sync --extra dev` first on a fresh checkout — a bare `uv run pytest`
  fails with `No module named 'robovla'`).
- The README status bullet "SmolVLA family pilot" is the authoritative summary.
- Full verification evidence + the 12-check harness: `~/smolvla-verify/`
  (this machine). Read it before touching the wrapper; see §5.

## 2. What to do, in order

1. **Wheel-arch probe (cheapest gate; do first).** Confirm the chosen cu126
   torch wheel carries Pascal kernels before building anything:
   `uv venv --python 3.12 /tmp/cu126probe/.venv` →
   `uv pip install torch==2.9.1+cu126 --index-url https://download.pytorch.org/whl/cu126`
   (2.7.1+cu126 and 2.8.0+cu126 also exist; 2.9.1+cu126 confirmed present on
   the index 2026-11-08) →
   `python -c "import torch; print(torch.cuda.get_arch_list()); x=torch.randn(4096,4096,device='cuda'); print(float((x@x).sum()))"`.
   Expect sm_50/sm_60 in the list (sm_60 SASS runs on sm_61 by minor-version
   binary compatibility). If sm_60 is absent from 2.9.1, walk down (2.8.0,
   2.7.1 — lerobot's floor is torch>=2.7,<2.16).
2. **Free the 1070.** It hosts the gtx1070-llama lane (judge + semantic_search
   containers) at ~6.6 GB of 8 GB. `docker/podman ps` as this user showed
   nothing — find the supervisor first (`systemctl --user list-units | grep -i
   -e llama -e container`, root `docker ps`, quadlet units, or cron). Kill it,
   confirm with `nvidia-smi` free memory, and REMEMBER to restart it at the
   end of the session (the judge-lane tool then fails loudly until it is back;
   do not interpret that as the diff being reviewed).
3. **fp32 profile (small core change).** The precision knob the house already
   has is the spec's `quant` field (`budgets.QUANT_PROFILES` = bnb8 | bf16).
   Add `"fp32"` to the profiles; `images/smolvla_finetune.py` currently
   HARDCODES `--accelerator.mixed_precision bf16` in `translate()` — map
   quant: bf16 → `bf16`, fp32 → `no` (policy weights stay float32 either way;
   the wrapper never passes `--policy.dtype`). Keep bnb8 rejected for the
   smolvla family (no bnb in the image). Search for existing tests that pin
   the quant profiles and the wrapper argv; update them together (§11 — one
   catalog). Openvla family: fp32 should probably remain REJECTED (its engine
   implements bnb8/bf16 only) — reject in spec or engine, one site.
4. **Pascal image variant.** `images/Dockerfile.cuda-smolvla-pascal`: copy of
   `Dockerfile.cuda-smolvla` with the torch layer switched to the probe-verified
   `torch==X+cu126 torchvision==Y+cu126` (ONE command — the shared-resolve
   trap; cu126 index URL). Everything else (lerobot==0.7.0 extras, ffmpeg,
   wheel, sink alias, ENTRYPOINT `-m agent`) unchanged. cu126 deliberately
   omits sm_120 (Blackwell): NEVER float this into the fleet default or the
   openvla image — the campus 5090s need cu128.
5. **CUDA hardware run on the 1070.** Fastest path: rerun the verification
   harness (§5) with CUDA + fp32 instead of the CPU venv: fresh venv with the
   probe-verified cu126 torch, repo installed, and instead of the
   `cuda→cpu` rewrite in `verify.py`, pass a spec with `quant="fp32"` and
   skip the device rewrite entirely (mixed_precision=no loads fp32 weights on
   CUDA). Stage 3's loader override: device "cuda". Success bar = the same
   12 checks. Watch VRAM (`torch.cuda.max_memory_allocated` is in the sink
   row as gpu_mem_gb only when mixed logs it — otherwise measure in the
   probe). fp32 estimate 5-7 GB of 8: if it OOMs, the first lever is smaller
   images (features_spec is 224^2 already; SmolVLA pads to 512 internally —
   `--policy.resize_imgs_with_padding` is the knob to lower, document any
   such delta), NOT quant.
6. **Then** (repo side, after the run proves it): a `tests/hardware/` row for
   the smolvla smoke (pattern: `tests/hardware/test_selftest.py` pins the CDI
   device line + the `podman run` ENTRYPOINT append — read its house comments,
   all three lessons are load-bearing), and the fleet knobs so the scheduler
   cannot place a smolvla job on an openvla-image node (v1 = pin by
   fleet.yaml config, section 10 note in the README bullet).

## 3. Facts established this session (not all in code comments — trust these,
they were read from lerobot v0.7.0 sources or proven on the real stack)

- `lerobot-train --steps` counts MICRO-batch loop steps; grad accumulation is
  `--accelerator.gradient_accumulation.steps`.
- `--use_policy_training_preset=true` OVERWRITES cfg.optimizer/scheduler
  (configs/train.py validate():349) — hence the explicit
  `--optimizer.type adamw --optimizer.lr ... --scheduler.type
  cosine_decay_with_warmup ...` block in translate(). The scheduler
  auto-scales warmup/decay into short runs.
- `--policy.input_features null` is the documented reset to derive I/O
  features from the dataset (configs/policies.py comment line ~59);
  output_features are ALWAYS dataset-derived when a dataset is given.
- Checkpoint layout: `<output>/checkpoints/<step>/pretrained_model/` with
  `config.json`, `model.safetensors`, `policy_preprocessor*.json/safetensors`,
  `policy_postprocessor*`, **and a `tokenizer/` dir — the checkpoint is
  self-contained** (proven). `checkpoints/last` is a symlink (the wrapper
  resolves it).
- The wrapper trained 99,880,992 learnable / 450,046,176 total params — the
  paper's numbers (VLM frozen, ~100 M expert trained).
- Three defects the CPU suite could not see (fixed in b98723a; do not
  regress): lerobot `validate_frame` needs exact dtypes (float32 ndarray
  actions); `prepare_state` requires `observation.state` UNCONDITIONALLY
  (constant-zero 3-dim placeholder both sides; normalizer divides by std+eps
  so zero variance is exact); single-frame `preprocess` batches images+
  language but NOT the state — the serve seam reshapes it to (1,3).
- Serving: `policy.select_action` holds a 50-action chunk queue (proven: the
  2nd predict returned a different action with no recompute). Prompt = RAW
  task string (no openvla In:/Out: wrap).
- Trackio sink surface: init/log/finish + Artifact/log_artifact no-ops
  (wrapper passes --trackio.disable_artifact=true). lerobot logs
  `train/<key>` dicts every `--log_freq` steps.
- Model volume for offline serving/train: `lerobot/smolvla_base` AND
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (the loader prints
  "Loading HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights" from within
  from_pretrained — it needs the backbone present). `robovla cli
  model_download` (injectable fetch seam) needs its catalog extended for both.
- Pascal facts (agreed with the owner 2026-11-08): cu128 wheels carry NO
  sm_61 kernels; CUDA 12.6 is the last cu12x family with Pascal support;
  cu126 has no sm_120, so it must stay a variant; bf16 does NOT auto-upcast
  on Pascal — hard "not implemented for BFloat16" errors; fp16 autocast is
  technically legal but GeForce Pascal fp16 math is crippled — fp32 is the
  only sane profile there. openvla-7b will never fit/fit-train on the 1070
  (bnb8 alone measured 7.85 GB) — the 1070 is a SMOVLVA-ONLY box.

## 4. House rules the next agent must keep

- TDD first on the core change (quant profile), ≤500 lines/file, §9.6 error
  codes, lazy stack boundaries, fixed argv (no shell), judge review before
  declaring done, pin rule: new image tag + verified smoke before trusting.
- `uv run pytest -m "not hardware and not e2e"` must stay green.
- The split key is the DATASET-ID task (`spec.dataset.split("/")[1]`), not the
  episode's task string — the unit tests have a TASK vs TASK_ID trap that
  catches the mix-up; keep it.

## 5. Verification assets (this machine, outside the repo)

- `~/smolvla-verify/venv` — python 3.12, torch 2.9 CPU, lerobot 0.7.0, repo
  installed editable.
- `~/smolvla-verify/verify.py` — the 12-check end-to-end (fixture → writer →
  wrapper main() → sink rows → manifest → serving stamp → serve round-trip).
  For CUDA: keep everything, build the spec with `quant="fp32"` once the
  profile lands, delete the `cpu_translate` monkeypatch and the
  device_processor "cpu" override.
- `~/smolvla-verify/probe_seam.py` / `probe_serve.py` — focused serving-shape
  probes (reusable if the seam ever breaks on hardware).
- HF cache is warm (smolvla_base + SmolVLM2) — an offline CUDA rerun can set
  `HF_HUB_OFFLINE=1`.

## 6. Explicitly still open (do not silently expand scope)

- smolvla validate/merge/probe/held-out-eval (v1 gaps, documented in the
  README bullet + wrapper docstring).
- Scheduler awareness of node image families (v1 = pin by fleet.yaml).
- Standalone capture app / lesson-plan integration for the SmolVLA family
  (teaching docs untouched so far).
- torchcodec note: on CUDA images keep the apt ffmpeg layer (this machine's
  torchcodec failed to load libav and fell back to pyav — image mode never
  decodes video, but keep the belt).
