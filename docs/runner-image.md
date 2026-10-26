# Runner image — build & run per platform (podman)

How the per-node runner image is **created** and **run** on each platform this
system uses. The container is the only hardware boundary (design §6, §10):
the agent on `:8601` + the fixed-argv job programs (`no shell, ever`). The
quadlet (`systemd/robovla-runner.service`) does the GPU passthrough + mounts.

Standard layout (both platforms):

| Host path | Container path | What it is |
|---|---|---|
| `<workdir>` (fleet.yaml `nodes[].workdir`) | `/data` | node layout: `datasets/ runs/ artifacts/` |
| model volume | `/models` (ro) | the **HF hub-cache dir** — see "Model volume" |
| — | `:8601` | agent, console-only traffic |

## Model volume (offline model load)

The engine loads the base model by HF hub id (`openvla/openvla-7b`,
custom-code, `trust_remote_code`), which resolves through `HF_HUB_CACHE`.
So the model volume **is the HF hub-cache directory** (the dir that contains
`models--openvla--openvla-7b/…`), and the container env points at it:

```
HF_HUB_CACHE=/models        # one line in /etc/robovla/node.env — the quadlet
                            # passes the env file through (--env-file)
```

Populating the volume (once per node):

```
huggingface-cli download openvla/openvla-7b --cache-dir <model-volume>
# e.g. <model-volume> = /home/adam/robovla/models   (contains hub/models--openvla--openvla-7b/)
```

or simply let the first job run online (it fills the host cache, which *is*
the volume). ~15 GB (3 safetensors shards + custom code).
`robovla cli model_download` is a generic single-file URL fetcher — it is
**not** the HF layout; use `huggingface-cli` for the base model.

## NVIDIA (campus nodes + host65 scratch box)

**Image:** `robovla-runner-cuda:2026.10` — `images/Dockerfile.cuda`.
FINALIZED pins (Phase 1, probe-verified on an RTX 4070 sm_89):
torch 2.9.0+cu128 + torchvision 0.24.0+cu128 **in one uv command** (a
separate PyPI resolve floats both to the latest cu130 build and silently
breaks the pin), transformers 4.51.3, peft 0.14.0, bitsandbytes 0.50.2,
timm 0.9.10 (modeling hard-checks `{0.9.10, 0.9.11, 0.9.12, 0.9.16}`),
openvla code cloned (not pip-installed) at SHA `c8f03f48` — the engine uses
the vendored prompt builder + pure token math because `import prismatic`
drags draccus/tensorflow. New pin = new tag + `gpu_selftest` at node join.

### Host prerequisites (one-time, root)

```
# Ubuntu 22.04+ / Linux Mint 21+:
sudo apt install podman uidmap slirp4netns fuse-overlayfs cni-plugins netavark \
                 nvidia-container-toolkit
# /etc/subuid + /etc/subgid must list the node user (check: getent subuid <user>)
# cgroup v2: stat -fc %T /sys/fs/cgroup  ->  cgroup2fs
```

- **`uidmap` is load-bearing**: rootless podman sets up the user namespace
  via the **setuid** `newuidmap`/`newgidmap`; a non-setuid copy fails with
  `write to uid_map failed: Operation not permitted`. If the user namespace
  cannot be set up, *nothing* runs (build or run) — there is no unprivileged
  workaround (filecaps setuid needs root too).
- **GPU passthrough = CDI** (nvidia-container-toolkit ≥ 1.19; works rootless
  and rootful; the plain `--device /dev/nvidia*` lines only work with the
  legacy hook-mode runtime and rootful podman):

```
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   # regenerate on driver update
```

### Build

```
podman build -t robovla-runner-cuda:2026.10 -f images/Dockerfile.cuda .
# ~10-20 min: nvidia/cuda:12.8.1 base + torch/torchvision from the cu128 index
```

### Run (the designed path)

```
sudo cp systemd/robovla-runner.service /etc/systemd/system/
sudo -u <nodeuser> robovla-node install      # pulls image, writes node.env, enables unit
# /etc/robovla/node.env:
#   ROBOVLA_LAN_IP / ROBOVLA_AGENT_TOKEN / ROBOVLA_WORKDIR / ROBOVLA_MODEL_DIR
#   ROBOVLA_IMAGE=robovla-runner-cuda:2026.10
#   HF_HUB_CACHE=/models
systemctl enable --now robovla-runner
```

The quadlet runs rootless (`User=<nodeuser>`) with **`--device
nvidia.com/gpu=all`** (CDI) + workdir → `/data` + model volume →
`/models:ro`.

### Verify (Phase 1 gate, per node)

```
curl -s -H "Authorization: Bearer <token>" http://<node>:8601/health
#   -> {"ok": true, "gpu": "NVIDIA GeForce RTX 4070 11.99GB", ...}
robovla cli gpu_selftest --node <id> --dataset <sNN/task/vN> --token <tok>
#   8-bit load, 8-bit fwd/bwd, bf16 skipped where it doesn't fit (≤12 GB),
#   predict probe 7x[0,1], 10-step QLoRA at batch 1 (~12.2 GB peak on a 12 GB GPU)
GPU_TESTS=1 uv run pytest -m hardware    # in-image job rows (podman run, fixed argv)
```

### Scratch-box variant — no root at all (host65, built 2026-10-22)

host65 (avc) has a locked root account and a Mint-restricted sudoers allowlist,
so the user-local variant is staged there (works for everything *except* the
setuid step — see the uidmap note above):

1. **podman 5.8.5 + friends from CentOS Stream 9 rpms** (glibc 2.34 ≤ jammy's
   2.35): `podman buildah conmon crun netavark slirp4netns fuse-overlayfs`
   (AppStream) + `shadow-utils-subid` (libsubid), `shadow-utils`
   (newuidmap/newgidmap), `libslirp`, `libxcrypt` (BaseOS) from
   `mirror.stream.centos.org/9-stream/…/Packages/`. Extract with
   `python -m rpmfile` (+ `zstandard`); the shared libs are
   `*.so.N.M.K` with the versioned symlink (17-byte members are symlinks).
   Binaries → `~/robovla-p1/tools/bin`, libs → same dir,
   `export LD_LIBRARY_PATH=~/robovla-p1/tools/bin PATH=…:$PATH`.
2. **conmon path** (no `/usr/libexec/podman` to write): user config
   `~/.config/containers/containers.conf`:
   `[engine]` → `conmon_path = ['…/tools/bin/conmon']` (an **array**).
3. **NVIDIA toolkit 1.20.1** from the GitHub release tarball (rpm inside):
   `nvidia-ctk` + `nvidia-cdi-hook` + `libnvidia-container*.so` → same `bin/`;
   then `nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml`
   (rootless mode; sandboxutils/imex warnings are benign). Podman's CDI
   lookup is system-first — the root step also does
   `mkdir -p /etc/cdi && cp ~/.config/cdi/nvidia.yaml /etc/cdi/`.
4. **The one root action that unblocks everything** (any root shell — console,
   or a one-time admin): `apt-get install -y uidmap` (or `shadow-utils`) so
   the setuid `newuidmap`/`newgidmap` exist; until then every `podman`
   command dies at namespace setup.

**Verified end-to-end on host65 (2026-10-22)** — the root action landed
(`apt-get install -y uidmap`) and the container path is GREEN:

- PATH shadowing: podman finds `newuidmap` **via PATH** — `tools/bin` (the
  non-setuid rpm copy) shadows `/usr/bin`, so even with `uidmap` installed
  the namespace setup still fails. Fix: `ln -sf /usr/bin/newuidmap
  tools/bin/newuidmap` (same for `newgidmap`).
- User-level `~/.config/containers/policy.json`
  (`default: [{type: insecureAcceptAnything}]`) — no system policy.json on a
  box without distro podman; fleet nodes get the distro's signed policy.
- User-level `~/.config/containers/registries.conf`
  (`unqualified-search-registries = ["docker.io", "quay.io"]`) — without it
  short names (`nvidia/cuda:…`) fail to resolve.
- `[network] default_rootless_network_cmd = "slirp4netns"` in
  `containers.conf` — **podman 5.x's compiled-in rootless default is
  `pasta`** (config `default.go`, no fallback when the pasta binary is
  absent); without this every container dies at namespace setup with
  "could not find pasta".
- `[engine] helper_binaries_dir = ['…/tools/bin']` (an **array**) — podman
  resolves helper binaries like `rootlessport` here, not in `$PATH`.
- `rootlessport` (needed for rootless `-p` publishing) is not in the CS9
  rpm set: cross-compile it from podman's own source —
  `git clone --branch v5.8.0 --filter=blob:none --sparse
  https://github.com/containers/podman; git sparse-checkout set cmd vendor`…
  or full checkout + `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build
  -mod=vendor -o rootlessport ./cmd/rootlessport` → `tools/bin/`
  (statically linked, ~5 MB).

Results: alpine rootless smoke test OK; `podman build` completed all 15
steps (`robovla-runner-cuda:2026.10`); the containerized agent (CDI
`--device nvidia.com/gpu=all`, workdir → `/data`, HF hub cache →
`/models:ro`, `HF_HUB_CACHE=/models`, `-p <lan-ip>:8601:8601`) reports
`/health` with the 4070 visible; `robovla cli gpu_selftest` over the LAN
passes end-to-end (`j_20261022_0001`: probe 7×[0,1], 10-step QLoRA
12.20 → 2.08 at 12227 MB / 98% VRAM); `GPU_TESTS=1 uv run pytest -m
hardware` (venv with the console wheel + `pip install pytest`) = **1
passed** (CUDA row, in-image job via CDI) **/ 1 skipped** (ROCm row — no
image on this box, by design). The hardware rows also caught three
test-level defects the CPU suite can't see (pinned in
`tests/hardware/test_selftest.py`): the CUDA row's stale AMD device lines
(→ CDI), `podman run` appending the after-image argv to the image
ENTRYPOINT (agent boots instead of the job — now `--entrypoint
/opt/venv/bin/python`), and the missing selftest dataset making the job
exit 0 vacuously (now materialized via
`scripts/make_selftest_dataset.py` in `_prereqs`).

**Role of this box (2026-10-22):** until node2 (campus 5090 Ti) is online,
host65 is the **development/test box**: crosscheck wrapper development and
verification, the hardware rows, and any container/GPU work happen here —
nothing waits on the 5090 except the Blackwell pin verification (first
`gpu_selftest` at node2's join; the 50-episode crosscheck fits the 4070 at
batch 1, tight as the selftest predicts). pai1 is **reserved for LLM
development** (the assistant that drives development runs on it) and gets
no openvla work.

## Crosscheck on the scratch box — the official engine end-to-end (2026-10-22)

The 50-episode crosscheck — the console engine vs the **official**
`openvla` `finetune.py` on identical data (design §6; the student curve
comparison) — is green on host65/4070 (`j_20261022_0002`,
`tests/hardware/test_selftest.py::test_crosscheck_cuda`, batch 1).

**Image tag:** `robovla-runner-cuda:2026.10.2` (the wandb sink — below;
`2026.10.1` predates it, the `.10` tag has no official stack). Pin rule:
after any image rebuild, re-run `test_gpu_selftest_cuda` before trusting
the tag. Rebuilds are ~2 min (heavy stack layers are decoupled from the
`images/` source COPY).

**The official stack (why each pin/flag):**

| Piece | Pin / flag | Why |
|---|---|---|
| draccus / accelerate / wandb | 0.8.0 / 1.15.0 / 0.31.0 | the official repo's own pins (`finetune.py`'s config layer, the DDP `PartialState`, its logging — the crosscheck child SHADOWS it with the house sink, below; the package stays installed for import-chain completeness) |
| tensorflow / tensorflow_datasets | 2.19.1 / 4.9.3 | TFDS is the official pin; the official TF 2.15 predates py3.12 (cp312 wheels start at 2.17) and the image is 3.12 |
| tensorflow_metadata | 1.17.3 | tfds's floating dep; the latest (1.21.0) ships pb2 gencode built with protobuf 6.x, which fails the runtime-version check (TF 2.19.1 forces protobuf<6) |
| tensorflow_graphics | **stub** (`images/tfg_stub`) | the official pin (2021.12.3) needs tf-addons — cp39–cp311 wheels only, unresolvable on py3.12. The pinned openvla imports it at module level but only DROID/furniture-bench/iamlab/cmu transforms use it; the crosscheck path (`toto`, a pure `tf.concat`) never does. The stub satisfies the import and every leaf raises a clear RuntimeError |
| dlimp (official fork) | `--no-deps` @ 040105d | its setup.py hard-pins TF 2.15 (would downgrade the image's 2.19.1); its runtime import surface is {tensorflow, tensorflow_datasets, numpy} — all present |
| openvla + prismatic | `--no-deps` @ c8f03f48 | the official pyproject pins torch 2.2.0 / transformers 4.40.1 / peft 0.11.1 / TF 2.15 — following them re-opens the Phase 1 pins; the finetune import chain's surface is all present |
| `finetune.py` build-time patch | `images/finetune_patch.diff`, sha256-pinned (`a5600f58…`) | the #90 workaround — below |

**The #90 workaround (disclosed, build-time):** the pinned script cannot
complete ONE QLoRA step on this stack — its hardcoded DDP wrap (no
`static_graph`) + the gradient checkpointing that
`prepare_model_for_kbit_training` enables meet torch 2.9.0's implicit
**non-reentrant** checkpoint default and the first backward dies
(`RuntimeError: Expected to mark a variable ready only once`;
openvla/openvla#90 — a maintainer-acknowledged latent bug; no argv/env
reaches the DDP args). The Dockerfile applies the exact #90 fix at build
time (`git apply --check` against the SHA checkout, then the patched file
is sha256-pinned): `static_graph=True` + explicitly **reentrant**
checkpointing — mathematically neutral. Crosscheck results carry this
note (the wrapper docstring discloses it too).

**The RLDS materialization (`images/rlds_writer.py`):** the subset → one
TFDS example per trajectory under `<job_dir>/crosscheck_rlds/toto/` (the
OXE `toto` config — the only pure-concat standardization, so the course
action convention `(vx, vy, wz, 0, 0, 0, 0.0)` flows through verbatim;
the zero-padded dims are "unused" in the official BOUNDS_Q99
normalization and the gripper dim is excluded by the EEF_POS mask).
Fixes found host65-side (attempts 1–5, all CPU-pinned in
`tests/unit/test_rlds_writer.py`): subclass `tfds.core.GeneratorBasedBuilder`
(a raw `Builder` needs a filesystem dataset), `skip_registration` on the
dynamically-created builder class, pin its `__module__` (TFDS re-imports
via `__import__` of `__module__` — an un-pinned dynamic class comes out
as `'abc'`), yield **(key, example)** pairs (the TFDS 4.x contract —
shard assignment hashes the key), and `open_gripper` is a per-frame
**scalar** (the official transform does `tf.concat([wv, rd,
open_gripper[:, None]])` — it re-adds the axis itself). The wrapper also
forwards the official's captured stdout/stderr into its own streams
(host65 attempt 5: the swallowed child output hid the real failure
behind a bare exit code).

**The wandb sink (`images/wandb_sink.py`, 2026-10-23):** the pinned
`finetune.py` has exactly TWO wandb touchpoints, main process only:
`wandb.init(entity=…, project=…, name=…)` and `wandb.log(
{"train_loss", "action_accuracy", "l1_loss"}, step=<gradient step>)`
— every 10 gradient steps, `train_loss` smoothed over the grad-accum
window. LAN policy is credential-free wandb; under the old
`WANDB_MODE=disabled` those calls were no-ops and the official's per-step
loss was the one crosscheck metric lost (the follow-up below, resolved in
`2026.10.2`). The wrapper now shadows the real package: the image COPYs
the sink to `/opt/robovla/sink/wandb.py` and puts that dir FIRST on the
official child's `PYTHONPATH`, so `import wandb` resolves to the
stdlib-only house sink. It maps the official payload into the job's
STANDARD `metrics.jsonl` (house row keys: `loss ← train_loss`, `step` from
the official kwarg, `total`/`lr` from the sink env; the other
finite-numeric payload keys — `action_accuracy`, `l1_loss` — pass through
verbatim), so the agent's incremental poll → SSE `metrics` event →
run-detail sparkline consume the official's curve with no console/UI
change (one metrics channel). `init` prints one line to the job log and
writes nothing to `metrics.jsonl` (the agent reads every JSON dict line
there as a metrics row — a header line would corrupt the stream).
Fail-loud: unknown wandb attributes raise — the pinned surface is
init/log, so a re-pinned script with a larger surface fails the
crosscheck visibly, not silently. CPU-pinned in
`tests/unit/test_wandb_sink.py` (row mapping, the env contract, the
no-op default, the fail-loud edges, name independence of the aliased
file, and the Dockerfile-dest ↔ wrapper-`SINK_DIR` agreement); the
hardware row now asserts the sink's rows (`_assert_crosscheck_ok`).
Disclosed in the results like the #90 patch: the official's own process
is otherwise untouched — the sink is a logging stand-in.

**Known degradations (documented in the results):**
- TFDS 4.9.3 rejects the deliberately empty `val` split ("No examples
  were yielded") — the writer's primary (train + empty val, which would
  let the official use 100% of the subset) always falls back to
  train-only, and the official then trains `train[:95%]`: **47/50
  episodes** (the last 3 of the subset, deterministically held out).
  `dataset_statistics.json` is computed on all 50 (1000 transitions).
- The loss curve is the official's `train_loss` smoothed over the
  grad-accum window, logged every 10 gradient steps (the pinned script's
  own cadence — not per micro-batch); `action_accuracy` is TRAIN accuracy
  (the held-out comparison stays the console side's `val.json`).

**Silicon result (2026-10-22, host65/RTX 4070 12 GB, batch 1):**

| Metric | Value |
|---|---|
| total wall (row: podman + RLDS + load + train + checkpoint + CPU merge) | 1644.8 s (27:24) |
| training loop | 375 steps (125/epoch × 3) at 4.24 s/step (tqdm 26:32) |
| VRAM (nvidia-smi, stable) | 8971 MiB (73% of 12288) |
| LoRA footprint | 110,828,288 / 7,652,065,472 trainable (1.4483%); r32, α16 (the official's hardcoded `min(rank, 16)`), bias none |
| artifacts (file contract, workdir-relative) | adapter dir 968,788,735 B (`adapter_model.safetensors` 968,782,584 B); merged 15,086,381,550 B (4 safetensors shards + tokenizer/config + `dataset_statistics.json`) |
| exp_id | `openvla-7b+toto+b8+lr-0.0005+lora-r32+dropout-0.0+q-4bit--j_20261022_0002--image_aug` |
| loss curve | not captured (this run used `2026.10.1`, which predates the wandb sink — the next crosscheck re-run carries it via `metrics.jsonl`; the hardware row asserts it) |

**Official-vs-recipe deltas** (the comparison is shape/convergence, not
point-for-point):

| Axis | Official (crosscheck) | Console recipe |
|---|---|---|
| quant | 4-bit only (`use_quantization` → nf4) | 8-bit (bnb8) |
| LoRA α | 16 (hardcoded `min(rank, 16)`) | 32 |
| data order | unseeded TF shuffle | seeded split, in order |
| eff-batch | 8 (batch 1 × grad_accum 8) | 8 (same) |
| loss visibility | wandb sink → the job's `metrics.jsonl` (smoothed, every 10 grad steps; `action_accuracy`/`l1_loss` ride along) | per-step `metrics.jsonl` |

**Ops (this row):** LONG (27 min on the 4070 — run it detached:
`setsid nohup` + poll). The pytest log only grows when the podman process
exits (capture_output) — a ~272 B log mid-run is NORMAL, not a hang;
`pgrep -f test_crosscheck` self-matches its own command line — confirm
liveness via nvidia-smi (idle = 27 MiB) + the log. Re-kick:
`rm -rf <job_dir>/{crosscheck_rlds,crosscheck_out,metrics.jsonl}` first
(TFDS reuses an existing `toto/1.0.0` — stale data would poison the run;
the sink APPENDS — a same-job re-run would mix stale rows), then
`GPU_TESTS=1 ROBOVLA_IMAGE_CUDA=robovla-runner-cuda:2026.10.2
ROBOVLA_WORK_DIR=… ROBOVLA_MODEL_DIR=… setsid nohup .venv/bin/python -m
pytest tests/hardware/test_selftest.py::test_crosscheck_cuda -m hardware
-v -rs > /tmp/…log 2>&1 &`. The row persists the official's captured
output to `<job_dir>/official_output.log` (the agentless row captures
what the agent's job log would carry; on failure the `[rank0]` traceback
is in the same log — the forwarding fix works).

## AMD / ROCm (home Strix Halo 8060S, gfx1151)

**Status: designed, not yet verified** — the home lab is off-campus
(roboVLA is LAN-only; this is a self-contained dev lab, never a fleet node)
and the torch-rocm wheel is a custom gfx1151 build.

**Image:** `robovla-runner-rocm:2026.10` — `images/Dockerfile.rocm`.
bf16, no quantization (no bitsandbytes; the home selftest is the `home_selftest`
job: bf16 GEMM benchmark + bf16 load + 10-step bf16 fwd/bwd). The torch wheel
is a **build arg** (no public gfx1151 wheel):

```
podman build --build-arg TORCH_ROCM_WHEEL=<custom gfx1151 torch wheel URL> \
    -t robovla-runner-rocm:2026.10 -f images/Dockerfile.rocm .
```

**Host prerequisites:** same rootless podman stack as the NVIDIA section
(podman, uidmap + setuid newuidmap, slirp4netns, fuse-overlayfs, cni-plugins,
netavark; cgroup v2, subuid/subgid) — *without* the NVIDIA toolkit. GPU access
is plain device passthrough (no CDI): `/dev/kfd` + `/dev/dri` must be
accessible to the node user (usually `video`+`render` groups).

**Run:** the quadlet's ROCm variant — replace the four nvidia `--device`
lines with `--device /dev/kfd --device /dev/dri --security-opt
seccomp=unconfined`, and set `ROBOVLA_IMAGE=robovla-runner-rocm:2026.10`
in `node.env`.

**Verify:** `robovla cli gpu_selftest` on the node reports `home_selftest`
semantics when the node profile is the home lab (bf16 GEMM TFLOPS + bf16
fwd/bwd; no 8-bit section, no bnb).

## Troubleshooting (found the hard way)

| Symptom | Cause / fix |
|---|---|
| container agent exits instantly, exit 0, zero output | ENTRYPOINT was `-m agent.server` (no `__main__` guard) — must be `-m agent`; pinned by `tests/integration/test_agent_entrypoint.py` |
| job crashes `No such file or directory: /opt/robovla/bin/robovla_job` | `ROBOVLA_JOB_BIN` not wired (now `partial(job_argv, bin_path=cfg.job_bin)` in `agent/server.py`) |
| `ImportError: cannot import name 'load_episode' from 'robovla.episode'` | it is the file seam `robovla.fs` (`images/engine.py`) |
| `write to uid_map failed: Operation not permitted` | non-setuid `newuidmap` — install the distro `uidmap` as root; if `tools/bin` is first in `$PATH`, symlink `tools/bin/newuidmap` → `/usr/bin/newuidmap` (podman resolves it via PATH) |
| `could not find pasta, the network namespace can't be configured` | podman 5.x's compiled-in rootless default backend is `pasta` (no fallback) — set `[network] default_rootless_network_cmd = "slirp4netns"` in `containers.conf` |
| `could not find "rootlessport" in one of […]` / `Failed to obtain podman configuration: … helper_binaries_dir` | rootless `-p` needs the `rootlessport` binary in `[engine] helper_binaries_dir` (**array**, not `$PATH`); it is not in the CS9 set — cross-compile `cmd/rootlessport` from the matching podman tag (see scratch-box section) |
| torch 2.14+cu130 appears in the image | torch+torchvision installed separately (pin drift) — one uv command, cu128 index |
| `timm` version error at model load | pin timm to the hard-checked set (0.9.10 in the image) |
| bf16 fwd/bwd skipped in selftest | by design on ≤12 GB GPUs (`free < 20 GB` branch) |
| model re-downloads inside the container | `HF_HUB_CACHE=/models` missing from `node.env` (env file passes through) |
| crosscheck job: official trains fine but no loss curve in the run detail | the image predates the sink COPY (`/opt/robovla/sink/wandb.py` missing) — `import wandb` in the child hit the real package and no-opped; rebuild (`2026.10.2`) |
