# Runner image: build and run per platform (podman)

How the per-node runner image is **created** and **run** on each platform. The
container is the only hardware boundary (design section 6, section 10). The
agent runs on `:8601`. The job programs have a fixed argv (`no shell, ever`).
The quadlet (`systemd/robovla-runner.service`) does the GPU passthrough +
mounts.

Standard layout (both platforms):

| Host path | Container path | What it is |
|---|---|---|
| `<workdir>` (fleet.yaml `nodes[].workdir`) | `/data` | node layout: `datasets/ runs/ artifacts/` |
| model volume | `/models` (ro) | the **HF hub-cache dir** - see "Model volume" |
| - | `:8601` | agent, console-only traffic |

## Model volume (offline model load)

The engine loads the base model by HF hub id (`openvla/openvla-7b`,
custom code, `trust_remote_code`). It resolves through `HF_HUB_CACHE`.
So the model volume **is the HF hub-cache directory** (the dir that contains
`models--openvla--openvla-7b/...`). The container env points at it:

```
HF_HUB_CACHE=/models        # one line in /etc/robovla/node.env - the quadlet
                            # passes the env file through (--env-file)
```

Populating the volume (once per node):

```
huggingface-cli download openvla/openvla-7b --cache-dir <model-volume>
# e.g. <model-volume> = /home/adam/robovla/models   (contains hub/models--openvla--openvla-7b/)
```

or simply let the first job run online (it fills the host cache, which *is*
the volume). About 15 GB (3 safetensors shards + custom code).
`robovla cli model_download` is a generic single-file URL fetcher. It is
**not** the HF layout. Use `huggingface-cli` for the base model.

## NVIDIA (campus nodes + host65 scratch box)

**Image:** `robovla-runner-cuda:2026.10` - `images/Dockerfile.cuda`.
FINALIZED pins (Phase 1, probe-verified on an RTX 4070 sm_89).
torch 2.9.0+cu128 + torchvision 0.24.0+cu128 **in one uv command**.
A separate PyPI resolve floats both to the latest cu130 build.
That silently breaks the pin. transformers 4.51.3. peft 0.14.0.
bitsandbytes 0.50.2. timm 0.9.10 (modeling hard-checks
`{0.9.10, 0.9.11, 0.9.12, 0.9.16}`). openvla code cloned at SHA `c8f03f48`
(not pip-installed). The engine uses the vendored prompt builder +
pure token math because `import prismatic` pulls in draccus/tensorflow.
New pin = new tag + `gpu_selftest` at node join.

### Host prerequisites (one-time, root)

```
# Ubuntu 22.04+ / Linux Mint 21+:
sudo apt install podman uidmap slirp4netns fuse-overlayfs cni-plugins netavark \
                 nvidia-container-toolkit
# /etc/subuid + /etc/subgid must list the node user (check: getent subuid <user>)
# cgroup v2: stat -fc %T /sys/fs/cgroup  ->  cgroup2fs
```

- **`uidmap` is required**: rootless podman sets up the user namespace
  via the **setuid** `newuidmap`/`newgidmap`. A non-setuid copy fails with
  `write to uid_map failed: Operation not permitted`. If the user namespace
  cannot be set up, *nothing* runs (build or run). There is no unprivileged
  workaround (filecaps setuid needs root too).
- **GPU passthrough = CDI** (nvidia-container-toolkit >= 1.19. Works rootless
  and rootful. The plain `--device /dev/nvidia*` lines only work with the
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
nvidia.com/gpu=all`** (CDI) + workdir -> `/data` + model volume ->
`/models:ro`.

### Verify (Phase 1 gate, per node)

```
curl -s -H "Authorization: Bearer <token>" http://<node>:8601/health
#   -> {"ok": true, "gpu": "NVIDIA GeForce RTX 4070 11.99GB", ...}
robovla cli gpu_selftest --node <id> --dataset <sNN/task/vN> --token <tok>
#   8-bit load, 8-bit fwd/bwd, bf16 skipped where it doesn't fit (<= 12 GB),
#   predict probe 7x[0,1], 10-step QLoRA at batch 1 (~12.2 GB peak on a 12 GB GPU)
GPU_TESTS=1 uv run pytest -m hardware    # in-image job rows (podman run, fixed argv)
```

### Scratch-box variant: no root at all (host65, built 2026-10-22)

host65 (avc) had a locked root account and a Mint-restricted sudoers
allowlist (2026-10-22). The node user gained password sudo by 2026-10-26
(used for the setuid step, the CDI spec regen, and the `rpm2cpio`/`cpio`/
`zstd` extraction tools). The user-local variant is staged there anyway
(no distro podman usable. See the 2026-10-26 recovery note below). So
it works for everything *except* the setuid step (see the uidmap note
above):

1. **podman 5.8.5 + friends from CentOS Stream 9 rpms** (glibc 2.34 <= jammy's
   2.35): `podman buildah conmon crun netavark slirp4netns fuse-overlayfs`
   (AppStream) + `shadow-utils-subid` (libsubid), `shadow-utils`
   (newuidmap/newgidmap), `libslirp`, `libxcrypt` (BaseOS), from
   `mirror.stream.centos.org/9-stream/.../Packages/`. Extract with
   `python -m rpmfile` (+ `zstandard`). The shared libs are
   `*.so.N.M.K` with the versioned symlink (17-byte members are symlinks).
   Binaries -> `~/robovla-tools/bin`, libs -> same dir,
   `export LD_LIBRARY_PATH=~/robovla-tools/bin PATH=...:$PATH`.
   (2026-10-26: the original `~/robovla-p1/tools` was destroyed by a bad
   `rsync --delete` from pai1. The same rpm set was re-extracted to
   `~/robovla-tools/`. Two rpm flavors: the zstd-payload rpms (netavark,
   podman-catatonit, shadow-utils*, libxcrypt) need
   `rpm2cpio f.rpm | zstd -d | cpio -idm`. Plain `rpm2cpio | cpio` fails
   silently on them. netavark/catatonit land under `usr/libexec/podman/`
   in the rpms. Collect them into `bin/`. Podman 5.x resolves them via
   the `helper_binaries_dir` config key, not PATH.)
2. **conmon path** (no `/usr/libexec/podman` to write): user config
   `~/.config/containers/containers.conf`:
   `[engine]` has `conmon_path = ['.../tools/bin/conmon']` (an **array**).
3. **NVIDIA toolkit 1.20.1** from the GitHub release tarball (rpm inside):
   `nvidia-ctk` + `nvidia-cdi-hook` + `libnvidia-container*.so` -> same `bin/`.
   Then `nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml`
   (rootless mode. sandboxutils/imex warnings are benign). Podman's CDI
   lookup is system-first. The root step also does
   `mkdir -p /etc/cdi && cp ~/.config/cdi/nvidia.yaml /etc/cdi/`.
4. **The one root action that unblocks everything** (any root shell.
   Console, or a one-time admin): `apt-get install -y uidmap` (or
   `shadow-utils`). This makes the setuid `newuidmap`/`newgidmap` exist.
   Until then every `podman` command dies at namespace setup.

**Verified end-to-end on host65 (2026-10-22)**: the root action landed
(`apt-get install -y uidmap`) and the container path is GREEN:

- PATH shadowing: podman finds `newuidmap` **via PATH**. `tools/bin` (the
  non-setuid rpm copy) shadows `/usr/bin`. So even with `uidmap` installed
  the namespace setup still fails. Fix: `ln -sf /usr/bin/newuidmap
  tools/bin/newuidmap` (same for `newgidmap`).
- User-level `~/.config/containers/policy.json`
  (`default: [{type: insecureAcceptAnything}]`). No system policy.json on a
  box without distro podman. Fleet nodes get the distro's signed policy.
- User-level `~/.config/containers/registries.conf`
  (`unqualified-search-registries = ["docker.io", "quay.io"]`). Without it,
  short names (`nvidia/cuda:...`) fail to resolve.
- `[network] default_rootless_network_cmd = "slirp4netns"` in
  `containers.conf`. **Podman 5.x's compiled-in rootless default is
  `pasta`** (config `default.go`. No fallback when the pasta binary is
  absent). Without this, every container dies at namespace setup with
  "could not find pasta".
- `[engine] helper_binaries_dir = ['.../tools/bin']` (an **array**).
  Podman resolves helper binaries like `rootlessport` here, not in `$PATH`.
- `rootlessport` (needed for rootless `-p` publishing) is not in the CS9
  rpm set. Cross-compile it from podman's own source.
  `git clone --branch v5.8.0 --filter=blob:none --sparse
  https://github.com/containers/podman; git sparse-checkout set cmd vendor`
  ... or a full checkout + `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build
  -mod=vendor -o rootlessport ./cmd/rootlessport`. Output -> `tools/bin/`
  (statically linked, about 5 MB).

Results: alpine rootless smoke test OK. `podman build` completed all 15
steps (`robovla-runner-cuda:2026.10`). The containerized agent (CDI
`--device nvidia.com/gpu=all`, workdir -> `/data`, HF hub cache ->
`/models:ro`, `HF_HUB_CACHE=/models`, `-p <lan-ip>:8601:8601`) reports
`/health` with the 4070 visible. `robovla cli gpu_selftest` over the LAN
passes end-to-end (`j_20261022_0001`: probe 7x[0,1], 10-step QLoRA
12.20 -> 2.08 at 12227 MB / 98% VRAM). `GPU_TESTS=1 uv run pytest -m
hardware` (venv with the console wheel + `pip install pytest`) = **1
passed** (CUDA row, in-image job via CDI) **/ 1 skipped** (ROCm row.
No image on this box, by design). The hardware rows also caught three
test-level defects the CPU suite cannot see (pinned in
`tests/hardware/test_selftest.py`). One: the CUDA row's stale AMD device lines
(now CDI). One: `podman run` appending the after-image argv to the image
ENTRYPOINT (the agent boots instead of the job. Now `--entrypoint
/opt/venv/bin/python`). One: the missing selftest dataset makes the job
exit 0 with no check (now materialized via
`scripts/make_selftest_dataset.py` in `_prereqs`).

**Recovered + re-verified 2026-10-26 (stack moved to `~/robovla-tools/`):**
a bad `rsync --delete` from pai1 (multiple source dirs + `--delete` sweeps
the WHOLE destination, not just the named dirs) destroyed the original
`~/robovla-p1` tree (the `tools/` rpm set, the venv, the tree copy. The
images/store/model cache/workdir outside it survived). Recovery:
- the same CS9 rpm set re-extracted to `~/robovla-tools/` (podman 5.8.5-2,
  buildah 1.43.2-3, conmon 2.2.1-2, crun 1.29.1, netavark 1.17.2-3,
  slirp4netns 1.3.4, fuse-overlayfs 1.17, libslirp, libxcrypt,
  shadow-utils/subid + the NVIDIA toolkit 1.20.1. zstd-payload rpm
  handling + `libexec/podman/` collection. See step 1).
- Mint's apt `podman` is **3.4.4. No CDI** (the pinned rows use
  `--device nvidia.com/gpu=all`, a podman >= 4.0 surface). Rejected and
  purged. Its leftover `/usr/bin/crun` (0.17) then failed the runs with
  `crun: unknown version specified`. **Podman resolves the runtime from a
  fixed path list (`/usr/bin` first), not PATH.** Purged the leftover and
  symlinked the user-local `crun`/`slirp4netns`/`fuse-overlayfs` into
  `/usr/local/bin` (the next fixed-list slot).
- the repo is now a **git clone at `~/robovla`** (sync = `git pull`.
  No more tree copies). The hardware-row venv = `uv venv --python 3.12` +
  `uv pip install . pytest` (uv's managed CPython 3.12, `~/.local/bin/uv`).
- the CDI spec regenerated with root (`/etc/cdi/nvidia.yaml` +
  `~/.config/cdi/`. hook = `~/robovla-tools/bin/nvidia-cdi-hook`). The old
  spec pointed at the destroyed `tools/` dir.
- re-verified end-to-end: `podman run --device nvidia.com/gpu=all` in the
  image (`torch.cuda.is_available()` True, the 4070). `test_gpu_selftest_cuda`
  on `2026.10.2` (34.9 s warm). The LONG `test_crosscheck_cuda` green
  (27:25. The wandb sink's `metrics.jsonl` rows asserted, below).

**Disk cleanup 2026-10-26 (post-recovery):** the disk had grown to 87%
(23 GB free). The podman image store was 78.8 GB (the current `2026.10.2`
+ the stale `2026.10.1`/`2026.10` tags + five dangling `<none>` rebuild
intermediates). On top of the 15 GB merged crosscheck artifact
(`~/robovla-selftest/runs/j_20261022_0002`. **Kept**, the documented
artifact). Actions: `podman image prune -f` (the dangling intermediates).
`podman rmi localhost/robovla-runner-cuda:2026.10` (the pre-official-stack
era. Superseded twice). And `:2026.10.1` (its reference role is done.
The `2026.10.2` run's bit-identical comparison closed `9841838`. The
artifacts stay on disk. The tag is rebuildable from the pinned Dockerfile).
`rm -rf ~/robovla-p1` (the stripped pre-git tree. 91 MB of a partial copy
left by the bad rsync. No `.git`, no live config references. Verified
before deleting). Result: disk 57% used (**76 GB free**). Image store
23.6 GB (`2026.10.2` + the base images only). The hardware row's default
tag now points at `2026.10.2` (`tests/hardware/test_selftest.py`). The
store no longer holds `2026.10.1`. The Ops recipe below always passes the
tag explicitly anyway. Pre-git leftovers still in `~`
(`robovla-sync.tar`, the 2026-10-21 probe scripts + the `.robovla-p1-token`
file) are unreferenced. Left for the owner to remove.

**Role of this box (2026-10-22):** until node2 (campus 5090 Ti) is online,
host65 is the **development/test box**. Crosscheck wrapper development and
verification, the hardware rows, and any container/GPU work happen here.
Nothing waits on the 5090 except the Blackwell pin verification (first
`gpu_selftest` at node2's join. The 50-episode crosscheck fits the 4070 at
batch 1, tight as the selftest predicts). pai1 is **reserved for LLM
development** (the assistant that drives development runs on it). It gets
no openvla work.

## Crosscheck on the scratch box: the official engine end-to-end (2026-10-22. Re-verified 2026-10-26 with the wandb sink)

The 50-episode crosscheck compares the console engine vs the **official**
`openvla` `finetune.py` on identical data (design section 6. The student
curve comparison). It is green on host65/4070 (`j_20261022_0002`,
`tests/hardware/test_selftest.py::test_crosscheck_cuda`, batch 1). The
row was re-run 2026-10-26 on `robovla-runner-cuda:2026.10.2` after the
wandb sink landed. The row now asserts the sink's `metrics.jsonl`.
The result table below carries the re-run.

**Image tag:** `robovla-runner-cuda:2026.10.2` (the wandb sink. Below.
`2026.10.1` predates it. The `.10` tag has no official stack). Pin rule:
after any image rebuild, re-run `test_gpu_selftest_cuda` before trusting
the tag. Rebuilds take about 2 min (the heavy stack layers are decoupled
from the `images/` source COPY). The rebuild + selftest re-run is done
2026-10-26 (selftest 34.9 s warm on the 4070). Tag trusted, crosscheck
re-run green.

**The official stack (why each pin/flag):**

| Piece | Pin / flag | Why |
|---|---|---|
| draccus / accelerate / wandb | 0.8.0 / 1.15.0 / 0.31.0 | the official repo's own pins: the `finetune.py` config layer (draccus), the DDP `PartialState` (accelerate), its logging (wandb). The crosscheck child SHADOWS wandb with the house sink (below). The package stays installed for import-chain completeness |
| tensorflow / tensorflow_datasets | 2.19.1 / 4.9.3 | TFDS is the official pin. The official TF 2.15 predates py3.12 (cp312 wheels start at 2.17). The image is 3.12 |
| tensorflow_metadata | 1.17.3 | tfds's floating dep. The latest (1.21.0) ships pb2 gencode built with protobuf 6.x. That fails the runtime-version check (TF 2.19.1 forces protobuf<6) |
| tensorflow_graphics | **stub** (`images/tfg_stub`) | the official pin (2021.12.3) needs tf-addons. cp39-cp311 wheels only, unresolvable on py3.12. The pinned openvla imports it at module level. Only DROID/furniture-bench/iamlab/cmu transforms use it. The crosscheck path (`toto`, a pure `tf.concat`) never does. The stub satisfies the import. Every leaf raises a clear RuntimeError |
| dlimp (official fork) | `--no-deps` @ 040105d | its setup.py hard-pins TF 2.15 (would downgrade the image's 2.19.1). Its runtime import surface is {tensorflow, tensorflow_datasets, numpy}. All present |
| openvla + prismatic | `--no-deps` @ c8f03f48 | the official pyproject pins torch 2.2.0 / transformers 4.40.1 / peft 0.11.1 / TF 2.15. Following them re-opens the Phase 1 pins. The finetune import chain's surface is all present |
| `finetune.py` build-time patch | `images/finetune_patch.diff`, sha256-pinned (`a5600f58...`) | the #90 workaround. Below |

**The #90 workaround (disclosed, build-time):** the pinned script cannot
complete ONE QLoRA step on this stack. Its hardcoded DDP wrap (no
`static_graph`) + the gradient checkpointing that
`prepare_model_for_kbit_training` enables meet torch 2.9.0's implicit
**non-reentrant** checkpoint default. The first backward dies
(`RuntimeError: Expected to mark a variable ready only once`).
openvla/openvla#90. This is a maintainer-acknowledged latent bug. No
argv/env reaches the DDP args. The Dockerfile applies the exact #90 fix at
build time (`git apply --check` against the SHA checkout. Then the patched
file is sha256-pinned): `static_graph=True` + explicitly **reentrant**
checkpointing. Mathematically neutral. The crosscheck results carry this
note (the wrapper docstring discloses it too).

**The RLDS materialization (`images/rlds_writer.py`):** the subset becomes
one TFDS example per trajectory. It lands under
`<job_dir>/crosscheck_rlds/toto/`. The OXE `toto` config is the only
pure-concat standardization. So the course action convention
`(vx, vy, wz, 0, 0, 0, 0.0)` flows through verbatim. The zero-padded dims
are "unused" in the official BOUNDS_Q99 normalization. The gripper dim is
excluded by the EEF_POS mask.
Fixes found host65-side (attempts 1-5. All CPU-pinned in
`tests/unit/test_rlds_writer.py`): subclass `tfds.core.GeneratorBasedBuilder`
(a raw `Builder` needs a filesystem dataset). `skip_registration` on the
dynamically-created builder class. Pin its `__module__` (TFDS re-imports
via `__import__` of `__module__`. An un-pinned dynamic class comes out
as `'abc'`). Yield **(key, example)** pairs (the TFDS 4.x contract.
Shard assignment hashes the key). And `open_gripper` is a per-frame
**scalar** (the official transform does `tf.concat([wv, rd,
open_gripper[:, None]])`. It re-adds the axis itself).
The wrapper also forwards the official's captured stdout/stderr into its
own streams. (host65 attempt 5: the swallowed child output hid the real
failure behind a bare exit code.)

**The wandb sink (`images/wandb_sink.py`, 2026-10-23):** the pinned
`finetune.py` has exactly TWO wandb touchpoints, main process only:
`wandb.init(entity=..., project=..., name=...)` and `wandb.log(
{"train_loss", "action_accuracy", "l1_loss"}, step=<gradient step>)`.
Every 10 gradient steps. `train_loss` is smoothed over the grad-accum
window. LAN policy is credential-free wandb. Under the old
`WANDB_MODE=disabled`, those calls were no-ops. The official's per-step
loss was the one crosscheck metric lost. (The follow-up below, resolved in
`2026.10.2`.) The wrapper now shadows the real package. The image COPYs
the sink to `/opt/robovla/sink/wandb.py`. It puts that dir FIRST on the
official child's `PYTHONPATH`. So `import wandb` resolves to the
stdlib-only house sink. It maps the official payload into the job's
STANDARD `metrics.jsonl` (house row keys: `loss <- train_loss`, `step` from
the official kwarg, `total`/`lr` from the sink env. The other
finite-numeric payload keys, `action_accuracy` and `l1_loss`, pass through
verbatim). So the agent's incremental poll -> SSE `metrics` event ->
run-detail sparkline consume the official's curve with no console/UI
change (one metrics channel). `init` prints one line to the job log. It
writes nothing to `metrics.jsonl` (the agent reads every JSON dict line
there as a metrics row. A header line would corrupt the stream).
Fail-loud: unknown wandb attributes raise. The pinned surface is
init/log. A re-pinned script with a larger surface fails the
crosscheck visibly, not silently. CPU-pinned in
`tests/unit/test_wandb_sink.py` (row mapping, the env contract, the
no-op default, the fail-loud edges, name independence of the aliased
file, and the Dockerfile-dest to wrapper-`SINK_DIR` agreement). The
hardware row now asserts the sink's rows (`_assert_crosscheck_ok`).
**Re-verified on silicon 2026-10-26 (host65/4070):** the crosscheck row
green on `2026.10.2`. 304 sink rows asserted (int step, finite loss,
monotone steps, `action_accuracy`/`l1_loss` passthrough. Loss 9.63 ->
1.11). The sink's one `init` job-log line captured in
`official_output.log` (`[wandb-sink] init: entity=local,
project=robovla-crosscheck, ... (sink: /data/runs/j_20261022_0002/metrics.jsonl)`).
Disclosed in the results like the #90 patch: the official's own process
is otherwise untouched. The sink is a logging stand-in.

**Known degradations (documented in the results):**
- TFDS 4.9.3 rejects the deliberately empty `val` split ("No examples
  were yielded"). The writer's primary (train + empty val, which would
  let the official use 100% of the subset) always falls back to
  train-only. The official then trains `train[:95%]`: **47/50
  episodes** (the last 3 of the subset, held out in a fixed order).
  `dataset_statistics.json` is computed on all 50 (1000 transitions).
- The loss curve is the official's `train_loss` smoothed over the
  grad-accum window. It logs every 10 gradient steps (the pinned script's
  own cadence. Not per micro-batch). `action_accuracy` is TRAIN accuracy.
  The held-out comparison stays the console side's `val.json`.

**Silicon result (host65/RTX 4070 12 GB, batch 1. The 2026-10-26 re-run
on `2026.10.2`. The original 2026-10-22 run on `2026.10.1` is identical
within noise):**

| Metric | Value |
|---|---|
| total wall (row: podman + RLDS + load + train + checkpoint + CPU merge) | 1645.4 s (27:25. 2026-10-22: 1644.8 s / 27:24) |
| training loop | 375 steps (125/epoch x 3) at 4.25 s/step (tqdm 26:32. 2026-10-22: 4.24 s/step) |
| VRAM (nvidia-smi, stable) | 8971 MiB (73% of 12288). Both runs |
| LoRA footprint | 110,828,288 / 7,652,065,472 trainable (1.4483%). r32, alpha 16 (the official's hardcoded `min(rank, 16)`), bias none |
| artifacts (file contract, workdir-relative) | adapter dir 968,788,735 B (`adapter_model.safetensors` 968,782,584 B). merged 15,086,381,550 B (4 safetensors shards + tokenizer/config + `dataset_statistics.json`). **Bit-identical across both runs** (deterministic subset + config) |
| exp_id | `openvla-7b+toto+b8+lr-0.0005+lora-r32+dropout-0.0+q-4bit--j_20261022_0002--image_aug`. Identical across both runs |
| loss curve | **captured via the wandb sink (2026-10-26 re-run. The 2026-10-22 run used `2026.10.1` and lost it)**. 304 rows in the job's `metrics.jsonl`. steps 0..370 (total 375): `loss` 9.63 -> 1.11 (min 0.97). `action_accuracy` 0.14 -> 0.75. `l1_loss` 0.56 -> 0.19. The hardware row asserts these (int step, finite loss, monotone steps, passthrough keys) |

**Official-vs-recipe deltas** (the comparison is shape/convergence, not
point-for-point):

| Axis | Official (crosscheck) | Console recipe |
|---|---|---|
| quant | 4-bit only (`use_quantization` -> nf4) | 8-bit (bnb8) |
| LoRA alpha | 16 (hardcoded `min(rank, 16)`) | 32 |
| data order | unseeded TF shuffle | seeded split, in order |
| eff-batch | 8 (batch 1 x grad_accum 8) | 8 (same) |
| loss visibility | wandb sink -> the job's `metrics.jsonl` (smoothed, every 10 grad steps. `action_accuracy`/`l1_loss` ride along) | per-step `metrics.jsonl` |

**Ops (this row):** LONG (27 min on the 4070. Run it detached:
`setsid nohup` + poll). The pytest log only grows when the podman process
exits (capture_output). A log of about 272 B mid-run is normal, not a hang.
`pgrep -f test_crosscheck` self-matches its own command line. Confirm
liveness via nvidia-smi (idle = 27 MiB) + the log. Re-kick:
`rm -rf <job_dir>/{crosscheck_rlds,crosscheck_out,metrics.jsonl}` first
(TFDS reuses an existing `toto/1.0.0`. Stale data would poison the run.
The sink APPENDS. A same-job re-run would mix stale rows). Then
`GPU_TESTS=1 ROBOVLA_IMAGE_CUDA=robovla-runner-cuda:2026.10.2
ROBOVLA_WORK_DIR=... ROBOVLA_MODEL_DIR=... setsid nohup .venv/bin/python -m
pytest tests/hardware/test_selftest.py::test_crosscheck_cuda -m hardware
-v -rs > /tmp/...log 2>&1 &`. The row persists the official's captured
output to `<job_dir>/official_output.log` (the agentless row captures
what the agent's job log would carry. On failure the `[rank0]` traceback
is in the same log. The forwarding fix works).

## AMD / ROCm (home Strix Halo 8060S, gfx1151)

**Status: designed, not yet verified** - the home lab is off-campus
(roboVLA is LAN-only. This is a self-contained dev lab, never a fleet node).
The torch-rocm wheel is a custom gfx1151 build.

**Image:** `robovla-runner-rocm:2026.10` - `images/Dockerfile.rocm`.
bf16, no quantization (no bitsandbytes). The home selftest is the
`home_selftest` job: bf16 GEMM benchmark + bf16 load + 10-step bf16 fwd/bwd.
The torch wheel is a **build arg** (no public gfx1151 wheel):

```
podman build --build-arg TORCH_ROCM_WHEEL=<custom gfx1151 torch wheel URL> \
    -t robovla-runner-rocm:2026.10 -f images/Dockerfile.rocm .
```

**Host prerequisites:** the same rootless podman stack as the NVIDIA section
(podman, uidmap + setuid newuidmap, slirp4netns, fuse-overlayfs, cni-plugins,
netavark. cgroup v2, subuid/subgid). *Without* the NVIDIA toolkit. GPU access
is plain device passthrough (no CDI): `/dev/kfd` + `/dev/dri` must be
accessible to the node user (usually the `video`+`render` groups).

**Run:** the quadlet's ROCm variant. Replace the four nvidia `--device`
lines with `--device /dev/kfd --device /dev/dri --security-opt
seccomp=unconfined`. Set `ROBOVLA_IMAGE=robovla-runner-rocm:2026.10`
in `node.env`.

**Verify:** `robovla cli gpu_selftest` on the node reports `home_selftest`
semantics when the node profile is the home lab (bf16 GEMM TFLOPS + bf16
fwd/bwd. No 8-bit section, no bnb).

## Troubleshooting (found the hard way)

| Symptom | Cause / fix |
|---|---|
| container agent exits instantly, exit 0, zero output | ENTRYPOINT was `-m agent.server` (no `__main__` guard). It must be `-m agent`. Pinned by `tests/integration/test_agent_entrypoint.py` |
| job crashes `No such file or directory: /opt/robovla/bin/robovla_job` | `ROBOVLA_JOB_BIN` not wired (now `partial(job_argv, bin_path=cfg.job_bin)` in `agent/server.py`) |
| `ImportError: cannot import name 'load_episode' from 'robovla.episode'` | it is the file seam `robovla.fs` (`images/engine.py`) |
| `write to uid_map failed: Operation not permitted` | non-setuid `newuidmap`. Install the distro `uidmap` as root. If `tools/bin` is first in `$PATH`, symlink `tools/bin/newuidmap` to `/usr/bin/newuidmap` (podman resolves it via PATH) |
| `could not find pasta, the network namespace can't be configured` | podman 5.x's compiled-in rootless default backend is `pasta` (no fallback). Set `[network] default_rootless_network_cmd = "slirp4netns"` in `containers.conf` |
| `could not find "rootlessport" in one of [...]` / `Failed to obtain podman configuration: ... helper_binaries_dir` | rootless `-p` needs the `rootlessport` binary in `[engine] helper_binaries_dir` (**array**, not `$PATH`). It is not in the CS9 set. Cross-compile `cmd/rootlessport` from the matching podman tag (see the scratch-box section) |
| torch 2.14+cu130 appears in the image | torch+torchvision installed separately (pin drift). One uv command, cu128 index |
| `timm` version error at model load | pin timm to the hard-checked set (0.9.10 in the image) |
| bf16 fwd/bwd skipped in selftest | by design on <= 12 GB GPUs (the `free < 20 GB` branch) |
| model re-downloads inside the container | `HF_HUB_CACHE=/models` missing from `node.env` (the env file passes through) |
| crosscheck job: official trains fine but no loss curve in the run detail | the image predates the sink COPY (`/opt/robovla/sink/wandb.py` missing). `import wandb` in the child hit the real package and no-opped. Rebuild (`2026.10.2`) |
