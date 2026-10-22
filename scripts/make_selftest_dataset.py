#!/usr/bin/env python3
"""Materialize a small deterministic synthetic dataset for the hardware rows.

The gpu_selftest's predict-action probe + 10-step QLoRA section and the
crosscheck row's 50-episode official finetune both need a real dataset at
``<workdir>/datasets/<dataset-id>/`` (engine._samples / _episodes_raw:
episodes/ep_*/{episode.json, actions.json, frame_*.jpg} + norm_stats.json).
This script materializes them so the rows execute on identical inputs
everywhere (deterministic seed — same data on every node).

Usage:
    python3 make_selftest_dataset.py --workdir /path/to/node-workdir \
        --dataset s01/stop-at-target/v1 [--episodes 6] [--frames 8]

    # the crosscheck row (tests/hardware): the 50-episode dataset
    python3 make_selftest_dataset.py --workdir /path/to/node-workdir \
        --dataset s01/crosscheck-50/v1 --episodes 50 --frames 20

Writes only under <workdir>/datasets/<dataset-id>/ (refuses to overwrite an
existing dataset dir). Frames are 224x224 JPEG (the prep seam's output size);
actions are smooth deterministic velocities so the norm stats are well-posed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError as e:  # the venv carries pillow; keep the failure clean
    print(f"error: pillow is required ({e})", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np  # vectorized frame generation (the venv/image carry it)
except ImportError as e:
    print(f"error: numpy is required ({e})", file=sys.stderr)
    sys.exit(1)

SEED = 20261021          # deterministic: identical dataset on every node
EP_ID = "ep_{:04d}"


def _actions(n: int, seed: int):
    """n smooth deterministic actions (t, vx, vy, wz) in [-1, 1]."""
    rng = random.Random(seed)
    phase = rng.uniform(0.0, 2 * math.pi)
    out = []
    for i in range(n):
        w = i * 0.35 + phase
        out.append({
            "t": round(i * 0.1, 6),
            "vx": round(0.7 * math.sin(w), 6),
            "vy": round(0.5 * math.sin(2 * w + 1.0), 6),
            "wz": round(0.6 * math.sin(3 * w + 2.0), 6),
        })
    return out


def _frame(path: Path, ep: int, f: int, seed: int) -> None:
    """A 224x224 JPEG: smooth gradient + seeded per-pixel noise.

    Vectorized (numpy): the 50-episode crosscheck dataset (1000 frames)
    must generate in seconds, not the 30+ minutes the per-pixel loop took.
    Same shape as before: R = 40 + x, G = 200y/224, B = 200 - x, uniform
    noise in [-18, 18], clipped to [0, 255] — the bytes themselves changed
    with the vectorization (deterministic per seed, as before).
    """
    rng = np.random.default_rng(seed * 100000 + ep * 1000 + f)
    xs = np.arange(224, dtype=np.int32)
    base = np.zeros((224, 224, 3), dtype=np.int32)
    base[..., 0] = 40 + xs[None, :]
    base[..., 1] = (200 * np.arange(224, dtype=np.int32)[:, None]) // 224
    base[..., 2] = 200 - xs[None, :]
    noise = rng.integers(-18, 19, size=(224, 224, 3))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, "JPEG", quality=90)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", required=True,
                    help="the node workdir (datasets/ goes under it)")
    ap.add_argument("--dataset", required=True,
                    help="dataset id, e.g. s01/stop-at-target/v1")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    ds_dir = Path(args.workdir) / "datasets" / args.dataset.replace("/", "__")
    if ds_dir.exists():
        # Idempotent by contract (the hardware rows re-materialize before
        # every job): present => leave it untouched and succeed. Never
        # overwrite — the job's data must be stable across runs.
        print(f"{ds_dir} already present — leaving it untouched (no overwrite)")
        return 0
    task = args.dataset.split("/")[1]

    episodes = ds_dir / "episodes"
    episodes.mkdir(parents=True)
    mins = [0.0] * 7
    maxs = [0.0] * 7
    for ep in range(1, args.episodes + 1):
        acts = _actions(args.frames, SEED + ep)
        d = episodes / EP_ID.format(ep)
        d.mkdir()
        (d / "episode.json").write_text(json.dumps(
            {"id": EP_ID.format(ep), "task": task, "success": True}, indent=2))
        (d / "actions.json").write_text(json.dumps(acts, indent=2))
        for f in range(args.frames):
            _frame(d / f"frame_{f + 1:04d}.jpg", ep, f, SEED)
        for a in acts:
            v7 = (a["vx"], a["vy"], a["wz"], 0.0, 0.0, 0.0, 0.0)
            mins = [min(m, v) for m, v in zip(mins, v7)]
            maxs = [max(m, v) for m, v in zip(maxs, v7)]
    # norm stats: the [0,1] contract bounds (min/max per dim, §5.1).
    for i in range(7):
        if maxs[i] == mins[i]:  # keep the affine map well-posed
            maxs[i] = mins[i] + 0.5
    (ds_dir / "norm_stats.json").write_text(json.dumps(
        {"min": mins, "max": maxs}, indent=2))
    n = args.episodes * args.frames
    print(f"dataset {args.dataset} at {ds_dir}: {args.episodes} episodes, "
          f"{n} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
