"""lerobot_writer — roboVLA episodes as a LeRobotDataset (SmolVLA family).

The SmolVLA fine-tune wrapper (``images/smolvla_finetune.py``) runs the
OFFICIAL ``lerobot-train`` (lerobot v0.7.0 pinned in
``images/Dockerfile.cuda-smolvla``), whose only dataset input is the
LeRobotDataset format: parquet rows + PNG images (``use_videos=False`` —
no ffmpeg in the runner image) + meta. This module converts the SAME
agent-pulled roboVLA bundle the console engine reads
(``episodes/ep_*/{episode.json, actions.json, frame_*.jpg}`` +
``norm_stats.json``) into that format, so both families train on
identical frames and raw actions.

Layout written (one episode per ``save_episode``):

    frame rows:  observation.images.camera1  uint8 (224, 224, 3) PNG
                 action                      float32 [vx, vy, wz]  (10 Hz)
                 task                        the episode task string

Deliberate choices (documented deltas from the openvla path):

  * NO ``observation.state``: the rover dataset carries no proprioception.
    A constant-zero state feature would hand mean/std normalization a
    zero std-dev; omitting the feature removes state from the policy's
    dataset-derived input features entirely (``make_policy`` derives
    features from the dataset; the wrapper passes
    ``--policy.input_features=null``).
  * 3-dim actions (``robovla.smolvla.to_smolvla_action`` slices the
    course 7-dim padding): SmolVLA pads to ``max_action_dim=32``
    internally, and the serving seam un-slices with the dataset stats.
  * images are the prep seam's 224^2 JPEGs decoded to uint8 arrays
    (SmolVLA pads them to 512^2 internally —
    ``resize_imgs_with_padding``).
  * only the seeded train split reaches the writer (the wrapper applies
    ``core.split.split``; the held-out episodes never train either
    family — the same split the console engine and the crosscheck use).

The pure builders below are CPU-tested; :func:`write_lerobot_dataset` is
the lazy stack boundary (lerobot imported on call — a missing stack is a
clean §9.6 ``internal``, never a traceback, per the §11 lazy-boundary
rule).
"""

from __future__ import annotations

import io
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from robovla.core.errors import RobovlaError
from robovla.smolvla import SMOLVLA_ACTION_DIM, SMOLVLA_CAMERA_KEY, \
    to_smolvla_action

#: The episode capture rate (the shared episode validator's contract:
#: actions.json and frames are 1:1 at 10 Hz).
FPS = 10

#: Frame size the prep seam writes (224^2; re-checked at the boundary —
#: a wrong-shaped batch would silently resize-pad and skew training).
FRAME_HW = (224, 224)


# -- pure builders (CPU-tested) ----------------------------------------------------


def features_spec(h: int = FRAME_HW[0], w: int = FRAME_HW[1]) -> dict[str, dict]:
    """The LeRobotDataset feature schema for the course rover (write mode).

    ``dtype="image"`` stores PNG frames inside the dataset (no video
    encoding, no ffmpeg dependency). Action names carry the driving dims
    so dataset metadata stays self-describing for a class read-through.
    """
    return {
        SMOLVLA_CAMERA_KEY: {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (SMOLVLA_ACTION_DIM,),
            "names": ["vx", "vy", "wz"],
        },
    }


def decode_frame(frame_bytes: bytes,
                 *, size: tuple[int, int] = FRAME_HW):
    """One stored JPEG -> uint8 (h, w, 3) numpy array, size-checked.

    The prep seam guarantees 224^2 (upload-time prep); a mismatch is a
    corrupt bundle — fail loud rather than train on silent resizes.
    """
    import numpy as np
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    except Exception as e:
        raise RobovlaError("dataset_not_ready",
                           f"frame jpeg undecodable: {e.__class__.__name__}: {e}") from e
    if img.size != (size[1], size[0]):
        raise RobovlaError(
            "dataset_not_ready",
            f"frame is {img.size[0]}x{img.size[1]}, expected "
            f"{size[1]}x{size[0]} (the prep seam writes {size[1]}^2)")
    return np.asarray(img, dtype=np.uint8)


def build_episode_sample(episode_id: str, task: str,
                         frame_bytes: Sequence[bytes],
                         actions: Sequence[Sequence[float]]) -> dict[str, Any]:
    """One roboVLA episode -> the writer sample dict.

    ``frames`` are decoded uint8 arrays, ``actions`` the 3-dim raw
    driving rows (1:1, 10 Hz — the episode validator's contract). The
    task string rides along (the dataset language column).
    """
    if not frame_bytes or len(frame_bytes) != len(actions):
        raise RobovlaError(
            "validation",
            f"episode {episode_id}: {len(actions)} actions vs "
            f"{len(frame_bytes)} frames (1:1 required)")
    if not isinstance(task, str) or not task:
        raise RobovlaError("validation", f"episode {episode_id}: task required")
    return {
        "id": episode_id,
        "task": task,
        "frames": [decode_frame(b) for b in frame_bytes],
        "actions": [to_smolvla_action(a) for a in actions],
    }


def lerobot_steps(frames_total: int, batch: int, epochs: int) -> int:
    """The ``--steps`` value for one house (epochs, batch) recipe.

    lerobot-train's loop step is one MICRO-batch: with gradient
    accumulation the optimizer updates every N loop steps, so the loop
    must run ``epochs x (frames // batch)`` steps to give the house
    ``epochs`` over the data (floor 1 — a sub-batch dataset still takes
    one step, same rule as ``rlds_writer.max_steps_for``). Gradient
    accumulation changes the effective batch, never the sample count,
    so it does not appear here.
    """
    for v, what in ((frames_total, "frames_total"), (batch, "batch"),
                    (epochs, "epochs")):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise RobovlaError("validation",
                               f"lerobot_steps: {what} must be an int >= 1")
    if not math.isfinite(float(frames_total)):
        raise RobovlaError("validation", "lerobot_steps: frames_total finite")
    return max(1, (frames_total // batch) * epochs)


# -- the lerobot writer (the lazy stack boundary) -----------------------------------


def write_lerobot_dataset(root: str | Path, repo_id: str,
                          samples: Sequence[dict[str, Any]],
                          *, fps: int = FPS,
                          robot_type: str = "mecanum_rover") -> None:
    """Materialize ``samples`` as a LeRobotDataset at ``root``.

    ``repo_id`` is the local handle the trainer resolves together with
    ``--dataset.root`` (no Hub traffic — the node runs
    ``HF_HUB_OFFLINE=1``). A pre-existing ``root`` is REMOVED first (a
    re-run of the same job id over a crashed attempt must not merge into
    the partial dataset).
    """
    if not samples:
        raise RobovlaError("dataset_not_ready",
                           "no episodes to materialize as a LeRobot dataset")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise RobovlaError(
            "internal",
            f"smolvla stack unavailable (lerobot): {getattr(e, 'name', e)}") from e
    root = Path(root)
    try:
        if root.exists():
            shutil.rmtree(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        ds = LeRobotDataset.create(
            repo_id=repo_id, fps=fps, features=features_spec(),
            root=root, robot_type=robot_type, use_videos=False)
        for ep in samples:
            for frame, action in zip(ep["frames"], ep["actions"]):
                ds.add_frame({SMOLVLA_CAMERA_KEY: frame,
                              "action": action, "task": ep["task"]})
            ds.save_episode()
        ds.finalize()
    except RobovlaError:
        raise
    except Exception as e:
        # The §11 lazy-boundary rule: an unexpected stack failure is a
        # clean §9.6 internal — never a raw exception the wrapper's
        # spec-read handler could mislabel.
        raise RobovlaError(
            "internal",
            f"LeRobot dataset materialization failed: "
            f"{type(e).__name__}: {e}") from e
