"""rlds_writer — the crosscheck subset as an official RLDS/TFDS dataset (§6).

The official ``vla-scripts/finetune.py`` (pinned openvla SHA
c8f03f48af692657d3060c19588038c7220e9af9) accepts RLDS/TFDS input only:
``RLDSDataset`` resolves ``tfds.builder(dataset_name, data_dir=data_root_dir)``
and dlimp's ``from_rlds`` reads nested-``steps`` trajectories (the OXE layout).
The crosscheck wrapper materializes the deterministic 50-episode
``core.crosscheck_subset`` here so the official engine sees exactly the
episodes the console engine trains on.

We borrow the official OXE dataset config name ``toto`` (7-dim EEF_POS). Its
standardization transform is the only one that is a pure concat of
``action.{world_vector, rotation_delta, open_gripper}`` + language
pass-through (no timestep slicing, no gripper relabeling), so the course
action convention ``(vx, vy, wz, 0, 0, 0, 0.0)`` flows through the official
pipeline verbatim: the zero-padded dims are "unused" dims in the official
``BOUNDS_Q99`` normalization (``min == max`` dims map to 0 — see
``normalize_action_and_proprio``) and the last dim is excluded from
normalization by the EEF_POS mask.

Raw layout written (one TFDS example per trajectory):

    steps:
      observation.image                          jpeg bytes, 224^2 (prep size)
      observation.natural_language_instruction   task string
      action.world_vector / rotation_delta / open_gripper   float32 (3/3/1)
      first / last / is_terminal / episode_index / frame_index / next

A deliberately EMPTY ``val`` split is written alongside ``train``: the
official loader uses split ``train`` only when a ``val`` split exists (else
it slices ``train[:95%]``) — with the empty val the crosscheck trains on
100% of the subset, the same data as the console engine. (If the TFDS build
ever rejects an empty split, the writer falls back to a train-only dataset
and the official side trains ``train[:95%]`` — documented degradation.)

The pure builders below are CPU-tested; :func:`write_rlds` is the lazy stack
boundary (tensorflow / tensorflow_datasets imported on call — a box without
them fails with a clean §9.6 ``internal``, never a traceback, per the
engine's lazy-boundary rule §11). TF never touches the GPU:
``set_visible_devices([], "GPU")`` at the boundary (the official rlds loader
does the same; torch owns the GPU).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Sequence

from robovla.core.errors import RobovlaError

from .contract import ACTION_DIMS

#: The official OXE config name our materialized dataset borrows (7-dim
#: EEF_POS; see the module docstring for why this one and not another).
RLDS_NAME = "toto"

#: The TFDS dataset version (the official loader reads <root>/<name>/1.0.0/).
RLDS_VERSION = "1.0.0"

_EP_ID_RE = re.compile(r"^ep_(\d+)$")


# -- pure builders (CPU-tested) ----------------------------------------------------


def action_to_rlds(action: Sequence[float]) -> dict[str, list[float]]:
    """One 7-dim course action -> the official EEF_POS action parts.

    ``(vx, vy, wz, 0, 0, 0, 0.0)`` -> ``world_vector (3) + rotation_delta (3)
    + open_gripper (1)`` — the official ``toto`` standardize transform
    concats them back in exactly this order, so the official pipeline sees
    our 7-dim row unchanged.
    """
    if len(action) != ACTION_DIMS:
        raise RobovlaError("validation",
                           f"rlds action must have {ACTION_DIMS} dims, "
                           f"got {len(action)}")
    out = [0.0] * ACTION_DIMS
    for i, v in enumerate(action):
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            raise RobovlaError("validation",
                               f"rlds action dim {i} must be a finite number")
        out[i] = float(v)
    return {"world_vector": out[0:3], "rotation_delta": out[3:6],
            "open_gripper": out[6:7]}


def _episode_index(episode_id: str) -> int:
    """``ep_0007`` -> 7 (the ep id pattern is enforced upstream: fs seam)."""
    m = _EP_ID_RE.fullmatch(episode_id)
    if m is None:
        raise RobovlaError("validation",
                           f"episode id {episode_id!r} must look like 'ep_0042'")
    return int(m.group(1))


def build_episode_sample(
    episode_id: str,
    task: str,
    frame_bytes: Sequence[bytes],
    actions: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """One trajectory as the TFDS example dict the writer serializes.

    ``frame_bytes`` / ``actions`` are the 1:1 frame/action rows (10 Hz —
    the episode validator's contract); the RLDS metadata (first/last/
    episode_index/frame_index/next) follows the standard OXE convention.
    The sample carries EXACTLY the writer schema's keys (the TFDS
    serializer rejects extras).
    """
    if not frame_bytes or len(frame_bytes) != len(actions):
        raise RobovlaError(
            "validation",
            f"episode {episode_id}: {len(actions)} actions vs "
            f"{len(frame_bytes)} frames (1:1 required)")
    task_bytes = task.encode("utf-8")
    ep = _episode_index(episode_id)
    n = len(actions)
    steps = []
    for i in range(n):
        last = i == n - 1
        if last:
            # OXE convention: the last step's `next` points at the next
            # trajectory's first step (never dereferenced by the finetune
            # path: window_size=1, goal relabeling unused).
            nxt = {"first": True, "last": False,
                   "episode_index": ep + 1, "frame_index": 0}
        else:
            nxt = {"first": False, "last": i + 1 == n - 1,
                   "episode_index": ep, "frame_index": i + 1}
        steps.append({
            "observation": {
                "image": frame_bytes[i],
                "natural_language_instruction": task_bytes,
            },
            "action": action_to_rlds(actions[i]),
            "first": i == 0,
            "last": last,
            "is_terminal": last,
            "episode_index": ep,
            "frame_index": i,
            "next": nxt,
        })
    return {"steps": steps}


def max_steps_for(total_frames: int, batch: int, grad_accum: int,
                  epochs: int) -> int:
    """The official loop's stopping point in optimizer steps.

    The official script runs ``max_steps`` gradient steps over an infinite
    (repeating) RLDS dataset — it has no epoch flag. With an effective
    batch of ``batch x grad_accum``: ``epochs x (total_frames // eff)``
    (the trailing partial chunk is dropped by the official loop, same as
    the console engine's micro-batch chunks). Floor 1: a sub-effective-
    batch dataset still trains one step.
    """
    for v, what in ((total_frames, "total_frames"), (batch, "batch"),
                    (grad_accum, "grad_accum"), (epochs, "epochs")):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise RobovlaError("validation",
                               f"max_steps: {what} must be an int >= 1")
    per_epoch = total_frames // (batch * grad_accum)
    return max(1, per_epoch * epochs)


def exp_id(model_id: str, dataset_name: str, eff_batch: int, lr: float,
           lora_rank: int, lora_dropout: float, use_quant: bool,
           note: str | None, image_aug: bool) -> str:
    """The official finetune.py exp_id verbatim (use_lora is always True for
    crosscheck — the QLoRA recipe). The wrapper resolves the official
    output dirs from this; keep in lockstep with the pinned script."""
    e = (f"{model_id.split('/')[-1]}+{dataset_name}"
         f"+b{eff_batch}+lr-{lr}")
    e += f"+lora-r{lora_rank}+dropout-{lora_dropout}"
    if use_quant:
        e += "+q-4bit"
    if note is not None:
        e += f"--{note}"
    if image_aug:
        e += "--image_aug"
    return e


# -- the TFDS writer (the lazy stack boundary) -------------------------------------


def write_rlds(root: str | Path, name: str,
               samples: Sequence[dict[str, Any]]) -> None:
    """Materialize ``samples`` as ``<root>/<name>/1.0.0/`` (TFDS arrow).

    Lazy by design: tensorflow / tensorflow_datasets are imported HERE (the
    runner image carries them; the CPU suite never does) — a missing stack
    is a clean §9.6 ``internal``, not a traceback. Writes the ``train``
    split (all trajectories) + an empty ``val`` split (see the module
    docstring: it makes the official loader use 100% of the subset).
    """
    if not samples:
        raise RobovlaError("dataset_not_ready",
                           "no episodes to materialize as an RLDS dataset")
    try:
        import numpy as np
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as e:
        raise RobovlaError(
            "internal",
            f"crosscheck stack unavailable (tensorflow): {e}") from e
    # torch owns the GPU on this node (the official rlds loader does the same)
    tf.config.set_visible_devices([], "GPU")

    step_features = tfds.features.FeaturesDict({
        "observation": tfds.features.FeaturesDict({
            "image": tfds.features.Image(
                shape=(224, 224, 3), dtype=np.uint8, encoding_format="jpeg"),
            "natural_language_instruction": tfds.features.Text(),
        }),
        "action": tfds.features.FeaturesDict({
            "world_vector": tfds.features.Tensor(shape=(3,), dtype=np.float32),
            "rotation_delta": tfds.features.Tensor(shape=(3,), dtype=np.float32),
            "open_gripper": tfds.features.Tensor(shape=(1,), dtype=np.float32),
        }),
        "first": tfds.features.Scalar(dtype=np.bool_),
        "last": tfds.features.Scalar(dtype=np.bool_),
        "is_terminal": tfds.features.Scalar(dtype=np.bool_),
        "episode_index": tfds.features.Scalar(dtype=np.int64),
        "frame_index": tfds.features.Scalar(dtype=np.int64),
        "next": tfds.features.FeaturesDict({
            "first": tfds.features.Scalar(dtype=np.bool_),
            "last": tfds.features.Scalar(dtype=np.bool_),
            "episode_index": tfds.features.Scalar(dtype=np.int64),
            "frame_index": tfds.features.Scalar(dtype=np.int64),
        }),
    })
    features = tfds.features.FeaturesDict({"steps":
                                           tfds.features.Dataset(step_features)})

    def _example(sample: dict[str, Any]) -> dict[str, Any]:
        """The pure sample -> the schema-typed example (dtype casts at the
        boundary; jpeg bytes / strings pass through)."""
        return {
            "steps": [
                {
                    "observation": {
                        "image": st["observation"]["image"],
                        "natural_language_instruction":
                            st["observation"]["natural_language_instruction"],
                    },
                    "action": {k: np.array(v, dtype=np.float32)
                               for k, v in st["action"].items()},
                    "first": st["first"],
                    "last": st["last"],
                    "is_terminal": st["is_terminal"],
                    "episode_index": st["episode_index"],
                    "frame_index": st["frame_index"],
                    "next": {k: st["next"][k]
                             for k in ("first", "last", "episode_index",
                                       "frame_index")},
                }
                for st in sample["steps"]
            ]
        }

    # Class name lowercases to `name` (the TFDS output dir) AND the explicit
    # `name` attribute — belt and suspenders for the tfds name derivation.
    builder_cls = type(name[0].upper() + name[1:],
                       (tfds.core.DatasetBuilder,), {
                           "name": name,
                           "VERSION": tfds.core.Version(RLDS_VERSION),
                           "RELEASE_NOTES": {RLDS_VERSION:
                                             "roboVLA crosscheck subset "
                                             "(deterministic, seeded per "
                                             "student + dataset version)."},
                           "_info": lambda self: self.dataset_info_from_configs(
                               features=features,
                               description="roboVLA crosscheck subset "
                                           "(images/rlds_writer.py)."),
                           "_split_generators": lambda self, dl_manager: {
                               "train": iter(map(_example, samples)),
                               "val": iter(()),  # empty on purpose (module
                               # docstring): the official loader then uses
                               # split "train" in full, not "train[:95%]".
                           },
                       })
    builder = builder_cls(data_dir=str(root))
    try:
        builder.download_and_prepare()
        return
    except Exception:
        # Documented fallback: if this TFDS version rejects an empty split,
        # write train-only (the official side then trains "train[:95%]").
        print(f"rlds_writer: empty val split rejected — falling back to a "
              f"train-only dataset (official side will use train[:95%])",
              flush=True)
    builder_cls = type(name[0].upper() + name[1:],
                       (tfds.core.DatasetBuilder,), {
                           "name": name,
                           "VERSION": tfds.core.Version(RLDS_VERSION),
                           "RELEASE_NOTES": {RLDS_VERSION:
                                             "roboVLA crosscheck subset."},
                           "_info": lambda self: self.dataset_info_from_configs(
                               features=features,
                               description="roboVLA crosscheck subset "
                                           "(images/rlds_writer.py)."),
                           "_split_generators": lambda self, dl_manager: {
                               "train": iter(map(_example, samples)),
                           },
                       })
    builder_cls(data_dir=str(root)).download_and_prepare()
