"""smolvla — the shared pure data plane for the SmolVLA model family.

One place for the facts that must agree across the training wrapper
(``images/smolvla_finetune.py``), the dataset converter
(``images/lerobot_writer.py``), the serving seam (``agent/serve.py``) and
the QA tests. The wheel ships this module to the console AND the runner
image, so the wrapper and the agent can never disagree on the mapping.

The mapping (the §9.5 contract stays openvla-shaped for every consumer —
the agent poll, the SSE events, ``/api/robot/predict``, the 7-float
probe):

  * course action ``(vx, vy, wz)`` (setup §6) <-> SmolVLA 3-dim action;
  * the house ``action_norm`` is the 7-dim ``[0, 1]`` vector:
    ``normalize((vx, vy, wz, 0, 0, 0, 0), dataset min/max stats)`` — the
    same affine map the openvla path uses, so the console denormalizes
    with the SAME dataset stats and the Pi sees the SAME ``[-1, 1]``
    velocities;
  * the serving metadata rides INSIDE the trained artifact dir
    (``serving.json``): the wrapper writes it at success, the serving
    seam reads it at load. An artifact without it is not servable —
    loud failure beats a silently wrong denormalization.

SmolVLA gets NO proprioceptive state: the rover dataset has none, and a
constant-zero state feature would give mean/std normalization a zero
std-dev. The converter therefore writes images + actions + task only.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .core.errors import RobovlaError
from .core.norm_stats import NormStats, normalize

#: The base checkpoint (the model card id; also the family marker — see
#: ``core.spec.model_family``).
SMOLVLA_MODEL_ID = "lerobot/smolvla_base"

#: The single camera feature key the converter writes and the serving seam
#: reads (lerobot ``OBS_IMAGES`` convention; the policy derives its input
#: features from the dataset, so this string IS the interface).
SMOLVLA_CAMERA_KEY = "observation.images.camera1"

#: Course driving action width (vx, vy, wz) — the SmolVLA action width for
#: this embodiment (padded internally to ``max_action_dim``).
SMOLVLA_ACTION_DIM = 3

#: The house action contract width (``images/contract.ACTION_DIMS``). The
#: wheel cannot import ``images`` (not packaged), so the value lives here;
#: a unit test pins the two constants against each other.
HOUSE_ACTION_DIM = 7

#: Serving-metadata file name inside the trained artifact dir.
SERVING_META_NAME = "serving.json"


def to_smolvla_action(a7: Sequence[float]) -> list[float]:
    """House 7-dim action -> the 3-dim driving action (first three dims).

    The course convention pads dims 3-6 with zeros (``episode.to_action_7d``);
    slicing them off is the exact inverse. Validates like the norm-stats
    layer: finite numbers, correct width.
    """
    if isinstance(a7, (str, bytes)) or len(a7) < SMOLVLA_ACTION_DIM:
        raise RobovlaError(
            "validation",
            f"action needs >= {SMOLVLA_ACTION_DIM} dims (vx, vy, wz)")
    out = []
    for i, v in enumerate(a7[:SMOLVLA_ACTION_DIM]):
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            raise RobovlaError("validation",
                               f"action dim {i} must be a finite number")
        out.append(float(v))
    return out


def to_house_action(raw: Sequence[float], stats: NormStats) -> tuple[float, ...]:
    """Raw robot action -> the house 7-dim ``action_norm`` (the §9.5 agent
    return): pad to 7 with zeros, then normalize with the dataset min/max.
    Accepts the raw 3-dim driving vector or a wider padded vector.

    Overshoot note (family delta): flow-matching output is CONTINUOUS and
    is not clamped to the training range, so a served action can leave
    [0, 1] (an openvla bin never does). No clamp here on purpose — the
    Pi side already clamps velocities to the autonomy caps (setup §6),
    and clamping twice hides the model being wrong from the class."""
    if isinstance(raw, (str, bytes)) or len(raw) < SMOLVLA_ACTION_DIM:
        raise RobovlaError(
            "validation",
            f"served action needs >= {SMOLVLA_ACTION_DIM} dims, "
            f"got {len(raw) if hasattr(raw, '__len__') else type(raw).__name__}")
    a7 = [0.0] * HOUSE_ACTION_DIM
    vals = list(raw)[:HOUSE_ACTION_DIM]
    a7[:len(vals)] = [float(v) for v in vals]
    return normalize(a7, stats)


def serving_meta_dict(stats: NormStats) -> dict[str, Any]:
    """The ``serving.json`` payload: family + action width + the dataset
    min/max stats (the exact 7-dim house stats file contents)."""
    return {
        "family": "smolvla",
        "action_dim": SMOLVLA_ACTION_DIM,
        "camera": SMOLVLA_CAMERA_KEY,
        "norm_stats": stats.to_dict(),
    }


def parse_serving_meta(d: Any) -> NormStats:
    """Validate one ``serving.json`` payload -> the NormStats the serving
    seam uses. Anything malformed is a ``validation`` error (the artifact
    is not servable)."""
    if not isinstance(d, dict):
        raise RobovlaError("validation", "serving meta must be an object")
    if d.get("family") != "smolvla":
        raise RobovlaError("validation",
                           f"serving meta family must be 'smolvla', "
                           f"got {d.get('family')!r}")
    if d.get("action_dim") != SMOLVLA_ACTION_DIM:
        raise RobovlaError("validation",
                           f"serving meta action_dim must be "
                           f"{SMOLVLA_ACTION_DIM}, got {d.get('action_dim')!r}")
    ns = d.get("norm_stats")
    if not isinstance(ns, dict):
        raise RobovlaError("validation", "serving meta needs norm_stats")
    try:
        stats = NormStats(min=tuple(ns["min"]), max=tuple(ns["max"]))
    except (KeyError, TypeError, RobovlaError) as e:
        raise RobovlaError("validation",
                           f"serving meta norm_stats unusable: {e}") from e
    return stats
