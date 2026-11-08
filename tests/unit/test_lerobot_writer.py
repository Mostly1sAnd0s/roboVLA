"""images.lerobot_writer — roboVLA episodes -> LeRobotDataset (SmolVLA
family input).

The pure builders are tested against REAL prep-seam JPEGs (224^2 — the
size contract is enforced, not assumed); the writer itself is the lazy
lerobot boundary (the CPU suite never imports lerobot: the recorder seam
and the clean §9.6 internal on a missing stack are what is pinned here —
the real write lands at the hardware row).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from images import lerobot_writer
from images.lerobot_writer import (FPS, SMOLVLA_CAMERA_KEY,
                                   SMOLVLA_STATE_KEY, build_episode_sample,
                                   decode_frame, features_spec,
                                   lerobot_steps, write_lerobot_dataset)
from robovla.core.errors import RobovlaError


def _jpeg(w: int = 224, h: int = 224) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


# -- features + steps (pure) ------------------------------------------------------


def test_features_spec_single_camera_three_dim_action():
    f = features_spec()
    assert set(f) == {SMOLVLA_CAMERA_KEY, "action", SMOLVLA_STATE_KEY}
    cam = f[SMOLVLA_CAMERA_KEY]
    assert cam["dtype"] == "image" and cam["shape"] == (224, 224, 3)
    act = f["action"]
    assert act["dtype"] == "float32" and act["shape"] == (3,)
    assert act["names"] == ["vx", "vy", "wz"]


def test_state_feature_is_the_required_zero_placeholder():
    # prepare_state reads observation.state UNCONDITIONALLY (the CPU
    # verification crashed the first train step without it, 2026-11-08);
    # the rover has no proprioception, so the placeholder is 3 zeros and
    # the normalizer's std+eps keeps it numerically exact.
    st = features_spec()[SMOLVLA_STATE_KEY]
    assert st["dtype"] == "float32" and st["shape"] == (3,)


@pytest.mark.parametrize("frames,batch,epochs,want", [
    (100, 4, 3, 75),   # (100 // 4) * 3  (loop steps = micro-batches)
    (100, 4, 1, 25),
    (3, 4, 1, 1),      # sub-batch dataset still takes one step
    (8, 8, 10, 10),
])
def test_lerobot_steps(frames, batch, epochs, want):
    assert lerobot_steps(frames, batch, epochs) == want


@pytest.mark.parametrize("args", [(0, 4, 1), (10, 0, 1), (10, 4, 0),
                                  (True, 4, 1), (10.0, 4, 1)])
def test_lerobot_steps_rejects(args):
    with pytest.raises(RobovlaError):
        lerobot_steps(*args)


def test_fps_is_the_episode_convention():
    assert FPS == 10


# -- sample building (pure over decoded frames) ------------------------------------


def test_build_episode_sample_decodes_and_slices():
    ep = build_episode_sample(
        "ep_0007", "park at the cone", [_jpeg(), _jpeg(), _jpeg()],
        [[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
         [0.2, -0.1, 0.4, 0.0, 0.0, 0.0, 0.0],
         [0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]])
    assert ep["id"] == "ep_0007" and ep["task"] == "park at the cone"
    assert len(ep["frames"]) == 3
    assert ep["frames"][0].dtype == np.uint8
    assert ep["frames"][0].shape == (224, 224, 3)
    assert ep["actions"][0] == [0.1, -0.2, 0.3]  # 7-dim padding sliced off


def test_decode_frame_rejects_wrong_size():
    with pytest.raises(RobovlaError) as ei:
        decode_frame(_jpeg(160, 120))
    assert ei.value.code == "dataset_not_ready"
    assert "224" in ei.value.message


def test_decode_frame_rejects_garbage():
    with pytest.raises(RobovlaError) as ei:
        decode_frame(b"not a jpeg")
    assert ei.value.code == "dataset_not_ready"


@pytest.mark.parametrize("args", [
    ("ep_1", "t", [], []),                # no frames
    ("ep_1", "t", [_jpeg()], []),         # count mismatch
    ("ep_1", "", [_jpeg()], [[0] * 7]),   # empty task
])
def test_build_episode_sample_rejects(args):
    with pytest.raises(RobovlaError):
        build_episode_sample(*args)


# -- the lazy boundary --------------------------------------------------------------


def test_missing_lerobot_is_a_clean_internal(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_lerobot(name, *a, **kw):
        if name.startswith("lerobot"):
            raise ImportError(name=name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_lerobot)
    sample = build_episode_sample("ep_0001", "t", [_jpeg()], [[0.0] * 7])
    with pytest.raises(RobovlaError) as ei:
        write_lerobot_dataset(tmp_path / "ds", "robovla/j_x", [sample])
    assert ei.value.code == "internal"
    assert "lerobot" in ei.value.message
