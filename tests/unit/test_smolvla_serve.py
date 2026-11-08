"""agent.serve — the SmolVLA serving seam (the family's half of the Model
protocol) + the factory's FAMILY routing.

The stack load is injected (the CPU suite has no torch/lerobot): the
fakes return plain objects, and what is pinned is the SEAM contract —
raw action -> house 7-dim [0,1] via the dataset stats stamped in
serving.json (the §9.5 promise: the console denormalizes with the SAME
stats the wrapper trained with), the loud failures (no policy dir, no
serving.json, empty prompt, unreadable meta), and the factory choosing
the model class from the run's spec.json model id — the same
``model_family`` derivation the agent argv uses (one source).
"""

from __future__ import annotations

import base64 as b64
import io
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent.serve import (SmolVlaModel, TorchVlaModel,
                         make_default_model_factory)
from robovla.core.errors import RobovlaError
from robovla.core.norm_stats import NormStats
from robovla.smolvla import SERVING_META_NAME, serving_meta_dict

RAW_ACTION = [0.5, -0.25, 0.1]


def _fake_torch():
    """The minimal torch surface SmolVlaModel.predict touches."""
    @contextmanager
    def inference_mode():
        yield
    return SimpleNamespace(inference_mode=inference_mode,
                           from_numpy=lambda a: a,
                           asarray=lambda a: np.asarray(a))


def _fake_bundle(action=RAW_ACTION):
    """(policy, preprocess, postprocess, torch) with a recorded flow."""
    seen = {}

    def preprocess(frame):
        seen["frame"] = frame
        return frame

    class _Policy:
        def select_action(self, batch):
            seen["batch"] = batch
            return np.array([[action]], dtype=np.float32)  # [1, 1, 3]

    def postprocess(out):
        seen["post_in"] = out
        return np.array([action], dtype=np.float32)

    return _Policy(), preprocess, postprocess, _fake_torch(), seen


def _jpeg_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (3, 4, 5)).save(buf, format="JPEG")
    return b64.b64encode(buf.getvalue()).decode()


def _policy_dir(tmp_path, stats: NormStats | None = None) -> Path:
    d = tmp_path / "pretrained_model"
    d.mkdir(parents=True, exist_ok=True)
    # realistic course stats: driving dims span [-1, 1], padded dims are
    # the degenerate (0, 0) the dataset always carries
    stats = stats or NormStats(min=(-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
                               max=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    (d / SERVING_META_NAME).write_text(json.dumps(serving_meta_dict(stats)))
    return d


# -- SmolVlaModel -------------------------------------------------------------------


def test_predict_maps_raw_to_house_norm(tmp_path):
    d = _policy_dir(tmp_path)
    policy, pre, post, torch, seen = _fake_bundle()
    m = SmolVlaModel("lerobot/smolvla_base", d,
                     loader=lambda _d: (policy, pre, post, torch))
    out = m.predict(_jpeg_b64(), "park at the cone")
    assert len(out) == 7
    assert out[0] == pytest.approx(0.75)   # (0.5 - -1) / 2
    assert out[1] == pytest.approx(0.375)
    assert out[2] == pytest.approx(0.55)
    assert out[3:] == [0.0, 0.0, 0.0, 0.0]  # degenerate padded dims
    # the frame reached the preprocessor as CHW float [0,1] + raw prompt
    frame = seen["frame"]
    img = frame["observation.images.camera1"]
    assert img.shape == (3, 48, 64) and img.dtype == np.float32
    assert img.max() <= 1.0
    assert frame["task"] == "park at the cone"


def test_predict_uses_the_stamped_dataset_stats(tmp_path):
    stats = NormStats(min=(-0.8, -0.5, -1.2, 0.0, 0.0, 0.0, 0.0),
                      max=(0.9, 0.6, 1.1, 0.0, 0.0, 0.0, 0.0))
    d = _policy_dir(tmp_path, stats)
    policy, pre, post, torch, _seen = _fake_bundle()
    m = SmolVlaModel("lerobot/smolvla_base", d,
                     loader=lambda _d: (policy, pre, post, torch))
    out = m.predict(_jpeg_b64(), "t")
    assert out[0] == pytest.approx((0.5 + 0.8) / 1.7)
    assert out[2] == pytest.approx((0.1 + 1.2) / 2.3)


@pytest.mark.parametrize("prompt", ["", None])
def test_empty_prompt_rejected(tmp_path, prompt):
    d = _policy_dir(tmp_path)
    policy, pre, post, torch, _seen = _fake_bundle()
    m = SmolVlaModel("lerobot/smolvla_base", d,
                     loader=lambda _d: (policy, pre, post, torch))
    with pytest.raises(RobovlaError) as ei:
        m.predict(_jpeg_b64(), prompt)
    assert ei.value.code == "validation"


def test_undecodable_image_rejected(tmp_path):
    d = _policy_dir(tmp_path)
    policy, pre, post, torch, _seen = _fake_bundle()
    m = SmolVlaModel("lerobot/smolvla_base", d,
                     loader=lambda _d: (policy, pre, post, torch))
    with pytest.raises(RobovlaError) as ei:
        m.predict("not-base64-jpeg", "t")
    assert ei.value.code == "validation"


def test_missing_policy_dir_or_meta_fails_loud(tmp_path):
    with pytest.raises(RobovlaError) as ei:
        SmolVlaModel("lerobot/smolvla_base", tmp_path / "absent")
    assert ei.value.code == "not_found"
    d = tmp_path / "no_meta"
    d.mkdir()
    with pytest.raises(RobovlaError) as ei:
        SmolVlaModel("lerobot/smolvla_base", d)
    assert ei.value.code == "not_found"
    assert SERVING_META_NAME in ei.value.message
    (d / SERVING_META_NAME).write_text("{ broken json")
    with pytest.raises(RobovlaError) as ei:
        SmolVlaModel("lerobot/smolvla_base", d)
    assert ei.value.code == "validation"


def test_stack_import_failure_is_a_clean_internal(tmp_path):
    d = _policy_dir(tmp_path)

    def boom(_d):
        raise ImportError("lerobot", name="lerobot")

    m = SmolVlaModel("lerobot/smolvla_base", d, loader=boom)
    with pytest.raises(RobovlaError) as ei:
        m.predict(_jpeg_b64(), "t")
    assert ei.value.code == "internal"


def test_unload_clears_the_bundle(tmp_path):
    d = _policy_dir(tmp_path)
    policy, pre, post, torch, _seen = _fake_bundle()
    m = SmolVlaModel("lerobot/smolvla_base", d,
                     loader=lambda _d: (policy, pre, post, torch))
    m.predict(_jpeg_b64(), "t")
    assert m._bundle is not None
    m.unload()  # the no-GPU path swallows the empty_cache attempt
    assert m._bundle is None


# -- factory routing -----------------------------------------------------------------


def _run_spec(workdir: Path, run_id: str, model: str) -> None:
    d = workdir / "runs" / run_id
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"model": model}))


def test_factory_routes_smolvla(tmp_path):
    _run_spec(tmp_path, "j_20261107_0001", "lerobot/smolvla_base")
    d = _policy_dir(tmp_path)
    m = make_default_model_factory(tmp_path)("j_20261107_0001",
                                             d.relative_to(tmp_path).as_posix())
    assert isinstance(m, SmolVlaModel)


def test_factory_defaults_to_openvla_and_missing_spec_stays_openvla(tmp_path):
    adapter = tmp_path / "adapter.bin"
    adapter.write_bytes(b"x")
    f = make_default_model_factory(tmp_path)
    assert isinstance(f("j_unknown", "adapter.bin"), TorchVlaModel)
    _run_spec(tmp_path, "j_20261021_0004", "openvla/openvla-7b")
    assert isinstance(f("j_20261021_0004", "adapter.bin"), TorchVlaModel)
