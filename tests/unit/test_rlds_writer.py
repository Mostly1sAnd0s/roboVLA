"""images/rlds_writer — the pure half of the crosscheck RLDS materialization.

The official finetune.py (pinned openvla SHA) consumes RLDS/TFDS only, so the
crosscheck subset is materialized in the official OXE raw layout under the
"toto" dataset config (7-dim EEF_POS; its standardize transform is a pure
concat of action.{world_vector, rotation_delta, open_gripper}). The pure
builders here are CPU-pinned; write_rlds is the lazy tensorflow boundary and
fails with a clean §9.6 ``internal`` on a box without the stack (never a
bare traceback — the engine's lazy-boundary rule, §11).
"""

from __future__ import annotations

import builtins
import pathlib
import sys

import pytest

# The repo root hosts the `images` package (baked into the image; not in the
# wheel) — the same importlib-free path pattern as test_robovla_job.py.
REPO = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from robovla.core.errors import RobovlaError  # noqa: E402
from images import rlds_writer as rw  # noqa: E402


# -- the 7-dim action split (course convention -> official EEF_POS parts) --------


def test_action_split_course_convention():
    a = rw.action_to_rlds((0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0))
    assert a == {"world_vector": [0.1, -0.2, 0.3],
                 "rotation_delta": [0.0, 0.0, 0.0],
                 "open_gripper": [0.0]}


def test_action_split_rejects_bad_dims():
    for bad in ([0.0] * 6, [0.0] * 8, [0.0] * 3 + [float("nan")] * 4):
        with pytest.raises(RobovlaError) as ei:
            rw.action_to_rlds(bad)
        assert ei.value.code == "validation"


# -- the per-episode sample layout (one TFDS example per trajectory) -------------


def _sample(n=3, ep="ep_0007", task="stop at the target"):
    return rw.build_episode_sample(
        ep, task, [b"frame-%d" % i for i in range(n)],
        [(0.1 * (i + 1), 0.2, 0.3, 0.0, 0.0, 0.0, 0.0) for i in range(n)])


def test_build_episode_sample_layout():
    steps = _sample()["steps"]
    assert len(steps) == 3
    st = steps[0]
    assert st["observation"]["image"] == b"frame-0"
    assert st["observation"]["natural_language_instruction"] == \
        b"stop at the target"
    assert st["action"]["world_vector"] == [0.1, 0.2, 0.3]
    assert st["first"] is True and st["last"] is False
    assert st["is_terminal"] is False
    assert st["episode_index"] == 7 and st["frame_index"] == 0
    assert st["next"] == {"first": False, "last": False,
                          "episode_index": 7, "frame_index": 1}
    last = steps[-1]
    assert last["first"] is False and last["last"] is True
    assert last["is_terminal"] is True
    assert last["next"] == {"first": True, "last": False,
                            "episode_index": 8, "frame_index": 0}
    # the sample carries exactly the schema keys (the TFDS writer is strict)
    assert set(steps[0].keys()) == {
        "observation", "action", "first", "last", "is_terminal",
        "episode_index", "frame_index", "next"}


def test_build_episode_sample_mismatch_rejected():
    with pytest.raises(RobovlaError) as ei:
        rw.build_episode_sample("ep_0007", "t", [b"x"] * 2, [(0.0,) * 7])
    assert ei.value.code == "validation"
    with pytest.raises(RobovlaError):
        rw.build_episode_sample("ep_0007", "t", [], [(0.0,) * 7])
    with pytest.raises(RobovlaError):
        rw.build_episode_sample("not-an-ep-id", "t", [b"x"], [(0.0,) * 7])


# -- the official step math --------------------------------------------------------


def test_max_steps():
    assert rw.max_steps_for(150, 4, 2, 3) == 54  # 150 // 8 = 18/epoch x 3
    assert rw.max_steps_for(7, 1, 8, 3) == 1     # sub-effective-batch -> floor 1
    assert rw.max_steps_for(100, 4, 2, 2) == 24  # 100 // 8 = 12/epoch x 2
    for bad in ((0, 1, 8, 3), (100, 0, 8, 3), (100, 1, 0, 3), (100, 1, 8, 0)):
        with pytest.raises(RobovlaError):
            rw.max_steps_for(*bad)


def test_exp_id_formula():
    # the official finetune.py exp_id verbatim (use_lora is always True here)
    e = rw.exp_id("openvla/openvla-7b", "toto", 8, 0.0005, 32, 0.0,
                  True, "j_20261022_0001", True)
    assert e == ("openvla-7b+toto+b8+lr-0.0005+lora-r32+dropout-0.0"
                 "+q-4bit--j_20261022_0001--image_aug")
    e2 = rw.exp_id("openvla/openvla-7b", "toto", 8, 0.0005, 32, 0.0,
                   False, None, False)
    assert e2 == "openvla-7b+toto+b8+lr-0.0005+lora-r32+dropout-0.0"


# -- the dynamic TFDS builder class (CPU-pinned with a fake tfds) --------------
#
# The first host65 crosscheck run (2026.10.1, 2026-10-22) failed at
# instantiation: in TFDS 4.9.3 the raw tfds.core.DatasetBuilder leaves
# `_download_and_prepare` / `_as_dataset` abstract, so a class with only
# `_info` + `_split_generators` cannot be built. The generator base
# (tfds.core.GeneratorBasedBuilder — the same base the official dlimp
# fork's MultiThreadedDatasetBuilder builds on) implements both on top of
# `_split_generators`, which is what the base _download_and_prepare calls.


def _fake_tfds():
    """Just enough of the tfds surface for build_builder_class: the class
    construction is pure Python over the injected module object (no real
    TF import — the CPU suite never carries the stack)."""
    import types

    class _GeneratorBasedBuilder:
        pass

    class _DatasetBuilder:
        pass

    class _Version:
        def __init__(self, s):
            self.s = s

    return types.SimpleNamespace(core=types.SimpleNamespace(
        GeneratorBasedBuilder=_GeneratorBasedBuilder,
        DatasetBuilder=_DatasetBuilder,
        Version=_Version))


def test_build_builder_class_contract():
    fake = _fake_tfds()
    samples = [rw.build_episode_sample("ep_%04d" % i, "t", [b"x"],
                                       [(0.0,) * 7]) for i in range(1, 51)]
    cls = rw.build_builder_class("toto", samples, fake, features=None)
    # The base must be GeneratorBasedBuilder (never the raw DatasetBuilder).
    assert issubclass(cls, fake.core.GeneratorBasedBuilder)
    assert not issubclass(cls, fake.core.DatasetBuilder)
    assert cls.__name__ == "Toto" and cls.name == "toto"
    assert str(cls.VERSION.s) == "1.0.0"
    b = cls.__new__(cls)  # the (data_dir) __init__ is the TFDS boundary
    splits = cls._split_generators(b, None)
    assert set(splits) == {"train", "val"}
    assert sum(1 for _ in splits["train"]) == 50
    assert sum(1 for _ in splits["val"]) == 0  # empty on purpose
    # _generate_examples must stay in lockstep with _split_generators (the
    # base writer calls the latter; the former only satisfies the abstract
    # method).
    assert set(cls._generate_examples(b)) == {"train", "val"}
    train_only = rw.build_builder_class("toto", samples, fake, features=None,
                                        val_empty=False)
    assert set(train_only._split_generators(
        train_only.__new__(train_only), None)) == {"train"}


def test_build_builder_class_example_layout():
    fake = _fake_tfds()
    sample = rw.build_episode_sample("ep_0007", "stop at the target",
                                     [b"frame-0", b"frame-1"],
                                     [(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0),
                                      (0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0)])
    cls = rw.build_builder_class("toto", [sample], fake, features=None)
    (example,) = list(cls._split_generators(cls.__new__(cls), None)["train"])
    assert len(example["steps"]) == 2
    s0, s1 = example["steps"]
    assert s0["observation"]["image"] == b"frame-0"
    assert s0["observation"]["natural_language_instruction"] == \
        b"stop at the target"
    assert s0["first"] is True and s1["last"] is True
    assert s1["is_terminal"] is True
    assert s0["episode_index"] == 7 and s1["frame_index"] == 1
    assert s1["next"] == {"first": True, "last": False,
                          "episode_index": 8, "frame_index": 0}
    # dtype casts happen at this boundary (the schema demands float32)
    import numpy as np
    act = s0["action"]["world_vector"]
    assert isinstance(act, np.ndarray) and act.dtype == np.float32
    assert list(act) == [0.1, 0.2, 0.3]


# -- the lazy boundary: no tensorflow -> clean §9.6 internal ----------------------


def test_write_rlds_clean_error_without_stack(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("No module named 'tensorflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RobovlaError) as ei:
        rw.write_rlds(pathlib.Path("/tmp/no-such-dir"), rw.RLDS_NAME,
                      [rw.build_episode_sample("ep_0001", "t", [b"x"],
                                               [(0.0,) * 7])])
    assert ei.value.code == "internal" and "tensorflow" in ei.value.message
