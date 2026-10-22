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
                 # a per-frame SCALAR (host65 attempt 5, 2026-10-22): the
                 # official toto transform does open_gripper[:, None] — a
                 # (1,)-shaped per-frame value reads back [F,1,1] through
                 # the concat and breaks the official pipeline's rank
                 "open_gripper": 0.0}


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


def _fake_tfds(dp_failures=None):
    """Just enough of the tfds surface for build_builder_class + the
    write_rlds boundary: the class construction is pure Python over the
    injected module object (no real TF import — the CPU suite never
    carries the stack).

    ``core.registered`` mimics tensorflow_datasets.core.registered (the
    host65 attempt-2 failure, 2026-10-22): builder classes register by
    name at creation and a second class with the same name raises
    ValueError; ``skip_registration()`` opts out. ``dp_failures`` queues
    ``download_and_prepare()`` failures (primary, then fallback)."""
    import types

    class _Registered:
        def __init__(self):
            self.registry: dict[str, type] = {}
            self.entered = 0
            self.dp_done: list[str] = []
            self.dp_failures: list[str] = list(dp_failures or [])
            self._skipping = False

        class _skip_ctx:
            def __init__(self, owner):
                self._owner = owner

            def __enter__(self):
                self._owner._skipping = True
                self._owner.entered += 1
                return self

            def __exit__(self, *exc):
                self._owner._skipping = False
                return False

        def skip_registration(self):
            return self._skip_ctx(self)

    registry = _Registered()

    # The ABCMeta metaclass is the REAL one (tfds's builders are ABC-based):
    # type() with such a base creates the class inside the 'abc' module's
    # frame, so __module__ comes out 'abc' unless pinned — the host65
    # attempt-3 failure (epath.resource_path('abc') -> SpecPath).
    import abc as _abc

    class _GeneratorBasedBuilder(metaclass=_abc.ABCMeta):
        def __init_subclass__(cls, **kwargs):
            name = getattr(cls, "name", None)
            if registry._skipping or name is None:
                return
            if name in registry.registry:
                raise ValueError(
                    f"Dataset with name {name} already registered.")
            registry.registry[name] = cls

        def __init__(self, data_dir=None):
            self.data_dir = data_dir

        def download_and_prepare(self):
            if registry.dp_failures:
                raise RuntimeError(registry.dp_failures.pop(0))
            registry.dp_done.append(str(self.data_dir))

    class _DatasetBuilder:
        pass

    class _Version:
        def __init__(self, s):
            self.s = s

    def identity(*args, **kwargs):
        # identity features (the fake write never validates the schema)
        return args[0] if args else kwargs

    return types.SimpleNamespace(core=types.SimpleNamespace(
        GeneratorBasedBuilder=_GeneratorBasedBuilder,
        DatasetBuilder=_DatasetBuilder,
        Version=_Version,
        registered=registry),
        features=types.SimpleNamespace(
            FeaturesDict=identity, Image=identity, Text=identity,
            Tensor=identity, Scalar=identity, Dataset=identity))


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
    train_pairs = list(splits["train"])
    assert len(train_pairs) == 50
    # The TFDS 4.x (pinned 4.9.3) contract: {split: iterable of
    # (key, example) PAIRS} — the base _download_and_prepare hands each
    # per-split generator to split_builder.submit_split_generation, which
    # unpacks `for key, example in generator` (host65 attempt 4, 2026-10-22:
    # raw examples broke that unpack). Keys index the trajectories.
    keys = [key for key, _ex in train_pairs]
    assert keys == list(range(50))
    assert all(set(ex) == {"steps"} for _k, ex in train_pairs)
    assert sum(1 for _ in splits["val"]) == 0  # empty on purpose
    # _generate_examples (the legacy per-split contract, never executed in
    # our dict-form path) must stay in lockstep with _split_generators'
    # train generator — the same (key, example) pairs.
    legacy = list(cls._generate_examples(b))
    assert len(legacy) == len(train_pairs)
    for (k1, e1), (k2, e2) in zip(legacy, train_pairs):
        assert k1 == k2
        assert e1["steps"][0]["observation"]["image"] == \
            e2["steps"][0]["observation"]["image"]
        assert list(e1["steps"][0]["action"]["world_vector"]) == \
            list(e2["steps"][0]["action"]["world_vector"])
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
    (_key, example) = next(iter(cls._split_generators(
        cls.__new__(cls), None)["train"]))
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
    # open_gripper crosses the boundary as a 0-dim float32 (a per-frame
    # SCALAR — the official toto transform's open_gripper[:, None] breaks
    # on a (1,)-shaped value; host65 attempt 5, 2026-10-22)
    og = s0["action"]["open_gripper"]
    assert isinstance(og, np.ndarray) and og.ndim == 0 \
        and og.dtype == np.float32
    assert float(og) == 0.0


# -- TFDS name registration (host65 attempt 2, 2026-10-22) -------------------
#
# The writer creates the builder class TWICE under one name: the primary
# write (train + empty val) and the train-only fallback. TFDS registers
# builder classes by name at creation (tensorflow_datasets/core/
# registered.py — a second class with the same name raises ValueError),
# so the fallback's class hit 'Dataset with name toto already registered'
# and the wrapper mislabeled it 'unreadable spec'. The official loader
# never sees a writer-side class — its process resolves the dataset from
# files (tfds.builder's file-based fallback, read_only_builder.
# builder_from_files) — so both creations must skip registration.


def test_builder_class_creation_skips_registration():
    fake = _fake_tfds()
    samples = [_sample()]
    primary = rw.build_builder_class("toto", samples, fake, features=None)
    fallback = rw.build_builder_class("toto", samples, fake, features=None,
                                      val_empty=False)
    assert primary.__name__ == fallback.__name__ == "Toto"
    # both creations ran inside skip_registration (entered, then balanced)
    reg = fake.core.registered
    assert reg.entered == 2 and reg._skipping is False
    # nothing leaked into the registry: the official process must resolve
    # the dataset from files, never from a writer-side class
    assert "toto" not in reg.registry


def test_builder_class_module_is_writer_module():
    """host65 attempt 3 (2026-10-22): type() with the ABCMeta-based tfds
    builder resolves the class inside the 'abc' module's frame, so
    __module__ comes out 'abc' unless pinned. TFDS's
    _get_builder_datadir_path() then does
    epath.resource_path('abc').joinpath(*[]) — the stdlib 'abc' is a
    single-file module, so etils returns an importlib.resources
    SpecPath whose joinpath() requires an argument -> TypeError. Pin the
    true module so tfds resolves the writer package's real directory."""
    fake = _fake_tfds()
    cls = rw.build_builder_class("toto", [_sample()], fake, features=None)
    assert cls.__module__ == "images.rlds_writer"


# -- the lazy boundary: the write path (fake stack, no real TF) ---------------


def _fake_stack(monkeypatch, fake_tf, fake_tfds):
    """Route the lazy ``import tensorflow`` / ``import tensorflow_datasets``
    inside write_rlds to fakes (the CPU suite never carries the stack)."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            return fake_tf
        if name == "tensorflow_datasets":
            return fake_tfds
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _fake_tf_module():
    import types
    return types.SimpleNamespace(config=types.SimpleNamespace(
        set_visible_devices=lambda *a, **k: None))


def test_write_rlds_primary_failure_falls_back(tmp_path, capsys,
                                               monkeypatch):
    """The primary write (train + empty val) fails -> the REAL error is
    surfaced (host65 attempt 2 swallowed it behind 'empty val split
    rejected'), the partial dir is cleared, and the train-only fallback
    completes."""
    fake_tfds = _fake_tfds(dp_failures=["empty val split rejected (fake)"])
    _fake_stack(monkeypatch, _fake_tf_module(), fake_tfds)
    root = tmp_path / "rlds"
    root.mkdir()
    (root / rw.RLDS_NAME).mkdir()  # the partial dir the failed write left
    rw.write_rlds(root, rw.RLDS_NAME, [_sample()])
    out = capsys.readouterr().out
    assert "primary write failed" in out
    assert "RuntimeError: empty val split rejected (fake)" in out
    reg = fake_tfds.core.registered
    assert reg.dp_done == [str(root)]  # the fallback write completed
    assert reg.entered == 2 and "toto" not in reg.registry


def test_write_rlds_both_writes_fail_is_clean_internal(tmp_path, monkeypatch):
    """Primary AND fallback fail -> a clean §9.6 internal carrying the
    REAL fallback error (host65 attempt 2: the fallback's raw ValueError
    escaped the boundary and the wrapper mislabeled it 'unreadable
    spec' — never a bare traceback)."""
    fake_tfds = _fake_tfds(dp_failures=["primary boom", "fallback boom"])
    _fake_stack(monkeypatch, _fake_tf_module(), fake_tfds)
    root = tmp_path / "rlds"
    root.mkdir()
    with pytest.raises(RobovlaError) as ei:
        rw.write_rlds(root, rw.RLDS_NAME, [_sample()])
    assert ei.value.code == "internal"
    assert "fallback boom" in ei.value.message


def test_write_stack_open_gripper_is_scalar_feature(tmp_path, monkeypatch):
    """The writer's schema must declare open_gripper WITHOUT a Tensor shape
    (a per-frame scalar). host65 attempt 5 (2026-10-22): a (1,)-shaped
    per-frame value made the official toto_dataset_transform's
    ``tf.concat([wv (F,3), rd (F,3), open_gripper[:, None] (F,1,1)])`` fail
    with 'Shape must be rank 2 but is rank 3' — the real OXE toto data
    carries one scalar per frame."""
    fake_tfds = _fake_tfds()
    _fake_stack(monkeypatch, _fake_tf_module(), fake_tfds)
    captured = {}

    def capture(name, samples, tfds, features, **kw):
        captured["features"] = features
        return fake_tfds.core.GeneratorBasedBuilder

    monkeypatch.setattr(rw, "build_builder_class", capture)
    rw._write_with_stack(fake_tfds, tmp_path, "toto", [_sample()])
    act = captured["features"]["steps"]["action"]
    assert act["world_vector"]["shape"] == (3,)
    assert act["rotation_delta"]["shape"] == (3,)
    assert "shape" not in act["open_gripper"]  # scalar per frame


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
