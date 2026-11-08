"""SmolVLA family — the shared pure layer (core.spec.model_family routing
rule + robovla.smolvla data plane).

Pinned here: the family derivation (the model id is the ONLY family
signal — Appendix B stays frozen), the per-family catalog lr, the
3-dim <-> house-7-dim action mapping (the §9.5 contract every consumer
keeps), and the serving-metadata schema the wrapper stamps and the
serving seam reads.
"""

from __future__ import annotations

import pytest

from robovla.core.errors import RobovlaError
from robovla.core.norm_stats import NormStats
from robovla.core.spec import (FAMILY_DEFAULT_LR, MODEL_FAMILY_OPENVLA,
                               MODEL_FAMILY_SMOLVLA, build_job_spec,
                               model_family)
from robovla.smolvla import (HOUSE_ACTION_DIM, SMOLVLA_ACTION_DIM,
                             SMOLVLA_CAMERA_KEY, parse_serving_meta,
                             serving_meta_dict, to_house_action,
                             to_smolvla_action)


# -- family routing (core.spec) --------------------------------------------------


@pytest.mark.parametrize("model_id,expected", [
    ("lerobot/smolvla_base", MODEL_FAMILY_SMOLVLA),
    ("hf-user/My-SmolVLA-parking-ft", MODEL_FAMILY_SMOLVLA),  # case-insensitive
    ("openvla/openvla-7b", MODEL_FAMILY_OPENVLA),
    ("", MODEL_FAMILY_OPENVLA),
    (None, MODEL_FAMILY_OPENVLA),
    (123, MODEL_FAMILY_OPENVLA),
])
def test_model_family_derivation(model_id, expected):
    assert model_family(model_id) == expected


def test_family_default_lr():
    # one catalog: openvla QLoRA 5e-4, SmolVLA preset 1e-4
    assert FAMILY_DEFAULT_LR[MODEL_FAMILY_OPENVLA] == 5e-4
    assert FAMILY_DEFAULT_LR[MODEL_FAMILY_SMOLVLA] == 1e-4


def _spec(model: str, **kw) -> dict:
    return build_job_spec(
        job_id="j_20261107_0001", student="s07",
        job_type=kw.pop("job_type", "train"),
        dataset="s07/park-at-target/v1", node_id="avc",
        node_agent="10.0.0.9:8601", budget_gb=18, model=model, **kw,
    ).to_dict()


def test_build_spec_picks_lr_by_family():
    assert _spec("openvla/openvla-7b")["optim"]["lr"] == 5e-4
    assert _spec("lerobot/smolvla_base")["optim"]["lr"] == 1e-4


def test_build_spec_family_lr_overridable():
    s = _spec("lerobot/smolvla_base", overrides={"lr": 2e-4})
    assert s["optim"]["lr"] == 2e-4


# -- the data plane (robovla.smolvla) ---------------------------------------------


def test_to_smolvla_action_slices_the_course_padding():
    assert to_smolvla_action((0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0)) == \
        [0.1, -0.2, 0.3]


@pytest.mark.parametrize("bad", [[0.1, 0.2], "abc", [1.0, float("nan"), 0.0],
                                 [1.0, True, 0.0]])
def test_to_smolvla_action_rejects(bad):
    with pytest.raises(RobovlaError) as ei:
        to_smolvla_action(bad)
    assert ei.value.code == "validation"


def test_to_house_action_pads_and_normalizes():
    # real dataset stats: the padded course dims are constant-zero, so
    # their min == max == 0 (compute_norm_stats over to_action_7d rows)
    stats = NormStats(min=(-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
                      max=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    house = to_house_action([0.5, -0.25, 0.1], stats)
    assert len(house) == HOUSE_ACTION_DIM == 7
    assert house[0] == pytest.approx(0.75)
    assert house[1] == pytest.approx(0.375)
    assert house[2] == pytest.approx(0.55)
    # padded dims: min == max == 0 -> the documented degenerate mapping 0.0
    assert house[3:] == (0.0, 0.0, 0.0, 0.0)


def test_to_house_action_roundtrips_the_console_denormalization():
    # the §9.5 promise: agent normalize -> console denormalize = identity
    from robovla.core.norm_stats import denormalize
    stats = NormStats(min=(-0.8, -0.5, -1.2, 0.0, 0.0, 0.0, 0.0),
                      max=(0.9, 0.6, 1.1, 0.0, 0.0, 0.0, 0.0))
    raw = [0.3, -0.4, 0.75]
    back = denormalize(to_house_action(raw, stats), stats)
    assert back[:3] == pytest.approx(raw)


def test_serving_meta_roundtrip():
    stats = NormStats(min=(-1.0,) * 7, max=(1.0,) * 7)
    meta = serving_meta_dict(stats)
    assert meta["family"] == "smolvla"
    assert meta["action_dim"] == SMOLVLA_ACTION_DIM == 3
    assert meta["camera"] == SMOLVLA_CAMERA_KEY
    parsed = parse_serving_meta(meta)
    assert parsed.min == stats.min and parsed.max == stats.max


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("family"),
    lambda d: d.update(family="openvla"),
    lambda d: d.update(action_dim=7),
    lambda d: d.pop("norm_stats"),
    lambda d: d.update(norm_stats={"min": [0.0]}),
])
def test_serving_meta_rejects(mutate):
    d = serving_meta_dict(NormStats(min=(-1.0,) * 7, max=(1.0,) * 7))
    mutate(d)
    with pytest.raises(RobovlaError) as ei:
        parse_serving_meta(d)
    assert ei.value.code == "validation"


# -- cross-layer constant agreement (§11 DRY: one value, pinned) -------------------


def test_house_action_dim_matches_the_job_file_contract():
    from images import contract  # the image-side constant (not in the wheel)
    assert contract.ACTION_DIMS == HOUSE_ACTION_DIM
