"""core.budgets — the VRAM budget calculator (design §3).

``train_gb``/``infer_gb`` in fleet.yaml are the budgets at the node's **max
allowed batch**; smaller batches scale the estimate down (linear in the
variable part: fixed model/optimizer cost + activations). A job is admitted
only where measured free VRAM >= its estimate (the budget guard).
"""

import pytest

from robovla.core.budgets import (
    FIXED_GB,
    estimate_job_gb,
    estimate_train_gb,
    fits,
)
from robovla.core.errors import RobovlaError


class TestTrainEstimate:
    def test_max_batch_costs_full_base(self):
        # pai1 profile: 32 GB node, train_gb 18 at batch 8.
        assert estimate_train_gb(18.0, "bnb8", 8, 8) == pytest.approx(18.0)

    def test_smaller_batch_scales_down(self):
        # fixed 10 GB + (18-10) * 1/8 = 11 GB for batch 1.
        assert estimate_train_gb(18.0, "bnb8", 1, 8) == pytest.approx(11.0)
        # batch 4: 10 + 8 * 4/8 = 14 GB.
        assert estimate_train_gb(18.0, "bnb8", 4, 8) == pytest.approx(14.0)

    def test_never_below_fixed_never_above_base(self):
        e = estimate_train_gb(18.0, "bnb8", 1, 8)
        assert FIXED_GB["bnb8"] <= e <= 18.0

    def test_batch_above_max_clamps_to_base(self):
        assert estimate_train_gb(18.0, "bnb8", 16, 8) == pytest.approx(18.0)

    def test_bf16_has_higher_fixed_cost(self):
        # 15 GB bf16 weights (design §1: "bf16 training doesn't fit (15 GB weights)").
        assert estimate_train_gb(22.0, "bf16", 1, 2) == pytest.approx(15.0 + (22.0 - 15.0) * 0.5)
        assert FIXED_GB["bf16"] == pytest.approx(15.0)

    @pytest.mark.parametrize("args", [
        (5.0, "bnb8", 1, 8),   # base below the model's fixed floor
        (18.0, "fp42", 1, 8),  # unknown quant profile
        (18.0, "bnb8", 0, 8),  # batch < 1
        (18.0, "bnb8", 1, 0),  # max_batch < 1
        (-1.0, "bnb8", 1, 8),  # negative base
    ])
    def test_invalid_inputs_rejected(self, args):
        with pytest.raises(RobovlaError) as e:
            estimate_train_gb(*args)
        assert e.value.code == "validation"


class TestJobEstimate:
    def test_infer_uses_infer_budget(self):
        assert estimate_job_gb(18.0, 11.0, "bnb8", "infer", 1, 8) == pytest.approx(11.0)

    def test_train_types_scale_with_batch(self):
        for jt in ("smoke", "train", "validate", "crosscheck"):
            assert estimate_job_gb(18.0, 11.0, "bnb8", jt, 1, 8) == pytest.approx(11.0)
            assert estimate_job_gb(18.0, 11.0, "bnb8", jt, 8, 8) == pytest.approx(18.0)

    def test_merge_and_selftest_use_full_train_budget(self):
        # merge loads model + adapter + merged export; selftest loads the model.
        assert estimate_job_gb(18.0, 11.0, "bnb8", "merge", None, 8) == pytest.approx(18.0)
        assert estimate_job_gb(18.0, 11.0, "bnb8", "gpu_selftest", None, 8) == pytest.approx(18.0)

    def test_unknown_job_type_rejected(self):
        with pytest.raises(RobovlaError):
            estimate_job_gb(18.0, 11.0, "bnb8", "retrain", 1, 8)


class TestTestMatrixScenarios:
    """Design §11 test-matrix rows: 16 GB node -> batch-1 preset; 2-slot
    admission on 32 GB; budget guard blocking what doesn't fit."""

    def test_16gb_node_batch1_preset_fits(self):
        # node3 profile: 16 GB, train_gb 14 at max batch 1 -> batch-1 job = 14 GB.
        job = estimate_job_gb(14.0, 11.0, "bnb8", "train", 1, 1)
        assert job == pytest.approx(14.0)
        assert fits(16.0, job)

    def test_32gb_node_two_concurrent_batch1_train_slots(self):
        per = estimate_job_gb(18.0, 11.0, "bnb8", "train", 1, 8)
        assert fits(32.0, 2 * per)

    def test_32gb_node_two_full_batch_slots_do_not_fit(self):
        per = estimate_job_gb(18.0, 11.0, "bnb8", "train", 8, 8)
        assert not fits(32.0, 2 * per)
        assert fits(32.0, per)

    def test_guard_blocks_when_free_is_below_budget(self):
        # pai1 with the 27B assistant loaded: ~2.5 GB free -> nothing trains.
        job = estimate_job_gb(18.0, 11.0, "bnb8", "train", 1, 8)
        assert not fits(2.5, job)

    def test_fits_is_inclusive(self):
        assert fits(11.0, 11.0)
        assert not fits(10.999, 11.0)
