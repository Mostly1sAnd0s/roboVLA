"""core.spec — the JobSpec contract: one schema, one validator, one preset
catalog (design §11, Appendix B). UI, scheduler, runners, docs, and the
crosscheck job all consume this; nothing else defines a JobSpec shape.
"""

import copy
import uuid

import pytest

from robovla.core.errors import RobovlaError
from robovla.core.spec import (
    ALLOWED_BATCHES,
    JOB_TYPES,
    MAX_SEQ_LEN,
    QUANT_PROFILES,
    RECIPE_EFFECTIVE_BATCH,
    build_job_spec,
    validate_job_spec,
)


def _valid_dict(**over):
    """A fully valid JobSpec dict (Appendix B shape); override any field."""
    d = {
        "job_id": "j_20261021_0001",
        "student": "s07",
        "type": "train",
        "model": "openvla/openvla-7b",
        "dataset": "s07/task-stop-at-target/v2",
        "node": {"id": "pai1", "agent": "10.0.0.10:8601"},
        "quant": "bnb8",
        "lora": {"rank": 32, "alpha": 32},
        "optim": {"lr": 0.0005, "epochs": 3, "batch": 4, "grad_accum": 2, "max_seq_len": 287},
        "image_aug": True,
        "seed": 42,
        "prompt_template": "In: What action should the robot take to {task}?\nOut:",
        "limits": {"max_min": 180, "budget_gb": 18.0},
        "resume_from": None,
        "idempotency_key": str(uuid.uuid4()),
    }
    for k, v in over.items():
        d[k] = v
    return d


class TestContractConstants:
    def test_job_types(self):
        assert JOB_TYPES == frozenset(
            {"smoke", "train", "validate", "merge", "infer", "crosscheck", "gpu_selftest", "home_selftest"}
        )

    def test_max_seq_len_fixed_at_287(self):
        assert MAX_SEQ_LEN == 287  # official OpenVLA context (256 visual tokens + text)

    def test_allowed_batches_and_quant(self):
        assert ALLOWED_BATCHES == (1, 4, 8)  # §3: allowed batch (1 / 4 / 8)
        assert QUANT_PROFILES == frozenset({"bnb8", "bf16"})
        assert RECIPE_EFFECTIVE_BATCH == 8


class TestValidator:
    def test_valid_spec_round_trips(self):
        d = _valid_dict()
        spec = validate_job_spec(d)
        assert spec.job_id == d["job_id"]
        assert spec.student == "s07"
        assert spec.type == "train"
        assert spec.node.id == "pai1"
        assert spec.node.agent == "10.0.0.10:8601"
        assert spec.lora.rank == 32 and spec.lora.alpha == 32
        assert spec.optim.max_seq_len == 287
        assert spec.limits.budget_gb == pytest.approx(18.0)
        assert spec.resume_from is None

    def test_not_a_dict_rejected(self):
        with pytest.raises(RobovlaError) as e:
            validate_job_spec(["nope"])
        assert e.value.code == "validation"

    @pytest.mark.parametrize("key", [
        "job_id", "student", "type", "model", "dataset", "node", "quant",
        "lora", "optim", "image_aug", "seed", "prompt_template", "limits",
        "resume_from", "idempotency_key",
    ])
    def test_missing_required_key_rejected(self, key):
        d = _valid_dict()
        del d[key]
        with pytest.raises(RobovlaError) as e:
            validate_job_spec(d)
        assert e.value.code == "validation"
        assert key in e.value.message

    def test_unknown_key_rejected(self):
        d = _valid_dict(**{"lora_rank": 8})
        with pytest.raises(RobovlaError) as e:
            validate_job_spec(d)
        assert e.value.code == "validation"

    @pytest.mark.parametrize("job_type", sorted(JOB_TYPES))
    def test_every_job_type_accepted(self, job_type):
        assert validate_job_spec(_valid_dict(type=job_type)).type == job_type

    def test_bad_job_type_rejected(self):
        with pytest.raises(RobovlaError):
            validate_job_spec(_valid_dict(type="retrain"))

    def test_id_patterns(self):
        for bad in ("j_20261021_1", "j_2026-10-21_0001", "J_20261021_0001", "x"):
            with pytest.raises(RobovlaError):
                validate_job_spec(_valid_dict(job_id=bad))
        for bad in ("s7", "S07", "s007", "t01"):
            with pytest.raises(RobovlaError):
                validate_job_spec(_valid_dict(student=bad))
        for bad in ("s07/task_stop/v1", "s07//v1", "team/task/v1", "s07/task/v1 ", "s07/Task/v1"):
            with pytest.raises(RobovlaError):
                validate_job_spec(_valid_dict(dataset=bad))
        assert validate_job_spec(_valid_dict(dataset="s12/longer-task-name-2/v10")).dataset

    def test_agent_endpoint_pattern(self):
        for bad in ("10.0.0.10", "10.0.0.10:0", "10.0.0.10:70000", "notanip:8601", "10.0.0.256:8601"):
            d = _valid_dict()
            d["node"] = {"id": "pai1", "agent": bad}
            with pytest.raises(RobovlaError):
                validate_job_spec(d)

    def test_quant_profiles(self):
        assert validate_job_spec(_valid_dict(quant="bf16")).quant == "bf16"
        with pytest.raises(RobovlaError):
            validate_job_spec(_valid_dict(quant="fp16"))

    def test_max_seq_len_must_be_287(self):
        d = _valid_dict()
        d["optim"] = {**d["optim"], "max_seq_len": 256}
        with pytest.raises(RobovlaError) as e:
            validate_job_spec(d)
        assert "287" in e.value.message

    def test_bnb8_batch_and_grad_accum_coupling(self):
        # recipe effective batch 8: batch must divide 8, grad_accum auto-matched.
        for batch, accum in ((1, 8), (4, 2), (8, 1)):
            d = _valid_dict()
            d["optim"] = {**d["optim"], "batch": batch, "grad_accum": accum}
            spec = validate_job_spec(d)
            assert spec.optim.grad_accum == 8 // spec.optim.batch
        d = _valid_dict()
        d["optim"] = {**d["optim"], "batch": 2}  # 2 is not an allowed batch
        with pytest.raises(RobovlaError):
            validate_job_spec(d)
        d = _valid_dict()
        d["optim"] = {**d["optim"], "batch": 4, "grad_accum": 1}  # mismatch
        with pytest.raises(RobovlaError):
            validate_job_spec(d)

    def test_bf16_grad_accum_is_free(self):
        d = _valid_dict(quant="bf16")
        d["optim"] = {**d["optim"], "batch": 1, "grad_accum": 2}
        assert validate_job_spec(d).optim.grad_accum == 2

    @pytest.mark.parametrize("lr", [0.0, -1e-4, 0.011])
    def test_lr_bounds(self, lr):
        d = _valid_dict()
        d["optim"] = {**d["optim"], "lr": lr}
        with pytest.raises(RobovlaError):
            validate_job_spec(d)

    @pytest.mark.parametrize("epochs", [0, 11, 2.5])
    def test_epochs_bounds(self, epochs):
        d = _valid_dict()
        d["optim"] = {**d["optim"], "epochs": epochs}
        with pytest.raises(RobovlaError):
            validate_job_spec(d)

    def test_lora_bounds(self):
        for rank, alpha in ((3, 32), (32, 3), (32, 129)):
            d = _valid_dict()
            d["lora"] = {"rank": rank, "alpha": alpha}
            with pytest.raises(RobovlaError):
                validate_job_spec(d)

    def test_seed_must_be_non_negative_int(self):
        d = _valid_dict()
        d["seed"] = -1
        with pytest.raises(RobovlaError):
            validate_job_spec(d)

    def test_image_aug_must_be_bool(self):
        d = _valid_dict()
        d["image_aug"] = "yes"
        with pytest.raises(RobovlaError):
            validate_job_spec(d)

    def test_prompt_template_requires_task_slot(self):
        d = _valid_dict()
        d["prompt_template"] = "do the thing"
        with pytest.raises(RobovlaError):
            validate_job_spec(d)

    def test_limits_bounds(self):
        for max_min, budget in ((0, 18.0), (1441, 18.0), (180, 0.0), (180, -1.0)):
            d = _valid_dict()
            d["limits"] = {"max_min": max_min, "budget_gb": budget}
            with pytest.raises(RobovlaError):
                validate_job_spec(d)

    def test_resume_from_nullable_string(self):
        assert validate_job_spec(_valid_dict(resume_from="ckpt_step_1500")).resume_from == "ckpt_step_1500"
        d = _valid_dict()
        d["resume_from"] = ""
        with pytest.raises(RobovlaError):
            validate_job_spec(d)

    def test_idempotency_key_must_be_uuid_or_null(self):
        assert validate_job_spec(_valid_dict(idempotency_key=None)).idempotency_key is None
        d = _valid_dict()
        d["idempotency_key"] = "not-a-uuid"
        with pytest.raises(RobovlaError):
            validate_job_spec(d)


class TestPresetCatalog:
    def test_default_recipe(self):
        spec = build_job_spec(
            job_id="j_20261021_0002",
            student="s01",
            job_type="train",
            dataset="s01/task-a/v1",
            node_id="pai1",
            node_agent="10.0.0.10:8601",
            budget_gb=18.0,
        )
        # The QLoRA default recipe, defined once here (design §6/§11):
        # 8-bit, rank 32, alpha 32, lr 5e-4, aug on, max_seq_len 287.
        assert spec.quant == "bnb8"
        assert spec.lora.rank == 32 and spec.lora.alpha == 32
        assert spec.optim.lr == pytest.approx(5e-4)
        assert spec.optim.epochs == 3
        assert spec.image_aug is True
        assert spec.optim.max_seq_len == 287
        assert spec.model == "openvla/openvla-7b"
        # {task} slot stays in the spec; the agent substitutes at runtime.
        assert spec.prompt_template == "In: What action should the robot take to {task}?\nOut:"

    def test_batch_drives_grad_accum(self):
        for batch, accum in ((1, 8), (4, 2), (8, 1)):
            spec = build_job_spec(
                job_id="j_20261021_0003",
                student="s01",
                job_type="train",
                dataset="s01/task-a/v1",
                node_id="pai1",
                node_agent="10.0.0.10:8601",
                batch=batch,
                budget_gb=18.0,
            )
            assert spec.optim.batch == batch
            assert spec.optim.grad_accum == accum

    def test_overrides_apply_to_recipe(self):
        spec = build_job_spec(
            job_id="j_20261021_0004",
            student="s01",
            job_type="train",
            dataset="s01/task-a/v1",
            node_id="pai1",
            node_agent="10.0.0.10:8601",
            batch=4,
            budget_gb=18.0,
            seed=7,
            overrides={"rank": 16, "alpha": 16, "lr": 1e-4, "epochs": 5, "image_aug": False},
        )
        assert spec.lora.rank == 16 and spec.lora.alpha == 16
        assert spec.optim.lr == pytest.approx(1e-4)
        assert spec.optim.epochs == 5
        assert spec.image_aug is False
        assert spec.seed == 7

    def test_overrides_cannot_touch_fixed_fields(self):
        with pytest.raises(RobovlaError):
            build_job_spec(
                job_id="j_20261021_0005",
                student="s01",
                job_type="train",
                dataset="s01/task-a/v1",
                node_id="pai1",
                node_agent="10.0.0.10:8601",
                budget_gb=18.0,
                overrides={"max_seq_len": 512},
            )

    def test_smoke_gets_shorter_default_limit(self):
        spec = build_job_spec(
            job_id="j_20261021_0006",
            student="s01",
            job_type="smoke",
            dataset="s01/task-a/v1",
            node_id="pai1",
            node_agent="10.0.0.10:8601",
            budget_gb=18.0,
        )
        assert spec.limits.max_min == 15

    def test_result_is_always_valid(self):
        # build_job_spec must route through the same validator (DRY).
        spec = build_job_spec(
            job_id="j_20261021_0007",
            student="s02",
            job_type="crosscheck",
            dataset="s02/task-b/v1",
            node_id="pai1",
            node_agent="10.0.0.10:8601",
            budget_gb=18.0,
            idempotency_key=str(uuid.uuid4()),
        )
        assert validate_job_spec(spec.to_dict()) == spec

    def test_spec_to_dict_is_stable(self):
        spec = build_job_spec(
            job_id="j_20261021_0008",
            student="s03",
            job_type="train",
            dataset="s03/task-c/v1",
            node_id="pai1",
            node_agent="10.0.0.10:8601",
            budget_gb=18.0,
        )
        d1 = spec.to_dict()
        d2 = copy.deepcopy(d1)
        assert d1 == d2
