"""Edge-case coverage for the pure core (design §11: core unit-tested to
>= 95% line coverage). Each class targets one module's uncovered branches —
validation paths, accessor edges, and the config-diff matrix.
"""

import copy
import math

import pytest

from robovla.core.budgets import estimate_job_gb, estimate_train_gb
from robovla.core.config import Change, FleetConfig, diff_fleet, validate_fleet
from robovla.core.errors import RobovlaError
from robovla.core.norm_stats import NormStats
from robovla.core.scheduler import Job, NodeState, Scheduler, default_duration_s
from robovla.core.spec import build_job_spec, validate_job_spec
from robovla.core.states import apply_transition
from robovla.core.ticket import MAX_FIELD_LEN, create_ticket, set_outcome
from robovla.core.audit import AuditLog, parse_rfc3339
from robovla.episode import Episode, validate_episode


def _node(nid="nodeA", priority=0, train_slots=1, infer_slots=1, enabled=True):
    from robovla.core.config import (
        AgentConfig,
        GpuBudgets,
        GpuConfig,
        NodeConfig,
        Presets,
    )

    return NodeConfig(
        id=nid,
        label=nid,
        image="img",
        workdir="/w",
        agent=AgentConfig(lan_ip="10.9.9.9", port=8601, token_ref="tok"),
        gpu=GpuConfig(32.0, "bnb8", train_slots, infer_slots, GpuBudgets(18.0, 11.0)),
        presets=Presets((1, 4, 8)),
        priority=priority,
        student_selectable=False,
        enabled=enabled,
    )


def _job(jid="j_20261021_0001", **kw):
    kw.setdefault("student", "s07")
    kw.setdefault("job_type", "train")
    kw.setdefault("target_node", None)
    kw.setdefault("run_now", False)
    kw.setdefault("submitted_at", 1.0)
    kw.setdefault("budget_gb", 11.0)
    kw.setdefault("est_duration_s", 60.0)
    return Job(job_id=jid, **kw)


class TestSchedulerEdges:
    def test_empty_fleet_rejected(self):
        with pytest.raises(RobovlaError) as e:
            Scheduler([])
        assert e.value.code == "validation"

    def test_unknown_job_not_found(self):
        s = Scheduler([_node()])
        with pytest.raises(RobovlaError) as e:
            s.job("j_20990101_0001")
        assert e.value.code == "not_found"

    def test_accessors(self):
        s = Scheduler([_node("a"), _node("b", priority=5)])
        s.add(_job("j_20261021_0001"))
        s.add(_job("j_20261021_0002"))
        assert [j.job_id for j in s.jobs()] == ["j_20261021_0001", "j_20261021_0002"]
        assert s.node_ids() == ("a", "b")

    def test_add_validation_edges(self):
        s = Scheduler([_node()])
        with pytest.raises(RobovlaError):
            s.add(42)
        with pytest.raises(RobovlaError):
            s.add(_job(jid=""))
        with pytest.raises(RobovlaError):
            s.add(_job(jid=123))
        with pytest.raises(RobovlaError):
            s.add(_job(student=""))
        with pytest.raises(RobovlaError):
            s.add(_job(est_duration_s=-1.0))
        with pytest.raises(RobovlaError):
            s.add(_job(submitted_at=-1.0))
        with pytest.raises(RobovlaError) as e:
            s.add(_job(state="running"))
        assert e.value.code == "conflict"

    def test_set_target_unknown_node(self):
        s = Scheduler([_node()])
        s.add(_job())
        with pytest.raises(RobovlaError) as e:
            s.set_target("j_20261021_0001", "ghost")
        assert e.value.code == "validation"

    def test_admission_node_edges(self):
        s = Scheduler([_node("nodeA", priority=5, enabled=False), _node("nodeB", priority=0)])
        s.add(_job(target_node="nodeB"))
        states_off = {"nodeA": NodeState(False, 32.0), "nodeB": NodeState(False, 32.0)}
        assert s.admission_node("j_20261021_0001", states_off) is None
        states_short = {"nodeA": NodeState(True, 32.0), "nodeB": NodeState(True, 5.0)}
        assert s.admission_node("j_20261021_0001", states_short) is None
        # slot full
        s.add(_job("j_20261021_0000", submitted_at=0.5))
        s.transition("j_20261021_0000", "start", node_id="nodeB")
        assert s.admission_node("j_20261021_0001", states_short) is None
        # non-queued job
        assert s.admission_node("j_20261021_0000", states_short) is None

    def test_blockers_edges(self):
        s = Scheduler([_node("nodeA", priority=0), _node("nodeB", priority=5)])
        s.add(_job("j_20261021_0001", submitted_at=0.5))
        s.transition("j_20261021_0001", "start", node_id="nodeA")
        assert s.admission_blockers("j_20261021_0001", {}) == []  # non-queued
        # auto job: A busy, B free -> admissible now -> []
        s.add(_job("j_20261021_0002"))
        assert s.admission_blockers("j_20261021_0002", {"nodeB": NodeState(True, 32.0)}) == []

    def test_position_terminal_is_zero(self):
        s = Scheduler([_node()])
        s.add(_job())
        s.transition("j_20261021_0001", "start", node_id="nodeA")
        assert s.position("j_20261021_0001", {"nodeA": NodeState(True, 32.0)}) == 0

    def test_eta_pinned_to_disabled_or_offline_node(self):
        s = Scheduler([_node("nodeA", enabled=False)])
        s.add(_job(target_node="nodeA"))
        assert (
            s.eta_s("j_20261021_0001", {"nodeA": NodeState(True, 32.0)}) == float("inf")
        )
        s2 = Scheduler([_node("nodeA"), _node("nodeB", priority=5)])
        s2.add(_job(target_node="nodeA"))
        states = {"nodeA": NodeState(False, 32.0), "nodeB": NodeState(True, 32.0)}
        # pinned to A: B's free VRAM doesn't help; A offline -> inf
        assert s2.eta_s("j_20261021_0001", states) == float("inf")

    def test_duration_defaults_validation(self):
        with pytest.raises(RobovlaError):
            default_duration_s("train")  # needs batch + epochs
        with pytest.raises(RobovlaError):
            default_duration_s("crosscheck", batch=4)
        with pytest.raises(RobovlaError):
            default_duration_s("train", batch=0, epochs=3)
        with pytest.raises(RobovlaError):
            default_duration_s("train", batch=4, epochs=0)


class TestConfigEdges:
    FLEET = None  # reuse the canonical fixture from test_config

    def _fleet_dict(self):
        import test_config

        return copy.deepcopy(test_config.FLEET)

    def test_nan_budget_rejected(self):
        d = self._fleet_dict()
        d["nodes"][0]["gpu"]["budgets"]["train_gb"] = math.nan
        with pytest.raises(RobovlaError):
            validate_fleet(d)

    def test_gpu_and_agent_must_be_objects(self):
        d = self._fleet_dict()
        d["nodes"][0]["gpu"] = [1]
        with pytest.raises(RobovlaError) as e:
            validate_fleet(d)
        assert "object" in e.value.message
        d = self._fleet_dict()
        d["nodes"][0]["agent"] = [1]
        with pytest.raises(RobovlaError):
            validate_fleet(d)

    def test_priority_defaults_to_zero(self):
        d = self._fleet_dict()
        del d["nodes"][0]["priority"]
        f = validate_fleet(d)
        assert f.nodes_by_id()["pai1"].priority == 0

    def test_diff_matrix(self):
        base = validate_fleet(self._fleet_dict())
        d = self._fleet_dict()
        n0 = d["nodes"][0]
        n0["gpu"]["quant"] = "bf16"
        n0["gpu"]["vram_gb"] = 64
        n0["gpu"]["train_slots"] = 2
        n0["gpu"]["infer_slots"] = 1
        n0["gpu"]["budgets"]["infer_gb"] = 12
        n0["presets"]["batch"] = [1]
        n0["agent"]["lan_ip"] = "10.0.0.50"
        c = d["console"]
        c["bind"] = "10.0.0.1"
        c["port"] = 9000
        c["tls"] = True
        c["data_dir"] = "/x"
        c["session_ttl_h"] = 6
        c["limits"]["admin_rate"] = "30/min"
        c["retention"]["run_logs_per_student"] = 5
        c["retention"]["merged_export_days"] = 7
        c["retention"]["backup"]["enabled"] = False
        c["retention"]["backup"]["at"] = "03:00"
        c["retention"]["backup"]["dest"] = "/y"
        changes = diff_fleet(base, validate_fleet(d))
        got = {(c_.scope, c_.field) for c_ in changes}
        for expected in (
            ("nodes.pai1.gpu", "quant"),
            ("nodes.pai1.gpu", "vram_gb"),
            ("nodes.pai1.gpu", "train_slots"),
            ("nodes.pai1.gpu", "infer_slots"),
            ("nodes.pai1.gpu", "infer_gb"),
            ("nodes.pai1.presets", "batch"),
            ("nodes.pai1.agent", "endpoint"),
            ("console", "bind"),
            ("console", "port"),
            ("console", "tls"),
            ("console", "data_dir"),
            ("console", "session_ttl_h"),
            ("console.limits", "admin_rate"),
            ("console.retention", "run_logs_per_student"),
            ("console.retention", "merged_export_days"),
            ("console.retention.backup", "enabled"),
            ("console.retention.backup", "at"),
            ("console.retention.backup", "dest"),
        ):
            assert expected in got, f"missing diff for {expected}"
        assert all(isinstance(c_, Change) for c_ in changes)


class TestBudgetsEdges:
    def test_infer_budget_must_be_positive(self):
        with pytest.raises(RobovlaError):
            estimate_job_gb(18.0, 0.0, "bnb8", "infer", 1, 8)

    def test_train_base_must_be_positive_for_full_base(self):
        with pytest.raises(RobovlaError):
            estimate_job_gb(-1.0, 11.0, "bnb8", "merge", None, 8)

    def test_nan_base_rejected(self):
        with pytest.raises(RobovlaError):
            estimate_train_gb(math.nan, "bnb8", 1, 8)


class TestNormStatsEdges:
    def test_wrong_dim_count(self):
        with pytest.raises(RobovlaError):
            NormStats((0.0,) * 6, (1.0,) * 6)

    def test_non_finite(self):
        with pytest.raises(RobovlaError):
            NormStats((math.nan,) + (0.0,) * 6, (1.0,) * 7)

    def test_min_above_max(self):
        with pytest.raises(RobovlaError):
            NormStats((1.0,) * 7, (0.0,) * 7)

    def test_to_dict(self):
        s = NormStats((0.0,) * 7, (1.0,) * 7)
        assert s.to_dict() == {"dims": 7, "min": [0.0] * 7, "max": [1.0] * 7}


class TestSpecEdges:
    def test_model_too_long(self):
        d = _valid()
        d["model"] = "x" * 201
        with pytest.raises(RobovlaError) as e:
            validate_job_spec(d)
        assert "model" in e.value.message

    def test_nan_lr(self):
        d = _valid()
        d["optim"]["lr"] = math.nan
        with pytest.raises(RobovlaError):
            validate_job_spec(d)

    def test_node_must_be_object(self):
        d = _valid()
        d["node"] = ["pai1"]
        with pytest.raises(RobovlaError) as e:
            validate_job_spec(d)
        assert "object" in e.value.message

    def test_lora_missing_key(self):
        d = _valid()
        del d["lora"]["alpha"]
        with pytest.raises(RobovlaError) as e:
            validate_job_spec(d)
        assert "alpha" in e.value.message

    def test_bf16_batch_outside_catalog(self):
        d = _valid(quant="bf16")
        d["optim"]["batch"] = 16
        d["optim"]["grad_accum"] = 1
        with pytest.raises(RobovlaError):
            validate_job_spec(d)

    def test_build_catalog_validation(self):
        with pytest.raises(RobovlaError):
            build_job_spec(
                job_id="j_20261021_0001", student="s01", job_type="retrain",
                dataset="s01/task/v1", node_id="pai1", node_agent="10.0.0.10:8601",
                budget_gb=18.0,
            )
        with pytest.raises(RobovlaError):
            build_job_spec(
                job_id="j_20261021_0001", student="s01", job_type="train",
                dataset="s01/task/v1", node_id="pai1", node_agent="10.0.0.10:8601",
                budget_gb=18.0, quant="fp42",
            )
        with pytest.raises(RobovlaError):
            build_job_spec(
                job_id="j_20261021_0001", student="s01", job_type="train",
                dataset="s01/task/v1", node_id="pai1", node_agent="10.0.0.10:8601",
                budget_gb=18.0, batch=2,
            )
        with pytest.raises(RobovlaError):
            build_job_spec(
                job_id="j_20261021_0001", student="s01", job_type="train",
                dataset="s01/task/v1", node_id="pai1", node_agent="10.0.0.10:8601",
                budget_gb=18.0, quant="bf16", batch=16,
            )
        # bf16 happy path: grad_accum defaults to 1
        spec = build_job_spec(
            job_id="j_20261021_0001", student="s01", job_type="train",
            dataset="s01/task/v1", node_id="pai1", node_agent="10.0.0.10:8601",
            budget_gb=18.0, quant="bf16", batch=1,
        )
        assert spec.optim.grad_accum == 1


def _valid(quant="bnb8"):
    import test_spec

    d = test_spec._valid_dict(quant=quant)
    if quant == "bf16":
        d["optim"]["batch"] = 1
        d["optim"]["grad_accum"] = 1
    return d


class TestStatesEdges:
    def test_unknown_state_in_transition(self):
        with pytest.raises(RobovlaError) as e:
            apply_transition("paused", "start")
        assert e.value.code == "validation"


class TestTicketEdges:
    def test_to_dict(self):
        t = create_ticket(
            ticket_id="t_0001", owner="s07",
            payload={
                "hypothesis": "h", "config_summary": "c", "prediction": "p",
            },
            at="2026-10-21T12:00:00Z",
        )
        d = t.to_dict()
        assert d["state"] == "unlinked" and d["linked_job"] is None
        assert d["hypothesis"] == "h"

    def test_payload_must_be_object(self):
        with pytest.raises(RobovlaError):
            create_ticket(ticket_id="t_1", owner="s07", payload=["nope"], at="2026-10-21T12:00:00Z")

    def test_outcome_too_long(self):
        t = create_ticket(
            ticket_id="t_1", owner="s07",
            payload={"hypothesis": "h", "config_summary": "c", "prediction": "p"},
            at="2026-10-21T12:00:00Z",
        )
        with pytest.raises(RobovlaError):
            set_outcome(t, "x" * (MAX_FIELD_LEN + 1), at="2026-10-21T13:00:00Z")


class TestAuditEdges:
    def test_iter(self):
        log = AuditLog()
        log.record("2026-10-21T09:00:00Z", "a", "b", "c")
        assert [e.action for e in log] == ["b"]

    def test_rfc3339_parse_invalid_date(self):
        with pytest.raises(RobovlaError):
            parse_rfc3339("2026-02-30T00:00:00Z")  # regex passes, calendar fails


class TestEpisodeEdges:
    def test_action_range_detail(self):
        from robovla.episode import Action

        ep = Episode(
            id="ep_0001",
            frame_count=20,
            actions=tuple(
                Action(t=i * 0.1, vx=1.5 if i == 5 else 0.2, vy=0.0, wz=0.0) for i in range(20)
            ),
        )
        r = validate_episode(ep)
        assert r.status == "fail"
        issue = next(i for i in r.issues if i.rule == "action_range")
        assert "1.5" in issue.detail

    def test_single_action_skips_timing(self):
        from robovla.episode import Action

        ep = Episode(id="ep_0001", frame_count=1, actions=(Action(0.0, 0.2, 0.0, 0.0),))
        r = validate_episode(ep)
        assert r.status == "ok"

    def test_non_episode_input_rejected(self):
        with pytest.raises(RobovlaError):
            validate_episode({"not": "an episode"})
