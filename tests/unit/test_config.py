"""core.config — fleet.yaml schema + loader + diff/reload rules (design §10,
Appendix A). Core stays pure: ``parse_fleet`` takes YAML *text*; reading the
file happens at the seam (app/cli). ``diff`` feeds the admin audit log.
"""

import copy

import pytest
import yaml

from robovla.core.config import (
    FLEET_VERSION,
    NODE_ID_RE,
    Change,
    FleetConfig,
    diff_fleet,
    parse_fleet,
    validate_fleet,
)
from robovla.core.errors import RobovlaError

# The shipped default fleet (Appendix A) as a dict — the single fixture.
# Placeholder IPs from the spec become concrete values here; code never
# hardcodes node names or IPs (DRY).
FLEET = {
    "version": 1,
    "console": {
        "bind": "0.0.0.0",  # pai1-only default (single LAN interface; §9.8)
        "port": 8501,
        "tls": False,
        "data_dir": "/home/adam/robovla-console",
        "session_ttl_h": 12,
        "limits": {
            "student": {
                "max_running": 1,
                "max_queued": 2,
                "max_dataset_bytes": 2147483648,
                "max_run_min": 180,
                "run_now": True,
            },
            "admin_rate": "60/min",
        },
        "retention": {
            "run_logs_per_student": 20,
            "merged_export_days": 14,
            "backup": {"enabled": True, "at": "02:30", "dest": "/home/adam/robovla-backups"},
        },
    },
    "nodes": [
        {
            "id": "pai1",
            "label": "pai1 (RTX 5090, primary)",
            "image": "robovla-runner-cuda:2026.10",
            "workdir": "/home/adam/robovla",
            "agent": {"lan_ip": "10.0.0.10", "port": 8601, "token_ref": "pai1-agent"},
            "gpu": {
                "vram_gb": 32,
                "quant": "bnb8",
                "train_slots": 1,
                "infer_slots": 2,
                "budgets": {"train_gb": 18, "infer_gb": 11},
            },
            "presets": {"batch": [1, 4, 8]},
            "priority": 0,
            "student_selectable": True,
            "enabled": True,
        },
        {
            "id": "node2",
            "label": "campus RTX 5090 Ti",
            "image": "robovla-runner-cuda:2026.10",
            "workdir": "/home/adam/robovla",
            "agent": {"lan_ip": "10.0.0.11", "port": 8601, "token_ref": "node2-agent"},
            "gpu": {
                "vram_gb": 32,
                "quant": "bnb8",
                "train_slots": 1,
                "infer_slots": 2,
                "budgets": {"train_gb": 18, "infer_gb": 11},
            },
            "presets": {"batch": [1, 4, 8]},
            "priority": 0,
            "student_selectable": False,
            "enabled": False,  # until join probe passes
        },
        {
            "id": "node3",
            "label": "campus RTX 4070 Ti (batch-1 only)",
            "image": "robovla-runner-cuda:2026.10",
            "workdir": "/home/adam/robovla",
            "agent": {"lan_ip": "10.0.0.12", "port": 8601, "token_ref": "node3-agent"},
            "gpu": {
                "vram_gb": 16,
                "quant": "bnb8",
                "train_slots": 1,
                "infer_slots": 1,
                "budgets": {"train_gb": 14, "infer_gb": 11},
            },
            "presets": {"batch": [1]},  # grad-accum auto-matched to the recipe
            "priority": 0,
            "student_selectable": False,
            "enabled": False,  # until join probe passes
        },
    ],
}


def _fleet():
    return validate_fleet(copy.deepcopy(FLEET))


class TestParse:
    def test_shipped_default_fleet(self):
        f = _fleet()
        assert f.version == FLEET_VERSION == 1
        assert f.console.port == 8501
        assert f.console.bind == "0.0.0.0"
        assert f.console.tls is False
        assert f.console.session_ttl_h == 12
        s = f.console.limits.student
        assert (s.max_running, s.max_queued, s.max_run_min) == (1, 2, 180)
        assert s.max_dataset_bytes == 2147483648
        assert s.run_now is True
        assert f.console.limits.admin_rate == "60/min"
        assert f.console.limits.admin_rate_per_min == 60
        r = f.console.retention
        assert r.run_logs_per_student == 20
        assert r.merged_export_days == 14
        assert r.backup.enabled and r.backup.at == "02:30"

        # launch fleet: pai1 only enabled (Appendix A note)
        assert [n.id for n in f.nodes] == ["pai1", "node2", "node3"]
        assert [n.id for n in f.enabled_nodes()] == ["pai1"]
        pai1 = f.nodes_by_id()["pai1"]
        assert pai1.student_selectable is True
        assert pai1.gpu.vram_gb == 32 and pai1.gpu.quant == "bnb8"
        assert (pai1.gpu.train_slots, pai1.gpu.infer_slots) == (1, 2)
        assert (pai1.gpu.budgets.train_gb, pai1.gpu.budgets.infer_gb) == (18.0, 11.0)
        assert pai1.presets.batch == (1, 4, 8)
        assert f.nodes_by_id()["node3"].presets.batch == (1,)
        assert f.nodes_by_id()["node3"].gpu.vram_gb == 16

    def test_yaml_round_trip_through_loader(self):
        # the seam reads the file; parse_fleet is the pure text -> config step
        text = yaml.safe_dump(copy.deepcopy(FLEET), sort_keys=False)
        f = parse_fleet(text)
        assert [n.id for n in f.nodes] == ["pai1", "node2", "node3"]
        assert f.console.port == 8501

    def test_nodes_by_priority_order(self):
        d = copy.deepcopy(FLEET)
        d["nodes"][0]["priority"] = 0
        d["nodes"][1]["priority"] = 5
        d["nodes"][2]["priority"] = 3
        f = validate_fleet(d)
        assert [n.id for n in f.nodes_by_priority()] == ["node2", "node3", "pai1"]

    def test_priority_ties_keep_file_order(self):
        f = _fleet()  # all priorities 0
        assert [n.id for n in f.nodes_by_priority()] == ["pai1", "node2", "node3"]


class TestValidation:
    def test_not_a_mapping_rejected(self):
        with pytest.raises(RobovlaError) as e:
            validate_fleet([1, 2, 3])
        assert e.value.code == "validation"

    def test_version_must_be_one(self):
        d = copy.deepcopy(FLEET)
        d["version"] = 2
        with pytest.raises(RobovlaError) as e:
            validate_fleet(d)
        assert "version" in e.value.message

    @pytest.mark.parametrize("bad_text", ["not: [valid", "nodes: [", ""])
    def test_malformed_yaml_rejected(self, bad_text):
        with pytest.raises(RobovlaError) as e:
            parse_fleet(bad_text)
        assert e.value.code == "validation"

    def test_duplicate_node_id_rejected(self):
        d = copy.deepcopy(FLEET)
        d["nodes"][1]["id"] = "pai1"
        with pytest.raises(RobovlaError) as e:
            validate_fleet(d)
        assert "pai1" in e.value.message

    def test_node_id_format(self):
        assert NODE_ID_RE.fullmatch("pai1")
        assert NODE_ID_RE.fullmatch("node-2")
        assert not NODE_ID_RE.fullmatch("2node")
        assert not NODE_ID_RE.fullmatch("NODE")
        d = copy.deepcopy(FLEET)
        d["nodes"][0]["id"] = "Bad ID"
        with pytest.raises(RobovlaError):
            validate_fleet(d)

    def test_no_nodes_rejected(self):
        d = copy.deepcopy(FLEET)
        d["nodes"] = []
        with pytest.raises(RobovlaError):
            validate_fleet(d)

    @pytest.mark.parametrize(
        "mutate, needle",
        [
            (lambda n: n["gpu"].update({"quant": "fp16"}), "quant"),
            (lambda n: n["gpu"].update({"train_slots": 3}), "train_slots"),
            (lambda n: n["gpu"].update({"infer_slots": 0}), "infer_slots"),
            (lambda n: n["gpu"]["budgets"].update({"train_gb": 5}), "train_gb"),
            (lambda n: n["gpu"]["budgets"].update({"infer_gb": 0}), "infer_gb"),
            (lambda n: n["gpu"].update({"vram_gb": 0}), "vram_gb"),
            (lambda n: n["presets"].update({"batch": []}), "batch"),
            (lambda n: n["presets"].update({"batch": [1, 16]}), "batch"),
            (lambda n: n["agent"].update({"lan_ip": "300.0.0.1"}), "lan_ip"),
            (lambda n: n["agent"].update({"port": 70000}), "port"),
            (lambda n: n["agent"].update({"token_ref": ""}), "token_ref"),
            (lambda n: n.update({"priority": "high"}), "priority"),
            (lambda n: n.update({"enabled": "yes"}), "enabled"),
            (lambda n: n.update({"image": ""}), "image"),
        ],
    )
    def test_node_field_bounds(self, mutate, needle):
        d = copy.deepcopy(FLEET)
        mutate(d["nodes"][0])
        with pytest.raises(RobovlaError) as e:
            validate_fleet(d)
        assert needle in e.value.message

    def test_bf16_budget_must_meet_bf16_floor(self):
        # bf16 weights are 15 GB (budgets.FIXED_GB) — a 10 GB train budget is
        # below the model's floor and must be rejected at config time.
        d = copy.deepcopy(FLEET)
        d["nodes"][0]["gpu"]["quant"] = "bf16"
        d["nodes"][0]["gpu"]["budgets"]["train_gb"] = 10
        with pytest.raises(RobovlaError) as e:
            validate_fleet(d)
        assert "floor" in e.value.message

    @pytest.mark.parametrize(
        "mutate, needle",
        [
            (lambda c: c["limits"]["student"].update({"max_running": -1}), "max_running"),
            (lambda c: c["limits"]["student"].update({"max_queued": -1}), "max_queued"),
            (lambda c: c["limits"]["student"].update({"max_dataset_bytes": 0}), "max_dataset_bytes"),
            (lambda c: c["limits"]["student"].update({"max_run_min": 0}), "max_run_min"),
            (lambda c: c["limits"]["student"].update({"run_now": "yes"}), "run_now"),
            (lambda c: c["limits"].update({"admin_rate": "60/sec"}), "admin_rate"),
            (lambda c: c["limits"].update({"admin_rate": "per/min"}), "admin_rate"),
            (lambda c: c.update({"port": 0}), "port"),
            (lambda c: c.update({"session_ttl_h": 0}), "session_ttl_h"),
            (lambda c: c["retention"]["backup"].update({"at": "25:00"}), "at"),
            (lambda c: c["retention"]["backup"].update({"at": "2:30"}), "at"),
            (lambda c: c["retention"].update({"merged_export_days": 0}), "merged_export_days"),
        ],
    )
    def test_console_field_bounds(self, mutate, needle):
        d = copy.deepcopy(FLEET)
        mutate(d["console"])
        with pytest.raises(RobovlaError) as e:
            validate_fleet(d)
        assert needle in e.value.message

    def test_missing_top_level_rejected(self):
        d = copy.deepcopy(FLEET)
        del d["nodes"]
        with pytest.raises(RobovlaError) as e:
            validate_fleet(d)
        assert "nodes" in e.value.message


class TestDiff:
    def test_no_changes(self):
        assert diff_fleet(_fleet(), _fleet()) == []

    def test_node_enable_flagged(self):
        d = copy.deepcopy(FLEET)
        d["nodes"][1]["enabled"] = True
        changes = diff_fleet(_fleet(), validate_fleet(d))
        assert Change("nodes.node2", "enabled", False, True) in changes

    def test_limit_change_flagged(self):
        d = copy.deepcopy(FLEET)
        d["console"]["limits"]["student"]["max_queued"] = 3
        changes = diff_fleet(_fleet(), validate_fleet(d))
        assert Change("console.limits.student", "max_queued", 2, 3) in changes

    def test_added_and_removed_nodes(self):
        d = copy.deepcopy(FLEET)
        node4 = copy.deepcopy(d["nodes"][1])
        node4["id"] = "node4"
        node4["agent"] = {"lan_ip": "10.0.0.13", "port": 8601, "token_ref": "node4-agent"}
        d["nodes"].append(node4)
        d["nodes"] = [d["nodes"][0], d["nodes"][1], node4]  # drop node3, keep node4
        changes = diff_fleet(_fleet(), validate_fleet(d))
        assert Change("nodes", "added", None, "node4") in changes
        assert Change("nodes", "removed", "node3", None) in changes

    def test_budget_change_flagged(self):
        d = copy.deepcopy(FLEET)
        d["nodes"][0]["gpu"]["budgets"]["train_gb"] = 20
        changes = diff_fleet(_fleet(), validate_fleet(d))
        assert Change("nodes.pai1.gpu", "train_gb", 18.0, 20.0) in changes

    def test_diff_is_flat_and_audit_shape(self):
        d = copy.deepcopy(FLEET)
        d["nodes"][1]["enabled"] = True
        d["nodes"][1]["priority"] = 2
        changes = diff_fleet(_fleet(), validate_fleet(d))
        assert all(isinstance(c, Change) for c in changes)
        # before/after JSON-serializable for the audit log (§8: before→after)
        import json
        json.dumps([c.to_dict() for c in changes])
