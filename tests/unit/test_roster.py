"""roster.csv — the shipped 12-student roster (design §8.1/§9.7: ids
s01–s12; created **inactive** until the teacher activates; s00 is reserved
for system jobs and must never be a roster account). The header is exactly
the admin import contract (id,name,email,role,active).
"""

import csv
import io
import re
import subprocess
from pathlib import Path

from robovla.core.errors import RobovlaError
import pytest

REPO = Path(__file__).resolve().parents[2]
ROSTER = REPO / "roster.csv"
STUDENT_IDS = tuple(f"s{i:02d}" for i in range(1, 13))
HEADER = ["id", "name", "email", "role", "active"]


def _rows() -> list[dict]:
    with ROSTER.open(newline="") as f:
        return list(csv.DictReader(f))


class TestShippedRoster:
    def test_header_is_the_import_contract(self):
        with ROSTER.open(newline="") as f:
            header = next(csv.reader(f))
        assert header == HEADER

    def test_twelve_students_s01_to_s12(self):
        rows = _rows()
        assert [r["id"] for r in rows] == list(STUDENT_IDS)

    def test_s00_is_reserved_not_in_roster(self):
        assert "s00" not in [r["id"] for r in _rows()]

    def test_ids_match_the_student_format(self):
        # sNN is the account shape the rest of the system assumes (storage
        # users, dataset ids sNN/<task>/vN, JobSpec.student)
        for r in _rows():
            assert re.fullmatch(r"s\d{2}", r["id"])

    def test_all_students_inactive_by_default(self):
        # §8.1: created inactive until activated (teacher lever)
        assert all(r["active"].strip().lower() in ("", "false", "0", "no")
                   for r in _rows())

    def test_roles_are_student(self):
        assert all(r["role"].strip() == "student" for r in _rows())

    def test_every_row_has_a_display_name(self):
        assert all(r["name"].strip() for r in _rows())


class TestOpsArtifacts:
    """The operator artifacts ship with the repo and stay syntactically
    sound: the robovla-node host script (§8.2/§10) and the systemd units
    (Appendix C). No node name/IP/token may be hardcoded in the units
    (hardware lives in fleet.yaml / node env, §10)."""

    NODE = REPO / "robovla-node"
    CONSOLE_UNIT = REPO / "systemd" / "robovla-console.service"
    RUNNER_UNIT = REPO / "systemd" / "robovla-runner.service"

    def test_node_script_is_valid_bash(self):
        r = subprocess.run(["bash", "-n", str(self.NODE)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    def test_node_script_covers_the_documented_one_liners(self):
        text = self.NODE.read_text()
        for cmd in ("install", "update", "restart", "status", "gpu-reset"):
            assert re.search(rf'^\s*{cmd}\)\s*$', text, re.M), f"missing {cmd}"

    def test_node_script_gpu_reset_needs_confirm(self):
        text = self.NODE.read_text()
        assert "--confirm" in text
        # the guard: refuse while the container runs a job process
        assert "pgrep" in text

    def test_console_unit_runs_serve_and_restarts(self):
        text = self.CONSOLE_UNIT.read_text()
        assert "robovla serve" in text
        assert "Restart=on-failure" in text
        assert "[Install]" in text and "multi-user.target" in text

    def test_runner_unit_wraps_podman_with_gpu_passthrough(self):
        text = self.RUNNER_UNIT.read_text()
        assert "podman run" in text
        # GPU passthrough (the same shape the GPU_TESTS rows use)
        assert "/dev/nvidia0" in text
        assert "SYS_PTRACE" in text
        # per-node values come from the env file, never hardcoded here
        assert "EnvironmentFile" in text

    def test_units_hardcode_no_ip_or_token_value(self):
        for p in (self.CONSOLE_UNIT, self.RUNNER_UNIT):
            text = p.read_text()
            assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text), \
                f"{p.name}: hardcoded IP"
            # env var *names* may appear; a value assigned in the unit may not
            assert not re.search(r"TOKEN=\S+", text), \
                f"{p.name}: token value where the env var name belongs"
