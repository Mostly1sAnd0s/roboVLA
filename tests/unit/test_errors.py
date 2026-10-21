"""core.errors — the ONE error type (design §9.6, DRY: one error shape)."""

import pytest

from robovla.core.errors import (
    CODES,
    STATUS_BY_CODE,
    RobovlaError,
)


class TestErrorType:
    def test_carries_code_and_message(self):
        err = RobovlaError("conflict", "ticket already linked")
        assert err.code == "conflict"
        assert err.message == "ticket already linked"
        assert str(err) == "ticket already linked"

    def test_is_a_value_error(self):
        assert isinstance(RobovlaError("internal", "boom"), ValueError)

    def test_payload_shape_matches_9_6(self):
        err = RobovlaError("node_offline", "agent unreachable")
        assert err.to_payload() == {"error": {"code": "node_offline", "message": "agent unreachable"}}

    def test_requires_known_code(self):
        with pytest.raises(RobovlaError):
            RobovlaError("bogus_code", "nope")

    def test_code_and_message_required(self):
        with pytest.raises(RobovlaError):
            RobovlaError("", "x")
        with pytest.raises(RobovlaError):
            RobovlaError("conflict", "")


class TestStableCodeSet:
    def test_contains_all_9_6_codes(self):
        for code in (
            "auth",
            "forbidden",
            "not_found",
            "conflict",
            "dataset_not_ready",
            "budget_unfit",
            "limit_exceeded",
            "rate_limited",
            "node_offline",
            "internal",
        ):
            assert code in CODES

    def test_status_mapping(self):
        assert STATUS_BY_CODE["auth"] == 401
        assert STATUS_BY_CODE["forbidden"] == 403
        assert STATUS_BY_CODE["not_found"] == 404
        # domain conflicts -> 409 (test matrix: "409 if already linked")
        for code in ("conflict", "dataset_not_ready", "budget_unfit", "limit_exceeded"):
            assert STATUS_BY_CODE[code] == 409
        assert STATUS_BY_CODE["rate_limited"] == 429
        assert STATUS_BY_CODE["node_offline"] == 503
        assert STATUS_BY_CODE["internal"] == 500
        # malformed request -> 400 (test matrix: "train: 400 without ticket")
        assert STATUS_BY_CODE["validation"] == 400

    def test_every_code_has_a_status(self):
        assert set(CODES) == set(STATUS_BY_CODE)
