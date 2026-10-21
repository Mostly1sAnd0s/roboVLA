"""The ONE error type for roboVLA (design §9.6, DRY checklist: "one error type").

Every layer (core, storage, runner, http) raises :class:`RobovlaError`; the
HTTP seam renders it as the §9.6 shape ``{"error": {"code", "message"}}`` with
the status from :data:`STATUS_BY_CODE`. No module defines its own error type
or status mapping.

Stable code set — the Pi client and tests depend on these (do not rename).
``validation`` (400) was added beyond the §9.6 list for malformed requests;
the test matrix requires 400 for e.g. "train submitted without ticket".
"""

from __future__ import annotations

#: The stable §9.6 code set (plus ``validation`` for 400-class schema errors).
CODES = frozenset(
    {
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
        "validation",
    }
)

#: code -> HTTP status (single mapping; consumed only by the http seam).
STATUS_BY_CODE = {
    "auth": 401,
    "forbidden": 403,
    "not_found": 404,
    "conflict": 409,
    "dataset_not_ready": 409,
    "budget_unfit": 409,
    "limit_exceeded": 409,
    "rate_limited": 429,
    "node_offline": 503,
    "internal": 500,
    "validation": 400,
}


class RobovlaError(ValueError):
    """Domain error with a stable §9.6 code.

    ``code`` must be one of :data:`CODES`; ``message`` is a short,
    human-readable sentence (never contains secrets, node IPs, or stack
    traces — those belong in the service log, not in API responses).
    """

    def __init__(self, code: str, message: str) -> None:
        if code not in CODES:
            raise RobovlaError("internal", f"unknown error code: {code!r}")
        if not message:
            raise RobovlaError("internal", "error message must be non-empty")
        self.code = code
        self.message = message
        super().__init__(message)

    def to_payload(self) -> dict:
        """The §9.6 JSON error shape: ``{"error": {"code", "message"}}``."""
        return {"error": {"code": self.code, "message": self.message}}
