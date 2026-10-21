"""HttpRunner — the single real Runner implementation (design §4, §10).

Console -> node agent over LAN HTTP (default port 8601, console-only
traffic), authenticated with the per-node pre-shared agent token (Bearer;
the console verifies it against the argon2 hash in its DB for the reverse
pull, §9.9). The agent API: jobs/serve/vram/health + /predict (§4).

Errors: connection failure / timeout -> ``node_offline``; HTTP 4xx/5xx ->
the agent's §9.6 code (agent errors mirror the console's shape). A custom
``httpx.Client`` can be injected (tests use ``MockTransport``).
"""

from __future__ import annotations

from typing import Any

import httpx

from ..core.errors import RobovlaError
from .interface import Runner

#: Agent-side error code set mirrors the console's (§9.6, stable).
_CODE_BY_STATUS = {
    401: "auth",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    429: "rate_limited",
    503: "node_offline",
}

DEFAULT_TIMEOUT_S = 10.0
PREDICT_TIMEOUT_S = 5.0  # robot steps are latency-sensitive (§9.5: p95 < 400 ms target)


class HttpRunner:
    """One node's agent over HTTP (implements :class:`Runner`)."""

    def __init__(
        self,
        base_url: str,
        token: str,
        client: httpx.Client | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise RobovlaError("validation", f"agent base_url must be http(s): {base_url!r}")
        self._client = client or httpx.Client(
            base_url=base_url, timeout=timeout_s,
        )
        # one source for auth: the header is set here, not by callers
        self._client.headers["Authorization"] = f"Bearer {token}"
        self._owns_client = client is None
        self._predict_timeout = PREDICT_TIMEOUT_S

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- transport ---------------------------------------------------------------

    def _request(self, method: str, path: str, *, json: Any = None,
                 params: Any = None, timeout: float | None = None) -> Any:
        try:
            resp = self._client.request(method, path, json=json, params=params, timeout=timeout)
        except httpx.HTTPError as e:
            raise RobovlaError("node_offline", f"agent unreachable: {e.__class__.__name__}") from e
        if resp.status_code >= 400:
            code = _CODE_BY_STATUS.get(resp.status_code, "internal")
            try:
                body = resp.json()
                err = body.get("error", {})
                code = err.get("code", code) if isinstance(err, dict) else code
                message = err.get("message", resp.text[:200]) if isinstance(err, dict) else resp.text[:200]
            except (ValueError, AttributeError):
                message = resp.text[:200] or f"HTTP {resp.status_code}"
            raise RobovlaError(code, f"agent {path}: {message}")
        return resp.json() if resp.content else None

    # -- node probes ---------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def vram(self) -> dict[str, Any]:
        return self._request("GET", "/vram")

    # -- jobs -----------------------------------------------------------------------

    def submit(self, spec: dict[str, Any]) -> None:
        self._request("POST", "/jobs", json={"spec": spec})

    def poll(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}")

    def kill(self, job_id: str) -> None:
        self._request("POST", f"/jobs/{job_id}/kill")

    def tail(self, job_id: str, offset: int = 0) -> str:
        data = self._request("GET", f"/jobs/{job_id}/logs", params={"offset": offset})
        return data["log"] if isinstance(data, dict) else str(data)

    def fetch_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/jobs/{job_id}/artifacts")
        return data.get("artifacts", []) if isinstance(data, dict) else data

    # -- inference slots ------------------------------------------------------------------

    def serve(self, run_id: str, adapter: str) -> str:
        data = self._request("POST", "/serve", json={"run_id": run_id, "adapter": adapter})
        return data["handle"]

    def release(self, handle: str) -> None:
        self._request("DELETE", f"/serve/{handle}")

    def predict(self, handle: str, image_b64: str, prompt: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/predict",
            json={"handle": handle, "image": image_b64, "prompt": prompt},
            timeout=self._predict_timeout,
        )
