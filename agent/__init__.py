"""roboVLA runner agent (design §4, §10, Appendix C).

Runs inside the per-node podman runner container: a small HTTP API on the
node's LAN port (default 8601, console-only traffic) that supervises job
processes (fixed argv, no shell, ever), probes VRAM, pulls dataset bundles
from the console at dispatch, and holds inference slot adapters.

Hardware lives in config (env), not code: every endpoint/IP/token/port is an
environment variable set by the systemd unit / quadlet (§9.8, §10). The
console talks to this agent via ``robovla.runner.HttpRunner``; the protocol
is ``robovla.runner.interface.Runner`` (the FakeRunner is the CPU oracle).
"""

__version__ = "0.1.0"
