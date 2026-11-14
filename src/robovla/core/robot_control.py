"""The robot command ladder and the per-robot config (§9.10 extension).

Pure logic, no storage and no I/O: the rules that decide what the robot is
allowed to do right now, so the safety order is testable on its own.

The ladder, highest first:

1. **stop** — latched, from any authorized human, wins over everything and
   stays until someone explicitly clears it.
2. **teleop** — a human command with a short expiry. Expired is the same as
   absent: no command means no motion, so a dead browser stops the robot.
3. **model** — the policy acts only when neither of the above applies.

Safety controls bypass the model and the prompt: nothing a prompt or a policy
can emit reaches ``cmd_vel`` while a stop is latched.

Expiry is compared against the console clock passed in by the caller. The
robot applies its OWN deadman too (the reply can be lost in Wi-Fi land); the
two watchdogs are redundant on purpose.
"""

from __future__ import annotations

from typing import Any

from .audit import parse_rfc3339
from .errors import RobovlaError

#: Classroom motion limits for a human-driven ROSBot. Anything above these is
#: clamped, never passed through: a stuck key must not become a full-speed run.
TELEOP_LIMITS = {"vx": (-0.3, 0.3), "vy": (-0.1, 0.1), "wz": (-0.9, 0.9)}

#: How long a teleop command stays alive if the browser stops refreshing it.
DEFAULT_DEADMAN_S = 0.5

#: What a robot asks for at bringup; every field is overridable per student.
DEFAULT_CONFIG: dict[str, Any] = {
    "ros_domain_id": 99,
    "rate_hz": 1.0,             # model loop
    "telemetry_rate_hz": 5.0,   # frame + command round trip
    "frame_size": 224,
    "prompt": "",
    "teleop": {"enabled": False, "deadman_s": DEFAULT_DEADMAN_S},
    "record": {"enabled": True},
}

# ranges the console will accept: a typo here should not brick a drive session
_RANGES = {"ros_domain_id": (0, 232), "rate_hz": (0.2, 10.0),
           "telemetry_rate_hz": (0.5, 10.0), "frame_size": (112, 640),
           "teleop.deadman_s": (0.2, 5.0)}


def _num(value: Any, lo: float, hi: float, fallback: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(hi, max(lo, v))


def merge_config(stored: dict[str, Any] | None) -> dict[str, Any]:
    """Stored JSON over defaults, deep for the two nested blocks, all clamped.

    Unknown keys are dropped, so a config written by an older console cannot
    smuggle a field the node does not understand.
    """
    out: dict[str, Any] = {
        "ros_domain_id": DEFAULT_CONFIG["ros_domain_id"],
        "rate_hz": DEFAULT_CONFIG["rate_hz"],
        "telemetry_rate_hz": DEFAULT_CONFIG["telemetry_rate_hz"],
        "frame_size": DEFAULT_CONFIG["frame_size"],
        "prompt": DEFAULT_CONFIG["prompt"],
        "teleop": dict(DEFAULT_CONFIG["teleop"]),
        "record": dict(DEFAULT_CONFIG["record"]),
    }
    src = stored or {}
    for key in ("ros_domain_id", "rate_hz", "telemetry_rate_hz", "frame_size"):
        if key in src:
            lo, hi = _RANGES[key]
            out[key] = _num(src[key], lo, hi, out[key])
    out["ros_domain_id"] = int(out["ros_domain_id"])
    out["frame_size"] = int(out["frame_size"])
    if isinstance(src.get("prompt"), str):
        out["prompt"] = src["prompt"][:200]
    tel = src.get("teleop") or {}
    if isinstance(tel, dict):
        out["teleop"]["enabled"] = bool(tel.get("enabled", out["teleop"]["enabled"]))
        out["teleop"]["deadman_s"] = _num(tel.get("deadman_s"),
                                          *_RANGES["teleop.deadman_s"],
                                          out["teleop"]["deadman_s"])
    rec = src.get("record") or {}
    if isinstance(rec, dict):
        out["record"]["enabled"] = bool(rec.get("enabled", out["record"]["enabled"]))
    return out


def clamp_cmd(vx: Any, vy: Any, wz: Any) -> list[float]:
    """One teleop command, clamped to the classroom limits. Junk becomes zero."""
    out = []
    for key, v in (("vx", vx), ("vy", vy), ("wz", wz)):
        lo, hi = TELEOP_LIMITS[key]
        try:
            out.append(min(hi, max(lo, float(v))))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _alive(cmd: dict[str, Any] | None, now: str) -> bool:
    """A teleop command is alive while its expiry is still in the future."""
    if not cmd:
        return False
    try:
        return parse_rfc3339(cmd["expires_at"]) > parse_rfc3339(now)
    except (KeyError, TypeError, ValueError, RobovlaError):
        return False


def control_for(ctl: dict[str, Any] | None, now: str) -> dict[str, Any]:
    """The control block for one telemetry reply.

    ``ctl`` is the stored per-robot control row (``{"stop":…, "teleop":…,
    "record":…}``). The reply always states the mode, so the robot never has to
    guess, and the mode never depends on the prompt.
    """
    ctl = ctl or {}
    stop = ctl.get("stop")
    if stop:
        return {"mode": "stop", "cmd_vel": [0.0, 0.0, 0.0],
                "stop": {"by": stop.get("by", "?"), "at": stop.get("at"),
                         "reason": stop.get("reason", "")},
                "record": ctl.get("record")}
    teleop = ctl.get("teleop")
    if _alive(teleop, now):
        return {"mode": "teleop",
                "cmd_vel": clamp_cmd(teleop.get("vx"), teleop.get("vy"),
                                     teleop.get("wz")),
                "expires_in_s": round(max(
                    0.0, (parse_rfc3339(teleop["expires_at"])
                          - parse_rfc3339(now)).total_seconds()), 3),
                "record": ctl.get("record")}
    return {"mode": "model", "cmd_vel": None, "record": ctl.get("record")}
