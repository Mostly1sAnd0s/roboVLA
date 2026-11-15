# Robot operation from the browser — plan

Goal: a student runs the whole loop in the console — drive, record, upload,
train, serve, test — without a terminal. The only terminal step left is
`ssh pi3` then `rosbot` (the one bringup command), as the recipe states.

## Status

Built (2026-11-14) and running: `core/robot_control.py` (the ladder and the
config, pure and unit-tested), `http/robot.py` (`/api/robot/config`, `/status`,
`/teleop`, `/stop`, `/record`), the control block on the telemetry reply,
`/ui/drive.html` (keyboard + gamepad, deadman bar, latched e-stop, teacher
broadcast, record start/stop, config editor). Robot side:
`rosbot_vla/control.py`, the teleop/stop/record handling inside
`vla_drive_node`, `record_episode_node session:=true`, and
`rosbot_vla/console_config.py` for the bringup domain read.

Verified on hardware the same day: browser teleop drove the robot, the camera
stream reached the Drive page, and the domain read returned the seat's value
(99) with a live token — plus the fail-safe cases on the Pi (no token file, and
an unreachable console both fall back to 99, exit 0, no stall).

NOT yet verified on hardware — do not treat as done:

* the **e-stop drill**: spacebar latch, teacher broadcast, clear, and the
  robot-side deadman with the console killed mid-drive;
* `record_episode_node session:=true` driven from the browser (the recorder
  itself has recorded real episodes; the console-driven start/stop has not);
* a second robot on a second domain, which is what makes per-seat assignment
  more than a rule on paper.

Tests: roboVLA `tests/unit/test_robot_control.py`,
`tests/integration/test_robot_ops.py`; rosBot `test/test_control.py`,
`test/test_modules.py`.

## Invariants that constrain every design choice

1. **The robot initiates all HTTP.** It polls `/api/robot/telemetry`; the
   console never opens a socket to the Pi. No WebSocket to the robot.
2. **Console → robot commands ride the telemetry reply.** Anything the browser
   wants the robot to do is written into a per-student pending-action row and
   returned on the robot's next poll.
3. **Safety bypasses model and prompt.** Nothing a prompt or a policy can emit
   may outrank a stop. A lost heartbeat must stop the robot by itself.

Consequence: browser control latency equals the telemetry poll interval, and
every command needs an expiry so a dead browser cannot leave the robot moving.

## Robot-side work (rosBot package) — as built

### Domain per seat (built)

Twelve robots on one domain are twelve robots that can publish to each other's
`/cmd_vel`, so the console ASSIGNS the domain: seat *n* owns `99 + (n - 1)`
(`s01` = 99 … `s12` = 110, inside the legal 0-232). Seat 01 keeps 99 on purpose,
so the pilot rig and every existing operator shell are unaffected — the rule is
adopted without a migration. A stored value in the robot config always wins,
which is how a multi-robot lesson or a lab with its own AP pins a rig.

`robot_up.sh` asks the console and exports the answer before ROS starts
(`python3 -m rosbot_vla.console_config --shell`); the read is best-effort, 3 s,
exits 0 with the fallback, and can only ever emit digits. A human shell needs the
same number, so derive it the same way instead of typing 99 — a mismatch is the
documented silent-empty-room failure.

## Console-side work

- `POST /api/robot/teleop {vx, vy, wz, hold_s}` → pending action, expires
  `hold_s` later (default 0.4 s). Keyboard repeat or a gamepad poll refreshes
  it. Return the applied value and the queue latency so the UI can show it.
- `POST /api/robot/estop {clear: false}` and `{clear: true}` — teacher or the
  owning student only. Logged to the audit trail like a kill.
- `POST /api/robot/record {action: "start"|"stop", task, note}` → the node
  starts or closes an episode session; the console learns the episode count
  through the telemetry reply.
- `GET /api/robot/config` and the teacher-side editor for it.
- Teleop is gated: an active infer slot or an explicit teleop-enabled flag, a
  student who owns the robot, and `dry_run` honoured on the node side.

## UI work

- `/ui/drive.html` for the owning student (teacher sees any robot): camera
  frame, speed/heading readout, keyboard WASD + arrow keys with a visible
  deadman, optional gamepad through the Gamepad API, big e-stop, record
  start/stop with the session episode counter, prompt box, dry-run toggle, and
  a strip of the last N episodes with a link to datasets.
- The keyboard handler must stop on window blur and on tab hide, and it must
  send a zero command when released, not merely stop sending.
- `/ui/robot.html` keeps the passive view; driving moves to its own page so a
  student cannot lose the controls by scrolling.

## Verification

- Unit: pending-action expiry, RBAC on teleop/estop/record, config round trip.
- Integration: telemetry reply carries the pending command exactly once and
  drops it after expiry; e-stop overrides a live teleop stream.
- Hardware (`tests/hardware`): robot stops within `deadman_s` when the console
  is killed mid-drive; a latched e-stop survives a policy that keeps asking for
  forward motion; recorded episodes reach the console through the UI upload.

## Decisions taken (2026-11-14)

1. **Drive path: through the console.** The invariant holds — the Pi runs no
   server and listens on no port. Cost accepted: command latency equals the
   telemetry poll, and the browser's picture arrives at that same cadence.
2. **Frame rate over MJPEG.** Raising the telemetry rate (default 5 Hz, cap 12)
   gives a driveable view without a new channel. Streaming from the Pi would
   have meant a listening port and no logging; a console-side relay would add no
   frames per second, since the rate is set by how often the robot posts.
   Revisit only with measurements, not impressions.
3. **Stop authority: teacher broadcast always wins**, students get spacebar on
   their own robot. The node ladder is stop > teleop > model, the stop is
   repeated on every reply until explicitly cleared, and the robot's own deadman
   zeroes velocity if commands stop arriving. Hardware e-stop remains final.
