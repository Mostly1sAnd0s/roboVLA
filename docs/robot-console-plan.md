# Robot operation from the browser — plan

Goal: a student runs the whole loop in the console — drive, record, upload,
train, serve, test — without a terminal. The only terminal step left is
`ssh pi3` then `rosbot` (the one bringup command), as the recipe states.

Status: console pages, endpoint board, dataset upload, training, serving and
the robot view work from the browser. Teleop, remote record start/stop and the
UI e-stop do not exist yet. This file is the build plan for those.

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

Robot-side items below are proposals against the rosBot tree as read on
2026-11-14 (`vla_drive_node` takes `console_url`, `token_file`, `prompt`,
`rate_hz`, `dry_run`, `gain`, `exposure` as ROS params, set by `robot_up.sh`).
Confirm each against the node before implementing.

## Robot-side work (rosBot package)

- `robot_up.sh` reads a one-line console URL and the Pi token from
  `~/rosbot_ws/config.env` (written once by the teacher). Everything else comes
  from the console: `GET /api/robot/config` → `{ros_domain_id, rate_hz, prompt,
  teleop: {enabled, deadman_s}, record: {...}}`. That is how the UI sets the
  endpoint and domain ID: it edits stored per-student config, the robot picks it
  up at next bringup.
- `vla_drive_node` gains three subscriptions it acts on locally:
  `cmd_vel_teleop` ( Twist from the console channel, watchdog `deadman_s`),
  `record/start`, `record/stop`, and `estop` (latched, zero velocity, wins over
  everything, cleared only by a second explicit clear).
- The teleop watchdog lives on the robot, not in the console: if no teleop
  command arrives within `deadman_s`, the node stops publishing and zeroes
  velocity. Same for a telemetry POST that fails.

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

## Open questions for the teacher

1. Teleop transport on the LAN: keep the console as the only path (one channel,
   logged, but 1–3 fps video and poll-interval latency), or let the browser talk
   to the Pi directly for the drive session (lower latency, breaks invariant 1,
   unlogged)?
2. Camera for driving: raise the telemetry frame rate during a drive session, or
   let the Pi stream MJPEG to the browser directly (LAN-only, again outside
   invariant 1)?
3. Who may e-stop: owning student only, any student, or does the teacher's
   broadcast stop win?
