# YAM WebSocket Control

This repo exposes a bimanual YAM robot and camera feeds through simple
WebSocket APIs. The robot control server owns the bimanual CAN bridge process,
so there is one process responsible for bridge startup, socket health, robot
connection, and reconnect.

## Setup

```bash
git clone --recursive git@github.com:herculesggimenes/can-mac-hackathon.git
cd can-mac-hackathon
uv sync
```

## Robot Control WebSocket

Start the bimanual API server:

```bash
uv run yamctl stop viewer model --hard-stop
uv run yamctl control-server --background
```

Endpoint:

```text
ws://<robot-mac-ip>:8780/control
```

Quick smoke test:

```bash
uv run python scripts/yam_ws_client.py --method get_joint_pos
uv run python scripts/yam_ws_client.py \
  --method command_joint_pos \
  --params '{"joint_pos":[0,0,0,0,0,0,0.3,0,0,0,0,0,0,0.3]}'
```

The API uses JSON-RPC-style messages with method names matching the YAM robot
object. It is bimanual-only: joint position reads and writes are 14D vectors in
`left[0:7] + right[0:7]` order.

```json
{"id":"1","method":"get_joint_pos","params":{}}
{"id":"2","method":"command_joint_pos","params":{"joint_pos":[0,0,0,0,0,0,0.3,0,0,0,0,0,0,0.3]}}
{"id":"3","method":"get_observations","params":{}}
{"id":"4","method":"get_robot_info","params":{}}
{"id":"5","method":"num_dofs","params":{}}
{"id":"6","method":"zero_torque_mode","params":{}}
{"id":"7","method":"get_status","params":{}}
{"id":"8","method":"reconnect","params":{}}
```

Joint commands are fourteen numbers. Each arm has seven joints. Joint 7 of each
arm is the normalized gripper command and is clamped by default to `[0.01,
0.59]`.

Run a bounded WebSocket motion demo:

```bash
uv run python scripts/yam_samba_dance_ws.py --dry-run
uv run python scripts/yam_samba_dance_ws.py --duration 8 --loops 1
```

The demo checks that both arms are connected, ramps motion in/out, caps per-step
joint deltas, leaves both grippers unchanged, and restores the observed baseline
pose by default.

## Reconnect Behavior

The server owns and supervises the bimanual bridge plus both arms:

- If the bridge exits or sockets disappear, the server restarts the bridge.
- If a robot call fails, that arm is marked disconnected.
- The WebSocket server keeps running.
- `get_status` reports bridge state plus per-arm `connected`, `last_error`,
  `last_disconnected_at`, `next_reconnect_at`, and `reconnect_attempts`.
- The server retries disconnected arms with exponential backoff.
- Command calls fail fast if either arm is disconnected.
- You can force reconnect with the `reconnect` method.

Tune reconnect timing:

```bash
uv run yamctl control-server \
  --reconnect-initial-delay 0.5 \
  --reconnect-max-delay 5.0
```

## Bridge And Arm Mapping

Two stock YAM arms need separate CAN buses unless the motor ids are remapped,
because each arm uses motor ids `1..7`.

By default, `yamctl control-server` starts `can-bridge/start_bimanual_bridges.sh`
inside the API server process and uses:

```text
left:can0,right:can1
```

Override only if the physical mapping is different:

```bash
uv run yamctl control-server \
  --arm-specs left:can1,right:can0 \
  --host 0.0.0.0
```

The API still expects exactly two arms and 14D commands.

## Camera WebSocket

Start the multi-camera WebSocket endpoint:

```bash
sudo /opt/homebrew/bin/uv run python scripts/multi_camera_ws_server.py \
  --host 0.0.0.0 \
  --port 8770 \
  --camera-specs 'right:1,left:2,top:orbbec:color,depth:orbbec:depth' \
  --quality 80
```

Endpoint:

```text
ws://<robot-mac-ip>:8770/cameras
```

Protocol:

```json
{"type":"get_all_frames","bundle":true}
{"type":"get_frame","camera_id":"top"}
{"type":"reset_camera","camera_id":"depth"}
{"type":"subscribe","fps":5,"cameras":"all","bundle":true}
{"type":"stop"}
```

Use `reset_camera` for a stuck feed. OpenCV cameras reset independently; Orbbec
logical feeds restart the shared Orbbec SDK worker.

Orbbec depth frames include the normal JPEG preview plus raw metric depth when
available:

```json
{
  "camera_id": "right_depth",
  "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 200.0},
  "depth": {
    "format": "uint16",
    "width": 640,
    "height": 400,
    "scale_to_meters": 0.001,
    "data": "<base64 little-endian uint16>"
  }
}
```

Convert a depth pixel to a 3D camera-frame point:

```bash
uv run python scripts/yam_depth_geometry.py logs/yam-snapshots/latest/camera_payload.json \
  --camera-id right_depth \
  --u 320 \
  --v 200
```

Collect one Codex-optimized snapshot before planning robot actions. This writes
`codex_snapshot.json`, saves all camera images, summarizes robot state, and
samples depth into compact camera-frame 3D points:

```bash
uv run python scripts/yam_codex_snapshot.py \
  --robot-url wss://2d37-12-125-194-54.ngrok-free.app/control \
  --camera-url ws://127.0.0.1:8770/cameras \
  --depth-grid 3
```

Use `codex_snapshot.json` as the primary planning input. It includes the 14D
joint order, both-arm gripper values, wrist camera joints, saved image paths,
depth quality, center depth, sampled depth grid points, and recommended command
primitives. It also writes `codex_contact_sheet.jpg`, a single large image with
all cameras tiled together for faster visual inspection.

Plan a bounded IK move from a depth pixel. This is dry-run unless `--execute`
is passed:

```bash
uv run python scripts/yam_move_to_depth_pixel.py logs/yam-snapshots/latest/camera_payload.json \
  --camera-id depth \
  --u 320 \
  --v 200 \
  --calibration config/yam_camera_calibration.json \
  --control-url wss://2d37-12-125-194-54.ngrok-free.app/control \
  --arm right \
  --target-offset 0,0,0.06
```

The calibration file must contain a 4x4 camera-to-robot transform:

```json
{
  "T_robot_camera": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

Run a calibration-free local visual-servo touch test. This tracks the selected
depth-preview pixel, probes tiny joint movements to estimate a local image
Jacobian, then takes bounded steps toward the image center and desired depth.
It does not need `T_robot_camera`, but it must be run with `--execute` to learn
the local Jacobian from real motion:

```bash
uv run python scripts/yam_visual_servo_touch.py \
  --control-url wss://2d37-12-125-194-54.ngrok-free.app/control \
  --camera-url ws://127.0.0.1:8770/cameras \
  --camera-id depth \
  --arm right \
  --target-u 297 \
  --target-v 88 \
  --desired-depth-m 0.035 \
  --max-joint-step 0.02 \
  --iterations 1 \
  --execute
```

Use Cartesian IK for smoother manual control instead of per-joint deltas. This
reads the current 14D state, applies an end-effector delta, solves IK for one
arm, and sends an interpolated full-14D trajectory:

```bash
uv run python scripts/yam_cartesian_control.py \
  --control-url wss://2d37-12-125-194-54.ngrok-free.app/control \
  --arm right \
  --frame local \
  --delta 0,0,-0.03 \
  --max-joint-delta 0.10 \
  --steps 20 \
  --hz 60 \
  --execute
```

Use `--frame local` for gripper-relative moves and `--frame world` for robot
world-axis moves. Omit `--execute` to inspect the planned 14D command first.
The execution path keeps one WebSocket open for the full trajectory; use
`--hz` plus `--steps` to tune smoothness and speed.
For wrist-camera aiming, use the camera rotation convenience flags. `--wrist-only`
keeps joints 1-3 fixed after IK so joints 4-6 do the pitch/yaw/roll work:

```bash
uv run python scripts/yam_cartesian_control.py \
  --control-url wss://2d37-12-125-194-54.ngrok-free.app/control \
  --arm right \
  --frame local \
  --delta 0,0,0 \
  --camera-pitch-deg -12 \
  --wrist-only \
  --max-segment-rad 0.08 \
  --max-joint-delta 0.10 \
  --steps 20 \
  --hz 80 \
  --execute
```

Use `--camera-yaw-deg` and `--camera-roll-deg` the same way. If the wrist camera
mount is inverted, flip the sign of the pitch or yaw command.
For larger task-level moves, pass the full desired delta and let the script
split it into Cartesian IK segments:

```bash
uv run python scripts/yam_cartesian_control.py \
  --control-url wss://2d37-12-125-194-54.ngrok-free.app/control \
  --arm right \
  --frame local \
  --delta 0,0,-0.12 \
  --max-segment-m 0.03 \
  --max-joint-delta 0.20 \
  --steps 20 \
  --hz 80 \
  --no-stop-on-ik-failure \
  --allow-nonconverged-segments \
  --execute
```

Debug all frames:

```bash
uv run python scripts/multi_camera_ws_client.py \
  ws://127.0.0.1:8770/cameras \
  --output-dir logs/multi-camera-check
```

Open the browser viewer:

```bash
open web/multi_camera_viewer.html
```

For ngrok, expose port `8770` and connect the viewer to:

```text
wss://<ngrok-id>.ngrok-free.app/cameras
```
