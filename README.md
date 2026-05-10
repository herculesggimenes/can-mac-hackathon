# YAM WebSocket Control

This repo exposes the YAM robot and camera feeds through simple WebSocket APIs
so external agents can control one or two arms without running the local viewer.

## Setup

```bash
git clone --recursive git@github.com:herculesggimenes/can-mac-hackathon.git
cd can-mac-hackathon
uv sync
```

## Robot Control WebSocket

Start the API server for one YAM arm:

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
  --params '{"joint_pos":[0,0,0,0,0,0,0.3]}'
```

The API uses JSON-RPC-style messages with method names matching the YAM robot
object:

```json
{"id":"1","method":"get_joint_pos","params":{}}
{"id":"2","method":"command_joint_pos","params":{"joint_pos":[0,0,0,0,0,0,0.3]}}
{"id":"3","method":"get_observations","params":{}}
{"id":"4","method":"get_robot_info","params":{}}
{"id":"5","method":"num_dofs","params":{}}
{"id":"6","method":"zero_torque_mode","params":{}}
{"id":"7","method":"get_status","params":{}}
{"id":"8","method":"reconnect","params":{"arm":"left"}}
```

Legacy command-style messages are also supported:

```json
{"type":"get_state"}
{"type":"get_status"}
{"type":"command_joints","q":[0,0,0,0,0,0,0.3]}
{"type":"command_delta","dq":[0,0,0,0,0,0,0.02]}
{"type":"subscribe_state","fps":10}
{"type":"hard_stop"}
```

Joint commands are seven numbers. Joint 7 is the normalized gripper command and
is clamped by default to `[0.01, 0.59]`.

## Reconnect Behavior

The server manages each arm independently:

- If a CAN socket or robot call fails, only that arm is marked disconnected.
- The WebSocket server keeps running.
- `get_status` and `get_state` report `connected`, `last_error`,
  `last_disconnected_at`, `next_reconnect_at`, and `reconnect_attempts`.
- The server retries disconnected arms with exponential backoff.
- Command calls fail fast while the target arm is disconnected.
- You can force reconnect with the `reconnect` method.

Tune reconnect timing:

```bash
uv run yamctl control-server \
  --reconnect-initial-delay 0.5 \
  --reconnect-max-delay 5.0
```

## Two YAM Arms

Two stock YAM arms need separate CAN buses unless the motor ids are remapped,
because each arm uses motor ids `1..7`.

Start one bridge per adapter/socket:

```bash
uv run python can-bridge/slcan_bridge.py \
  --serial /dev/cu.LEFT_CANABLE \
  --socket /tmp/can0.sock \
  --bitrate 1000000

uv run python can-bridge/slcan_bridge.py \
  --serial /dev/cu.RIGHT_CANABLE \
  --socket /tmp/can1.sock \
  --bitrate 1000000
```

Then start the API with explicit arm ids:

```bash
uv run yamctl control-server \
  --no-bridge \
  --arm-specs left:can0,right:can1 \
  --host 0.0.0.0
```

When multiple arms are connected, write commands must include the target arm:

```json
{"id":"left-open","method":"command_joint_pos","params":{"arm":"left","joint_pos":[0,0,0,0,0,0,0.59]}}
{"id":"right-state","method":"get_joint_pos","params":{"arm":"right"}}
```

Read-only calls without `arm` return a map for every connected arm.

## Camera WebSocket

Start the single-camera helper:

```bash
uv run yamctl camera \
  --camera-index auto \
  --host 0.0.0.0 \
  --port 8766
```

HTTP JPEG endpoint:

```text
http://<robot-mac-ip>:8766/frame.jpg
```

Start the multi-camera WebSocket endpoint:

```bash
uv run yamctl cameras \
  --camera-specs front:0,top:1,wrist:2 \
  --host 0.0.0.0 \
  --port 8770
```

Endpoint:

```text
ws://<robot-mac-ip>:8770/cameras
```

Protocol:

```json
{"type":"get_all_frames","bundle":true}
{"type":"get_frame","camera_id":"top"}
{"type":"subscribe","fps":5,"cameras":"all","bundle":true}
{"type":"stop"}
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
