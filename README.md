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
