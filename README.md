# YAM Teleop Viewer

![Viewer](https://raw.githubusercontent.com/quantbagel/can-mac/main/viewer.png)

Bidirectional teleop bridge for the [YAM](https://github.com/i2rt/i2rt) arm with a browser-based MuJoCo viewer (via [viser](https://github.com/nerfstudio-project/viser) + [mjviser](https://pypi.org/project/mjviser/)).

Mirror the real arm in sim, or control the real arm from sim sliders.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [Rust](https://rustup.rs/) (for the CAN bridge)
- CANable 2.0 USB adapter

## Setup

```bash
git clone --recursive https://github.com/quantbagel/can-mac.git
cd can-mac
uv sync
```

### CAN bridge (requires a CANable 2.0 adapter)

Single arm (gs_usb firmware, USB IDs **1d50:606f**):

```bash
cd can-bridge && cargo build --release
./target/release/can-bridge  # listens on /tmp/can0.sock
```

Two arms on macOS: start both Unix sockets with **`can-bridge/start_bimanual_bridges.sh`**. That script runs the Rust bridge on **`can0`** and either a second Rust **`can-bridge 1`** or, if the second dongle is CDC/SLCAN only, **`slcan_bridge.py`** on **`/tmp/can1.sock`**. Set **`YAM_SLCAN_SERIAL=/dev/cu.usbmodem…`** if the wrong serial device is chosen. **`uv run yamctl status`** reports **`can_socket`** and **`can1_socket`**.

## Run

```bash
uv run teleop_viewer.py
# Open http://localhost:8080
```

## Operator CLI

This checkout also provides `yamctl`, a thin wrapper around the local camera,
SLCAN bridge, viewer, one-arm LeRobot policy server, and legacy Modal bridge.

```bash
uv run yamctl status
uv run yamctl camera
uv run yamctl bridge
uv run yamctl viewer
```

Expose the same simple YAM joint API over WebSocket when another process needs
to drive the arm instead of the local viewer:

```bash
uv run yamctl stop viewer model --hard-stop
uv run yamctl control-server --background

uv run python scripts/yam_ws_client.py --method get_joint_pos
uv run python scripts/yam_ws_client.py \
  --method command_joint_pos \
  --params '{"joint_pos":[0,0,0,0,0,0,0.3]}'
```

The WebSocket endpoint is:

```text
ws://<robot-mac-ip>:8780/control
```

It accepts JSON-RPC-style calls with the YAM method names:

```json
{"id":"1","method":"get_joint_pos","params":{}}
{"id":"2","method":"command_joint_pos","params":{"joint_pos":[0,0,0,0,0,0,0.3]}}
{"id":"3","method":"get_observations","params":{}}
{"id":"4","method":"zero_torque_mode","params":{}}
{"id":"5","method":"get_status","params":{}}
{"id":"6","method":"reconnect","params":{"arm":"left"}}
```

If a CAN socket or arm drops, the server keeps running, marks that arm as
disconnected in `get_status` / `get_state`, and retries with exponential
backoff. Tune reconnect timing with `--reconnect-initial-delay` and
`--reconnect-max-delay`. Command calls fail fast while an arm is disconnected;
read calls still show which arms are healthy.

For two YAM arms, run one SLCAN bridge per CAN adapter/socket and start the
control server with explicit arm ids:

```bash
uv run python can-bridge/slcan_bridge.py \
  --serial /dev/cu.LEFT_CANABLE \
  --socket /tmp/can0.sock \
  --bitrate 1000000

uv run python can-bridge/slcan_bridge.py \
  --serial /dev/cu.RIGHT_CANABLE \
  --socket /tmp/can1.sock \
  --bitrate 1000000

uv run yamctl control-server \
  --no-bridge \
  --arm-specs left:can0,right:can1 \
  --host 0.0.0.0
```

Two arms need separate CAN buses unless the motor ids are remapped, because each
stock YAM uses ids `1..7`. With multiple arms connected, write commands must
include the target arm:

```json
{"id":"left-open","method":"command_joint_pos","params":{"arm":"left","joint_pos":[0,0,0,0,0,0,0.59]}}
{"id":"right-state","method":"get_joint_pos","params":{"arm":"right"}}
```

Read-only calls without `arm` return a map for every connected arm.

Expose the camera as HTTP JPEG plus a WebSocket stream:

```bash
uv run yamctl camera \
  --camera-index auto \
  --host 0.0.0.0 \
  --port 8766 \
  --ws-port 8767
```

The HTTP endpoint remains:

```text
http://<robot-mac-ip>:8766/frame.jpg
```

The WebSocket endpoint is:

```text
ws://<robot-mac-ip>:8767/camera
```

Send one of these JSON messages:

```json
{"type":"get_frame","encoding":"base64"}
{"type":"subscribe","fps":10,"encoding":"binary"}
{"type":"stop"}
```

For ngrok, expose the WebSocket port:

```bash
ngrok http 8767
```

Then connect with:

```text
wss://<ngrok-id>.ngrok-free.app/camera
```

Debug a WebSocket frame request:

```bash
uv run python scripts/ws_camera_client.py \
  ws://127.0.0.1:8767/camera \
  --output logs/ws-camera-frame.jpg
```

For three cameras, prefer the centralized multiplexed endpoint:

```bash
uv run yamctl cameras \
  --camera-specs front:0,top:1,wrist:2 \
  --host 0.0.0.0 \
  --port 8770
```

Specs can mix normal OpenCV cameras with Orbbec SDK feeds:

```bash
uv run yamctl cameras \
  --camera-specs top:opencv:2,left:opencv:3,right:orbbec:all \
  --host 0.0.0.0 \
  --port 8770
```

`right:orbbec:all` exports separate logical feeds for `right_color`,
`right_depth`, `right_ir`, `right_left_ir`, `right_right_ir`, and
`right_dual_ir`.

Or auto-pick the first three camera indexes that OpenCV can read:

```bash
uv run yamctl cameras --auto-count 3 --host 0.0.0.0 --port 8770
```

Expose that single socket through ngrok:

```bash
ngrok http 8770
```

Connect to:

```text
wss://<ngrok-id>.ngrok-free.app/cameras
```

Protocol:

```json
{"type":"get_all_frames","bundle":true}
{"type":"get_frame","camera_id":"top"}
{"type":"subscribe","fps":5,"cameras":"all","bundle":true}
{"type":"stop"}
```

Debug all three frames:

```bash
uv run python scripts/multi_camera_ws_client.py \
  ws://127.0.0.1:8770/cameras \
  --output-dir logs/multi-camera-check
```

Open a real-time browser viewer:

```bash
open web/multi_camera_viewer.html
```

For ngrok, paste the `wss://.../cameras` URL into the viewer, or open it with a
query string:

```text
web/multi_camera_viewer.html?ws=wss://<ngrok-id>.ngrok-free.app/cameras&autoconnect=1
```

The default hybrid path is now the local one-arm LeRobot ACT endpoint. Start it
with a trained policy checkpoint:

```bash
uv run yamctl policy-server \
  --policy-path outputs/train/yam-bread-toaster-act/checkpoints/last/pretrained_model \
  --background

uv run yamctl hybrid "put bread in toaster" \
  --ensure-camera \
  --observe-only \
  --max-iterations 1
```

By default, model runs stop if any motor MOS temperature exceeds `55 C` or
rotor temperature exceeds `100 C`.
Gripper commands are clamped to normalized joint 7 range `[0.01, 0.59]`.

For longer rollouts, tune iteration count and how many trajectory rows you execute per inference:

```bash
uv run yamctl hybrid "put bread in toaster" \
  --max-iterations 30 \
  --execute-action-steps 3 \
  --hz 0.5
```

The old Modal MolmoAct2 bimanual path is still available only when explicitly
requested:

```bash
uv run yamctl run "pick up the hat on the table" \
  --allow-modal-bimanual \
  --http-url "$YAM_MODAL_POLICY_HTTP_URL" \
  --profile fast
```

Run the legacy hybrid Modal + local verification loop by passing
`--policy-kind modal`. Start with observe-only to verify the live camera/state
payload and Modal action metadata without moving the robot:

```bash
uv run yamctl hybrid "grab a chip inside the box" \
  --policy-kind modal \
  --http-url "$YAM_MODAL_POLICY_HTTP_URL" \
  --ensure-camera \
  --observe-only \
  --max-iterations 1 \
  --trace-dir logs/hybrid-observe-test
```

When the trace shows `input_source: payload`, finite actions, and the expected
`[1, 30, 14]` action shape, enable execution. `--codex-corrections` adds local
camera-based alignment corrections, and `--auto-grasp` allows local close/lift
when the jaws appear aligned with the chip-box opening:

```bash
uv run yamctl hybrid "grab a chip inside the box" \
  --policy-kind modal \
  --http-url "$YAM_MODAL_POLICY_HTTP_URL" \
  --ensure-camera \
  --codex-corrections \
  --auto-grasp \
  --max-iterations 8 \
  --trace-dir logs/hybrid-chip-box
```

Stop a model rollout:

```bash
uv run yamctl hard-stop
uv run yamctl stop --hard-stop
uv run yamctl clear-stop
```

## Bread-to-toaster ACT policy

The recommended path for `put bread in toaster` is a task-specific LeRobot ACT
policy trained on this exact camera, arm, gripper, toaster, and bread setup.
The teleop viewer records raw YAM episodes with synchronized camera frames:

```bash
uv run yamctl camera
uv run yamctl viewer
```

Open <http://localhost:8080>, switch to `Control from Sim`, set the recording
task to `put bread in toaster`, then record successful demonstrations. Keep the
camera fixed and make sure the bread, gripper, and toaster slot are visible from
the single camera for the whole episode.

Validate recorded episodes:

```bash
uv run yamctl dataset summary
```

After installing LeRobot in this environment, export the raw episodes:

```bash
uv pip install 'lerobot[training]'

uv run yamctl dataset export \
  --repo-id local/yam-bread-toaster \
  --output-root lerobot-data \
  --fps 10
```

Train ACT locally:

```bash
uv run yamctl dataset train-act \
  --repo-id local/yam-bread-toaster \
  --dataset-root lerobot-data \
  --device mps
```

Start with 50-100 clean successful demonstrations before tuning model size or
trying SmolVLA. ACT is the first baseline because the task is narrow and the
dataset has one camera plus 7D state/action.
