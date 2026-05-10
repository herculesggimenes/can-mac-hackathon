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
