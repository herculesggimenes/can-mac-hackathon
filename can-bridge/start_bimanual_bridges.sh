#!/usr/bin/env bash
# Start two Unix-socket CAN bridges for bimanual YAM on macOS.
#
# Arm 1 -> /tmp/can0.sock  : Rust gs_usb (CANable 1d50:606f), device index 0
# Arm 2 -> /tmp/can1.sock : second gs_usb if present; otherwise SLCAN via USB-serial
#   (many CANable2 builds show up as /dev/cu.usbmodem* with non-gs_usb firmware).
#
# Environment:
#   YAM_SLCAN_SERIAL=/dev/cu.usbmodem…   Force SLCAN device for arm 2 (when not second gs_usb).
#   YAM_ONLY_CAN1=1                     Skip starting can0; only create can1 (can0 must already exist).
#                                         Use after can-bridge 0 is running, when adding the second dongle.
#
# Examples:
#   ./start_bimanual_bridges.sh
#   YAM_SLCAN_SERIAL=/dev/cu.usbmodem206D338A594E1 ./start_bimanual_bridges.sh
#   YAM_ONLY_CAN1=1 ./start_bimanual_bridges.sh    # can0 already up; only wire can1

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
export CARGO_TARGET_DIR="$ROOT/target"
BIN="$ROOT/target/release/can-bridge"

ONLY_CAN1="${YAM_ONLY_CAN1:-0}"

if [[ ! -x "$BIN" ]]; then
  echo "Building release binary -> $CARGO_TARGET_DIR ..."
  (cd "$ROOT" && cargo build --release)
fi
if [[ ! -x "$BIN" ]]; then
  echo "error: expected executable at $BIN after cargo build --release" >&2
  exit 1
fi

cleanup() {
  echo ""
  echo "Stopping bridges..."
  [[ -n "${PID0:-}" ]] && kill "$PID0" 2>/dev/null || true
  [[ -n "${PID1:-}" ]] && kill "$PID1" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PID0=""
PID1=""

# --- can1: second gs_usb, else SLCAN. Sets PID1 on success; returns 0 if /tmp/can1.sock exists.
start_can1_bridge() {
  rm -f /tmp/can1.sock

  echo "Starting can-bridge 1 -> /tmp/can1.sock (second gs_usb if available)"
  "$BIN" 1 &
  PID1=$!

  sleep 0.9
  if [[ -S /tmp/can1.sock ]]; then
    echo "can1: second gs_usb bridge up."
    return 0
  fi

  kill "$PID1" 2>/dev/null || true
  wait "$PID1" 2>/dev/null || true
  PID1=""

  local SLCAN_PORT="${YAM_SLCAN_SERIAL:-}"
  if [[ -z "$SLCAN_PORT" ]]; then
    SLCAN_PORT="$(ls /dev/cu.usbmodem* 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "$SLCAN_PORT" || ! -e "$SLCAN_PORT" ]]; then
    echo "ERROR: No second gs_usb device, and no USB modem for SLCAN fallback." >&2
    echo "  Plug the second dongle or set YAM_SLCAN_SERIAL=/dev/cu.usbmodem..." >&2
    return 1
  fi

  echo "Second adapter via SLCAN: $SLCAN_PORT -> /tmp/can1.sock"
  (
    cd "$REPO_ROOT" && uv run python "$ROOT/slcan_bridge.py" --serial "$SLCAN_PORT" --socket /tmp/can1.sock --bitrate 1000000
  ) &
  PID1=$!
  return 0
}

if [[ "$ONLY_CAN1" == "1" ]]; then
  if [[ ! -S /tmp/can0.sock ]]; then
    echo "ERROR: YAM_ONLY_CAN1=1 requires an existing /tmp/can0.sock (start can-bridge 0 first)." >&2
    exit 1
  fi
  echo "YAM_ONLY_CAN1: leaving can0 untouched; starting can1 only."
  trap - EXIT
  trap cleanup INT TERM
  if ! start_can1_bridge; then
    echo "  can1 could not be started." >&2
    exit 1
  fi
else
  for i in 0 1; do
    rm -f "/tmp/can${i}.sock"
  done

  echo "Starting can-bridge 0 -> /tmp/can0.sock (gs_usb device #0)"
  "$BIN" 0 &
  PID0=$!

  for _ in $(seq 1 80); do
    [[ -S /tmp/can0.sock ]] && break
    sleep 0.1
  done
  if [[ ! -S /tmp/can0.sock ]]; then
    echo "ERROR: /tmp/can0.sock never appeared. Is the gs_usb CANable (1d50:606f) plugged in?" >&2
    kill "$PID0" 2>/dev/null || true
    exit 1
  fi

  if ! start_can1_bridge; then
    echo "  Leaving can0 (PID ${PID0}) running — only can1 is missing." >&2
    trap - EXIT
    exit 1
  fi
fi

ok=1
for _ in $(seq 1 50); do
  [[ -S /tmp/can0.sock ]] && [[ -S /tmp/can1.sock ]] && ok=0 && break
  sleep 0.15
done

echo ""
if [[ "$ok" -ne 0 ]]; then
  echo "ERROR: Expected both sockets within ~8s." >&2
  [[ -S /tmp/can0.sock ]] || echo "  Missing /tmp/can0.sock" >&2
  [[ -S /tmp/can1.sock ]] || echo "  Missing /tmp/can1.sock (check YAM_SLCAN_SERIAL / firmware)" >&2
  if [[ -S /tmp/can0.sock ]] && [[ ! -S /tmp/can1.sock ]]; then
    echo "  Leaving can0 running; fix can1 then: YAM_ONLY_CAN1=1 $0" >&2
    trap - EXIT
  fi
  exit 1
fi

echo "Both bridges ready (can0 + can1). PIDs: ${PID0:-none} ${PID1}. Leave this terminal open."
echo "Then: cd .. && uv run yamctl hybrid ... --arms bimanual --left-can can0 --right-can can1"
echo "Press Ctrl+C here to stop both."
wait
