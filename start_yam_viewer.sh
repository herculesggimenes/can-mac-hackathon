#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SERIAL_PORT="${SERIAL_PORT:-/dev/cu.usbmodem206D338A594E1}"
CAN_BITRATE="${CAN_BITRATE:-1000000}"

mkdir -p logs

if [ -S /tmp/can0.sock ]; then
  rm -f /tmp/can0.sock
fi

.venv/bin/python can-bridge/slcan_bridge.py \
  --serial "$SERIAL_PORT" \
  --bitrate "$CAN_BITRATE" \
  > logs/slcan_bridge.log 2>&1 &

BRIDGE_PID=$!
echo "$BRIDGE_PID" > logs/slcan_bridge.pid

for _ in $(seq 1 50); do
  if [ -S /tmp/can0.sock ]; then
    break
  fi
  sleep 0.1
done

CAN_MAC_PATCH=1 .venv/bin/python teleop_viewer.py
