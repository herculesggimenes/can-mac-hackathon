#!/usr/bin/env python3
"""Drive a YAM WebSocket arm from an SO-100/SO-101 serial leader.

This is a relative bridge for mismatched hardware. On startup it captures the
current SO leader pose and the selected YAM WebSocket arm pose, then applies
bounded raw-servo deltas to the YAM arm slice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]
HACKATHON = ROOT.parent
LEROBOT = HACKATHON / "lerobot-MakerMods"
sys.path.insert(0, str(LEROBOT / "src"))

ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
SO_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return

    class Device:
        def __init__(self, device_type: str = "cpu"):
            self.type = device_type

        def __str__(self) -> str:
            return self.type

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class Backends:
        class mps:
            @staticmethod
            def is_available() -> bool:
                return False

    sys.modules["torch"] = types.SimpleNamespace(
        Tensor=object,
        device=Device,
        dtype=object,
        float64="float64",
        float32="float32",
        cuda=Cuda,
        backends=Backends,
    )


def _rpc(ws: Any, method: str, params: dict[str, Any] | None = None, *, wait: bool = True) -> Any:
    ws.send(json.dumps({"id": f"{method}-{time.time_ns()}", "method": method, "params": params or {}}))
    if not wait:
        return None
    while True:
        response = json.loads(ws.recv())
        if response.get("type") == "hello":
            continue
        if not response.get("ok"):
            raise RuntimeError(response.get("error", response))
        return response.get("result")


def _drain_rpc_responses(ws: Any) -> int:
    drained = 0
    while True:
        try:
            response = json.loads(ws.recv(timeout=0))
        except TimeoutError:
            return drained
        if response.get("type") == "hello":
            continue
        drained += 1
        if not response.get("ok", True):
            print(f"websocket command error: {response.get('error', response)}", file=sys.stderr, flush=True)


def _write_status(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    tmp_path = f"{path}.tmp"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


class SOLoader:
    def __init__(self, port: str, kind: str):
        _install_torch_stub()
        if kind == "so100":
            from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig

            self.teleop = SO100Leader(SO100LeaderConfig(port=port, id="ws_bridge_so100_leader"))
        elif kind == "so101":
            from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

            self.teleop = SO101Leader(SO101LeaderConfig(port=port, id="ws_bridge_so101_leader"))
        else:
            raise ValueError(f"unsupported leader kind: {kind}")
        # Avoid SO leader calibration/configuration reads on every teleop start.
        # They are slow and this USB serial bus sometimes returns a malformed
        # packet under rapid restarts. For this bridge we only need raw present
        # positions, so opening the bus is enough.
        self.teleop.bus.connect(handshake=False)

    def close(self) -> None:
        self.teleop.bus.disconnect(disable_torque=False)

    def read_raw(self) -> np.ndarray:
        last_exc: Exception | None = None
        for _ in range(5):
            try:
                values = self.teleop.bus.sync_read("Present_Position", normalize=False, num_retry=2)
                return np.asarray([float(values[motor]) for motor in SO_MOTORS], dtype=float)
            except Exception as exc:
                last_exc = exc
                time.sleep(0.02)
        assert last_exc is not None
        raise last_exc


def _parse_ints(raw: str, expected: int, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} expects {expected} comma-separated integers")
    return values


def _parse_floats(raw: str, expected: int, name: str) -> np.ndarray:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} expects {expected} comma-separated numbers")
    return np.asarray(values, dtype=float)


def _parse_lock_joints(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    invalid = sorted(value for value in values if value < 0 or value > 6)
    if invalid:
        raise ValueError(f"--lock-joints entries must be YAM arm indexes in [0, 6], got {invalid}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge an SO serial leader to a YAM WebSocket arm.")
    parser.add_argument("--control-url", default="ws://127.0.0.1:8780/control")
    parser.add_argument("--port", default="/dev/tty.usbmodem5B140318401")
    parser.add_argument("--kind", choices=["so100", "so101"], default="so100")
    parser.add_argument("--arm", choices=["left", "right", "both"], default="right")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--raw-to-rad", type=float, default=0.0012)
    parser.add_argument("--joint-map", default="0,1,2,3,4", help="YAM arm joint indexes driven by the 5 SO arm joints.")
    parser.add_argument("--joint-signs", default="1,1,1,1,1")
    parser.add_argument(
        "--sixth-joint-source",
        choices=["none", "gripper", "wrist_roll"],
        default="none",
        help="Optional source for YAM arm joint 5. SO leaders only have 5 arm axes.",
    )
    parser.add_argument("--sixth-joint-sign", type=float, default=1.0)
    parser.add_argument("--sixth-joint-scale", type=float, default=None)
    parser.add_argument("--max-joint-delta", type=float, default=0.45)
    parser.add_argument("--max-step", type=float, default=0.025)
    parser.add_argument("--max-gripper-step", type=float, default=0.015)
    parser.add_argument("--min-gripper", type=float, default=0.01)
    parser.add_argument("--max-gripper", type=float, default=0.59)
    parser.add_argument("--lock-joints", default="", help="Comma-separated YAM arm indexes [0..6] to hold at startup pose.")
    parser.add_argument(
        "--sync-samples",
        type=int,
        default=5,
        help="Number of leader samples to average at startup when synchronizing the leader baseline.",
    )
    parser.add_argument(
        "--fire-and-forget",
        action="store_true",
        help="Do not block on every command_joint_pos response. Responses are still drained to keep the socket healthy.",
    )
    parser.add_argument("--max-in-flight", type=int, default=2, help="Maximum unacknowledged commands in no-wait mode.")
    parser.add_argument("--status-path", default="", help="Optional JSON file for live leader axis diagnostics.")
    parser.add_argument("--execute", action="store_true", help="Actually send WebSocket commands. Default is dry-run.")
    args = parser.parse_args()

    joint_map = _parse_ints(args.joint_map, 5, "--joint-map")
    if any(index < 0 or index > 5 for index in joint_map):
        raise ValueError("--joint-map entries must be YAM arm joint indexes in [0, 5]")
    signs = _parse_floats(args.joint_signs, 5, "--joint-signs")
    locked_joints = _parse_lock_joints(args.lock_joints)

    leader = SOLoader(args.port, args.kind)
    try:
        with connect(
            args.control_url,
            open_timeout=15,
            max_size=16 * 1024 * 1024,
            ping_interval=None,
        ) as ws:
            # Synchronization point: sample both live robot and leader after the
            # previous bridge has fully stopped, then treat that pair as zero
            # relative motion. This prevents a startup jump from stale offsets.
            q14_base = np.asarray(_rpc(ws, "get_joint_pos"), dtype=float)
            if q14_base.shape != (14,):
                raise RuntimeError(f"expected 14D websocket state, got {q14_base.shape}")
            arm_names = ["left", "right"] if args.arm == "both" else [args.arm]
            arm_bases = {name: q14_base[ARM_SLICES[name]].copy() for name in arm_names}
            samples = [leader.read_raw()]
            for _ in range(max(args.sync_samples, 1) - 1):
                time.sleep(0.02)
                samples.append(leader.read_raw())
            leader_base = np.mean(np.stack(samples, axis=0), axis=0)
            command_arms = {name: arm_bases[name].copy() for name in arm_names}

            mode = "EXECUTE" if args.execute else "dry-run"
            print(
                f"so_leader_ws_bridge ready ({mode}): {args.port} -> websocket {args.arm}. "
                f"Synchronized leader baseline from {len(samples)} samples; Ctrl+C to stop.",
                flush=True,
            )

            dt = 1.0 / max(args.hz, 1.0)
            in_flight = 0
            while True:
                start = time.monotonic()
                if args.fire_and_forget:
                    in_flight = max(0, in_flight - _drain_rpc_responses(ws))
                raw = leader.read_raw()
                delta_raw = raw[:5] - leader_base[:5]
                q14 = q14_base.copy()
                for arm_name in arm_names:
                    arm_base = arm_bases[arm_name]
                    target_arm = arm_base.copy()
                    for src_index, yam_index in enumerate(joint_map):
                        if yam_index in locked_joints:
                            continue
                        delta = float(
                            np.clip(
                                delta_raw[src_index] * args.raw_to_rad * signs[src_index],
                                -args.max_joint_delta,
                                args.max_joint_delta,
                            )
                        )
                        target_arm[yam_index] = arm_base[yam_index] + delta

                    if args.sixth_joint_source != "none" and 5 not in locked_joints:
                        source_index = 5 if args.sixth_joint_source == "gripper" else 4
                        scale = args.sixth_joint_scale if args.sixth_joint_scale is not None else args.raw_to_rad
                        delta = float(
                            np.clip(
                                (raw[source_index] - leader_base[source_index]) * scale * args.sixth_joint_sign,
                                -args.max_joint_delta,
                                args.max_joint_delta,
                            )
                        )
                        target_arm[5] = arm_base[5] + delta

                    if 6 not in locked_joints:
                        gripper_norm = float(np.clip((raw[5] - 400.0) / 3200.0, 0.0, 1.0))
                        target_arm[6] = args.min_gripper + gripper_norm * (args.max_gripper - args.min_gripper)

                    step_limits = np.full(7, args.max_step, dtype=float)
                    step_limits[6] = args.max_gripper_step
                    command_arms[arm_name] = command_arms[arm_name] + np.clip(
                        target_arm - command_arms[arm_name],
                        -step_limits,
                        step_limits,
                    )
                    q14[ARM_SLICES[arm_name]] = command_arms[arm_name]
                if args.execute:
                    if args.fire_and_forget:
                        if in_flight < max(args.max_in_flight, 1):
                            _rpc(ws, "command_joint_pos", {"joint_pos": q14.tolist()}, wait=False)
                            in_flight += 1
                    else:
                        _rpc(ws, "command_joint_pos", {"joint_pos": q14.tolist()})
                else:
                    print(json.dumps({"raw": raw.tolist(), "target": q14.tolist()}), flush=True)
                _write_status(
                    args.status_path,
                    {
                        "updated_at": time.time(),
                        "motors": SO_MOTORS,
                        "raw": raw.tolist(),
                        "leader_base": leader_base.tolist(),
                        "delta_raw": (raw - leader_base).tolist(),
                        "joint_map": joint_map,
                        "joint_signs": signs.tolist(),
                        "locked_joints": sorted(locked_joints),
                        "target": q14.tolist(),
                        "in_flight": in_flight,
                    },
                )

                elapsed = time.monotonic() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    finally:
        leader.close()


if __name__ == "__main__":
    raise SystemExit(main())
