#!/usr/bin/env python3
"""Drive one YAM WebSocket arm from a local YAM leader arm.

The robot WebSocket is bimanual and always accepts 14D commands in
left[0:7] + right[0:7] order. This bridge reads a local teaching-handle leader
and updates only the selected follower slice, leaving the other arm unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ARM_SLICES = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}


def _rpc(ws: Any, method: str, params: dict[str, Any] | None = None) -> Any:
    request = {"id": f"{method}-{time.time_ns()}", "method": method, "params": params or {}}
    ws.send(json.dumps(request))
    while True:
        response = json.loads(ws.recv())
        if response.get("type") == "hello":
            continue
        if not response.get("ok"):
            raise RuntimeError(response.get("error", response))
        return response.get("result")


class LeaderReader:
    def __init__(self, channel: str, ee_mass: float | None):
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType
        from yam_control_ws_server import _install_can_patch

        _install_can_patch()
        self.robot = get_yam_robot(
            channel=channel,
            gripper_type=GripperType.YAM_TEACHING_HANDLE,
            ee_mass=ee_mass,
            zero_gravity_mode=False,
        )
        self.motor_chain = self.robot.motor_chain

    def close(self) -> None:
        self.robot.close()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self.robot.get_observations()
        encoder_obs = self.motor_chain.get_same_bus_device_states()
        q_arm = np.asarray(obs["joint_pos"], dtype=float)
        handle = encoder_obs[0]
        gripper = 1.0 - float(handle.position)
        q7 = np.concatenate([q_arm[:6], [gripper]])
        buttons = np.asarray(handle.io_inputs, dtype=float)
        return q7, buttons


def _clip_step(current: np.ndarray, target: np.ndarray, max_step: float, max_gripper_step: float) -> np.ndarray:
    limits = np.full(7, max_step, dtype=float)
    limits[6] = max_gripper_step
    delta = np.clip(target - current, -limits, limits)
    return current + delta


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge a local YAM leader arm to one arm in the YAM WebSocket API.")
    parser.add_argument("--control-url", default="ws://127.0.0.1:8780/control")
    parser.add_argument("--leader-can", default="can1")
    parser.add_argument("--arm", choices=sorted(ARM_SLICES), default="right")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--max-joint-step", type=float, default=0.025)
    parser.add_argument("--max-gripper-step", type=float, default=0.015)
    parser.add_argument("--min-gripper", type=float, default=0.01)
    parser.add_argument("--max-gripper", type=float, default=0.59)
    parser.add_argument("--button-index", type=int, default=0)
    parser.add_argument("--deadman", action="store_true", help="Require holding the button instead of toggle-to-sync.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ee-mass", type=float, default=None)
    args = parser.parse_args()

    dt = 1.0 / max(args.hz, 1.0)
    arm_slice = ARM_SLICES[args.arm]

    leader = LeaderReader(args.leader_can, args.ee_mass)
    synchronized = False
    previous_pressed = False

    try:
        with connect(args.control_url, max_size=16 * 1024 * 1024) as ws:
            current_14 = np.asarray(_rpc(ws, "get_joint_pos"), dtype=float)
            if current_14.shape != (14,):
                raise RuntimeError(f"expected 14D websocket joint state, got {current_14.shape}")
            command_14 = current_14.copy()
            command_arm = command_14[arm_slice].copy()

            print(
                f"leader_ws_bridge ready: leader={args.leader_can} -> websocket {args.arm} arm. "
                f"{'Holding' if args.deadman else 'Press'} button {args.button_index} to sync.",
                flush=True,
            )

            while True:
                start = time.monotonic()
                leader_q, buttons = leader.read()
                if args.button_index >= len(buttons):
                    raise RuntimeError(f"button-index {args.button_index} unavailable; handle inputs={buttons.tolist()}")

                pressed = bool(buttons[args.button_index] > 0.5)
                if args.deadman:
                    synchronized = pressed
                elif pressed and not previous_pressed:
                    synchronized = not synchronized
                    state = "SYNC ON" if synchronized else "sync off"
                    print(state, flush=True)
                    current_14 = np.asarray(_rpc(ws, "get_joint_pos"), dtype=float)
                    command_14 = current_14.copy()
                    command_arm = command_14[arm_slice].copy()
                previous_pressed = pressed

                leader_q = leader_q.copy()
                leader_q[6] = float(np.clip(leader_q[6], args.min_gripper, args.max_gripper))

                if synchronized:
                    command_arm = _clip_step(command_arm, leader_q, args.max_joint_step, args.max_gripper_step)
                    command_14[arm_slice] = command_arm
                    if args.dry_run:
                        print(json.dumps({"target": command_14.tolist(), "leader": leader_q.tolist()}), flush=True)
                    else:
                        _rpc(ws, "command_joint_pos", {"joint_pos": command_14.tolist()})

                elapsed = time.monotonic() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    finally:
        leader.close()


if __name__ == "__main__":
    raise SystemExit(main())
