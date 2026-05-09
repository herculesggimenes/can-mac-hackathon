#!/usr/bin/env python3
"""Direct local YAM joint control without Modal policy inference."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import can
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "can-bridge"))
STOP_FILE = ROOT / "HARD_STOP"
STOP_REQUESTED = False


def _request_stop(_signum=None, _frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


def _install_can_patch() -> None:
    from can_bridge import CanBridgeBus
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface
    from i2rt.robots.utils import GripperType

    orig_bus = can.interface.Bus

    def patched_bus(*args, **kwargs):
        interface = kwargs.get("interface", kwargs.get("bustype"))
        channel = kwargs.get("channel", args[0] if args else None)
        if interface == "socketcan" and channel in {"can0", 0, None}:
            return CanBridgeBus(channel=0)
        return orig_bus(*args, **kwargs)

    can.interface.Bus = patched_bus

    orig_close = DMChainCanInterface.close

    def patched_close(self):
        orig_close(self)
        motor_interface = getattr(self, "motor_interface", None)
        if motor_interface is not None:
            motor_interface.close()

    DMChainCanInterface.close = patched_close

    orig_get_limits = GripperType.get_gripper_limits
    orig_get_cal = GripperType.get_gripper_needs_calibration

    def patched_limits(self):
        if self == GripperType.LINEAR_4310:
            return (0.0, -4.20)
        return orig_get_limits(self)

    def patched_cal(self):
        if self == GripperType.LINEAR_4310:
            return False
        return orig_get_cal(self)

    GripperType.get_gripper_limits = patched_limits
    GripperType.get_gripper_needs_calibration = patched_cal


def _parse_vector(value: str, *, name: str) -> np.ndarray:
    parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(f"{name} needs exactly 7 numbers")
    return np.asarray([float(part) for part in parts], dtype=float)


def _safety_report(robot) -> dict:
    with robot._state_lock:
        joint_state = robot._joint_state
        if joint_state is None:
            return {"available": False}
        return {
            "available": True,
            "temp_mos": np.asarray(joint_state.temp_mos, dtype=float).tolist(),
            "temp_rotor": np.asarray(joint_state.temp_rotor, dtype=float).tolist(),
            "max_temp_mos": float(np.max(joint_state.temp_mos)),
            "max_temp_rotor": float(np.max(joint_state.temp_rotor)),
        }


def _safety_block_reason(report: dict, args: argparse.Namespace) -> dict | None:
    if not report.get("available", False):
        return {"execution_blocked": "joint_state_unavailable"}
    if report["max_temp_mos"] > args.max_temp_mos:
        return {"execution_blocked": "mos_temperature_too_high", "max_temp_mos": report["max_temp_mos"], "limit": args.max_temp_mos}
    if report["max_temp_rotor"] > args.max_temp_rotor:
        return {"execution_blocked": "rotor_temperature_too_high", "max_temp_rotor": report["max_temp_rotor"], "limit": args.max_temp_rotor}
    return None


def _build_target(current: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    target = np.asarray(current, dtype=float).copy()
    if args.target is not None:
        target = args.target.copy()
    if args.delta is not None:
        target += args.delta
    if args.gripper is not None:
        target[6] = args.gripper
    target[6] = float(np.clip(target[6], args.min_gripper_command, args.max_gripper_command))
    return target


def _limit_target(current: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    limited = np.asarray(current, dtype=float).copy()
    limited += np.clip(np.asarray(target, dtype=float) - limited, -args.max_delta, args.max_delta)
    limited[6] = float(np.clip(limited[6], args.min_gripper_command, args.max_gripper_command))
    return limited


def run(args: argparse.Namespace) -> int:
    if STOP_FILE.exists() and not args.ignore_hard_stop:
        print(f"hard stop exists at {STOP_FILE}; run `yamctl clear-stop` first", file=sys.stderr)
        return 2

    _install_can_patch()

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    robot = get_yam_robot(channel="can0", gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=False)
    try:
        current = np.asarray(robot.get_joint_pos(), dtype=float)
        safety = _safety_report(robot)
        print(json.dumps({"current": current.tolist(), "safety": safety}), flush=True)
        block_reason = _safety_block_reason(safety, args)
        if block_reason is not None:
            print(json.dumps(block_reason), flush=True)
            return 1
        if args.read_only:
            return 0

        requested = _build_target(current, args)
        target = _limit_target(current, requested, args)
        print(json.dumps({"requested": requested.tolist(), "target": target.tolist(), "delta": (target - current).tolist()}), flush=True)

        steps = max(1, args.steps)
        for index in range(1, steps + 1):
            if STOP_REQUESTED or STOP_FILE.exists():
                print(json.dumps({"stopped": True, "step": index}), flush=True)
                return 130
            safety = _safety_report(robot)
            block_reason = _safety_block_reason(safety, args)
            if block_reason is not None:
                print(json.dumps({"safety": safety}), flush=True)
                print(json.dumps(block_reason), flush=True)
                return 1
            command = current + (index / steps) * (target - current)
            robot.command_joint_pos(command)
            print(json.dumps({"step": index, "steps": steps, "commanded": command.tolist()}), flush=True)
            if args.duration > 0:
                time.sleep(args.duration / steps)
        return 0
    finally:
        robot.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct local YAM joint control without Modal.")
    parser.add_argument("--target", type=lambda v: _parse_vector(v, name="--target"))
    parser.add_argument("--delta", type=lambda v: _parse_vector(v, name="--delta"))
    parser.add_argument("--gripper", type=float, help="Set normalized joint 7/gripper command.")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-delta", type=float, default=0.20)
    parser.add_argument("--max-temp-mos", type=float, default=55.0)
    parser.add_argument("--max-temp-rotor", type=float, default=100.0)
    parser.add_argument("--min-gripper-command", type=float, default=0.01)
    parser.add_argument("--max-gripper-command", type=float, default=0.59)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--ignore-hard-stop", action="store_true")
    args = parser.parse_args()
    if not args.read_only and args.target is None and args.delta is None and args.gripper is None:
        parser.error("pass --target, --delta, --gripper, or --read-only")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
