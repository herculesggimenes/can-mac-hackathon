#!/usr/bin/env python3
"""Move left and right YAM arms together (same timestep for each joint command)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "can-bridge"))
sys.path.insert(0, str(ROOT / "scripts"))

STOP_FILE = ROOT / "HARD_STOP"
STOP_REQUESTED = False


def _request_stop(_signum=None, _frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


def _install_can_patch() -> None:
    from local_modal_robot_bridge import _install_can_patch as _patch

    _patch()


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


def _safety_pair(robot_l, robot_r) -> dict:
    a = _safety_report(robot_l)
    b = _safety_report(robot_r)
    if not a.get("available", False) or not b.get("available", False):
        return {"available": False}
    return {
        "available": True,
        "max_temp_mos": max(a["max_temp_mos"], b["max_temp_mos"]),
        "max_temp_rotor": max(a["max_temp_rotor"], b["max_temp_rotor"]),
        "temp_mos_left": a["max_temp_mos"],
        "temp_mos_right": b["max_temp_mos"],
    }


def _safety_block_reason_pair(report: dict, args: argparse.Namespace) -> dict | None:
    if not report.get("available", False):
        return {"execution_blocked": "joint_state_unavailable"}
    if report["max_temp_mos"] > args.max_temp_mos:
        return {
            "execution_blocked": "mos_temperature_too_high",
            "max_temp_mos": report["max_temp_mos"],
            "limit": args.max_temp_mos,
        }
    if report["max_temp_rotor"] > args.max_temp_rotor:
        return {
            "execution_blocked": "rotor_temperature_too_high",
            "max_temp_rotor": report["max_temp_rotor"],
            "limit": args.max_temp_rotor,
        }
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

    robot_l = get_yam_robot(
        channel=args.left_can,
        gripper_type=GripperType.LINEAR_4310,
        zero_gravity_mode=False,
        clip_commands_to_joint_limits=False,
    )
    robot_r = get_yam_robot(
        channel=args.right_can,
        gripper_type=GripperType.LINEAR_4310,
        zero_gravity_mode=False,
        clip_commands_to_joint_limits=False,
    )
    try:
        robot_l._limit_gripper_force = -1.0
        robot_r._limit_gripper_force = -1.0

        cur_l = np.asarray(robot_l.get_joint_pos(), dtype=float)
        cur_r = np.asarray(robot_r.get_joint_pos(), dtype=float)
        safety = _safety_pair(robot_l, robot_r)
        print(
            json.dumps(
                {
                    "left": cur_l.tolist(),
                    "right": cur_r.tolist(),
                    "safety": safety,
                    "left_can": args.left_can,
                    "right_can": args.right_can,
                }
            ),
            flush=True,
        )
        block = _safety_block_reason_pair(safety, args)
        if block is not None:
            print(json.dumps(block), flush=True)
            return 1

        if args.read_only:
            print(
                json.dumps(
                    {
                        "direct_both_ok": True,
                        "note": "Shutdown noise after exit is common; separate DM faults per arm.",
                    }
                ),
                flush=True,
            )
            return 0

        req_l = _build_target(cur_l, args)
        req_r = _build_target(cur_r, args)
        tgt_l = _limit_target(cur_l, req_l, args)
        tgt_r = _limit_target(cur_r, req_r, args)
        print(
            json.dumps(
                {
                    "requested_left": req_l.tolist(),
                    "requested_right": req_r.tolist(),
                    "target_left": tgt_l.tolist(),
                    "target_right": tgt_r.tolist(),
                    "delta_left": (tgt_l - cur_l).tolist(),
                    "delta_right": (tgt_r - cur_r).tolist(),
                }
            ),
            flush=True,
        )

        steps = max(1, args.steps)
        for index in range(1, steps + 1):
            if STOP_REQUESTED or STOP_FILE.exists():
                print(json.dumps({"stopped": True, "step": index}), flush=True)
                return 130
            safety = _safety_pair(robot_l, robot_r)
            block = _safety_block_reason_pair(safety, args)
            if block is not None:
                print(json.dumps(block), flush=True)
                return 1
            alpha = index / steps
            cmd_l = cur_l + alpha * (tgt_l - cur_l)
            cmd_r = cur_r + alpha * (tgt_r - cur_r)
            robot_l.command_joint_pos(cmd_l)
            robot_r.command_joint_pos(cmd_r)
            print(
                json.dumps(
                    {
                        "step": index,
                        "steps": steps,
                        "commanded_left": cmd_l.tolist(),
                        "commanded_right": cmd_r.tolist(),
                    }
                ),
                flush=True,
            )
            if args.duration > 0:
                time.sleep(args.duration / steps)
        print(json.dumps({"direct_both_done": True}), flush=True)
        return 0
    finally:
        for arm, name in ((robot_l, "left"), (robot_r, "right")):
            try:
                arm.close()
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({f"{name}_close_error": repr(exc)}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Command both YAM arms in parallel (same interpolation clock).",
    )
    parser.add_argument("--left-can", default=os.environ.get("YAM_BIMANUAL_LEFT_CAN", "can0"))
    parser.add_argument("--right-can", default=os.environ.get("YAM_BIMANUAL_RIGHT_CAN", "can1"))
    parser.add_argument("--target", type=lambda v: _parse_vector(v, name="--target"))
    parser.add_argument("--delta", type=lambda v: _parse_vector(v, name="--delta"))
    parser.add_argument("--gripper", type=float)
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
        parser.error(
            "need --read-only, --target, --delta, or --gripper. "
            "Example: uv run yamctl direct-both --delta \"0,0,0.35,0,0,0,0\" --duration 5 --steps 100 --max-delta 0.5"
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
