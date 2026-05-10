#!/usr/bin/env python3
"""Cartesian IK control for the bimanual YAM WebSocket API.

This is a smoother operator-facing layer over ``command_joint_pos``:
read 14D joints, split a task-level end-effector delta into Cartesian segments,
solve single-arm IK for each segment, then send full 14D commands as
interpolated trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "i2rt"))

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml


ARM_SLICES = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}


def _rpc(ws, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    ws.send(json.dumps({"id": method, "method": method, "params": params or {}}))
    response = json.loads(ws.recv())
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "rpc_failed"))
    return response


def _get_joint_pos(control_url: str) -> np.ndarray:
    with connect(control_url, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        response = _rpc(ws, "get_joint_pos")
    q = np.asarray(response["result"], dtype=float)
    if q.shape != (14,):
        raise RuntimeError(f"expected 14D joints, got {q.shape}")
    return q


def _get_status(control_url: str) -> dict[str, Any]:
    with connect(control_url, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        return _rpc(ws, "get_status")["result"]


def _assert_motion_ready(status: dict[str, Any]) -> None:
    if status.get("zero_gravity_mode"):
        raise RuntimeError("refusing to execute while zero_gravity_mode is enabled")
    arms = status.get("arms") or {}
    for arm_name in ("left", "right"):
        arm = arms.get(arm_name) or {}
        if not arm.get("connected"):
            raise RuntimeError(f"refusing to execute: {arm_name} arm is not connected")


def _command_joint_pos(control_url: str, q: np.ndarray) -> None:
    with connect(control_url, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        _rpc(ws, "command_joint_pos", {"joint_pos": q.tolist()})


def _command_joint_pos_open(ws, q: np.ndarray) -> None:
    _rpc(ws, "command_joint_pos", {"joint_pos": q.tolist()})


def _parse_vec3(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return np.asarray(parts, dtype=float)


def _combined_xml_path(arm: str, gripper: str) -> str:
    arm_type = ArmType.from_string_name(arm)
    gripper_type = GripperType.from_string_name(gripper)
    return combine_arm_and_gripper_xml(arm_type.get_xml_path(), gripper_type.get_xml_path())


def _command_q_to_model_q(xml_path: str, q7: np.ndarray) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(xml_path)
    q_model = np.zeros(model.nq, dtype=float)
    q_model[:6] = q7[:6]
    slide_values: list[float] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
            continue
        lo, hi = model.jnt_range[joint_id]
        slide_values.append(float(lo + q7[6] * (hi - lo)))
    if slide_values:
        q_model[6 : 6 + len(slide_values)] = slide_values
    elif model.nq > 6:
        q_model[6:] = q7[6]
    return q_model


def _rotation_delta_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    def rot_x(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    def rot_y(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    def rot_z(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def _bounded_arm_target(
    current: np.ndarray,
    desired: np.ndarray,
    max_joint_delta: float,
    *,
    wrist_only: bool = False,
) -> np.ndarray:
    desired = desired.copy()
    if wrist_only:
        desired[:3] = current[:3]
    desired[6] = current[6]
    if max_joint_delta <= 0:
        return desired
    delta = desired - current
    max_abs = float(np.max(np.abs(delta[:6])))
    if max_abs > max_joint_delta:
        desired[:6] = current[:6] + delta[:6] * (max_joint_delta / max_abs)
    return desired


def _plan_segment(
    args: argparse.Namespace,
    *,
    kin: Kinematics,
    xml_path: str,
    q14: np.ndarray,
    segment_delta: np.ndarray,
    segment_rot_delta: np.ndarray,
) -> dict[str, Any]:
    arm_slice = ARM_SLICES[args.arm]
    current_arm = q14[arm_slice].copy()
    current_model_q = _command_q_to_model_q(xml_path, current_arm)
    current_pose = kin.fk(current_model_q, args.site)

    target_pose = current_pose.copy()
    if args.frame == "local":
        target_pose[:3, 3] += current_pose[:3, :3] @ segment_delta
    else:
        target_pose[:3, 3] += segment_delta
    target_pose[:3, :3] = current_pose[:3, :3] @ _rotation_delta_xyz(*segment_rot_delta)

    ok, ik_q = kin.ik(
        target_pose,
        args.site,
        init_q=current_model_q,
        max_iters=args.max_iters,
        pos_threshold=args.pos_threshold,
        ori_threshold=args.ori_threshold,
    )
    desired_arm = current_arm.copy()
    desired_arm[:6] = ik_q[:6]
    bounded_arm = _bounded_arm_target(
        current_arm,
        desired_arm,
        args.max_joint_delta,
        wrist_only=args.wrist_only,
    )

    command = q14.copy()
    command[arm_slice] = bounded_arm
    return {
        "ok": bool(ok),
        "segment_delta_m": segment_delta.tolist(),
        "segment_rot_delta_rad": segment_rot_delta.tolist(),
        "current_pose": current_pose.tolist(),
        "target_pose": target_pose.tolist(),
        "current_joint_pos": q14.tolist(),
        "desired_arm_joint_pos": desired_arm.tolist(),
        "bounded_arm_joint_pos": bounded_arm.tolist(),
        "command_joint_pos": command.tolist(),
        "max_joint_delta": args.max_joint_delta,
        "wrist_only": bool(args.wrist_only),
    }


def _segment_count(args: argparse.Namespace) -> int:
    requested = max(1, int(args.segments))
    delta_norm = float(np.linalg.norm(args.delta))
    rot_norm = float(np.linalg.norm(args.rot_delta))
    by_translation = int(np.ceil(delta_norm / args.max_segment_m)) if args.max_segment_m > 0 else 1
    by_rotation = int(np.ceil(rot_norm / args.max_segment_rad)) if args.max_segment_rad > 0 else 1
    return max(requested, by_translation, by_rotation, 1)


def plan_cartesian(args: argparse.Namespace) -> dict[str, Any]:
    status = _get_status(args.control_url)
    q14 = _get_joint_pos(args.control_url)
    xml_path = _combined_xml_path(args.arm_type, args.gripper)
    kin = Kinematics(xml_path, args.site)
    segment_count = _segment_count(args)
    segment_delta = args.delta / segment_count
    segment_rot_delta = args.rot_delta / segment_count

    segments: list[dict[str, Any]] = []
    current_q = q14.copy()
    for index in range(segment_count):
        segment_plan = _plan_segment(
            args,
            kin=kin,
            xml_path=xml_path,
            q14=current_q,
            segment_delta=segment_delta,
            segment_rot_delta=segment_rot_delta,
        )
        segment_plan["index"] = index
        segments.append(segment_plan)
        current_q = np.asarray(segment_plan["command_joint_pos"], dtype=float)
        if not segment_plan["ok"] and args.stop_on_ik_failure:
            break

    return {
        "ok": all(bool(segment["ok"]) for segment in segments),
        "executable": bool(segments) and (
            all(bool(segment["ok"]) for segment in segments) or bool(args.allow_nonconverged_segments)
        ),
        "arm": args.arm,
        "frame": args.frame,
        "site": args.site,
        "delta_m": args.delta.tolist(),
        "rot_delta_rad": args.rot_delta.tolist(),
        "camera_rot_delta_deg": args.camera_rot_delta_deg.tolist(),
        "wrist_only": bool(args.wrist_only),
        "segment_count": len(segments),
        "requested_segments": int(args.segments),
        "max_segment_m": args.max_segment_m,
        "max_segment_rad": args.max_segment_rad,
        "initial_joint_pos": q14.tolist(),
        "initial_status": status,
        "command_joint_pos": current_q.tolist(),
        "max_joint_delta_per_segment": args.max_joint_delta,
        "segments": segments,
    }


def execute_trajectory_open(ws, start: np.ndarray, target: np.ndarray, steps: int, dt: float) -> None:
    steps = max(1, int(steps))
    for index in range(1, steps + 1):
        alpha = index / steps
        q = start + (target - start) * alpha
        _command_joint_pos_open(ws, q)
        if dt > 0:
            time.sleep(dt)


def execute_plan(control_url: str, plan: dict[str, Any], steps: int, dt: float) -> None:
    with connect(control_url, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        for segment in plan["segments"]:
            _assert_motion_ready(_rpc(ws, "get_status")["result"])
            execute_trajectory_open(
                ws,
                np.asarray(segment["current_joint_pos"], dtype=float),
                np.asarray(segment["command_joint_pos"], dtype=float),
                steps,
                dt,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Move one YAM end-effector by Cartesian IK delta.")
    parser.add_argument("--control-url", default="ws://127.0.0.1:8780/control")
    parser.add_argument("--arm", choices=sorted(ARM_SLICES), default="right")
    parser.add_argument("--arm-type", default="yam")
    parser.add_argument("--gripper", default="linear_4310")
    parser.add_argument("--site", default="grasp_site")
    parser.add_argument("--frame", choices=["world", "local"], default="local")
    parser.add_argument("--delta", type=_parse_vec3, required=True, help="XYZ delta in meters, e.g. 0,0,-0.03")
    parser.add_argument("--rot-delta", type=_parse_vec3, default=np.zeros(3), help="XYZ Euler delta in radians.")
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=0.0,
        help="Convenience camera-local pitch rotation in degrees, added to --rot-delta X.",
    )
    parser.add_argument(
        "--camera-yaw-deg",
        type=float,
        default=0.0,
        help="Convenience camera-local yaw rotation in degrees, added to --rot-delta Y.",
    )
    parser.add_argument(
        "--camera-roll-deg",
        type=float,
        default=0.0,
        help="Convenience camera-local roll rotation in degrees, added to --rot-delta Z.",
    )
    parser.add_argument(
        "--wrist-only",
        action="store_true",
        help="Keep arm joints 1-3 fixed after IK so joints 4-6 do camera aiming.",
    )
    parser.add_argument("--segments", type=int, default=1, help="Minimum Cartesian IK segments for the full delta.")
    parser.add_argument("--max-segment-m", type=float, default=0.04, help="Automatically split translation into chunks no larger than this.")
    parser.add_argument("--max-segment-rad", type=float, default=0.25, help="Automatically split rotation into chunks no larger than this.")
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.20,
        help="Per-segment joint cap, not total move cap. Use <=0 only for dry-run/debug, not live operation.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Interpolated commands per IK segment.")
    parser.add_argument("--hz", type=float, default=50.0, help="Command rate for interpolated trajectory.")
    parser.add_argument("--dt", type=float, default=None, help="Deprecated: seconds between commands. Overrides --hz when set.")
    parser.add_argument(
        "--ik-preset",
        choices=["strict", "live", "loose"],
        default="live",
        help="IK convergence preset. live is tolerant enough for real robot nudges; strict matches simulator-level checks.",
    )
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--pos-threshold", type=float, default=None)
    parser.add_argument("--ori-threshold", type=float, default=None)
    parser.add_argument("--stop-on-ik-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-nonconverged-segments",
        action="store_true",
        help="Allow execution of finite bounded segment commands even when IK misses the strict threshold.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    presets = {
        "strict": {"max_iters": 200, "pos_threshold": 1e-4, "ori_threshold": 1e-3},
        "live": {"max_iters": 350, "pos_threshold": 0.015, "ori_threshold": 0.08},
        "loose": {"max_iters": 500, "pos_threshold": 0.04, "ori_threshold": 0.18},
    }
    preset = presets[args.ik_preset]
    if args.max_iters is None:
        args.max_iters = preset["max_iters"]
    if args.pos_threshold is None:
        args.pos_threshold = preset["pos_threshold"]
    if args.ori_threshold is None:
        args.ori_threshold = preset["ori_threshold"]
    if args.execute and args.max_joint_delta <= 0:
        raise SystemExit("--max-joint-delta must be positive when --execute is used")
    args.camera_rot_delta_deg = np.asarray(
        [args.camera_pitch_deg, args.camera_yaw_deg, args.camera_roll_deg],
        dtype=float,
    )
    args.rot_delta = np.asarray(args.rot_delta, dtype=float) + np.deg2rad(args.camera_rot_delta_deg)

    plan = plan_cartesian(args)
    if args.execute:
        if not plan["executable"]:
            raise SystemExit("IK did not converge; pass --allow-nonconverged-segments to execute bounded segment commands")
        dt = float(args.dt) if args.dt is not None else 1.0 / max(float(args.hz), 1.0)
        execute_plan(args.control_url, plan, args.steps, dt)
        plan["executed"] = True
    else:
        plan["executed"] = False
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
