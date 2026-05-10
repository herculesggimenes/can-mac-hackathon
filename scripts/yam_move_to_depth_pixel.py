#!/usr/bin/env python3
"""Plan or execute a bounded YAM move toward a metric depth pixel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import mujoco
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "i2rt"))

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

from yam_depth_geometry import _load_frame, point_from_frame_payload, transform_point


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


def _load_transform(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        matrix = data
    else:
        matrix = (
            data.get("T_robot_camera")
            or data.get("T_base_camera")
            or data.get("T_world_camera")
            or data.get("matrix")
        )
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"expected a 4x4 camera-to-robot transform in {path}")
    return transform


def _parse_vec3(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return np.asarray(parts, dtype=float)


def _parse_joint_pos(value: str) -> np.ndarray:
    if Path(value).exists():
        data = json.loads(Path(value).read_text())
        if isinstance(data, dict) and "result" in data:
            data = data["result"]
    else:
        data = json.loads(value)
    q = np.asarray(data, dtype=float)
    if q.shape != (14,):
        raise ValueError(f"expected 14 joint values, got shape {q.shape}")
    return q


def _current_joint_pos(control_url: str) -> np.ndarray:
    with connect(control_url, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        response = _rpc(ws, "get_joint_pos")
    q = np.asarray(response["result"], dtype=float)
    if q.shape != (14,):
        raise RuntimeError(f"robot returned {q.shape}, expected 14D joints")
    return q


def _send_joint_pos(control_url: str, q: np.ndarray) -> None:
    with connect(control_url, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        _rpc(ws, "command_joint_pos", {"joint_pos": q.tolist()})


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


def _bounded_target(current: np.ndarray, desired: np.ndarray, max_joint_delta: float) -> np.ndarray:
    delta = desired - current
    max_abs = float(np.max(np.abs(delta[:6]))) if delta.size >= 6 else float(np.max(np.abs(delta)))
    if max_abs <= max_joint_delta or max_abs == 0:
        return desired.copy()
    scale = max_joint_delta / max_abs
    bounded = current + delta * scale
    bounded[6] = desired[6]
    return bounded


def plan_move(args: argparse.Namespace) -> dict[str, Any]:
    frame = _load_frame(Path(args.camera_payload).expanduser(), args.camera_id)
    point_info = point_from_frame_payload(frame, args.u, args.v, args.window)
    point_camera = np.asarray(point_info["point_camera_m"], dtype=float)
    t_robot_camera = _load_transform(Path(args.calibration).expanduser())
    point_robot = transform_point(point_camera, t_robot_camera)
    target_position = point_robot + args.target_offset

    current_14 = args.joint_pos if args.joint_pos is not None else _current_joint_pos(args.control_url)
    arm_slice = ARM_SLICES[args.arm]
    current_arm = current_14[arm_slice].copy()

    xml_path = _combined_xml_path(args.arm_type, args.gripper)
    kin = Kinematics(xml_path, args.site)
    current_model_q = _command_q_to_model_q(xml_path, current_arm)
    current_pose = kin.fk(current_model_q, args.site)
    target_pose = current_pose.copy()
    target_pose[:3, 3] = target_position

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
    desired_arm[6] = current_arm[6]
    bounded_arm = _bounded_target(current_arm, desired_arm, args.max_joint_delta)

    command_14 = current_14.copy()
    command_14[arm_slice] = bounded_arm

    return {
        "ok": bool(ok),
        "arm": args.arm,
        "site": args.site,
        "pixel": point_info["pixel"],
        "depth_m": point_info["depth_m"],
        "point_camera_m": point_camera.tolist(),
        "point_robot_m": point_robot.tolist(),
        "target_position_robot_m": target_position.tolist(),
        "current_joint_pos": current_14.tolist(),
        "desired_arm_joint_pos": desired_arm.tolist(),
        "bounded_arm_joint_pos": bounded_arm.tolist(),
        "command_joint_pos": command_14.tolist(),
        "max_joint_delta": args.max_joint_delta,
        "will_execute": bool(args.execute),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a raw depth pixel into a bounded YAM IK command. Dry-run by default."
    )
    parser.add_argument("camera_payload", help="Path to camera_payload.json containing raw depth data.")
    parser.add_argument("--camera-id", default="depth")
    parser.add_argument("--u", type=int, required=True)
    parser.add_argument("--v", type=int, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--calibration", required=True, help="JSON file with T_robot_camera 4x4 matrix.")
    parser.add_argument("--control-url", default="ws://127.0.0.1:8780/control")
    parser.add_argument("--arm", choices=sorted(ARM_SLICES), default="right")
    parser.add_argument("--arm-type", default="yam")
    parser.add_argument("--gripper", default="linear_4310")
    parser.add_argument("--site", default="grasp_site")
    parser.add_argument(
        "--target-offset",
        type=_parse_vec3,
        default=np.zeros(3),
        help="Robot-frame XYZ offset in meters, e.g. 0,0,0.06 for above target.",
    )
    parser.add_argument("--max-joint-delta", type=float, default=0.08)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--pos-threshold", type=float, default=1e-4)
    parser.add_argument("--ori-threshold", type=float, default=1e-3)
    parser.add_argument("--joint-pos", type=_parse_joint_pos, help="14D JSON list or path, for offline dry-runs.")
    parser.add_argument("--execute", action="store_true", help="Send the planned 14D command to the robot.")
    args = parser.parse_args()

    plan = plan_move(args)
    if args.execute:
        if not plan["ok"]:
            raise SystemExit("IK did not converge; refusing to execute")
        _send_joint_pos(args.control_url, np.asarray(plan["command_joint_pos"], dtype=float))
        plan["executed"] = True
    else:
        plan["executed"] = False

    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
