#!/usr/bin/env python3
"""Calibration-free local visual servoing for a wrist depth target.

The script tracks a small image patch around a target pixel, estimates a local
image Jacobian by making tiny joint probes, then plans small bounded joint
updates that move the tracked target toward a desired image/depth goal.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from websockets.sync.client import connect

ARM_SLICES = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}


@dataclass
class Observation:
    image: np.ndarray
    depth_u16: np.ndarray
    depth_scale: float
    uv: np.ndarray
    depth_m: float
    score: float


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


def _command_joint_pos(control_url: str, q: np.ndarray) -> None:
    with connect(control_url, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        _rpc(ws, "command_joint_pos", {"joint_pos": q.tolist()})


def _get_frames(camera_url: str) -> list[dict[str, Any]]:
    with connect(camera_url, open_timeout=10, max_size=128 * 1024 * 1024) as ws:
        ws.recv()
        ws.send(json.dumps({"type": "get_all_frames", "bundle": True}))
        payload = json.loads(ws.recv())
    if payload.get("type") != "frames":
        raise RuntimeError(f"expected frame bundle, got {payload.get('type')!r}")
    return payload.get("frames", [])


def _decode_jpeg(frame: dict[str, Any]) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(frame["data"]), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode JPEG for {frame.get('camera_id')}")
    return image


def _decode_depth(frame: dict[str, Any]) -> tuple[np.ndarray, float]:
    depth = frame.get("depth")
    if not depth:
        raise RuntimeError(f"frame {frame.get('camera_id')} has no raw depth")
    if depth.get("format") != "uint16":
        raise RuntimeError(f"unsupported depth format: {depth.get('format')!r}")
    width = int(depth["width"])
    height = int(depth["height"])
    raw = base64.b64decode(depth["data"])
    depth_u16 = np.frombuffer(raw, dtype="<u2").reshape((height, width))
    return depth_u16, float(depth.get("scale_to_meters", 0.001))


def _depth_at(depth_u16: np.ndarray, uv: np.ndarray, scale: float, window: int) -> float:
    u, v = [int(round(x)) for x in uv]
    half = max(0, window // 2)
    x0 = max(0, u - half)
    x1 = min(depth_u16.shape[1], u + half + 1)
    y0 = max(0, v - half)
    y1 = min(depth_u16.shape[0], v + half + 1)
    valid = depth_u16[y0:y1, x0:x1]
    valid = valid[valid > 0]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid)) * scale


def _extract_template(image: np.ndarray, uv: np.ndarray, radius: int) -> np.ndarray:
    u, v = [int(round(x)) for x in uv]
    x0 = max(0, u - radius)
    x1 = min(image.shape[1], u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(image.shape[0], v + radius + 1)
    template = image[y0:y1, x0:x1]
    if template.shape[0] < 5 or template.shape[1] < 5:
        raise ValueError("target template too close to image boundary")
    return template


def _track_template(image: np.ndarray, template: np.ndarray, last_uv: np.ndarray, search_radius: int) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape[:2]
    u, v = [int(round(x)) for x in last_uv]
    x0 = max(0, u - search_radius)
    y0 = max(0, v - search_radius)
    x1 = min(image.shape[1], u + search_radius)
    y1 = min(image.shape[0], v + search_radius)
    roi = gray[y0:y1, x0:x1]
    if roi.shape[0] < h or roi.shape[1] < w:
        roi = gray
        x0 = 0
        y0 = 0
    result = cv2.matchTemplate(roi, template_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    center = np.array([x0 + max_loc[0] + w / 2.0, y0 + max_loc[1] + h / 2.0], dtype=float)
    return center, float(max_val)


def observe(
    camera_url: str,
    camera_id: str,
    template: np.ndarray | None,
    uv_hint: np.ndarray,
    search_radius: int,
    depth_window: int,
) -> Observation:
    frames = _get_frames(camera_url)
    frame = next((item for item in frames if str(item.get("camera_id")) == camera_id), None)
    if frame is None:
        raise RuntimeError(f"camera {camera_id!r} not found")
    if frame.get("type") != "frame":
        raise RuntimeError(f"camera {camera_id!r} not ready: {frame.get('error')}")
    image = _decode_jpeg(frame)
    depth_u16, scale = _decode_depth(frame)
    if template is None:
        uv = uv_hint.astype(float)
        score = 1.0
    else:
        uv, score = _track_template(image, template, uv_hint, search_radius)
    depth_m = _depth_at(depth_u16, uv, scale, depth_window)
    return Observation(image=image, depth_u16=depth_u16, depth_scale=scale, uv=uv, depth_m=depth_m, score=score)


def _feature(obs: Observation, initial_depth_m: float) -> np.ndarray:
    depth_value = obs.depth_m if np.isfinite(obs.depth_m) else initial_depth_m
    return np.array([obs.uv[0], obs.uv[1], depth_value], dtype=float)


def _arm_joint_indices(arm: str, joints: list[int]) -> list[int]:
    offset = ARM_SLICES[arm].start
    return [offset + joint for joint in joints]


def estimate_jacobian(
    args: argparse.Namespace,
    q0: np.ndarray,
    obs0: Observation,
    template: np.ndarray,
    joint_indices: list[int],
) -> np.ndarray:
    baseline = _feature(obs0, obs0.depth_m)
    columns = []
    for joint_index in joint_indices:
        q_probe = q0.copy()
        q_probe[joint_index] += args.probe_delta
        if args.execute:
            _command_joint_pos(args.control_url, q_probe)
            time.sleep(args.settle_s)
            obs_probe = observe(args.camera_url, args.camera_id, template, obs0.uv, args.search_radius, args.depth_window)
            _command_joint_pos(args.control_url, q0)
            time.sleep(args.settle_s)
        else:
            obs_probe = obs0
        delta_feature = _feature(obs_probe, obs0.depth_m) - baseline
        delta_feature = np.nan_to_num(delta_feature, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(delta_feature / args.probe_delta)
    return np.column_stack(columns)


def plan_update(args: argparse.Namespace, obs: Observation, jacobian: np.ndarray) -> np.ndarray:
    desired = np.array([args.goal_u, args.goal_v, args.desired_depth_m], dtype=float)
    current = _feature(obs, obs.depth_m)
    error = desired - current
    weights = np.diag([args.pixel_weight, args.pixel_weight, args.depth_weight])
    weighted_j = weights @ jacobian
    weighted_error = weights @ error
    weighted_j = np.nan_to_num(weighted_j, nan=0.0, posinf=0.0, neginf=0.0)
    weighted_error = np.nan_to_num(weighted_error, nan=0.0, posinf=0.0, neginf=0.0)
    dq = np.linalg.pinv(weighted_j, rcond=args.pinv_rcond) @ weighted_error
    max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
    if max_abs > args.max_joint_step:
        dq *= args.max_joint_step / max_abs
    return dq


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration-free local visual servo touch test for YAM wrist depth.")
    parser.add_argument("--control-url", default="ws://127.0.0.1:8780/control")
    parser.add_argument("--camera-url", default="ws://127.0.0.1:8770/cameras")
    parser.add_argument("--camera-id", default="depth", help="Depth camera id that includes JPEG preview and raw depth.")
    parser.add_argument("--arm", choices=sorted(ARM_SLICES), default="right")
    parser.add_argument("--target-u", type=float, required=True, help="Initial target pixel x in the depth preview.")
    parser.add_argument("--target-v", type=float, required=True, help="Initial target pixel y in the depth preview.")
    parser.add_argument("--goal-u", type=float, help="Desired target pixel x. Defaults to image center x.")
    parser.add_argument("--goal-v", type=float, help="Desired target pixel y. Defaults to image center y.")
    parser.add_argument("--desired-depth-m", type=float, default=0.035, help="Desired target depth from wrist camera.")
    parser.add_argument("--joints", default="0,1,2,3,4,5", help="0-based arm joints to probe/control, comma-separated.")
    parser.add_argument("--template-radius", type=int, default=10)
    parser.add_argument("--search-radius", type=int, default=45)
    parser.add_argument("--depth-window", type=int, default=7)
    parser.add_argument("--probe-delta", type=float, default=0.015)
    parser.add_argument("--max-joint-step", type=float, default=0.025)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--settle-s", type=float, default=0.35)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=900.0, help="Meters-to-pixels scale for depth error.")
    parser.add_argument("--pinv-rcond", type=float, default=1e-3)
    parser.add_argument("--min-track-score", type=float, default=0.35)
    parser.add_argument("--execute", action="store_true", help="Actually probe and command the robot. Dry-run by default.")
    args = parser.parse_args()

    q = _get_joint_pos(args.control_url)
    uv0 = np.array([args.target_u, args.target_v], dtype=float)
    obs = observe(args.camera_url, args.camera_id, None, uv0, args.search_radius, args.depth_window)
    template = _extract_template(obs.image, obs.uv, args.template_radius)
    if args.goal_u is None:
        args.goal_u = obs.image.shape[1] / 2.0
    if args.goal_v is None:
        args.goal_v = obs.image.shape[0] / 2.0
    arm_joint_ids = _arm_joint_indices(args.arm, [int(item.strip()) for item in args.joints.split(",") if item.strip()])

    history: list[dict[str, Any]] = []
    for iteration in range(args.iterations):
        if iteration > 0:
            obs = observe(args.camera_url, args.camera_id, template, obs.uv, args.search_radius, args.depth_window)
        if obs.score < args.min_track_score:
            raise SystemExit(f"tracking score too low: {obs.score:.3f}")
        if not args.execute:
            history.append(
                {
                    "iteration": iteration,
                    "uv": obs.uv.tolist(),
                    "depth_m": obs.depth_m,
                    "track_score": obs.score,
                    "controlled_joint_indices": arm_joint_ids,
                    "probe_delta": args.probe_delta,
                    "max_joint_step": args.max_joint_step,
                    "note": "dry_run_only; pass --execute to physically probe joints and estimate the local image Jacobian",
                }
            )
            break
        jacobian = estimate_jacobian(args, q, obs, template, arm_joint_ids)
        dq = plan_update(args, obs, jacobian)
        q_next = q.copy()
        for index, joint_index in enumerate(arm_joint_ids):
            q_next[joint_index] += dq[index]
        history.append(
            {
                "iteration": iteration,
                "uv": obs.uv.tolist(),
                "depth_m": obs.depth_m,
                "track_score": obs.score,
                "controlled_joint_indices": arm_joint_ids,
                "jacobian": jacobian.tolist(),
                "dq": dq.tolist(),
                "command_joint_pos": q_next.tolist(),
            }
        )
        _command_joint_pos(args.control_url, q_next)
        time.sleep(args.settle_s)
        q = _get_joint_pos(args.control_url)

    print(
        json.dumps(
            {
                "executed": bool(args.execute),
                "arm": args.arm,
                "camera_id": args.camera_id,
                "initial_target_uv": uv0.tolist(),
                "goal_uv": [args.goal_u, args.goal_v],
                "desired_depth_m": args.desired_depth_m,
                "history": history,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
