#!/usr/bin/env python3
"""Depth-camera geometry helpers for YAM perception and IK planning."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np


def deproject_pixel(u: float, v: float, depth_m: float, intrinsics: dict[str, float]) -> np.ndarray:
    """Convert one depth pixel to a 3D point in the camera optical frame."""
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m], dtype=float)


def transform_point(point: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to a 3D point."""
    point_h = np.ones(4, dtype=float)
    point_h[:3] = point
    return (transform_4x4 @ point_h)[:3]


def depth_array_from_frame_payload(frame: dict[str, Any]) -> np.ndarray:
    depth = frame.get("depth")
    if not depth:
        raise ValueError("frame payload has no raw depth field")
    if depth.get("format") != "uint16":
        raise ValueError(f"unsupported depth format: {depth.get('format')!r}")
    width = int(depth["width"])
    height = int(depth["height"])
    raw = base64.b64decode(depth["data"])
    return np.frombuffer(raw, dtype="<u2").reshape((height, width))


def depth_at_pixel(depth_u16: np.ndarray, u: int, v: int, scale_to_meters: float, window: int = 5) -> float:
    """Return median valid depth near a pixel, in meters."""
    half = max(0, int(window) // 2)
    y0 = max(0, int(v) - half)
    y1 = min(depth_u16.shape[0], int(v) + half + 1)
    x0 = max(0, int(u) - half)
    x1 = min(depth_u16.shape[1], int(u) + half + 1)
    patch = depth_u16[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        raise ValueError(f"no valid depth samples near pixel ({u}, {v})")
    return float(np.median(valid)) * float(scale_to_meters)


def point_from_frame_payload(frame: dict[str, Any], u: int, v: int, window: int = 5) -> dict[str, Any]:
    depth = frame.get("depth") or {}
    intrinsics = frame.get("intrinsics")
    if not intrinsics:
        raise ValueError("frame payload has no camera intrinsics")
    depth_u16 = depth_array_from_frame_payload(frame)
    depth_m = depth_at_pixel(depth_u16, u, v, float(depth.get("scale_to_meters", 0.001)), window)
    point_camera = deproject_pixel(float(u), float(v), depth_m, intrinsics)
    return {
        "pixel": [int(u), int(v)],
        "depth_m": depth_m,
        "point_camera_m": point_camera.tolist(),
        "intrinsics": intrinsics,
    }


def _load_frame(path: Path, camera_id: str | None) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    frames = payload.get("frames", []) if payload.get("type") == "frames" else [payload]
    candidates = [frame for frame in frames if frame.get("type") == "frame" and frame.get("depth")]
    if camera_id is not None:
        candidates = [frame for frame in candidates if str(frame.get("camera_id")) == camera_id]
    if not candidates:
        raise ValueError("no matching raw depth frame found")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a raw depth frame pixel to a 3D camera-frame point.")
    parser.add_argument("payload", help="Path to camera_payload.json containing raw depth data.")
    parser.add_argument("--camera-id", help="Depth camera id to use when the payload is bundled.")
    parser.add_argument("--u", type=int, required=True, help="Pixel x coordinate.")
    parser.add_argument("--v", type=int, required=True, help="Pixel y coordinate.")
    parser.add_argument("--window", type=int, default=5, help="Median depth window size in pixels.")
    args = parser.parse_args()

    frame = _load_frame(Path(args.payload).expanduser(), args.camera_id)
    result = point_from_frame_payload(frame, args.u, args.v, args.window)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
