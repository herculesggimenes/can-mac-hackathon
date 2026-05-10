#!/usr/bin/env python3
"""Collect an LLM-facing YAM robot snapshot.

The raw APIs are good for programs but awkward for Codex-style decision making:
joint data, robot health, RGB images, depth payloads, and command limits are
spread across several calls and files. This script creates one compact
``codex_snapshot.json`` plus image/depth artifacts so a planner can reason from
one stable input.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from websockets.sync.client import connect

from yam_depth_geometry import deproject_pixel, depth_array_from_frame_payload, depth_at_pixel


ARM_SLICES = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}


def _rpc(ws, method: str, params: dict[str, Any] | None = None, request_id: str | None = None) -> dict[str, Any]:
    ws.send(json.dumps({"id": request_id or method, "method": method, "params": params or {}}))
    response = json.loads(ws.recv())
    if not response.get("ok"):
        raise RuntimeError(response.get("error", f"{method} failed"))
    return response


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _safe_camera_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _joint_summary(joint_pos: list[float] | None) -> dict[str, Any] | None:
    if joint_pos is None:
        return None
    q = np.asarray(joint_pos, dtype=float)
    if q.shape != (14,):
        return {"error": f"expected 14 joints, got {q.shape}", "joint_pos": joint_pos}

    summary: dict[str, Any] = {
        "order": "left[0:7] + right[0:7]",
        "joint_pos": q.tolist(),
        "arms": {},
    }
    for arm, slc in ARM_SLICES.items():
        arm_q = q[slc]
        summary["arms"][arm] = {
            "joints_1_to_6_rad": arm_q[:6].tolist(),
            "gripper_normalized": float(arm_q[6]),
            "camera_wrist_joints_4_to_6_rad": arm_q[3:6].tolist(),
        }
    return summary


def _robot_snapshot(robot_url: str) -> dict[str, Any]:
    with connect(robot_url, open_timeout=10, max_size=32 * 1024 * 1024) as ws:
        hello = json.loads(ws.recv())
        status = _rpc(ws, "get_status", request_id="status")["result"]
        joint_pos = _rpc(ws, "get_joint_pos", request_id="joint_pos")["result"]
        observations_response = _rpc(ws, "get_observations", request_id="observations")

    return {
        "hello": hello,
        "status": status,
        "joint_summary": _joint_summary(joint_pos),
        "observations": observations_response.get("result"),
    }


def _image_size(path: Path) -> list[int] | None:
    try:
        with Image.open(path) as image:
            return [int(image.width), int(image.height)]
    except Exception:
        return None


def _make_contact_sheet(camera_entries: list[dict[str, Any]], snapshot_dir: Path, tile_width: int = 640) -> str | None:
    image_entries = [entry for entry in camera_entries if entry.get("image_path")]
    if not image_entries:
        return None

    loaded: list[tuple[dict[str, Any], Image.Image]] = []
    for entry in image_entries:
        image_path = snapshot_dir / str(entry["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        scale = tile_width / max(image.width, 1)
        tile_height = max(1, int(round(image.height * scale)))
        image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        loaded.append((entry, image))

    if not loaded:
        return None

    label_height = 28
    columns = 2 if len(loaded) > 1 else 1
    rows = int(math.ceil(len(loaded) / columns))
    tile_height = max(image.height for _entry, image in loaded)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for index, (entry, image) in enumerate(loaded):
        col = index % columns
        row = index // columns
        x = col * tile_width
        y = row * (tile_height + label_height)
        label = str(entry.get("camera_id", "camera"))
        if entry.get("has_depth"):
            label += " depth"
            center_depth = entry.get("center_depth") or {}
            if center_depth.get("depth_m") is not None:
                label += f" center={float(center_depth['depth_m']):.3f}m"
        draw.rectangle((x, y, x + tile_width, y + label_height), fill=(28, 28, 28))
        draw.text((x + 10, y + 6), label, fill=(240, 240, 240), font=font)
        sheet.paste(image, (x, y + label_height))

    path = snapshot_dir / "codex_contact_sheet.jpg"
    sheet.save(path, quality=90)
    return _relative(path, snapshot_dir)


def _depth_quality(depth_u16: np.ndarray, scale_to_meters: float) -> dict[str, Any]:
    valid = depth_u16[depth_u16 > 0]
    total = int(depth_u16.size)
    if valid.size == 0:
        return {"valid_ratio": 0.0, "valid_pixels": 0, "total_pixels": total}
    meters = valid.astype(np.float64) * float(scale_to_meters)
    return {
        "valid_ratio": float(valid.size / max(total, 1)),
        "valid_pixels": int(valid.size),
        "total_pixels": total,
        "min_m": float(np.min(meters)),
        "median_m": float(np.median(meters)),
        "max_m": float(np.max(meters)),
    }


def _sample_depth_grid(frame: dict[str, Any], grid: int, window: int) -> list[dict[str, Any]]:
    depth = frame.get("depth") or {}
    intrinsics = frame.get("intrinsics")
    if not intrinsics:
        return []

    depth_u16 = depth_array_from_frame_payload(frame)
    scale = float(depth.get("scale_to_meters", 0.001))
    height, width = depth_u16.shape
    samples: list[dict[str, Any]] = []
    for row in range(grid):
        for col in range(grid):
            u = int(round((col + 0.5) * width / grid))
            v = int(round((row + 0.5) * height / grid))
            u = min(max(u, 0), width - 1)
            v = min(max(v, 0), height - 1)
            try:
                depth_m = depth_at_pixel(depth_u16, u, v, scale, window)
                point = deproject_pixel(float(u), float(v), depth_m, intrinsics)
                samples.append(
                    {
                        "pixel": [u, v],
                        "depth_m": depth_m,
                        "point_camera_m": point.tolist(),
                    }
                )
            except Exception as exc:
                samples.append({"pixel": [u, v], "error": str(exc)})
    return samples


def _center_depth(frame: dict[str, Any], window: int) -> dict[str, Any] | None:
    depth = frame.get("depth") or {}
    intrinsics = frame.get("intrinsics")
    if not depth or not intrinsics:
        return None
    depth_u16 = depth_array_from_frame_payload(frame)
    height, width = depth_u16.shape
    u = width // 2
    v = height // 2
    scale = float(depth.get("scale_to_meters", 0.001))
    try:
        depth_m = depth_at_pixel(depth_u16, u, v, scale, window)
    except Exception as exc:
        return {"pixel": [u, v], "error": str(exc)}
    return {
        "pixel": [u, v],
        "depth_m": depth_m,
        "point_camera_m": deproject_pixel(float(u), float(v), depth_m, intrinsics).tolist(),
    }


def _camera_snapshot(camera_url: str, snapshot_dir: Path, grid: int, window: int) -> dict[str, Any]:
    frames_dir = snapshot_dir / "frames"
    depth_dir = snapshot_dir / "depth"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with connect(camera_url, open_timeout=10, max_size=64 * 1024 * 1024) as ws:
        hello = json.loads(ws.recv())
        ws.send(json.dumps({"type": "get_all_frames", "bundle": True}))
        payload = json.loads(ws.recv())

    _write_json(snapshot_dir / "camera_payload.json", payload)

    camera_entries: list[dict[str, Any]] = []
    frames = payload.get("frames", []) if payload.get("type") == "frames" else [payload]
    for frame in frames:
        if frame.get("type") != "frame":
            camera_entries.append({"type": frame.get("type"), "error": frame.get("error"), "raw": frame})
            continue

        camera_id = str(frame["camera_id"])
        safe_id = _safe_camera_id(camera_id)
        image_path = frames_dir / f"{safe_id}.jpg"
        image_path.write_bytes(base64.b64decode(frame["data"]))

        entry: dict[str, Any] = {
            "camera_id": camera_id,
            "image_path": _relative(image_path, snapshot_dir),
            "image_size": _image_size(image_path),
            "captured_at": frame.get("captured_at"),
            "source": frame.get("source"),
            "has_depth": bool(frame.get("depth")),
            "intrinsics": frame.get("intrinsics"),
        }

        if frame.get("depth"):
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth = dict(frame["depth"])
            raw = base64.b64decode(depth.pop("data"))
            depth_path = depth_dir / f"{safe_id}.u16"
            depth_meta_path = depth_dir / f"{safe_id}.json"
            depth_path.write_bytes(raw)
            _write_json(depth_meta_path, depth)

            depth_u16 = depth_array_from_frame_payload(frame)
            entry["depth_raw_path"] = _relative(depth_path, snapshot_dir)
            entry["depth_metadata_path"] = _relative(depth_meta_path, snapshot_dir)
            entry["depth_quality"] = _depth_quality(depth_u16, float(depth.get("scale_to_meters", 0.001)))
            entry["center_depth"] = _center_depth(frame, window)
            entry["depth_grid_samples"] = _sample_depth_grid(frame, grid, window)

        camera_entries.append(entry)

    contact_sheet_path = _make_contact_sheet(camera_entries, snapshot_dir)

    return {
        "hello": hello,
        "payload_path": "camera_payload.json",
        "contact_sheet_path": contact_sheet_path,
        "cameras": camera_entries,
    }


def _affordances(args: argparse.Namespace) -> dict[str, Any]:
    max_step = float(args.max_joint_delta)
    return {
        "command_space": {
            "type": "absolute_14d_joint_pos",
            "order": "left[0:7] + right[0:7]",
            "gripper_index_per_arm": 6,
            "gripper_limits": [0.01, 0.59],
        },
        "recommended_primitives": [
            {
                "name": "small_joint_delta",
                "description": "Apply a bounded relative change to selected joints, then verify with a new snapshot.",
                "max_abs_delta_rad": max_step,
            },
            {
                "name": "cartesian_delta",
                "script": "scripts/yam_cartesian_control.py",
                "description": "Move one end effector by a small local/world Cartesian delta using IK.",
                "default_max_joint_delta_rad": min(max_step, 0.1),
            },
            {
                "name": "wrist_camera_aim",
                "script": "scripts/yam_cartesian_control.py --delta 0,0,0 --wrist-only --camera-pitch-deg ...",
                "description": "Aim the wrist camera with joints 4-6 while preserving joints 1-3.",
                "default_degrees_per_step": 5.0,
            },
        ],
        "decision_rules": [
            "Observe before moving.",
            "Use depth only from cameras with has_depth=true and valid depth samples.",
            "Prefer high-rate multi-step Cartesian/wrist moves over one large joint jump.",
            "After every physical command, collect another snapshot before deciding the next move.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect one Codex-optimized YAM robot/camera/depth snapshot.")
    parser.add_argument("--robot-url", default="ws://127.0.0.1:8780/control")
    parser.add_argument("--camera-url", default="ws://127.0.0.1:8770/cameras")
    parser.add_argument("--output-dir", default="logs/yam-codex-snapshots")
    parser.add_argument("--depth-grid", type=int, default=3, help="NxN sampled depth grid per depth camera.")
    parser.add_argument("--depth-window", type=int, default=7, help="Median window for sampled depth points.")
    parser.add_argument("--max-joint-delta", type=float, default=0.10)
    args = parser.parse_args()

    root = Path(args.output_dir).expanduser()
    snapshot_dir = root / time.strftime("%Y%m%d-%H%M%S")
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    snapshot: dict[str, Any] = {
        "schema": "yam_codex_snapshot.v1",
        "created_at": time.time(),
        "snapshot_dir": str(snapshot_dir),
        "robot_url": args.robot_url,
        "camera_url": args.camera_url,
        "affordances": _affordances(args),
    }

    try:
        snapshot["robot"] = _robot_snapshot(args.robot_url)
    except Exception as exc:
        snapshot["robot_error"] = str(exc)

    try:
        snapshot["perception"] = _camera_snapshot(args.camera_url, snapshot_dir, max(1, args.depth_grid), args.depth_window)
    except Exception as exc:
        snapshot["perception_error"] = str(exc)

    snapshot_path = snapshot_dir / "codex_snapshot.json"
    _write_json(snapshot_path, snapshot)
    print(json.dumps({"snapshot_path": str(snapshot_path), **snapshot}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
