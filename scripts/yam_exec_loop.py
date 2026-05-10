#!/usr/bin/env python3
"""Fast Codex-exec policy loop for right-arm YAM robot tasks.

This script keeps the LLM out of the hardware path. It collects one compact
snapshot, asks `codex exec` for exactly one JSON action, validates the action
against a small allowlist, and optionally executes it. Dry-run is the default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from websockets.sync.client import connect

from yam_depth_geometry import _load_frame, point_from_frame_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_URL = "wss://2d37-12-125-194-54.ngrok-free.app/control"
DEFAULT_CAMERA_URL = "ws://127.0.0.1:8770/cameras"
PROMPT_PATH = ROOT / "prompts" / "yam_exec_system.md"
SCHEMA_PATH = ROOT / "schemas" / "yam_exec_action.schema.json"
PIXEL_LOCATOR_SCHEMA_PATH = ROOT / "schemas" / "yam_pixel_locator.schema.json"
ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
GRIPPER_LIMITS = (0.01, 0.59)


def _task_target_phrase(task: str) -> str:
    normalized = " ".join(task.strip().split())
    lower = normalized.lower()
    prefixes = (
        "pick up the ",
        "pickup the ",
        "pick up ",
        "grab the ",
        "grab ",
        "touch the ",
        "touch ",
        "move to the ",
        "move to ",
    )
    for prefix in prefixes:
        if lower.startswith(prefix) and len(normalized) > len(prefix):
            return normalized[len(prefix) :]
    return normalized or "the task target"


def _run(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)


def _step_label(step: int | str) -> str:
    if isinstance(step, int):
        return f"step_{step:03d}"
    return str(step)


def _rpc(ws, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    ws.send(json.dumps({"id": method, "method": method, "params": params or {}}))
    response = json.loads(ws.recv())
    if not response.get("ok"):
        raise RuntimeError(response.get("error", f"{method} failed"))
    return response


def _robot_status(robot_url: str) -> dict[str, Any]:
    with connect(robot_url, open_timeout=10, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        return _rpc(ws, "get_status")["result"]


def _assert_motion_ready(status: dict[str, Any]) -> None:
    if status.get("zero_gravity_mode"):
        raise RuntimeError("refusing motion: zero_gravity_mode is enabled")
    arms = status.get("arms") or {}
    for arm_name in ("left", "right"):
        arm = arms.get(arm_name) or {}
        if not arm.get("connected"):
            raise RuntimeError(f"refusing motion: {arm_name} arm is not connected")


def _collect_snapshot(args: argparse.Namespace, run_dir: Path, step: int | str) -> dict[str, Any]:
    label = _step_label(step)
    output_dir = run_dir / "snapshots"
    cmd = [
        sys.executable,
        "scripts/yam_codex_snapshot.py",
        "--robot-url",
        args.robot_url,
        "--camera-url",
        args.camera_url,
        "--output-dir",
        str(output_dir),
        "--depth-grid",
        str(args.depth_grid),
        "--depth-window",
        str(args.depth_window),
    ]
    proc = _run(cmd, timeout=args.snapshot_timeout)
    (run_dir / f"{label}_snapshot.stdout").write_text(proc.stdout)
    (run_dir / f"{label}_snapshot.stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"snapshot failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


def _compact_status(status: dict[str, Any]) -> dict[str, Any]:
    arms = status.get("arms") or {}
    return {
        "zero_gravity_mode": status.get("zero_gravity_mode"),
        "bridge": status.get("bridge"),
        "arms_connected": {
            "left": bool((arms.get("left") or {}).get("connected")),
            "right": bool((arms.get("right") or {}).get("connected")),
        },
        "right_arm": arms.get("right"),
    }


def _compact_right_joint_summary(joint_summary: dict[str, Any]) -> dict[str, Any]:
    arms = joint_summary.get("arms") or {}
    right = arms.get("right") or {}
    return {
        "order_note": "Underlying robot API remains 14D left[0:7]+right[0:7], but this policy controls only right[0:7].",
        "right": right,
    }


def _parse_pixel(value: str | list[int] | tuple[int, int] | None) -> list[int] | None:
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"expected pixel as [u, v]; got {value!r}")
        return [int(value[0]), int(value[1])]
    parts = [int(float(part.strip())) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected pixel as u,v; got {value!r}")
    return parts


def _pixel_point(snapshot_dir: Path, camera_id: str, pixel: list[int] | None, window: int) -> dict[str, Any] | None:
    if pixel is None:
        return None
    payload_path = snapshot_dir / "camera_payload.json"
    try:
        frame = _load_frame(payload_path, camera_id)
        return point_from_frame_payload(frame, int(pixel[0]), int(pixel[1]), window)
    except Exception as exc:
        return {"pixel": pixel, "error": str(exc)}


def _top_depth_servo(args: argparse.Namespace, snapshot: dict[str, Any], compact: dict[str, Any]) -> dict[str, Any]:
    snapshot_dir = Path(str(snapshot.get("snapshot_dir", "")))
    cameras = (snapshot.get("perception") or {}).get("cameras", [])
    depth_cameras = [camera for camera in cameras if camera.get("has_depth")]
    preferred = next((camera for camera in depth_cameras if str(camera.get("camera_id")) == args.primary_depth_camera), None)
    primary = preferred or next((camera for camera in depth_cameras if str(camera.get("camera_id")) == "top"), None)
    primary = primary or (depth_cameras[0] if depth_cameras else None)
    if not primary:
        return {
            "enabled": False,
            "reason": "no depth-producing camera found",
            "preferred_camera_id": args.primary_depth_camera,
        }

    camera_id = str(primary.get("camera_id"))
    target_pixel = _parse_pixel(getattr(args, "_located_target_pixel", None) or args.target_pixel)
    gripper_pixel = _parse_pixel(getattr(args, "_located_gripper_pixel", None) or args.gripper_pixel)
    target_point = _pixel_point(snapshot_dir, camera_id, target_pixel, args.depth_window)
    gripper_point = _pixel_point(snapshot_dir, camera_id, gripper_pixel, args.depth_window)
    pixel_error = None
    camera_delta_m = None
    if target_pixel is not None and gripper_pixel is not None:
        pixel_error = [int(target_pixel[0] - gripper_pixel[0]), int(target_pixel[1] - gripper_pixel[1])]
    if (
        target_point
        and gripper_point
        and target_point.get("point_camera_m") is not None
        and gripper_point.get("point_camera_m") is not None
    ):
        camera_delta_m = (
            np.asarray(target_point["point_camera_m"], dtype=float)
            - np.asarray(gripper_point["point_camera_m"], dtype=float)
        ).tolist()

    return {
        "enabled": True,
        "camera_id": camera_id,
        "image_size": primary.get("image_size"),
        "has_metric_depth": True,
        "center_depth": primary.get("center_depth"),
        "depth_quality": primary.get("depth_quality"),
        "target_pixel": target_pixel,
        "gripper_pixel": gripper_pixel,
        "pixel_error_target_minus_gripper": pixel_error,
        "target_point_camera_m": target_point,
        "gripper_point_camera_m": gripper_point,
        "delta_target_minus_gripper_camera_m": camera_delta_m,
        "world_delta_available": False,
        "control_note": (
            "Use this top-depth camera as the primary table geometry source. "
            "Pixel/camera-frame deltas are reliable only in this camera frame until a top-depth-to-robot calibration is supplied."
        ),
    }


def _target_distance(compact: dict[str, Any]) -> dict[str, Any]:
    servo = ((compact.get("perception") or {}).get("top_depth_servo") or {})
    locator = servo.get("auto_pixel_locator") or {}
    pixel_error = servo.get("pixel_error_target_minus_gripper")
    camera_delta = servo.get("delta_target_minus_gripper_camera_m")
    distance_px = None
    distance_camera_m = None
    if pixel_error is not None:
        distance_px = float(np.linalg.norm(np.asarray(pixel_error[:2], dtype=float)))
    if camera_delta is not None:
        distance_camera_m = float(np.linalg.norm(np.asarray(camera_delta[:3], dtype=float)))
    primary = None
    primary_unit = None
    if distance_camera_m is not None:
        primary = distance_camera_m
        primary_unit = "m_camera_frame"
    elif distance_px is not None:
        primary = distance_px
        primary_unit = "px"
    return {
        "target_pixel": servo.get("target_pixel"),
        "gripper_pixel": servo.get("gripper_pixel"),
        "pixel_error_target_minus_gripper": pixel_error,
        "distance_px": distance_px,
        "distance_camera_m": distance_camera_m,
        "primary_distance": primary,
        "primary_unit": primary_unit,
        "locator_confidence": locator.get("confidence"),
        "locator_reason": locator.get("reason"),
        "valid": primary is not None,
        "note": "Distance is target-to-right-gripper in the top-depth camera frame when depth is available, otherwise image pixels.",
    }


def _add_target_distance(compact: dict[str, Any]) -> None:
    compact["perception"]["target_distance"] = _target_distance(compact)


def _progress_check(
    before: dict[str, Any],
    after: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    before_dist = ((before.get("perception") or {}).get("target_distance") or {})
    after_dist = ((after.get("perception") or {}).get("target_distance") or {})
    unit = None
    before_value = None
    after_value = None
    if before_dist.get("distance_camera_m") is not None and after_dist.get("distance_camera_m") is not None:
        unit = "m_camera_frame"
        before_value = before_dist.get("distance_camera_m")
        after_value = after_dist.get("distance_camera_m")
    elif before_dist.get("distance_px") is not None and after_dist.get("distance_px") is not None:
        unit = "px"
        before_value = before_dist.get("distance_px")
        after_value = after_dist.get("distance_px")
    if before_value is None or after_value is None or unit is None:
        return {
            "valid": False,
            "closer": None,
            "reason": "target distance was not measurable in the same unit before and after the action",
            "before": before_dist,
            "after": after_dist,
        }
    improvement = float(before_value) - float(after_value)
    tolerance = args.progress_tolerance_m if unit == "m_camera_frame" else args.progress_tolerance_px
    closer = improvement > float(tolerance)
    return {
        "valid": True,
        "closer": closer,
        "unit": unit,
        "before_distance": before_value,
        "after_distance": after_value,
        "improvement": improvement,
        "tolerance": tolerance,
        "reason": "distance decreased" if closer else "distance did not decrease enough",
        "before": before_dist,
        "after": after_dist,
    }


def _compact_snapshot(snapshot: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    perception = snapshot.get("perception") or {}
    robot = snapshot.get("robot") or {}
    status = robot.get("status") or {}
    joint_summary = robot.get("joint_summary") or {}
    compact: dict[str, Any] = {
        "schema": "yam_exec_observation.v1",
        "created_at": snapshot.get("created_at"),
        "snapshot_dir": snapshot.get("snapshot_dir"),
        "contact_sheet_path": None,
        "robot": {
            "status": _compact_status(status),
            "joint_summary": _compact_right_joint_summary(joint_summary),
        },
        "perception": {
            "cameras": [],
            "depth": None,
            "top_depth_servo": None,
        },
    }

    snapshot_dir = Path(str(snapshot.get("snapshot_dir", "")))
    contact = perception.get("contact_sheet_path")
    if contact:
        compact["contact_sheet_path"] = str((snapshot_dir / contact).resolve())

    for camera in perception.get("cameras", []):
        entry = {
            "camera_id": camera.get("camera_id"),
            "has_depth": camera.get("has_depth"),
            "image_size": camera.get("image_size"),
            "image_path": str((snapshot_dir / str(camera.get("image_path"))).resolve())
            if camera.get("image_path")
            else None,
            "center_depth": camera.get("center_depth"),
            "depth_quality": camera.get("depth_quality"),
        }
        compact["perception"]["cameras"].append(entry)
        if camera.get("has_depth") and (
            compact["perception"]["depth"] is None or str(camera.get("camera_id")) == args.primary_depth_camera
        ):
            compact["perception"]["depth"] = {
                "camera_id": camera.get("camera_id"),
                "center_depth": camera.get("center_depth"),
                "depth_quality": camera.get("depth_quality"),
                "depth_grid_samples": camera.get("depth_grid_samples", []),
            }
    compact["perception"]["top_depth_servo"] = _top_depth_servo(args, snapshot, compact)
    _add_target_distance(compact)
    return compact


def _depth_m(compact: dict[str, Any]) -> float | None:
    depth = ((compact.get("perception") or {}).get("depth") or {}).get("center_depth") or {}
    value = depth.get("depth_m")
    if value is None:
        return None
    return float(value)


def _load_prior_memory(memory_dir: Path, task: str, limit: int = 12) -> list[dict[str, Any]]:
    path = memory_dir / "trajectory_memory.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    task_key = task.lower()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        result = entry.get("result") or {}
        if not result.get("executed"):
            continue
        entry_task = str(entry.get("task", "")).lower()
        if task_key and task_key not in entry_task and entry_task not in task_key:
            continue
        action = entry.get("action") or {}
        entries.append(
            {
                "task": entry.get("task"),
                "step": entry.get("step"),
                "tool": action.get("tool"),
                "confidence": action.get("confidence"),
                "delta_m": action.get("delta_m"),
                "frame": action.get("frame"),
                "depth_center_m": (entry.get("observation") or {}).get("depth_center_m"),
                "executed": result.get("executed"),
                "ok": result.get("ok"),
                "reason": str(action.get("reason", ""))[:180],
            }
        )
    return entries[-limit:]


def _summarize_memory(
    transcript: list[dict[str, Any]],
    compact: dict[str, Any],
    prior_memory: list[dict[str, Any]],
    max_recent: int = 8,
) -> dict[str, Any]:
    tool_counts: dict[str, int] = {}
    recent: list[dict[str, Any]] = []
    cartesian_moves: list[dict[str, Any]] = []

    for record in transcript:
        action = record.get("action") or {}
        args = action.get("args") or {}
        tool = str(action.get("tool"))
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
        if tool == "move_right_cartesian":
            cartesian_moves.append(record)
        recent.append(
            {
                "step": record.get("step"),
                "tool": tool,
                "done": bool(action.get("done")),
                "confidence": action.get("confidence"),
                "delta_m": [args.get("dx"), args.get("dy"), args.get("dz")] if tool == "move_right_cartesian" else None,
                "frame": args.get("frame"),
                "wrist_delta_deg": [args.get("pitch_deg"), args.get("yaw_deg"), args.get("roll_deg")]
                if tool == "aim_right_wrist"
                else None,
                "executed": bool((record.get("result") or {}).get("executed", not (record.get("result") or {}).get("dry_run", True))),
                "target_distance_before": ((record.get("observation") or {}).get("target_distance") or {}).get("primary_distance"),
                "target_distance_after": (((record.get("post_observation") or {}).get("target_distance") or {}).get("primary_distance")),
                "progress_check": record.get("progress_check"),
                "reason": str(action.get("reason", ""))[:220],
            }
        )

    current_depth = _depth_m(compact)
    previous_depth = None
    for record in reversed(transcript):
        previous_depth = ((record.get("observation") or {}).get("depth_center_m"))
        if previous_depth is not None:
            break

    depth_trend = None
    if current_depth is not None and previous_depth is not None:
        delta = current_depth - float(previous_depth)
        depth_trend = {
            "previous_center_m": previous_depth,
            "current_center_m": current_depth,
            "delta_m": delta,
            "interpretation": "center depth increased" if delta > 0 else "center depth decreased" if delta < 0 else "center depth unchanged",
        }

    return {
        "schema": "yam_exec_trajectory_memory.v1",
        "steps_completed": len(transcript),
        "tool_counts": tool_counts,
        "last_action": recent[-1] if recent else None,
        "recent_actions": recent[-max_recent:],
        "retrieved_prior_trajectories": prior_memory,
        "cartesian_move_count": len(cartesian_moves),
        "depth_trend": depth_trend,
        "guidance": [
            "Every physical step must be followed by a distance check. Prefer actions predicted to reduce target_distance.primary_distance.",
            "If the last progress_check says closer=false, treat the prior movement direction as suspect and correct direction before continuing.",
            "Do not repeat moves that failed to improve visibility or target centering.",
            "Prefer a decisive Cartesian correction when recent actions are only wrist/status-like probing.",
            "Use the latest image as authority, but use recent_actions to avoid oscillating signs.",
        ],
    }


def _memory_record(
    task: str,
    run_dir: Path,
    step: int,
    compact: dict[str, Any],
    action: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    args = action.get("args") or {}
    return {
        "step": step,
        "task": task,
        "run_dir": str(run_dir),
        "created_at": compact.get("created_at"),
        "observation": {
            "snapshot_dir": compact.get("snapshot_dir"),
            "contact_sheet_path": compact.get("contact_sheet_path"),
            "depth_center_m": _depth_m(compact),
            "target_distance": ((compact.get("perception") or {}).get("target_distance")),
            "arms_connected": ((compact.get("robot") or {}).get("status") or {}).get("arms_connected"),
            "zero_gravity_mode": ((compact.get("robot") or {}).get("status") or {}).get("zero_gravity_mode"),
        },
        "action": {
            "tool": action.get("tool"),
            "confidence": action.get("confidence"),
            "done": action.get("done"),
            "delta_m": [args.get("dx"), args.get("dy"), args.get("dz")]
            if action.get("tool") == "move_right_cartesian"
            else None,
            "frame": args.get("frame"),
            "wrist_delta_deg": [args.get("pitch_deg"), args.get("yaw_deg"), args.get("roll_deg")]
            if action.get("tool") == "aim_right_wrist"
            else None,
            "reason": action.get("reason"),
        },
        "result": {
            "ok": result.get("ok"),
            "executed": result.get("executed"),
            "dry_run": result.get("dry_run"),
            "segment_count": result.get("segment_count"),
            "command_joint_pos": result.get("command_joint_pos"),
        },
    }


def _operator_correction(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "note": args.operator_note,
        "runtime_notes": getattr(args, "_runtime_operator_notes", []),
        "avoid_delta_signs": args.avoid_delta_signs,
        "prefer_delta_signs": args.prefer_delta_signs,
        "instruction": (
            "Operator correction overrides inferred visual direction. Avoid listed "
            "delta sign patterns unless the latest observation strongly contradicts them. "
            "Runtime notes come from post-action target-distance checks."
        ),
    }


def _locate_top_depth_pixels(
    args: argparse.Namespace,
    run_dir: Path,
    step: int | str,
    compact: dict[str, Any],
) -> dict[str, Any] | None:
    if not args.auto_locate_pixels:
        return None
    if args.target_pixel and args.gripper_pixel:
        return None

    servo = ((compact.get("perception") or {}).get("top_depth_servo") or {})
    if not servo.get("enabled"):
        return {"error": "top_depth_servo is not enabled", "confidence": 0.0}
    camera_id = str(servo.get("camera_id"))
    cameras = (compact.get("perception") or {}).get("cameras") or []
    camera = next((item for item in cameras if str(item.get("camera_id")) == camera_id), None)
    image_path = camera.get("image_path") if camera else None
    if not image_path:
        return {"error": f"no image path for depth camera {camera_id!r}", "confidence": 0.0}

    prompt = "\n".join(
        [
            "Locate two pixels in this top-depth/table camera image.",
            f"User task: {args.task}",
            f"Task target object or contact point: {_task_target_phrase(args.task)}",
            "Return pixel coordinates in this image's native coordinate system.",
            "target_u,target_v: the center of the task target object/contact point named above. Do not locate a toaster lever unless the user task is specifically about a toaster lever.",
            "gripper_u,gripper_v: the right gripper/end-effector tip closest to that target.",
            "If one point is not visible, set its u/v to null and lower confidence.",
            "Return only JSON matching the schema.",
        ]
    )
    label = _step_label(step)
    output_path = run_dir / f"{label}_pixel_locator.json"
    cmd = [
        "codex",
        "exec",
        "--cd",
        str(ROOT),
        "--sandbox",
        "read-only",
        "--output-schema",
        str(args.pixel_locator_schema),
        "--output-last-message",
        str(output_path),
        "--model",
        args.model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
        "--image",
        str(image_path),
        "-",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=args.pixel_locator_timeout,
        check=False,
    )
    (run_dir / f"{label}_pixel_locator.stdout").write_text(proc.stdout)
    (run_dir / f"{label}_pixel_locator.stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        return {
            "error": proc.stderr.strip() or proc.stdout.strip() or "pixel locator failed",
            "confidence": 0.0,
        }
    try:
        located = json.loads(output_path.read_text())
    except Exception as exc:
        return {"error": f"failed to parse pixel locator output: {exc}", "confidence": 0.0}

    target = None
    gripper = None
    if located.get("target_u") is not None and located.get("target_v") is not None:
        target = [int(located["target_u"]), int(located["target_v"])]
    if located.get("gripper_u") is not None and located.get("gripper_v") is not None:
        gripper = [int(located["gripper_u"]), int(located["gripper_v"])]
    if target is not None:
        args._located_target_pixel = target
    if gripper is not None:
        args._located_gripper_pixel = gripper
    located["target_pixel"] = target
    located["gripper_pixel"] = gripper
    located["camera_id"] = camera_id
    located["image_path"] = image_path
    return located


def _observe_with_locator(
    args: argparse.Namespace,
    run_dir: Path,
    step: int | str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    args._located_target_pixel = None
    args._located_gripper_pixel = None
    snapshot = _collect_snapshot(args, run_dir, step)
    compact = _compact_snapshot(snapshot, args)
    pixel_locator = _locate_top_depth_pixels(args, run_dir, step, compact)
    if pixel_locator:
        compact = _compact_snapshot(snapshot, args)
        compact["perception"]["top_depth_servo"]["auto_pixel_locator"] = pixel_locator
        _add_target_distance(compact)
    return snapshot, compact, pixel_locator


def _build_decision_prompt(
    system_prompt: str,
    task: str,
    compact: dict[str, Any],
    memory: dict[str, Any],
    operator_correction: dict[str, Any],
    step: int,
) -> str:
    return "\n\n".join(
        [
            system_prompt,
            f"User task: {task}",
            f"Step index: {step}",
            "Compact trajectory memory JSON:",
            json.dumps(memory, indent=2, sort_keys=True),
            "Operator correction JSON:",
            json.dumps(operator_correction, indent=2, sort_keys=True),
            "Latest compact observation JSON:",
            json.dumps(compact, indent=2, sort_keys=True),
            "Return exactly one JSON action.",
        ]
    )


def _codex_decide(
    args: argparse.Namespace,
    run_dir: Path,
    step: int,
    compact: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = Path(args.system_prompt).read_text()
    prompt = _build_decision_prompt(system_prompt, args.task, compact, memory, _operator_correction(args), step)
    prompt_path = run_dir / f"step_{step:03d}_prompt.md"
    output_path = run_dir / f"step_{step:03d}_decision.json"
    prompt_path.write_text(prompt)

    cmd = [
        "codex",
        "exec",
        "--cd",
        str(ROOT),
        "--sandbox",
        "read-only",
        "--output-schema",
        str(args.output_schema),
        "--output-last-message",
        str(output_path),
        "--model",
        args.model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
    ]
    image_path = compact.get("contact_sheet_path")
    if image_path:
        cmd.extend(["--image", str(image_path)])
    cmd.append("-")

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=args.codex_timeout,
        check=False,
    )
    (run_dir / f"step_{step:03d}_codex.stdout").write_text(proc.stdout)
    (run_dir / f"step_{step:03d}_codex.stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"codex exec failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(output_path.read_text())


def _num(action: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = action.get("args", {}).get(name, default)
    if value is None:
        return float(default)
    return float(value)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _execute_cartesian(args: argparse.Namespace, action: dict[str, Any], arm: str, run_dir: Path, step: int) -> dict[str, Any]:
    max_abs_delta = float(args.max_cartesian_delta)
    dx = _clamp(_num(action, "dx"), -max_abs_delta, max_abs_delta)
    dy = _clamp(_num(action, "dy"), -max_abs_delta, max_abs_delta)
    dz = _clamp(_num(action, "dz"), -max_abs_delta, max_abs_delta)
    requested_steps = action.get("args", {}).get("steps")
    requested_hz = action.get("args", {}).get("hz")
    steps = int(_clamp(float(args.trajectory_steps), int(args.min_trajectory_steps), 80))
    hz = _clamp(float(args.hz), float(args.min_hz), 160)
    cmd = [
        sys.executable,
        "scripts/yam_cartesian_control.py",
        "--control-url",
        args.robot_url,
        "--arm",
        arm,
        "--frame",
        str(action.get("args", {}).get("frame") or "world"),
        f"--delta={dx},{dy},{dz}",
        "--ik-preset",
        str(action.get("args", {}).get("ik_preset") or args.ik_preset),
        "--max-iters",
        str(args.max_ik_iters),
        "--pos-threshold",
        str(args.pos_threshold),
        "--ori-threshold",
        str(args.ori_threshold),
        "--no-stop-on-ik-failure",
        "--max-segment-m",
        str(args.max_segment_m),
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--steps",
        str(steps),
        "--hz",
        str(hz),
    ]
    if args.allow_nonconverged_segments:
        cmd.append("--allow-nonconverged-segments")
    if args.execute:
        cmd.append("--execute")
    proc = _run(cmd, timeout=args.motion_timeout)
    (run_dir / f"step_{step:03d}_motion.stdout").write_text(proc.stdout)
    (run_dir / f"step_{step:03d}_motion.stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"cartesian action failed: {proc.stderr.strip() or proc.stdout.strip()}")
    result = json.loads(proc.stdout)
    result["dry_run"] = not args.execute
    result["executor_timing"] = {
        "steps": steps,
        "hz": hz,
        "requested_steps_ignored": requested_steps,
        "requested_hz_ignored": requested_hz,
    }
    return result


def _execute_wrist_aim(args: argparse.Namespace, action: dict[str, Any], arm: str, run_dir: Path, step: int) -> dict[str, Any]:
    max_deg = float(args.max_wrist_deg)
    pitch = _clamp(_num(action, "pitch_deg"), -max_deg, max_deg)
    yaw = _clamp(_num(action, "yaw_deg"), -max_deg, max_deg)
    roll = _clamp(_num(action, "roll_deg"), -max_deg, max_deg)
    cmd = [
        sys.executable,
        "scripts/yam_cartesian_control.py",
        "--control-url",
        args.robot_url,
        "--arm",
        arm,
        "--frame",
        "local",
        "--delta=0,0,0",
        "--camera-pitch-deg",
        str(pitch),
        "--camera-yaw-deg",
        str(yaw),
        "--camera-roll-deg",
        str(roll),
        "--wrist-only",
        "--ik-preset",
        str(action.get("args", {}).get("ik_preset") or args.ik_preset),
        "--max-iters",
        str(args.max_ik_iters),
        "--pos-threshold",
        str(args.pos_threshold),
        "--ori-threshold",
        str(args.ori_threshold),
        "--no-stop-on-ik-failure",
        "--max-segment-rad",
        str(args.max_segment_rad),
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--steps",
        str(args.trajectory_steps),
        "--hz",
        str(args.hz),
    ]
    if args.allow_nonconverged_segments:
        cmd.append("--allow-nonconverged-segments")
    if args.execute:
        cmd.append("--execute")
    proc = _run(cmd, timeout=args.motion_timeout)
    (run_dir / f"step_{step:03d}_wrist.stdout").write_text(proc.stdout)
    (run_dir / f"step_{step:03d}_wrist.stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"wrist action failed: {proc.stderr.strip() or proc.stdout.strip()}")
    result = json.loads(proc.stdout)
    result["dry_run"] = not args.execute
    result["executor_timing"] = {
        "steps": int(args.trajectory_steps),
        "hz": float(args.hz),
    }
    return result


def _set_gripper(args: argparse.Namespace, action: dict[str, Any], arm: str) -> dict[str, Any]:
    value = _clamp(_num(action, "value", 0.3), *GRIPPER_LIMITS)
    with connect(args.robot_url, open_timeout=10, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        status = _rpc(ws, "get_status")["result"]
        _assert_motion_ready(status)
        q = np.asarray(_rpc(ws, "get_joint_pos")["result"], dtype=float)
        if q.shape != (14,):
            raise RuntimeError(f"expected 14D joint vector, got {q.shape}")
        q[ARM_SLICES[arm]][6] = value
        if args.execute:
            _rpc(ws, "command_joint_pos", {"joint_pos": q.tolist()})
    return {"arm": arm, "value": value, "dry_run": not args.execute}


def _hard_stop(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        return {"dry_run": True, "action": "hard_stop"}
    with connect(args.robot_url, open_timeout=10, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        response = _rpc(ws, "set_zero_gravity_mode", {"enabled": True})
    return {"dry_run": False, "action": "set_zero_gravity_mode", "response": response.get("result")}


def _execute_action(args: argparse.Namespace, action: dict[str, Any], run_dir: Path, step: int) -> dict[str, Any]:
    tool = action["tool"]
    if tool == "hard_stop":
        return _hard_stop(args)
    status = _robot_status(args.robot_url)
    _assert_motion_ready(status)
    if tool == "move_right_cartesian":
        return _execute_cartesian(args, action, "right", run_dir, step)
    if tool == "aim_right_wrist":
        return _execute_wrist_aim(args, action, "right", run_dir, step)
    if tool == "set_gripper_right":
        return _set_gripper(args, action, "right")
    raise RuntimeError(f"unsupported tool: {tool}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a constrained codex-exec YAM policy loop.")
    parser.add_argument("task", help="Robot task, e.g. 'touch the toaster lever'.")
    parser.add_argument("--robot-url", default=os.environ.get("YAM_ROBOT_URL", DEFAULT_ROBOT_URL))
    parser.add_argument("--camera-url", default=os.environ.get("YAM_CAMERA_WS_URL", DEFAULT_CAMERA_URL))
    parser.add_argument("--model", default=os.environ.get("YAM_EXEC_MODEL", "gpt-5.5"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("YAM_EXEC_REASONING_EFFORT", "low"))
    parser.add_argument("--operator-note", default=os.environ.get("YAM_OPERATOR_NOTE", ""))
    parser.add_argument("--avoid-delta-signs", action="append", default=[])
    parser.add_argument("--prefer-delta-signs", action="append", default=[])
    parser.add_argument("--primary-depth-camera", default=os.environ.get("YAM_PRIMARY_DEPTH_CAMERA", "top"))
    parser.add_argument("--target-pixel", help="Optional top-depth target pixel as u,v.")
    parser.add_argument("--gripper-pixel", help="Optional top-depth gripper pixel as u,v.")
    parser.add_argument("--auto-locate-pixels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pixel-locator-schema", type=Path, default=PIXEL_LOCATOR_SCHEMA_PATH)
    parser.add_argument("--system-prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--output-schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "logs" / "yam-exec-runs")
    parser.add_argument("--memory-dir", type=Path, default=ROOT / "logs" / "yam-exec-memory")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="Actually execute validated robot actions. Default is dry-run.")
    parser.add_argument("--depth-grid", type=int, default=5)
    parser.add_argument("--depth-window", type=int, default=7)
    parser.add_argument("--max-cartesian-delta", type=float, default=0.14)
    parser.add_argument("--max-wrist-deg", type=float, default=15.0)
    parser.add_argument("--max-segment-m", type=float, default=0.05)
    parser.add_argument("--max-segment-rad", type=float, default=0.26)
    parser.add_argument("--max-joint-delta", type=float, default=0.14)
    parser.add_argument("--trajectory-steps", type=int, default=18)
    parser.add_argument("--min-trajectory-steps", type=int, default=8)
    parser.add_argument("--hz", type=float, default=140.0)
    parser.add_argument("--min-hz", type=float, default=80.0)
    parser.add_argument("--ik-preset", choices=["strict", "live", "loose"], default="loose")
    parser.add_argument("--max-ik-iters", type=int, default=80)
    parser.add_argument("--pos-threshold", type=float, default=999.0)
    parser.add_argument("--ori-threshold", type=float, default=999.0)
    parser.add_argument(
        "--progress-tolerance-m",
        type=float,
        default=0.01,
        help="Minimum target-distance improvement in camera-frame meters required for a Cartesian step to count as progress.",
    )
    parser.add_argument(
        "--progress-tolerance-px",
        type=float,
        default=8.0,
        help="Minimum target-distance improvement in pixels required when metric distance is unavailable.",
    )
    parser.add_argument(
        "--stop-on-failed-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the run after any executed Cartesian move that does not reduce target distance.",
    )
    parser.add_argument(
        "--allow-nonconverged-segments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute finite bounded IK segment commands even if IK misses the convergence threshold.",
    )
    parser.add_argument("--snapshot-timeout", type=float, default=30.0)
    parser.add_argument("--pixel-locator-timeout", type=float, default=90.0)
    parser.add_argument("--codex-timeout", type=float, default=120.0)
    parser.add_argument("--motion-timeout", type=float, default=45.0)
    args = parser.parse_args()

    run_dir = args.run_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    args.memory_dir.mkdir(parents=True, exist_ok=True)
    transcript: list[dict[str, Any]] = []
    memory_jsonl = run_dir / "trajectory_memory.jsonl"
    shared_memory_jsonl = args.memory_dir / "trajectory_memory.jsonl"
    memory_summary_path = run_dir / "memory_summary.json"
    args._runtime_operator_notes = []

    for step in range(max(1, args.max_steps)):
        snapshot, compact, pixel_locator = _observe_with_locator(args, run_dir, step)
        prior_memory = _load_prior_memory(args.memory_dir, args.task)
        memory = _summarize_memory(transcript, compact, prior_memory)
        memory_summary_path.write_text(json.dumps(memory, indent=2, sort_keys=True) + "\n")
        (run_dir / f"step_{step:03d}_compact_observation.json").write_text(
            json.dumps(compact, indent=2, sort_keys=True) + "\n"
        )
        (run_dir / f"step_{step:03d}_memory_summary.json").write_text(
            json.dumps(memory, indent=2, sort_keys=True) + "\n"
        )
        action = _codex_decide(args, run_dir, step, compact, memory)
        result = _execute_action(args, action, run_dir, step)
        post_snapshot = None
        post_compact = None
        progress = None
        if action.get("tool") != "hard_stop" and bool(result.get("executed")):
            post_label = f"step_{step:03d}_post"
            post_snapshot, post_compact, _ = _observe_with_locator(args, run_dir, post_label)
            progress = _progress_check(compact, post_compact, args)
            (run_dir / f"{post_label}_compact_observation.json").write_text(
                json.dumps(post_compact, indent=2, sort_keys=True) + "\n"
            )
            (run_dir / f"{post_label}_progress_check.json").write_text(
                json.dumps(progress, indent=2, sort_keys=True) + "\n"
            )
            if (
                action.get("tool") == "move_right_cartesian"
                and progress.get("valid")
                and not progress.get("closer")
            ):
                move_args = action.get("args") or {}
                note = (
                    "Last Cartesian move failed the target-distance check: "
                    f"delta={[move_args.get('dx'), move_args.get('dy'), move_args.get('dz')]} "
                    f"changed distance from {progress.get('before_distance')} to "
                    f"{progress.get('after_distance')} {progress.get('unit')} "
                    f"(improvement {progress.get('improvement')}). Correct direction before any further approach."
                )
                args._runtime_operator_notes.append(note)
                args._runtime_operator_notes = args._runtime_operator_notes[-6:]
                if args.stop_on_failed_progress:
                    result["stopped_after_failed_progress"] = True
        record = {
            "step": step,
            "observation": {
                "snapshot_dir": compact.get("snapshot_dir"),
                "contact_sheet_path": compact.get("contact_sheet_path"),
                "depth_center_m": _depth_m(compact),
                "target_distance": ((compact.get("perception") or {}).get("target_distance")),
            },
            "post_observation": {
                "snapshot_dir": post_compact.get("snapshot_dir") if post_compact else None,
                "contact_sheet_path": post_compact.get("contact_sheet_path") if post_compact else None,
                "depth_center_m": _depth_m(post_compact) if post_compact else None,
                "target_distance": ((post_compact.get("perception") or {}).get("target_distance")) if post_compact else None,
            },
            "progress_check": progress,
            "action": action,
            "result": result,
        }
        transcript.append(record)
        memory_entry = _memory_record(args.task, run_dir, step, compact, action, result)
        with memory_jsonl.open("a") as f:
            f.write(json.dumps(memory_entry, sort_keys=True) + "\n")
        with shared_memory_jsonl.open("a") as f:
            f.write(json.dumps(memory_entry, sort_keys=True) + "\n")
        memory_summary_path.write_text(
            json.dumps(_summarize_memory(transcript, compact, prior_memory), indent=2, sort_keys=True) + "\n"
        )
        (run_dir / "transcript.json").write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        if action.get("done") or action.get("tool") == "hard_stop":
            break
        if bool((result or {}).get("stopped_after_failed_progress")):
            break

    print(json.dumps({"run_dir": str(run_dir), "execute": bool(args.execute), "steps": len(transcript)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
