#!/usr/bin/env python3
"""Hybrid Modal + local verification loop for YAM tasks.

Modal proposes actions. The local loop owns the robot, validates/model-debugs
the response, executes commands with gripper clamping, saves camera frames, and can apply
small task-specific corrections when visual progress stalls.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local_modal_robot_bridge import (  # noqa: E402
    STOP_FILE,
    _clip_command_for_hardware,
    _extract_bimanual_actions,
    _extract_single_arm_actions,
    _install_can_patch,
    _safety_block_reason,
    _safety_report,
    _safety_report_pair,
)
from http_camera_fetch import fetch_rgb_from_camera_url  # noqa: E402
from opencv_util import video_capture  # noqa: E402
from orbbec_wrist import OrbbecColorPipeline  # noqa: E402


STOP_REQUESTED = False


def _request_stop(_signum=None, _frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Trace:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.path = root / "trace.jsonl"
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **fields) -> None:
        record = {"t": time.time(), "event": event, **fields}
        line = json.dumps(record, default=_jsonable)
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _fetch_rgb(args: argparse.Namespace, camera_url: str) -> np.ndarray:
    return fetch_rgb_from_camera_url(
        camera_url.strip(),
        timeout=5.0,
        session=getattr(args, "_http_session", None),
        ws_registry=getattr(args, "_camera_ws_registry", None),
    )


def _resize_rgb_to_shape(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize RGB to (H, W) = target_hw."""

    th, tw = target_hw
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape[0] == th and rgb.shape[1] == tw:
        return rgb
    return cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _raw_wrist_rgb(
    args: argparse.Namespace,
    right_cap: cv2.VideoCapture | None,
    orbbec_right: OrbbecColorPipeline | None,
) -> np.ndarray:
    """Wrist feed for Molmo ``right`` — Orbbec, OpenCV index, or HTTP ``--right-camera-url`` only."""

    if orbbec_right is not None:
        return orbbec_right.read_rgb()
    if right_cap is not None:
        ok, bgr = right_cap.read()
        if not ok or bgr is None:
            raise RuntimeError("right/wrist camera read failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ru = getattr(args, "right_camera_url", None) or ""
    if ru.strip():
        return _fetch_rgb(args, ru)
    raise RuntimeError(
        "Molmo 'right' needs exactly one wrist source: --right-orbbec | --right-camera-index | --right-camera-url. "
        "--camera-url is only for overhead/top (HTTP)."
    )


def _top_rgb_for_policy(
    args: argparse.Namespace,
    top_cap: cv2.VideoCapture | None,
    wrist_rgb: np.ndarray,
    trace: Trace,
    iteration: int,
) -> np.ndarray:
    """Molmo ``top``: OpenCV ``--top-camera-index``, else HTTP ``--camera-url``, else black."""

    black = _black_rgb_same_shape(wrist_rgb)
    hw = (wrist_rgb.shape[0], wrist_rgb.shape[1])
    if top_cap is not None:
        try:
            return _read_webcam_rgb_resized(top_cap, hw)
        except RuntimeError as exc:
            trace.write("top_camera_read_failed", iteration=iteration, error=str(exc))
            return black
    try:
        raw_top = _fetch_rgb(args, args.camera_url)
        return _resize_rgb_to_shape(raw_top, hw)
    except Exception as exc:  # noqa: BLE001
        trace.write("top_camera_http_failed", iteration=iteration, error=repr(exc))
        return black


def _left_rgb_for_policy(
    args: argparse.Namespace,
    left_cap: cv2.VideoCapture | None,
    wrist_rgb: np.ndarray,
    trace: Trace,
    iteration: int,
) -> np.ndarray:
    """Molmo ``left``: OpenCV ``--left-camera-index``, else HTTP ``--left-camera-url``, else black."""

    black = _black_rgb_same_shape(wrist_rgb)
    hw = (wrist_rgb.shape[0], wrist_rgb.shape[1])
    if left_cap is not None:
        try:
            return _read_webcam_rgb_resized(left_cap, hw)
        except RuntimeError as exc:
            trace.write("left_camera_read_failed", iteration=iteration, error=str(exc))
            return black
    lu = getattr(args, "left_camera_url", None) or ""
    if lu.strip():
        try:
            raw_left = _fetch_rgb(args, lu)
            return _resize_rgb_to_shape(raw_left, hw)
        except Exception as exc:  # noqa: BLE001
            trace.write("left_camera_http_failed", iteration=iteration, error=repr(exc))
            return black
    return black


def _save_frame(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format="JPEG", quality=88)


def _encode_rgb(rgb: np.ndarray, quality: int = 85) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=max(1, min(95, int(quality))))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _black_rgb_same_shape(rgb: np.ndarray) -> np.ndarray:
    """Solid black RGB frame matching ``rgb`` resolution (H, W, 3), uint8."""

    h, w = np.asarray(rgb).shape[:2]
    return np.zeros((h, w, 3), dtype=np.uint8)


def _read_webcam_rgb_resized(cap: cv2.VideoCapture, target_hw: tuple[int, int]) -> np.ndarray:
    """Read one frame from an OpenCV capture and resize to (H, W) = ``target_hw``."""

    ok, bgr = cap.read()
    if not ok or bgr is None:
        raise RuntimeError("webcam read failed (device closed or no frame)")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    th, tw = target_hw
    if rgb.shape[0] != th or rgb.shape[1] != tw:
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def _image_features(rgb: np.ndarray) -> dict:
    image = np.asarray(rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)
    orange = (r > 120) & (g > 55) & (g < 185) & (b < 130) & ((r - b) > 45)
    dark = (r < 70) & (g < 70) & (b < 70)
    dark[: int(h * 0.02), :] = False
    dark[int(h * 0.62) :, :] = False

    def centroid(mask: np.ndarray) -> list[float] | None:
        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            return None
        return [float(xs.mean()), float(ys.mean())]

    orange_centroid = centroid(orange)
    jaw_centroid = centroid(dark)
    return {
        "width": w,
        "height": h,
        "orange_pixels": int(orange.sum()),
        "dark_pixels": int(dark.sum()),
        "orange_centroid": orange_centroid,
        "jaw_centroid": jaw_centroid,
        "x_error_px": None if orange_centroid is None or jaw_centroid is None else orange_centroid[0] - jaw_centroid[0],
        "y_error_px": None if orange_centroid is None or jaw_centroid is None else orange_centroid[1] - jaw_centroid[1],
    }


def _policy_request(args: argparse.Namespace, payload: dict) -> dict:
    session = getattr(args, "_http_session", None)
    post = session.post if session is not None else requests.post
    response = post(args.http_url, json=payload, timeout=args.http_timeout)
    try:
        body = response.json()
    except ValueError:
        response.raise_for_status()
        raise
    if response.status_code >= 400:
        return body
    return body


def _health_check(args: argparse.Namespace, trace: Trace) -> None:
    health_url = args.http_url.rsplit("/", 1)[0] + "/health"
    try:
        session = getattr(args, "_http_session", None)
        get = session.get if session is not None else requests.get
        response = get(health_url, timeout=10)
        trace.write("policy_health", policy_kind=args.policy_kind, url=health_url, status_code=response.status_code, body=response.json())
    except Exception as exc:  # noqa: BLE001
        trace.write("policy_health_failed", policy_kind=args.policy_kind, url=health_url, error=repr(exc))


def _warmup_modal_infer(args: argparse.Namespace, trace: Trace) -> None:
    """Run one full ``POST /infer`` before connecting to robots so cold-start GPU work happens once."""

    if args.policy_kind != "modal":
        return
    # Resolution similar to live camera JPEGs; black frames keep upload small vs real frames.
    tiny = Image.new("RGB", (640, 480), color=(0, 0, 0))
    buf = io.BytesIO()
    tiny.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    payload = {
        "type": "observation",
        "task": args.task,
        "state": [0.0] * 14,
        "state_format": "yam_bimanual_yam_14d",
        "images": {"top": b64, "left": b64, "right": b64},
        "num_steps": args.num_steps,
    }
    t0 = time.time()
    try:
        body = _policy_request(args, payload)
    except Exception as exc:
        trace.write("modal_warmup_infer", ok=False, error=repr(exc))
        raise
    elapsed_ms = (time.time() - t0) * 1000.0
    is_err = body.get("type") == "error"
    trace.write(
        "modal_warmup_infer",
        ok=not is_err,
        elapsed_ms=elapsed_ms,
        action_shape=body.get("action_shape"),
        source=body.get("source"),
        error=body.get("error"),
    )
    if is_err:
        raise RuntimeError(body.get("error", "warmup infer failed"))


def _validate_response(response: dict, args: argparse.Namespace) -> tuple[list[np.ndarray], dict]:
    action = np.asarray(response.get("action", []), dtype=float)
    arms = getattr(args, "arms", "bimanual")
    expected_source = "molmoact2_bimanual_yam" if args.policy_kind == "modal" else "yam_lerobot_act"
    diagnostics = {
        "type": response.get("type"),
        "source": response.get("source"),
        "input_source": response.get("input_source"),
        "repo_revision": response.get("repo_revision"),
        "norm_tag": response.get("norm_tag"),
        "execute_ok": response.get("execute_ok"),
        "state_shape": response.get("state_shape"),
        "action_shape": response.get("action_shape") or list(action.shape),
        "action_min": None,
        "action_max": None,
        "action_finite": bool(action.size and np.isfinite(action).all()),
        "warnings": [],
    }
    warnings = diagnostics["warnings"]
    if response.get("type") != "action":
        warnings.append("response_type_not_action")
    if response.get("source") != expected_source:
        warnings.append("unexpected_or_fallback_source")
    if response.get("input_source") != "payload":
        warnings.append("model_not_using_live_payload")
    if action.size:
        diagnostics["action_min"] = float(np.nanmin(action))
        diagnostics["action_max"] = float(np.nanmax(action))
    if action.size == 0 or not np.isfinite(action).all():
        warnings.append("invalid_action_values")
    if arms == "bimanual":
        targets = _extract_bimanual_actions(response, args.action_step, args.execute_action_steps)
        if not targets:
            warnings.append("no_bimanual_targets_extracted")
        for target in targets:
            if len(target) != 14:
                warnings.append("target_not_14d")
                break
    else:
        targets = _extract_single_arm_actions(response, args.arm_slice, args.action_step, args.execute_action_steps)
        if not targets:
            warnings.append("no_single_arm_targets_extracted")
        for target in targets:
            if len(target) != 7:
                warnings.append("target_not_7d")
                break
    return targets, diagnostics


def _command_target(robot, target: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    commanded = _clip_command_for_hardware(np.asarray(target, dtype=float), args)
    robot.command_joint_pos(commanded)
    return commanded, time.time()


def _split_bimanual_target_halves(target14: np.ndarray, swap_io: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (slice_for_robot_left, slice_for_robot_right) from the model's 14D vector."""

    t = np.asarray(target14, dtype=float).reshape(14)
    if swap_io:
        return t[7:14].copy(), t[:7].copy()
    return t[:7].copy(), t[7:14].copy()


def _command_bimanual(
    robot_l,
    robot_r,
    target14: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    swap_io = getattr(args, "bimanual_io_swap", False)
    tl_raw, tr_raw = _split_bimanual_target_halves(target14, swap_io)
    cmd_l = _clip_command_for_hardware(np.asarray(tl_raw, dtype=float), args)
    cmd_r = _clip_command_for_hardware(np.asarray(tr_raw, dtype=float), args)
    robot_l.command_joint_pos(cmd_l)
    robot_r.command_joint_pos(cmd_r)
    return cmd_l, cmd_r, time.time()


def _direct_delta(robot, delta: list[float], last_command: np.ndarray, args: argparse.Namespace, trace: Trace, label: str) -> np.ndarray:
    current = np.asarray(robot.get_joint_pos(), dtype=float)
    target = current + np.asarray(delta, dtype=float)
    target[6] = float(np.clip(target[6], args.min_gripper_command, args.max_gripper_command))
    steps = max(1, args.correction_steps)
    for index in range(1, steps + 1):
        command = current + (index / steps) * (target - current)
        command = _clip_command_for_hardware(command, args)
        robot.command_joint_pos(command)
        trace.write("codex_correction_command", label=label, step=index, steps=steps, commanded=command.tolist())
        time.sleep(args.correction_duration / steps)
    return target


def _maybe_correct_chip_box(robot, features: dict, last_command: np.ndarray, args: argparse.Namespace, trace: Trace) -> np.ndarray:
    if not args.codex_corrections:
        return last_command
    if "chip" not in args.task.lower() and "box" not in args.task.lower():
        return last_command
    x_error = features.get("x_error_px")
    y_error = features.get("y_error_px")
    if x_error is None or y_error is None:
        return last_command

    corrected = last_command
    if abs(x_error) > args.align_px:
        # Empirically in this camera setup, negative joint 1 moves the jaws
        # toward the chip-box opening when the orange target appears left of the jaws.
        direction = 1.0 if x_error > 0 else -1.0
        corrected = _direct_delta(
            robot,
            [direction * args.align_joint1_step, 0, 0, 0, 0, 0, 0],
            corrected,
            args,
            trace,
            "align_x",
        )
    elif y_error > args.descend_px:
        corrected = _direct_delta(
            robot,
            [0, 0, -args.descend_joint3_step, 0, 0, 0, 0],
            corrected,
            args,
            trace,
            "descend_toward_box",
        )
    elif args.auto_grasp:
        corrected = _direct_delta(
            robot,
            [0, 0, 0, 0, 0, 0, args.min_gripper_command - float(np.asarray(robot.get_joint_pos())[6])],
            corrected,
            args,
            trace,
            "close_gripper",
        )
        corrected = _direct_delta(
            robot,
            [0, 0, args.lift_joint3_step, 0, 0, 0, 0],
            corrected,
            args,
            trace,
            "lift_after_grasp",
        )
    return corrected


def run(args: argparse.Namespace) -> int:
    if STOP_FILE.exists() and not args.ignore_hard_stop:
        print(f"hard stop exists at {STOP_FILE}; run `yamctl clear-stop` first", file=sys.stderr)
        return 2
    if args.clear_stop:
        STOP_FILE.unlink(missing_ok=True)

    _install_can_patch()

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    trace_root = Path(args.trace_dir).expanduser()
    if not trace_root.is_absolute():
        trace_root = ROOT / trace_root
    trace = Trace(trace_root)
    args._camera_ws_registry = {}
    _ws = sum(
        [
            bool(getattr(args, "right_orbbec", False)),
            getattr(args, "right_camera_index", None) is not None,
            bool((getattr(args, "right_camera_url", None) or "").strip()),
        ]
    )
    if _ws != 1:
        print(
            "hybrid: specify exactly one wrist source for Molmo right:\n"
            "  --right-orbbec           Orbbec color stream\n"
            "  --right-camera-index N   USB webcam (OpenCV)\n"
            "  --right-camera-url URL   wrist JPEG over HTTP or WebSocket (ws/wss)\n"
            "(--camera-url is Molmo top HTTP/WebSocket only, not the wrist.)",
            file=sys.stderr,
        )
        trace.close()
        return 2
    _ls = sum(
        [
            getattr(args, "left_camera_index", None) is not None,
            bool((getattr(args, "left_camera_url", None) or "").strip()),
        ]
    )
    if _ls > 1:
        print(
            "hybrid: use at most one Molmo left source:\n"
            "  --left-camera-index N   OpenCV device\n"
            "  --left-camera-url URL   JPEG over HTTP or WebSocket (ws/wss)\n",
            file=sys.stderr,
        )
        trace.close()
        return 2
    # Reuse TLS connection to Modal — avoids redoing handshake every inference (~100–300ms each).
    if args.policy_kind == "modal":
        args._http_session = requests.Session()
    _health_check(args, trace)
    _warmup_modal_infer(args, trace)

    arms = getattr(args, "arms", "bimanual")
    robot = None
    robot_left = robot_right = None

    if arms == "bimanual":
        robot_left = get_yam_robot(
            channel=args.left_can,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=False,
            clip_commands_to_joint_limits=False,
        )
        robot_right = get_yam_robot(
            channel=args.right_can,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=False,
            clip_commands_to_joint_limits=False,
        )
        robot_left._limit_gripper_force = -1.0
        robot_right._limit_gripper_force = -1.0
        last_command_l = _clip_command_for_hardware(np.asarray(robot_left.get_joint_pos(), dtype=float), args)
        last_command_r = _clip_command_for_hardware(np.asarray(robot_right.get_joint_pos(), dtype=float), args)
    else:
        robot = get_yam_robot(
            channel=args.left_can,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=False,
            clip_commands_to_joint_limits=False,
        )
        robot._limit_gripper_force = -1.0
        last_command = _clip_command_for_hardware(np.asarray(robot.get_joint_pos(), dtype=float), args)

    orbbec_right: OrbbecColorPipeline | None = None
    if getattr(args, "right_orbbec", False):
        orbbec_right = OrbbecColorPipeline()
        try:
            orbbec_right.start()
        except Exception as exc:  # noqa: BLE001
            trace.write("orbbec_right_open_failed", error=repr(exc))
            print(f"failed to start Orbbec wrist camera: {exc}", file=sys.stderr)
            if robot is not None:
                robot.close()
            if robot_left is not None:
                robot_left.close()
            if robot_right is not None:
                robot_right.close()
            trace.close()
            return 1
        trace.write("right_camera_ready", source="orbbec")

    right_cap: cv2.VideoCapture | None = None
    if getattr(args, "right_camera_index", None) is not None:
        right_cap = video_capture(int(args.right_camera_index))
        if not right_cap.isOpened():
            right_cap.release()
            trace.write("right_camera_open_failed", index=args.right_camera_index)
            print(f"failed to open wrist webcam --right-camera-index {args.right_camera_index}", file=sys.stderr)
            if orbbec_right is not None:
                orbbec_right.stop()
            if robot is not None:
                robot.close()
            if robot_left is not None:
                robot_left.close()
            if robot_right is not None:
                robot_right.close()
            trace.close()
            return 1
        for _ in range(5):
            right_cap.read()
        trace.write("right_camera_ready", index=args.right_camera_index, source="opencv")

    top_cap: cv2.VideoCapture | None = None
    if getattr(args, "top_camera_index", None) is not None:
        top_cap = video_capture(int(args.top_camera_index))
        if not top_cap.isOpened():
            top_cap.release()
            trace.write("top_camera_open_failed", index=args.top_camera_index)
            print(f"failed to open webcam --top-camera-index {args.top_camera_index}", file=sys.stderr)
            if orbbec_right is not None:
                orbbec_right.stop()
            if right_cap is not None:
                right_cap.release()
            if robot is not None:
                robot.close()
            if robot_left is not None:
                robot_left.close()
            if robot_right is not None:
                robot_right.close()
            trace.close()
            return 1
        for _ in range(5):
            top_cap.read()
        trace.write("top_camera_ready", index=args.top_camera_index)

    left_cap: cv2.VideoCapture | None = None
    if getattr(args, "left_camera_index", None) is not None:
        left_cap = video_capture(int(args.left_camera_index))
        if not left_cap.isOpened():
            left_cap.release()
            trace.write("left_camera_open_failed", index=args.left_camera_index)
            print(f"failed to open webcam --left-camera-index {args.left_camera_index}", file=sys.stderr)
            if top_cap is not None:
                top_cap.release()
            if orbbec_right is not None:
                orbbec_right.stop()
            if right_cap is not None:
                right_cap.release()
            if robot is not None:
                robot.close()
            if robot_left is not None:
                robot_left.close()
            if robot_right is not None:
                robot_right.close()
            trace.close()
            return 1
        for _ in range(5):
            left_cap.read()
        trace.write("left_camera_ready", index=args.left_camera_index)

    try:
        for iteration in range(args.max_iterations):
            if STOP_REQUESTED or STOP_FILE.exists():
                trace.write("stopped", iteration=iteration)
                return 130

            if arms == "bimanual":
                assert robot_left is not None and robot_right is not None
                sl = np.asarray(robot_left.get_joint_pos(), dtype=float)
                sr = np.asarray(robot_right.get_joint_pos(), dtype=float)
                if getattr(args, "bimanual_io_swap", False):
                    state = np.concatenate([sr, sl])
                else:
                    state = np.concatenate([sl, sr])
                safety = _safety_report_pair(robot_left, robot_right)
            else:
                assert robot is not None
                state = np.asarray(robot.get_joint_pos(), dtype=float)
                safety = _safety_report(robot)

            block_reason = _safety_block_reason(safety, args)
            raw_wrist = _raw_wrist_rgb(args, right_cap, orbbec_right)
            rgb = np.asarray(raw_wrist, dtype=np.uint8)
            _save_frame(raw_wrist, trace_root / f"frame-{iteration:03d}-wrist-raw.jpg")
            frame_path = trace_root / f"frame-{iteration:03d}-before.jpg"
            _save_frame(rgb, frame_path)
            features = _image_features(rgb)
            trace.write(
                "iteration_start",
                iteration=iteration,
                arms=arms,
                state=state.tolist(),
                safety=safety,
                frame=str(frame_path),
                image_features=features,
            )
            if block_reason is not None:
                trace.write("execution_blocked", **block_reason)
                return 1

            jq = int(getattr(args, "policy_jpeg_quality", 85))
            images = {"front": _encode_rgb(rgb, quality=jq)}
            policy_image_paths: dict[str, str] = {}
            if args.policy_kind == "modal":
                # Molmo: top = OpenCV or HTTP --camera-url; left = optional cam or black; right = wrist.
                top_rgb = _top_rgb_for_policy(args, top_cap, rgb, trace, iteration)
                left_rgb = _left_rgb_for_policy(args, left_cap, rgb, trace, iteration)
                images = {
                    "top": _encode_rgb(top_rgb, quality=jq),
                    "left": _encode_rgb(left_rgb, quality=jq),
                    "right": _encode_rgb(rgb, quality=jq),
                }
                for key, arr in (("policy-top", top_rgb), ("policy-left", left_rgb), ("policy-right", rgb)):
                    p = trace_root / f"frame-{iteration:03d}-{key}.jpg"
                    _save_frame(arr, p)
                    policy_image_paths[key] = str(p)
            elif args.policy_kind == "lerobot-act":
                pf = trace_root / f"frame-{iteration:03d}-policy-front.jpg"
                _save_frame(rgb, pf)
                policy_image_paths["policy-front"] = str(pf)
            state_format = "yam_bimanual_yam_14d" if arms == "bimanual" else "single_arm_yam_7d"
            payload = {
                "type": "observation",
                "t": time.time(),
                "task": args.task,
                "state": state.tolist(),
                "state_format": state_format,
                "images": images,
                "num_steps": args.num_steps,
            }
            trace.write(
                "policy_payload",
                iteration=iteration,
                policy_kind=args.policy_kind,
                task=args.task,
                arms=arms,
                state_format=payload["state_format"],
                state_len=len(payload["state"]),
                image_keys=sorted(payload["images"].keys()),
                num_steps=args.num_steps,
                policy_jpeg_quality=jq,
                policy_image_paths=policy_image_paths,
                wrist_raw_path=str(trace_root / f"frame-{iteration:03d}-wrist-raw.jpg"),
                wrist_source=(
                    "orbbec"
                    if orbbec_right is not None
                    else ("opencv" if right_cap is not None else "http")
                ),
            )
            t_infer0 = time.time()
            response = _policy_request(args, payload)
            infer_ms = (time.time() - t_infer0) * 1000.0
            trace.write(
                "policy_infer_timing",
                iteration=iteration,
                policy_kind=args.policy_kind,
                elapsed_ms=round(infer_ms, 1),
            )
            targets, diagnostics = _validate_response(response, args)
            trace.write("policy_response", iteration=iteration, policy_kind=args.policy_kind, diagnostics=diagnostics)
            if diagnostics["warnings"] and args.stop_on_model_warning:
                return 1

            executed = 0
            if args.observe_only:
                trace.write("observe_only_skip_execute", iteration=iteration, targets=len(targets))
            elif arms == "bimanual":
                assert robot_left is not None and robot_right is not None
                for target in targets:
                    last_command_l, last_command_r, _ = _command_bimanual(robot_left, robot_right, target, args)
                    executed += 1
                    trace.write(
                        "policy_command",
                        iteration=iteration,
                        commanded_left=last_command_l.tolist(),
                        commanded_right=last_command_r.tolist(),
                        target=np.asarray(target).tolist(),
                    )
                    time.sleep(args.action_step_delay)
            else:
                assert robot is not None
                for target in targets:
                    last_command, _ = _command_target(robot, target, args)
                    executed += 1
                    trace.write("policy_command", iteration=iteration, commanded=last_command.tolist(), target=np.asarray(target).tolist())
                    time.sleep(args.action_step_delay)

            raw_after = _raw_wrist_rgb(args, right_cap, orbbec_right)
            after_rgb = np.asarray(raw_after, dtype=np.uint8)
            _save_frame(raw_after, trace_root / f"frame-{iteration:03d}-wrist-raw-after.jpg")
            after_path = trace_root / f"frame-{iteration:03d}-after-modal.jpg"
            _save_frame(after_rgb, after_path)
            after_features = _image_features(after_rgb)
            if arms == "bimanual":
                assert robot_left is not None and robot_right is not None
                esl = np.asarray(robot_left.get_joint_pos(), dtype=float)
                esr = np.asarray(robot_right.get_joint_pos(), dtype=float)
                if getattr(args, "bimanual_io_swap", False):
                    end_state = np.concatenate([esr, esl]).tolist()
                else:
                    end_state = np.concatenate([esl, esr]).tolist()
            else:
                assert robot is not None
                end_state = np.asarray(robot.get_joint_pos(), dtype=float).tolist()
            trace.write(
                "policy_step_complete",
                iteration=iteration,
                executed_targets=executed,
                frame=str(after_path),
                image_features=after_features,
                state=end_state,
            )

            if not args.observe_only and arms == "single":
                assert robot is not None
                last_command = _maybe_correct_chip_box(robot, after_features, last_command, args, trace)
            time.sleep(1.0 / args.hz)
        return 0
    finally:
        session = getattr(args, "_http_session", None)
        if session is not None:
            session.close()
        if left_cap is not None:
            left_cap.release()
        if top_cap is not None:
            top_cap.release()
        if right_cap is not None:
            right_cap.release()
        if orbbec_right is not None:
            orbbec_right.stop()
        trace.write("shutdown")
        trace.close()
        if robot is not None:
            robot.close()
        if robot_left is not None:
            robot_left.close()
        if robot_right is not None:
            robot_right.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid policy + local verification robot loop.")
    parser.add_argument("task", nargs="?", default="pick up the hat", help="Task prompt sent to the policy.")
    parser.add_argument("--http-url", required=True)
    parser.add_argument("--policy-kind", choices=["lerobot-act", "modal"], default="lerobot-act")
    parser.add_argument(
        "--camera-url",
        default="http://127.0.0.1:8766/frame.jpg",
        help="Molmo 'top' when --top-camera-index omitted: HTTP(S) JPEG URL or ws/wss WebSocket sending JPEG/PNG "
        "binary or JSON base64 frames. Wrist/right: --right-orbbec, --right-camera-index, or --right-camera-url.",
    )
    parser.add_argument(
        "--right-orbbec",
        action="store_true",
        help="Use Orbbec color stream for Molmo 'right' (wrist). Mutually exclusive with --right-camera-index / --right-camera-url.",
    )
    parser.add_argument(
        "--right-camera-index",
        type=int,
        default=None,
        help="OpenCV device index for wrist (Molmo 'right'). Mutually exclusive with --right-orbbec / --right-camera-url.",
    )
    parser.add_argument(
        "--right-camera-url",
        default=None,
        help="HTTP JPEG or ws/wss WebSocket URL for wrist (Molmo 'right'). Mutually exclusive with "
        "--right-orbbec / --right-camera-index.",
    )
    parser.add_argument(
        "--top-camera-index",
        type=int,
        default=None,
        help="OpenCV index for the overhead / Molmo 'top' view. If omitted, top uses --camera-url.",
    )
    parser.add_argument(
        "--left-camera-index",
        type=int,
        default=None,
        help="OpenCV index for Molmo 'left'. Mutually exclusive with --left-camera-url. If both omitted, left is black.",
    )
    parser.add_argument(
        "--left-camera-url",
        default=None,
        help="HTTP or ws/wss URL for Molmo 'left'. Mutually exclusive with --left-camera-index. If both omitted, left is black.",
    )
    parser.add_argument("--trace-dir", default="logs/hybrid-latest")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=3,
        help="Molmo predict_action num_steps (trajectory length / decode cost).",
    )
    parser.add_argument(
        "--policy-jpeg-quality",
        type=int,
        default=78,
        help="JPEG quality for images sent to Modal (lower = smaller JSON + less upload time; 70–85 typical).",
    )
    parser.add_argument("--http-timeout", type=float, default=300.0)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--max-temp-mos", type=float, default=55.0)
    parser.add_argument("--max-temp-rotor", type=float, default=100.0)
    parser.add_argument("--min-gripper-command", type=float, default=0.01)
    parser.add_argument("--max-gripper-command", type=float, default=0.59)
    parser.add_argument("--arm-slice", choices=["first", "second"], default="first")
    parser.add_argument("--action-step", type=int, default=0)
    parser.add_argument("--execute-action-steps", type=int, default=3)
    parser.add_argument("--action-step-delay", type=float, default=0.05)
    parser.add_argument("--stop-on-model-warning", action="store_true")
    parser.add_argument("--observe-only", action="store_true", help="Call policy and write diagnostics without commanding robot motion.")
    parser.add_argument("--codex-corrections", action="store_true")
    parser.add_argument("--align-px", type=float, default=55.0)
    parser.add_argument("--descend-px", type=float, default=70.0)
    parser.add_argument("--align-joint1-step", type=float, default=0.10)
    parser.add_argument("--descend-joint3-step", type=float, default=0.10)
    parser.add_argument("--lift-joint3-step", type=float, default=0.22)
    parser.add_argument("--correction-duration", type=float, default=0.8)
    parser.add_argument("--correction-steps", type=int, default=16)
    parser.add_argument("--auto-grasp", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--ignore-hard-stop", action="store_true")
    parser.add_argument(
        "--arms",
        choices=["single", "bimanual"],
        default=os.environ.get("YAM_ARMS", "bimanual"),
        help="single: one arm on --left-can. bimanual: two arms (14D state), --left-can and --right-can. "
        "Override with YAM_ARMS=single or --arms single for one bus.",
    )
    parser.add_argument(
        "--left-can",
        default=os.environ.get("YAM_BIMANUAL_LEFT_CAN", "can0"),
        help="SocketCAN device for the left arm (also the only arm when --arms single). On macOS can-bridge: typically can0.",
    )
    parser.add_argument(
        "--right-can",
        default=os.environ.get("YAM_BIMANUAL_RIGHT_CAN", "can1"),
        help="SocketCAN device for the right arm when --arms bimanual. On macOS can-bridge: typically can1.",
    )
    parser.add_argument(
        "--bimanual-io-swap",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("YAM_BIMANUAL_IO_SWAP", "").lower() in ("1", "true", "yes"),
        help="Swap left/right halves for BOTH policy state (14D) and actions. Try if arms behave mirrored; "
        "otherwise prefer fixing --left-can/--right-can to match USB wiring.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.hz) or args.hz <= 0:
        parser.error("--hz must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
