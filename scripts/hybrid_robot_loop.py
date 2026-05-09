#!/usr/bin/env python3
"""Hybrid Modal + local verification loop for YAM tasks.

Modal proposes actions. The local loop owns the robot, validates/model-debugs
the response, executes only clipped commands, saves camera frames, and can apply
small task-specific corrections when visual progress stalls.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local_modal_robot_bridge import (  # noqa: E402
    STOP_FILE,
    _clip_command_for_hardware,
    _clip_step,
    _extract_single_arm_actions,
    _install_can_patch,
    _prepare_target_for_execution,
    _safety_block_reason,
    _safety_report,
)


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


def _fetch_rgb(camera_url: str) -> np.ndarray:
    response = requests.get(camera_url, timeout=5)
    response.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))


def _save_frame(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format="JPEG", quality=88)


def _encode_rgb(rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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
    response = requests.post(args.http_url, json=payload, timeout=args.http_timeout)
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
        response = requests.get(health_url, timeout=10)
        trace.write("policy_health", policy_kind=args.policy_kind, url=health_url, status_code=response.status_code, body=response.json())
    except Exception as exc:  # noqa: BLE001
        trace.write("policy_health_failed", policy_kind=args.policy_kind, url=health_url, error=repr(exc))


def _validate_response(response: dict, args: argparse.Namespace) -> tuple[list[np.ndarray], dict]:
    action = np.asarray(response.get("action", []), dtype=float)
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
    targets = _extract_single_arm_actions(response, args.arm_slice, args.action_step, args.execute_action_steps)
    if not targets:
        warnings.append("no_single_arm_targets_extracted")
    for target in targets:
        if len(target) != 7:
            warnings.append("target_not_7d")
            break
    return targets, diagnostics


def _command_target(robot, target: np.ndarray, last_command: np.ndarray, last_time: float, args: argparse.Namespace) -> tuple[np.ndarray, float, dict | None]:
    state = robot.get_joint_pos()
    target, target_block = _prepare_target_for_execution(target, state, args)
    if target_block is not None:
        return last_command, last_time, target_block
    now = time.time()
    command_dt = args.command_dt if args.command_dt > 0 else now - last_time
    commanded = _clip_step(last_command, target, args.max_speed, command_dt)
    commanded = _clip_command_for_hardware(commanded, args)
    robot.command_joint_pos(commanded)
    return commanded, now, None


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
    _health_check(args, trace)

    robot = get_yam_robot(channel="can0", gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=False)
    last_command = _clip_command_for_hardware(np.asarray(robot.get_joint_pos(), dtype=float), args)
    last_time = time.time()
    try:
        for iteration in range(args.max_iterations):
            if STOP_REQUESTED or STOP_FILE.exists():
                trace.write("stopped", iteration=iteration)
                return 130

            state = np.asarray(robot.get_joint_pos(), dtype=float)
            safety = _safety_report(robot)
            block_reason = _safety_block_reason(safety, args)
            rgb = _fetch_rgb(args.camera_url)
            frame_path = trace_root / f"frame-{iteration:03d}-before.jpg"
            _save_frame(rgb, frame_path)
            features = _image_features(rgb)
            trace.write(
                "iteration_start",
                iteration=iteration,
                state=state.tolist(),
                safety=safety,
                frame=str(frame_path),
                image_features=features,
            )
            if block_reason is not None:
                trace.write("execution_blocked", **block_reason)
                return 1

            images = {"front": _encode_rgb(rgb)}
            if args.policy_kind == "modal":
                images = {"top": _encode_rgb(rgb), "left": _encode_rgb(rgb), "right": _encode_rgb(rgb)}
            payload = {
                "type": "observation",
                "t": time.time(),
                "task": args.task,
                "state": state.tolist(),
                "state_format": "single_arm_yam_7d",
                "images": images,
                "num_steps": args.num_steps,
            }
            trace.write(
                "policy_payload",
                iteration=iteration,
                policy_kind=args.policy_kind,
                task=args.task,
                state_format=payload["state_format"],
                state_len=len(payload["state"]),
                image_keys=sorted(payload["images"].keys()),
                num_steps=args.num_steps,
            )
            response = _policy_request(args, payload)
            targets, diagnostics = _validate_response(response, args)
            trace.write("policy_response", iteration=iteration, policy_kind=args.policy_kind, diagnostics=diagnostics)
            if diagnostics["warnings"] and args.stop_on_model_warning:
                return 1

            executed = 0
            if args.observe_only:
                trace.write("observe_only_skip_execute", iteration=iteration, targets=len(targets))
            else:
                for target in targets:
                    last_command, last_time, blocked = _command_target(robot, target, last_command, last_time, args)
                    if blocked is not None:
                        trace.write("policy_target_blocked", iteration=iteration, **blocked)
                        break
                    executed += 1
                    trace.write("policy_command", iteration=iteration, commanded=last_command.tolist(), target=np.asarray(target).tolist())
                    time.sleep(args.action_step_delay)

            after_rgb = _fetch_rgb(args.camera_url)
            after_path = trace_root / f"frame-{iteration:03d}-after-modal.jpg"
            _save_frame(after_rgb, after_path)
            after_features = _image_features(after_rgb)
            trace.write(
                "policy_step_complete",
                iteration=iteration,
                executed_targets=executed,
                frame=str(after_path),
                image_features=after_features,
                state=np.asarray(robot.get_joint_pos(), dtype=float).tolist(),
            )

            if not args.observe_only:
                last_command = _maybe_correct_chip_box(robot, after_features, last_command, args, trace)
            time.sleep(1.0 / args.hz)
        return 0
    finally:
        trace.write("shutdown")
        trace.close()
        robot.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid policy + local verification robot loop.")
    parser.add_argument("task")
    parser.add_argument("--http-url", required=True)
    parser.add_argument("--policy-kind", choices=["lerobot-act", "modal"], default="lerobot-act")
    parser.add_argument("--camera-url", default="http://127.0.0.1:8766/frame.jpg")
    parser.add_argument("--trace-dir", default="logs/hybrid-latest")
    parser.add_argument("--hz", type=float, default=0.5)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--http-timeout", type=float, default=300.0)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-speed", type=float, default=0.05)
    parser.add_argument("--max-target-delta", type=float, default=0.75)
    parser.add_argument("--max-temp-mos", type=float, default=55.0)
    parser.add_argument("--max-temp-rotor", type=float, default=100.0)
    parser.add_argument("--min-gripper-command", type=float, default=0.01)
    parser.add_argument("--max-gripper-command", type=float, default=0.59)
    parser.add_argument("--arm-slice", choices=["first", "second"], default="first")
    parser.add_argument("--action-step", type=int, default=0)
    parser.add_argument("--execute-action-steps", type=int, default=3)
    parser.add_argument("--action-step-delay", type=float, default=0.05)
    parser.add_argument("--command-dt", type=float, default=0.25)
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
    args = parser.parse_args()
    if not math.isfinite(args.hz) or args.hz <= 0:
        parser.error("--hz must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
