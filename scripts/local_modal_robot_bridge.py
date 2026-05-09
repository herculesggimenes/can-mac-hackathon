#!/usr/bin/env python3
"""Local YAM bridge for a Modal WebSocket policy service.

By default this is observe-only and will not command hardware. Pass --execute to
send returned actions to the robot after clipping and speed limiting.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import signal
import sys
import time
from pathlib import Path

import requests
import numpy as np
import websockets
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "can-bridge"))
STOP_FILE = ROOT / "HARD_STOP"
STOP_REQUESTED = False


def _request_stop(_signum=None, _frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


def _install_can_patch():
    import can
    from can_bridge import CanBridgeBus
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface
    from i2rt.robots.utils import GripperType

    orig = can.interface.Bus

    def patched_bus(*args, **kwargs):
        interface = kwargs.get("interface", kwargs.get("bustype"))
        channel = kwargs.get("channel", args[0] if args else None)
        if interface == "socketcan" and channel in {"can0", 0, None}:
            return CanBridgeBus(channel=0)
        return orig(*args, **kwargs)

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


def _clip_step(current: np.ndarray, target: np.ndarray, max_speed: float, dt: float) -> np.ndarray:
    max_step = max_speed * max(dt, 1e-3)
    return current + np.clip(target - current, -max_step, max_step)


def _clip_command_for_hardware(commanded: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    commanded = np.asarray(commanded, dtype=float).copy()
    commanded[6] = float(np.clip(commanded[6], args.min_gripper_command, args.max_gripper_command))
    return commanded


def _apply_current_gripper_cap(state: np.ndarray, args: argparse.Namespace) -> None:
    if not args.cap_gripper_at_current_open:
        return
    current_gripper = float(np.asarray(state, dtype=float)[6])
    current_open = float(np.clip(current_gripper, args.min_gripper_command, args.max_gripper_command))
    args.max_gripper_command = min(args.max_gripper_command, current_open)
    print(
        json.dumps(
            {
                "gripper_cap": "current_open",
                "current_gripper": current_gripper,
                "max_gripper_command": args.max_gripper_command,
            }
        ),
        flush=True,
    )


def _encode_rgb_jpeg_b64(rgb: np.ndarray, size: tuple[int, int] = (640, 360)) -> str:
    import cv2

    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    image = Image.fromarray(resized)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenCVCamera:
    def __init__(self, index: int, width: int = 640, height: int = 360):
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

    def read_rgb(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame read failed")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


def _rgb_from_camera_url(url: str) -> np.ndarray:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return np.asarray(image)


def _camera_payload(camera: OpenCVCamera | None, camera_url: str | None) -> dict:
    if camera is None and not camera_url:
        return {}
    rgb = _rgb_from_camera_url(camera_url) if camera_url else camera.read_rgb()
    encoded = _encode_rgb_jpeg_b64(rgb)
    # MolmoAct2-BimanualYAM expects top/left/right. With one local webcam we
    # intentionally duplicate the same frame for inspection; this is not the
    # trained camera geometry.
    return {"top": encoded, "left": encoded, "right": encoded}


def _summarize_response(response: dict) -> str:
    action = np.asarray(response.get("action", []), dtype=object)
    summary = {
        "type": response.get("type"),
        "source": response.get("source"),
        "input_source": response.get("input_source"),
        "execute_ok": response.get("execute_ok"),
        "action_shape": response.get("action_shape") or list(action.shape),
        "note": response.get("note"),
    }
    return json.dumps(summary)


def _extract_single_arm_target(response: dict, arm_slice: str, action_step: int) -> np.ndarray | None:
    action = np.asarray(response.get("action", []), dtype=float)
    if action.size == 0:
        return None
    action = np.squeeze(action)
    if action.ndim == 2:
        step = int(np.clip(action_step, 0, action.shape[0] - 1))
        action = action[step]
    if action.shape[0] < 14:
        return action[:7] if action.shape[0] >= 7 else None
    start = 0 if arm_slice == "first" else 7
    return action[start : start + 7]


def _extract_single_arm_actions(
    response: dict,
    arm_slice: str,
    action_step: int,
    execute_action_steps: int,
) -> list[np.ndarray]:
    action = np.asarray(response.get("action", []), dtype=float)
    if action.size == 0:
        return []
    action = np.squeeze(action)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2:
        return []

    start_step = int(np.clip(action_step, 0, action.shape[0] - 1))
    stop_step = min(action.shape[0], start_step + max(1, execute_action_steps))
    selected = action[start_step:stop_step]
    arm_start = 0 if arm_slice == "first" else 7

    targets: list[np.ndarray] = []
    for step in selected:
        if step.shape[0] >= 14:
            targets.append(step[arm_start : arm_start + 7])
        elif step.shape[0] >= 7:
            targets.append(step[:7])
    return targets


def _target_report(state: np.ndarray, target: np.ndarray | None) -> dict:
    if target is None:
        return {"target": None}
    delta = np.asarray(target, dtype=float) - np.asarray(state[:7], dtype=float)
    return {
        "target": np.asarray(target, dtype=float).tolist(),
        "delta": delta.tolist(),
        "max_abs_delta": float(np.max(np.abs(delta))),
    }


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


def _prepare_target_for_execution(target: np.ndarray, state: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict | None]:
    target = np.asarray(target, dtype=float).copy()
    requested_gripper = float(target[6])
    target[6] = float(np.clip(target[6], args.min_gripper_command, args.max_gripper_command))

    arm_delta = target[:6] - np.asarray(state[:6], dtype=float)
    max_abs_arm_delta = float(np.max(np.abs(arm_delta)))
    if max_abs_arm_delta > args.max_target_delta:
        return target, {
            "execution_blocked": "arm_target_delta_too_large",
            "max_abs_arm_delta": max_abs_arm_delta,
            "limit": args.max_target_delta,
            "requested_gripper": requested_gripper,
            "clamped_gripper": float(target[6]),
        }
    return target, None


async def _policy_request(args: argparse.Namespace, payload: dict) -> dict:
    if args.http_url:
        response = requests.post(args.http_url, json=payload, timeout=args.http_timeout)
        response.raise_for_status()
        return response.json()
    async with websockets.connect(args.ws_url, ping_interval=10, ping_timeout=10) as ws:
        await ws.send(json.dumps(payload))
        return json.loads(await ws.recv())


async def run_bridge(args: argparse.Namespace) -> None:
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    if args.sample_only:
        payload = {"type": "observation", "sample": True, "num_steps": args.num_steps}
        response = await _policy_request(args, payload)
        print(_summarize_response(response), flush=True)
        if response.get("type") == "error":
            print(json.dumps(response), flush=True)
            raise SystemExit(1)
        return

    _install_can_patch()

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    robot = get_yam_robot(
        channel="can0",
        gripper_type=GripperType.LINEAR_4310,
        zero_gravity_mode=False,
    )
    initial_state = robot.get_joint_pos()
    _apply_current_gripper_cap(initial_state, args)
    last_command = _clip_command_for_hardware(initial_state, args)
    last_time = time.time()
    camera = None
    if not args.no_camera and not args.camera_url:
        camera = OpenCVCamera(args.camera_index)

    try:
        print(f"Connected to {args.http_url or args.ws_url}")
        print(f"Hard stop: Ctrl-C, kill this process, or create {STOP_FILE}")
        iteration = 0
        while not STOP_REQUESTED and not STOP_FILE.exists():
            state = robot.get_joint_pos()
            safety = _safety_report(robot)
            print(json.dumps({"safety": safety}), flush=True)
            block_reason = _safety_block_reason(safety, args)
            if block_reason is not None:
                print(json.dumps(block_reason), flush=True)
                break
            payload = {
                "type": "observation",
                "t": time.time(),
                "task": args.task,
                "state": state.tolist(),
                "state_format": "single_arm_yam_7d",
                "images": _camera_payload(camera, args.camera_url),
                "num_steps": args.num_steps,
            }
            response = await _policy_request(args, payload)
            print(_summarize_response(response), flush=True)
            targets = _extract_single_arm_actions(
                response,
                args.arm_slice,
                args.action_step,
                args.execute_action_steps,
            )
            target = targets[0] if targets else None
            print(json.dumps(_target_report(state, target)), flush=True)

            if args.execute and not args.force_execute_unsafe:
                print("Execution blocked: pass --force-execute-unsafe after validating policy output.", flush=True)
            elif args.execute and not response.get("execute_ok", False) and not args.allow_bimanual_slice:
                print("Execution blocked: remote execute_ok=false. Pass --allow-bimanual-slice to test one 7D slice.", flush=True)
            elif args.execute:
                if targets:
                    executed_steps = 0
                    blocked = False
                    for target in targets:
                        state = robot.get_joint_pos()
                        safety = _safety_report(robot)
                        block_reason = _safety_block_reason(safety, args)
                        if block_reason is not None:
                            print(json.dumps({"safety": safety}), flush=True)
                            print(json.dumps(block_reason), flush=True)
                            blocked = True
                            break

                        target, target_block = _prepare_target_for_execution(target, state, args)
                        if target_block is not None:
                            print(json.dumps(target_block), flush=True)
                            blocked = True
                            break
                        now = time.time()
                        command_dt = args.command_dt if args.command_dt > 0 else now - last_time
                        commanded = _clip_step(last_command, target, args.max_speed, command_dt)
                        commanded = _clip_command_for_hardware(commanded, args)
                        robot.command_joint_pos(commanded)
                        last_command = commanded
                        last_time = now
                        executed_steps += 1
                        print(
                            json.dumps(
                                {
                                    "commanded": commanded.tolist(),
                                    "target": target.tolist(),
                                    "trajectory_step": executed_steps,
                                    "trajectory_steps_requested": len(targets),
                                }
                            ),
                            flush=True,
                        )
                        if args.action_step_delay > 0:
                            await asyncio.sleep(args.action_step_delay)
                    if blocked:
                        break
                elif target is not None and target.shape[0] >= 7:
                    target, target_block = _prepare_target_for_execution(target, state, args)
                    if target_block is not None:
                        print(json.dumps(target_block), flush=True)
                        break
                    now = time.time()
                    command_dt = args.command_dt if args.command_dt > 0 else now - last_time
                    commanded = _clip_step(last_command, target, args.max_speed, command_dt)
                    commanded = _clip_command_for_hardware(commanded, args)
                    robot.command_joint_pos(commanded)
                    last_command = commanded
                    last_time = now
                    print(json.dumps({"commanded": commanded.tolist(), "target": target.tolist()}), flush=True)
                else:
                    print("Skipping execute: could not extract a 7D single-arm action", flush=True)

            iteration += 1
            if args.max_iterations and iteration >= args.max_iterations:
                break
            await asyncio.sleep(1.0 / args.hz)
    finally:
        print("Hard stop / shutdown: closing robot interface.", flush=True)
        if camera is not None:
            camera.close()
        robot.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", help="Modal websocket URL, e.g. wss://...modal.run/ws")
    parser.add_argument("--http-url", help="Modal HTTP inference URL, e.g. https://...modal.run/infer")
    parser.add_argument("--task", default="put bread in toaster")
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--http-timeout", type=float, default=300.0)
    parser.add_argument("--max-speed", type=float, default=0.03, help="Max joint command speed in rad/s or normalized gripper/s")
    parser.add_argument("--max-target-delta", type=float, default=0.35, help="Refuse execution if any target joint is farther than this from current state.")
    parser.add_argument("--min-gripper-command", type=float, default=0.01, help="Clamp gripper policy commands to at least this normalized value.")
    parser.add_argument("--max-gripper-command", type=float, default=0.59, help="Clamp gripper policy commands to at most this normalized value.")
    parser.add_argument("--cap-gripper-at-current-open", action="store_true", help="Use the startup gripper position as the max normalized open command for this run.")
    parser.add_argument("--max-temp-mos", type=float, default=60.0, help="Stop if any motor MOS temperature exceeds this C.")
    parser.add_argument("--max-temp-rotor", type=float, default=100.0, help="Stop if any motor rotor temperature exceeds this C.")
    parser.add_argument("--max-iterations", type=int, default=0, help="Stop after N policy calls; 0 means run until hard stop.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-url", help="HTTP URL returning a JPEG/PNG frame, e.g. http://127.0.0.1:8766/frame.jpg")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--arm-slice", choices=["first", "second"], default="first")
    parser.add_argument("--action-step", type=int, default=0)
    parser.add_argument("--execute-action-steps", type=int, default=1, help="Execute this many consecutive policy trajectory steps per inference.")
    parser.add_argument("--action-step-delay", type=float, default=0.0, help="Delay between local trajectory step commands.")
    parser.add_argument("--command-dt", type=float, default=0.0, help="Fixed dt for speed clipping each trajectory command; 0 uses wall-clock dt.")
    parser.add_argument("--sample-only", action="store_true", help="Call Modal with the official MolmoAct2 sample. Does not touch the robot or CAN.")
    parser.add_argument("--execute", action="store_true", help="Actually command the local robot. Default is observe-only.")
    parser.add_argument("--force-execute-unsafe", action="store_true", help="Required in addition to --execute; still requires remote execute_ok=true.")
    parser.add_argument("--allow-bimanual-slice", action="store_true", help="Allow executing one 7D slice of a 14D bimanual policy output.")
    args = parser.parse_args()
    if not args.ws_url and not args.http_url:
        parser.error("pass --http-url or --ws-url")
    asyncio.run(run_bridge(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
