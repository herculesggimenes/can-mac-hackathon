#!/usr/bin/env python3
"""Expose the YAM robot object's simple API over WebSocket."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "can-bridge"))
STOP_FILE = ROOT / "HARD_STOP"
DEFAULT_ARM_SPECS = "left:can0,right:can1"
CAN_SOCKETS = [Path("/tmp/can0.sock"), Path("/tmp/can1.sock")]


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    channel: str


@dataclass
class ArmRuntime:
    spec: ArmSpec
    robot: Any | None = None
    connected: bool = False
    reconnecting: bool = False
    last_error: str | None = None
    last_connected_at: float | None = None
    last_disconnected_at: float | None = None
    next_reconnect_at: float = 0.0
    reconnect_attempts: int = 0


class BridgeSupervisor:
    def __init__(self, *, startup_timeout: float):
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None
        self.last_error: str | None = None
        self.last_started_at: float | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None and self.ready():
            return
        self.stop()
        script = ROOT / "can-bridge" / "start_bimanual_bridges.sh"
        self.process = subprocess.Popen(
            ["bash", str(script)],
            cwd=ROOT,
            start_new_session=True,
        )
        self.last_started_at = time.time()
        if not self.wait_ready():
            code = self.process.poll() if self.process is not None else None
            self.last_error = f"bimanual CAN bridge did not create sockets; process_code={code}"
            raise RuntimeError(self.last_error)
        self.last_error = None

    def ready(self) -> bool:
        return all(path.exists() for path in CAN_SOCKETS)

    def wait_ready(self) -> bool:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.ready():
                return True
            if self.process is not None and self.process.poll() is not None:
                return False
            time.sleep(0.1)
        return self.ready()

    def ensure(self) -> None:
        if self.process is None or self.process.poll() is not None or not self.ready():
            self.start()

    def status(self) -> dict[str, Any]:
        return {
            "managed": True,
            "pid": self.process.pid if self.process is not None else None,
            "running": self.process is not None and self.process.poll() is None,
            "sockets": {str(path): path.exists() for path in CAN_SOCKETS},
            "last_started_at": self.last_started_at,
            "last_error": self.last_error,
        }

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.process = None


def _bridge_channel(channel: Any) -> int | None:
    if channel is None:
        return 0
    if isinstance(channel, int):
        return channel
    text = str(channel)
    if text.startswith("can") and text[3:].isdigit():
        return int(text[3:])
    if text.isdigit():
        return int(text)
    return None


def _install_can_patch() -> None:
    import can
    from can_bridge import CanBridgeBus
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface
    from i2rt.robots.utils import GripperType

    orig_bus = can.interface.Bus

    def patched_bus(*args, **kwargs):
        interface = kwargs.get("interface", kwargs.get("bustype"))
        channel = kwargs.get("channel", args[0] if args else None)
        bridge_channel = _bridge_channel(channel)
        if interface == "socketcan" and bridge_channel is not None:
            return CanBridgeBus(channel=bridge_channel)
        return orig_bus(*args, **kwargs)

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
            raw_limits = os.environ.get("YAM_GRIPPER_RAW_LIMITS", "0.0,-4.20")
            closed, open_ = (float(part.strip()) for part in raw_limits.split(",", 1))
            return closed, open_
        return orig_get_limits(self)

    def patched_cal(self):
        if self == GripperType.LINEAR_4310:
            return False
        return orig_get_cal(self)

    GripperType.get_gripper_limits = patched_limits
    GripperType.get_gripper_needs_calibration = patched_cal


class YamControlServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.arm_specs = _parse_arm_specs(args)
        self.lock = threading.Lock()
        self.arms = {spec.arm_id: ArmRuntime(spec=spec) for spec in self.arm_specs}
        self.closed = False
        self.robot_factory = None
        self.gripper_type = None
        self.bridge = BridgeSupervisor(startup_timeout=args.bridge_startup_timeout)

    def start(self) -> None:
        self.bridge.start()
        _install_can_patch()
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        self.robot_factory = get_yam_robot
        self.gripper_type = GripperType.LINEAR_4310
        for spec in self.arm_specs:
            self._connect_arm(spec.arm_id)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for runtime in self.arms.values():
                if runtime.robot is not None:
                    runtime.robot.close()
                runtime.robot = None
                runtime.connected = False
            self.bridge.stop()

    def _connect_arm(self, arm_id: str) -> bool:
        runtime = self.arms[arm_id]
        if self.closed:
            return False
        self.bridge.ensure()
        if self.robot_factory is None or self.gripper_type is None:
            raise RuntimeError("server_not_started")
        runtime.reconnecting = True
        try:
            robot = self.robot_factory(
                channel=runtime.spec.channel,
                gripper_type=self.gripper_type,
                zero_gravity_mode=False,
            )
            runtime.robot = robot
            runtime.connected = True
            runtime.reconnecting = False
            runtime.last_error = None
            runtime.last_connected_at = time.time()
            runtime.next_reconnect_at = 0.0
            runtime.reconnect_attempts = 0
            return True
        except Exception as exc:
            runtime.robot = None
            runtime.connected = False
            runtime.reconnecting = False
            runtime.last_error = str(exc)
            runtime.last_disconnected_at = time.time()
            self._schedule_reconnect(runtime)
            return False

    def _schedule_reconnect(self, runtime: ArmRuntime) -> None:
        runtime.reconnect_attempts += 1
        delay = min(self.args.reconnect_max_delay, self.args.reconnect_initial_delay * (2 ** (runtime.reconnect_attempts - 1)))
        runtime.next_reconnect_at = time.time() + delay

    def _mark_arm_failed(self, arm_id: str, exc: Exception) -> None:
        runtime = self.arms[arm_id]
        runtime.last_error = str(exc)
        runtime.last_disconnected_at = time.time()
        runtime.connected = False
        if runtime.robot is not None:
            try:
                runtime.robot.close()
            except Exception:
                pass
        runtime.robot = None
        self._schedule_reconnect(runtime)

    def reconnect_due_arms(self) -> None:
        now = time.time()
        with self.lock:
            arm_ids = [
                arm_id
                for arm_id, runtime in self.arms.items()
                if not runtime.connected and runtime.next_reconnect_at <= now
            ]
            for arm_id in arm_ids:
                self._connect_arm(arm_id)

    def arm_status(self) -> dict[str, Any]:
        return {
            "bridge": self.bridge.status(),
            "arms": {
                arm_id: {
                    "arm": arm_id,
                    "channel": runtime.spec.channel,
                    "connected": runtime.connected,
                    "reconnecting": runtime.reconnecting,
                    "last_error": runtime.last_error,
                    "last_connected_at": runtime.last_connected_at,
                    "last_disconnected_at": runtime.last_disconnected_at,
                    "next_reconnect_at": runtime.next_reconnect_at or None,
                    "reconnect_attempts": runtime.reconnect_attempts,
                }
                for arm_id, runtime in self.arms.items()
            },
        }

    def clamp_q(self, q: list[float] | np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        if q_arr.shape != (7,):
            raise ValueError(f"expected q with 7 numbers, got shape {q_arr.shape}")
        q_arr = q_arr.copy()
        q_arr[6] = float(np.clip(q_arr[6], self.args.min_gripper, self.args.max_gripper))
        return q_arr

    def _require_robot(self, arm_id: str):
        if arm_id not in self.arms:
            raise RuntimeError(f"robot_not_connected: {arm_id}")
        runtime = self.arms[arm_id]
        if runtime.robot is None or not runtime.connected:
            self.reconnect_due_arms()
            runtime = self.arms[arm_id]
        if runtime.robot is None or not runtime.connected:
            raise RuntimeError(f"robot_not_connected: {arm_id}; {runtime.last_error or 'waiting_for_reconnect'}")
        return arm_id, runtime.robot

    def _arm_ids(self) -> list[str]:
        return list(self.arms)

    def _joint_pos_one(self, arm: str) -> list[float]:
        _, robot = self._require_robot(arm)
        try:
            return np.asarray(robot.get_joint_pos(), dtype=float).tolist()
        except Exception as exc:
            self._mark_arm_failed(arm, exc)
            raise

    def get_joint_pos(self) -> list[float]:
        ids = self._arm_ids()
        values = {arm_id: self._joint_pos_one(arm_id) for arm_id in ids}
        return [value for arm_id in ids for value in values[arm_id]]

    def command_joint_pos(self, joint_pos: list[float] | np.ndarray) -> None:
        arm_ids = self._arm_ids()
        q_arr = np.asarray(joint_pos, dtype=float)
        expected = 7 * len(arm_ids)
        if q_arr.shape != (expected,):
            raise ValueError(f"expected {expected} joint values for arms {arm_ids}, got shape {q_arr.shape}")
        targets = {
            arm_id: self.clamp_q(q_arr[index * 7 : (index + 1) * 7])
            for index, arm_id in enumerate(arm_ids)
        }

        robots = {arm_id: self._require_robot(arm_id)[1] for arm_id in targets}
        for arm_id, target in targets.items():
            robot = robots[arm_id]
            with self.lock:
                if arm_id not in self.arms:
                    raise RuntimeError(f"robot_not_connected: {arm_id}")
                try:
                    robot.command_joint_pos(target)
                except Exception as exc:
                    self._mark_arm_failed(arm_id, exc)
                    raise

    def get_observations(self) -> dict[str, Any]:
        def one(arm_id: str) -> dict[str, Any]:
            _, robot = self._require_robot(arm_id)
            try:
                observations = robot.get_observations()
                return {key: np.asarray(value).tolist() for key, value in observations.items()}
            except Exception as exc:
                self._mark_arm_failed(arm_id, exc)
                raise

        return {arm_id: one(arm_id) for arm_id in self._arm_ids()}

    def get_robot_info(self) -> dict[str, Any]:
        def one(arm_id: str) -> dict[str, Any]:
            _, robot = self._require_robot(arm_id)
            try:
                return _json_safe(robot.get_robot_info())
            except Exception as exc:
                self._mark_arm_failed(arm_id, exc)
                raise

        return {arm_id: one(arm_id) for arm_id in self._arm_ids()}

    def num_dofs(self) -> int:
        def one(arm_id: str) -> int:
            _, robot = self._require_robot(arm_id)
            try:
                return int(robot.num_dofs())
            except Exception as exc:
                self._mark_arm_failed(arm_id, exc)
                raise

        return sum(one(arm_id) for arm_id in self._arm_ids())

    def zero_torque_mode(self) -> None:
        for arm_id in self._arm_ids():
            _, robot = self._require_robot(arm_id)
            try:
                robot.zero_torque_mode()
            except Exception as exc:
                self._mark_arm_failed(arm_id, exc)
                raise

    def channels_by_arm(self) -> dict[str, str]:
        return {spec.arm_id: spec.channel for spec in self.arm_specs}


def _parse_arm_specs(args: argparse.Namespace) -> list[ArmSpec]:
    raw = args.arm_specs
    specs: list[ArmSpec] = []
    seen: set[str] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid arm spec {part!r}; expected arm_id:channel")
        arm_id, channel = (item.strip() for item in part.split(":", 1))
        if not arm_id or not channel:
            raise ValueError(f"invalid arm spec {part!r}; expected arm_id:channel")
        if arm_id in seen:
            raise ValueError(f"duplicate arm id: {arm_id}")
        seen.add(arm_id)
        specs.append(ArmSpec(arm_id=arm_id, channel=channel))
    if not specs:
        raise ValueError("at least one arm spec is required")
    if len(specs) != 2:
        raise ValueError("this server is bimanual-only; expected exactly two arm specs")
    return specs


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _rpc_result(request_id: Any, result: Any = None) -> dict[str, Any]:
    return {"id": request_id, "ok": True, "result": _json_safe(result)}


def _rpc_error(request_id: Any, error: str) -> dict[str, Any]:
    return {"id": request_id, "ok": False, "error": error}


def _json_error(message: str, *, request_id: Any = None) -> str:
    payload: dict[str, Any] = {"id": request_id, "ok": False, "error": message}
    return json.dumps(payload)


def _handle_message(server: YamControlServer, message: dict[str, Any]) -> dict[str, Any]:
    method = message.get("method")
    if method:
        return _handle_rpc_method(server, message)

    return _rpc_error(message.get("id"), "missing method; this endpoint only accepts JSON-RPC-style messages")


def _handle_rpc_method(server: YamControlServer, message: dict[str, Any]) -> dict[str, Any]:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}
    try:
        if method == "get_joint_pos":
            return _rpc_result(request_id, server.get_joint_pos())
        if method == "get_status":
            return _rpc_result(request_id, server.arm_status())
        if method == "reconnect":
            with server.lock:
                for arm_id in server.arms:
                    server._connect_arm(arm_id)
            return _rpc_result(request_id, server.arm_status())
        if method == "command_joint_pos":
            joint_pos = params.get("joint_pos", params.get("q"))
            server.command_joint_pos(joint_pos)
            return _rpc_result(request_id, None)
        if method == "get_observations":
            return _rpc_result(request_id, server.get_observations())
        if method == "get_robot_info":
            return _rpc_result(request_id, server.get_robot_info())
        if method == "num_dofs":
            return _rpc_result(request_id, server.num_dofs())
        if method == "zero_torque_mode":
            server.zero_torque_mode()
            return _rpc_result(request_id, None)
        if method == "close":
            server.close()
            return _rpc_result(request_id, None)
        return _rpc_error(request_id, f"unknown method: {method}")
    except Exception as exc:
        return _rpc_error(request_id, str(exc))


def _serve_ws(server: YamControlServer, host: str, port: int) -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    def handler(ws) -> None:
        ws.send(
            json.dumps(
                {
                    "type": "hello",
                    "version": 1,
                    "arms": server.channels_by_arm(),
                    "methods": [
                        "get_joint_pos",
                        "command_joint_pos",
                        "get_observations",
                        "get_robot_info",
                        "get_status",
                        "reconnect",
                        "num_dofs",
                        "zero_torque_mode",
                        "close",
                    ],
                    "command_space": "yam_7d_absolute_normalized_gripper",
                    "gripper": {"min": server.args.min_gripper, "max": server.args.max_gripper},
                    "example": {
                        "method": "command_joint_pos",
                        "id": "cmd-1",
                        "params": {"joint_pos": [0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0.3]},
                    },
                }
            )
        )
        try:
            while True:
                raw = ws.recv()
                try:
                    message = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                except Exception:
                    ws.send(_json_error("expected JSON message"))
                    continue

                response = _handle_message(server, message)
                ws.send(json.dumps(response))
                if message.get("method") == "close":
                    return
        except (ConnectionClosed, EOFError):
            return

    with serve(handler, host, port) as ws_server:
        print(f"YAM control WebSocket: ws://{host}:{port}/control", flush=True)
        ws_server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Expose YAM joint controls over WebSocket.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument(
        "--arm-specs",
        default=DEFAULT_ARM_SPECS,
        help="Comma-separated arm_id:channel list. Defaults to left:can0,right:can1.",
    )
    parser.add_argument("--min-gripper", type=float, default=0.01)
    parser.add_argument("--max-gripper", type=float, default=0.59)
    parser.add_argument("--reconnect-initial-delay", type=float, default=0.5)
    parser.add_argument("--reconnect-max-delay", type=float, default=5.0)
    parser.add_argument("--bridge-startup-timeout", type=float, default=10.0)
    args = parser.parse_args()

    if STOP_FILE.exists():
        STOP_FILE.unlink()

    server = YamControlServer(args)

    def shutdown(_signum=None, _frame=None):
        server.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.start()
    try:
        _serve_ws(server, args.host, args.port)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
