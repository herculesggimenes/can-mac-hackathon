"""Command line operator tools for the local YAM control stack."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
STOP_FILE = ROOT / "HARD_STOP"
CAN_SOCKET = Path("/tmp/can0.sock")

DEFAULT_SERIAL_PORT = os.environ.get("YAM_SERIAL_PORT", "/dev/cu.usbmodem206D338A594E1")
DEFAULT_CAMERA_URL = os.environ.get("YAM_CAMERA_URL", "http://127.0.0.1:8766/frame.jpg")
DEFAULT_ONE_ARM_POLICY_HTTP_URL = os.environ.get("YAM_ONE_ARM_POLICY_HTTP_URL", "http://127.0.0.1:8777/infer")
DEFAULT_MODAL_POLICY_HTTP_URL = os.environ.get("YAM_MODAL_POLICY_HTTP_URL", "")
DEFAULT_POLICY_HTTP_URL = os.environ.get("YAM_POLICY_HTTP_URL", DEFAULT_MODAL_POLICY_HTTP_URL)


def _python() -> str:
    return sys.executable


def _pid_path(name: str) -> Path:
    return LOG_DIR / f"{name}.pid"


def _log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


def _read_pid(name: str) -> int | None:
    path = _pid_path(name)
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _process_lines() -> list[str]:
    result = subprocess.run(
        ["ps", "-axo", "pid,command"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    patterns = (
        "hybrid_robot_loop.py",
        "direct_robot_control.py",
        "local_modal_robot_bridge.py",
        "slcan_bridge.py",
        "teleop_viewer.py",
        "camera_http_server.py",
        "yam_lerobot_policy_server.py",
        "modal_molmoact2_service.py",
    )
    return [line.strip() for line in result.stdout.splitlines() if any(p in line for p in patterns)]


def _pid_from_process_line(line: str) -> int | None:
    try:
        return int(line.strip().split(maxsplit=1)[0])
    except (IndexError, ValueError):
        return None


def _discover_pid(pattern: str) -> int | None:
    if not pattern:
        return None
    ignored = ("/opt/homebrew/bin/nvim", " rg ", "rg ")
    for line in _process_lines():
        if pattern in line and not any(ignore in line for ignore in ignored):
            pid = _pid_from_process_line(line)
            if _pid_running(pid):
                return pid
    return None


def _robot_owner_lines() -> list[str]:
    owners = ("hybrid_robot_loop.py", "local_modal_robot_bridge.py", "teleop_viewer.py", "direct_robot_control.py")
    ignored = ("/opt/homebrew/bin/nvim", " rg ", "rg ")
    return [
        line
        for line in _process_lines()
        if any(owner in line for owner in owners) and not any(ignore in line for ignore in ignored)
    ]


def _ensure_no_robot_owner(*, allow: bool = False) -> bool:
    owners = _robot_owner_lines()
    if not owners or allow:
        return True
    print("another robot owner is already running; stop it before commanding hardware:", file=sys.stderr)
    for line in owners:
        print(f"  {line}", file=sys.stderr)
    return False


def _start_background(name: str, cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    LOG_DIR.mkdir(exist_ok=True)
    pid = _read_pid(name)
    if _pid_running(pid):
        print(f"{name} already running: pid={pid}")
        return int(pid)
    patterns = {
        "camera": ["camera_http_server.py"],
        "bridge": ["slcan_bridge.py"],
        "viewer": ["teleop_viewer.py"],
        "model": ["hybrid_robot_loop.py", "local_modal_robot_bridge.py"],
        "policy": ["yam_lerobot_policy_server.py"],
    }
    discovered = next(
        (pid for pattern in patterns.get(name, []) if (pid := _discover_pid(pattern)) is not None),
        None,
    )
    if discovered is not None:
        _pid_path(name).write_text(f"{discovered}\n")
        print(f"{name} already running: pid={discovered}")
        return discovered

    log = _log_path(name).open("ab")
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, **(env or {})},
    )
    _pid_path(name).write_text(f"{process.pid}\n")
    print(f"started {name}: pid={process.pid}, log={_log_path(name)}")
    return process.pid


def _stop_pid(name: str, *, sig: int = signal.SIGTERM) -> None:
    pid = _read_pid(name)
    if not _pid_running(pid):
        print(f"{name} not running")
        return
    assert pid is not None
    os.killpg(pid, sig)
    print(f"stopped {name}: pid={pid}")


def _wait_for_socket(timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if CAN_SOCKET.exists():
            return True
        time.sleep(0.1)
    return CAN_SOCKET.exists()


def _compose_task(task: str, context: str | None, context_file: str | None) -> str:
    parts = [task.strip()]
    if context_file:
        text = Path(context_file).expanduser().read_text().strip()
        if text:
            parts.append(f"Context: {text}")
    if context:
        parts.append(f"Context: {context.strip()}")
    return "\n".join(part for part in parts if part)


def cmd_status(_args: argparse.Namespace) -> int:
    status = {
        "hard_stop": STOP_FILE.exists(),
        "can_socket": CAN_SOCKET.exists(),
        "pid_files": {
            name: {"pid": _read_pid(name), "running": _pid_running(_read_pid(name))}
            for name in ("camera", "bridge", "viewer", "model", "policy")
        },
        "processes": _process_lines(),
    }
    print(json.dumps(status, indent=2))
    return 0


def cmd_hard_stop(_args: argparse.Namespace) -> int:
    STOP_FILE.write_text(f"requested_at={time.time()}\n")
    print(f"hard stop set: {STOP_FILE}")
    return 0


def cmd_clear_stop(_args: argparse.Namespace) -> int:
    STOP_FILE.unlink(missing_ok=True)
    print(f"hard stop cleared: {STOP_FILE}")
    return 0


def cmd_start_camera(args: argparse.Namespace) -> int:
    cmd = [
        _python(),
        "scripts/camera_http_server.py",
        "--camera-index",
        str(args.camera_index),
        "--max-camera-index",
        str(args.max_camera_index),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    _start_background("camera", cmd)
    return 0


def cmd_camera_snapshot(args: argparse.Namespace) -> int:
    if args.ensure_camera:
        camera_args = argparse.Namespace(camera_index=args.camera_index, max_camera_index=args.max_camera_index, host="127.0.0.1", port=8766)
        cmd_start_camera(camera_args)

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + args.timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(args.camera_url, timeout=1.0) as response:
                body = response.read()
            output.write_bytes(body)
            print(json.dumps({"camera_url": args.camera_url, "output": str(output), "bytes": len(body)}))
            return 0
        except Exception as exc:  # noqa: BLE001 - report final camera fetch failure.
            last_error = exc
            time.sleep(0.2)
    print(f"camera snapshot failed from {args.camera_url}: {last_error}", file=sys.stderr)
    return 1


def cmd_start_bridge(args: argparse.Namespace) -> int:
    if CAN_SOCKET.exists() and not any("slcan_bridge.py" in line for line in _process_lines()):
        CAN_SOCKET.unlink()

    cmd = [
        _python(),
        "can-bridge/slcan_bridge.py",
        "--serial",
        args.serial_port,
        "--bitrate",
        str(args.bitrate),
    ]
    _start_background("bridge", cmd)
    if not _wait_for_socket():
        print(f"bridge did not create {CAN_SOCKET}; check {_log_path('bridge')}", file=sys.stderr)
        return 1
    return 0


def cmd_policy_server(args: argparse.Namespace) -> int:
    cmd = [
        _python(),
        "scripts/yam_lerobot_policy_server.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device",
        args.device,
    ]
    if args.policy_path:
        cmd.extend(["--policy-path", args.policy_path])
    if args.background:
        _start_background("policy", cmd)
        return 0
    return subprocess.call(cmd, cwd=ROOT)


def cmd_start_viewer(args: argparse.Namespace) -> int:
    bridge_args = argparse.Namespace(serial_port=args.serial_port, bitrate=args.bitrate)
    if cmd_start_bridge(bridge_args) != 0:
        return 1
    env = {"CAN_MAC_PATCH": "1", "YAM_RECORD_CAMERA_URL": args.record_camera_url}
    cmd = [_python(), "teleop_viewer.py"]
    if args.background:
        _start_background("viewer", cmd, env=env)
        return 0
    return subprocess.call(cmd, cwd=ROOT, env={**os.environ, **env})


def cmd_stop(args: argparse.Namespace) -> int:
    if args.hard_stop:
        STOP_FILE.write_text(f"requested_at={time.time()}\n")
    for name in args.targets:
        _stop_pid(name)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.allow_modal_bimanual:
        print(
            "yamctl run is the legacy MolmoAct2 bimanual path. "
            "Use `yamctl hybrid ...` for the one-arm policy, or pass "
            "`--allow-modal-bimanual` to run the legacy path explicitly.",
            file=sys.stderr,
        )
        return 2
    if not _ensure_no_robot_owner(allow=args.allow_concurrent_owner):
        return 3
    if args.clear_stop:
        STOP_FILE.unlink(missing_ok=True)
    if STOP_FILE.exists() and not args.ignore_hard_stop:
        print(f"hard stop exists at {STOP_FILE}; run `yamctl clear-stop` first", file=sys.stderr)
        return 2

    if args.profile == "fast":
        args.hz = args.hz if args.hz != 0.25 else 0.5
        args.num_steps = args.num_steps if args.num_steps != 5 else 8
        args.max_speed = args.max_speed if args.max_speed != 0.02 else 0.05
        args.max_target_delta = args.max_target_delta if args.max_target_delta != 0.50 else 0.75
        args.max_iterations = args.max_iterations if args.max_iterations != 10 else 30
        args.execute_action_steps = args.execute_action_steps if args.execute_action_steps != 1 else 3
        args.action_step_delay = args.action_step_delay if args.action_step_delay != 0.0 else 0.05
        args.command_dt = args.command_dt if args.command_dt != 0.0 else 0.25

    if args.ensure_camera:
        camera_args = argparse.Namespace(camera_index=args.camera_index, max_camera_index=args.max_camera_index, host="127.0.0.1", port=8766)
        cmd_start_camera(camera_args)
    bridge_args = argparse.Namespace(serial_port=args.serial_port, bitrate=args.bitrate)
    if cmd_start_bridge(bridge_args) != 0:
        return 1

    task = _compose_task(args.task, args.context, args.context_file)
    cmd = [
        _python(),
        "scripts/local_modal_robot_bridge.py",
        "--http-url",
        args.http_url,
        "--task",
        task,
        "--camera-url",
        args.camera_url,
        "--hz",
        str(args.hz),
        "--num-steps",
        str(args.num_steps),
        "--max-speed",
        str(args.max_speed),
        "--max-target-delta",
        str(args.max_target_delta),
        "--max-temp-mos",
        str(args.max_temp_mos),
        "--max-temp-rotor",
        str(args.max_temp_rotor),
        "--min-gripper-command",
        str(args.min_gripper_command),
        "--max-gripper-command",
        str(args.max_gripper_command),
        "--max-iterations",
        str(args.max_iterations),
        "--execute-action-steps",
        str(args.execute_action_steps),
        "--action-step-delay",
        str(args.action_step_delay),
        "--command-dt",
        str(args.command_dt),
        "--http-timeout",
        str(args.http_timeout),
        "--execute",
        "--force-execute-unsafe",
        "--allow-bimanual-slice",
    ]
    if args.cap_gripper_at_current_open:
        cmd.append("--cap-gripper-at-current-open")
    if args.background:
        _start_background("model", cmd)
        return 0
    return subprocess.call(cmd, cwd=ROOT)


def cmd_direct(args: argparse.Namespace) -> int:
    if not _ensure_no_robot_owner(allow=args.allow_concurrent_owner):
        return 3
    bridge_args = argparse.Namespace(serial_port=args.serial_port, bitrate=args.bitrate)
    if cmd_start_bridge(bridge_args) != 0:
        return 1

    cmd = [
        _python(),
        "scripts/direct_robot_control.py",
        "--duration",
        str(args.duration),
        "--steps",
        str(args.steps),
        "--max-delta",
        str(args.max_delta),
        "--max-temp-mos",
        str(args.max_temp_mos),
        "--max-temp-rotor",
        str(args.max_temp_rotor),
        "--min-gripper-command",
        str(args.min_gripper_command),
        "--max-gripper-command",
        str(args.max_gripper_command),
    ]
    if args.target is not None:
        cmd.extend(["--target", args.target])
    if args.delta is not None:
        cmd.extend(["--delta", args.delta])
    if args.gripper is not None:
        cmd.extend(["--gripper", str(args.gripper)])
    if args.read_only:
        cmd.append("--read-only")
    if args.ignore_hard_stop:
        cmd.append("--ignore-hard-stop")
    return subprocess.call(cmd, cwd=ROOT)


def cmd_hybrid(args: argparse.Namespace) -> int:
    if not _ensure_no_robot_owner(allow=args.allow_concurrent_owner):
        return 3
    if args.ensure_camera:
        camera_args = argparse.Namespace(camera_index=args.camera_index, max_camera_index=args.max_camera_index, host="127.0.0.1", port=8766)
        cmd_start_camera(camera_args)
    bridge_args = argparse.Namespace(serial_port=args.serial_port, bitrate=args.bitrate)
    if cmd_start_bridge(bridge_args) != 0:
        return 1

    http_url = args.http_url
    if args.policy_kind == "modal" and http_url == DEFAULT_ONE_ARM_POLICY_HTTP_URL:
        http_url = DEFAULT_MODAL_POLICY_HTTP_URL

    cmd = [
        _python(),
        "scripts/hybrid_robot_loop.py",
        _compose_task(args.task, args.context, args.context_file),
        "--http-url",
        http_url,
        "--policy-kind",
        args.policy_kind,
        "--camera-url",
        args.camera_url,
        "--trace-dir",
        args.trace_dir,
        "--hz",
        str(args.hz),
        "--num-steps",
        str(args.num_steps),
        "--http-timeout",
        str(args.http_timeout),
        "--max-iterations",
        str(args.max_iterations),
        "--max-speed",
        str(args.max_speed),
        "--max-target-delta",
        str(args.max_target_delta),
        "--max-temp-mos",
        str(args.max_temp_mos),
        "--max-temp-rotor",
        str(args.max_temp_rotor),
        "--min-gripper-command",
        str(args.min_gripper_command),
        "--max-gripper-command",
        str(args.max_gripper_command),
        "--arm-slice",
        args.arm_slice,
        "--action-step",
        str(args.action_step),
        "--execute-action-steps",
        str(args.execute_action_steps),
        "--action-step-delay",
        str(args.action_step_delay),
        "--command-dt",
        str(args.command_dt),
        "--align-px",
        str(args.align_px),
        "--descend-px",
        str(args.descend_px),
        "--align-joint1-step",
        str(args.align_joint1_step),
        "--descend-joint3-step",
        str(args.descend_joint3_step),
        "--lift-joint3-step",
        str(args.lift_joint3_step),
        "--correction-duration",
        str(args.correction_duration),
        "--correction-steps",
        str(args.correction_steps),
    ]
    if args.codex_corrections:
        cmd.append("--codex-corrections")
    if args.auto_grasp:
        cmd.append("--auto-grasp")
    if args.stop_on_model_warning:
        cmd.append("--stop-on-model-warning")
    if args.observe_only:
        cmd.append("--observe-only")
    if args.clear_stop:
        cmd.append("--clear-stop")
    if args.ignore_hard_stop:
        cmd.append("--ignore-hard-stop")
    if args.background:
        _start_background("model", cmd)
        return 0
    return subprocess.call(cmd, cwd=ROOT)


def cmd_dataset(args: argparse.Namespace) -> int:
    cmd = [_python(), "scripts/yam_lerobot_dataset.py", args.dataset_command]
    for name, value in vars(args).items():
        if name in {"func", "command", "dataset_command"} or value is None or value is False:
            continue
        option = "--" + name.replace("_", "-")
        if value is True:
            cmd.append(option)
        else:
            cmd.extend([option, str(value)])
    return subprocess.call(cmd, cwd=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yamctl", description="Operate the local YAM control stack.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show hard-stop, socket, and process state.")
    status.set_defaults(func=cmd_status)

    hard_stop = sub.add_parser("hard-stop", help="Create HARD_STOP so model loops exit.")
    hard_stop.set_defaults(func=cmd_hard_stop)

    clear_stop = sub.add_parser("clear-stop", help="Remove HARD_STOP.")
    clear_stop.set_defaults(func=cmd_clear_stop)

    camera = sub.add_parser("camera", help="Start the local HTTP camera helper.")
    camera.add_argument("--camera-index", default="0", help="OpenCV camera index, or 'auto' to probe indexes.")
    camera.add_argument("--max-camera-index", type=int, default=9)
    camera.add_argument("--host", default="127.0.0.1")
    camera.add_argument("--port", type=int, default=8766)
    camera.set_defaults(func=cmd_start_camera)

    snapshot = sub.add_parser("camera-snapshot", help="Capture one local camera frame to a JPEG.")
    snapshot.add_argument("--camera-url", default=DEFAULT_CAMERA_URL)
    snapshot.add_argument("--output", default="logs/latest-camera.jpg")
    snapshot.add_argument("--timeout", type=float, default=5.0)
    snapshot.add_argument("--camera-index", default="0")
    snapshot.add_argument("--max-camera-index", type=int, default=9)
    snapshot.add_argument("--ensure-camera", action="store_true", help="Start the camera helper before fetching.")
    snapshot.set_defaults(func=cmd_camera_snapshot)

    bridge = sub.add_parser("bridge", help="Start the SLCAN-to-/tmp/can0.sock bridge.")
    bridge.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    bridge.add_argument("--bitrate", type=int, default=1_000_000)
    bridge.set_defaults(func=cmd_start_bridge)

    policy = sub.add_parser("policy-server", help="Start the local one-arm LeRobot ACT policy server.")
    policy.add_argument("--policy-path", help="Local path or Hub id for a trained one-arm ACT policy.")
    policy.add_argument("--host", default="127.0.0.1")
    policy.add_argument("--port", type=int, default=8777)
    policy.add_argument("--device", default="mps")
    policy.add_argument("--background", action="store_true")
    policy.set_defaults(func=cmd_policy_server)

    viewer = sub.add_parser("viewer", help="Start the manual YAM viewer.")
    viewer.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    viewer.add_argument("--bitrate", type=int, default=1_000_000)
    viewer.add_argument("--record-camera-url", default=DEFAULT_CAMERA_URL)
    viewer.add_argument("--background", action="store_true")
    viewer.set_defaults(func=cmd_start_viewer)

    stop = sub.add_parser("stop", help="Stop background processes tracked by yamctl.")
    stop.add_argument(
        "targets",
        nargs="*",
        default=["model", "viewer", "policy", "bridge"],
        choices=["model", "viewer", "policy", "bridge", "camera"],
    )
    stop.add_argument("--hard-stop", action="store_true", help="Set HARD_STOP before stopping processes.")
    stop.set_defaults(func=cmd_stop)

    run = sub.add_parser("run", help="Legacy Modal bimanual model control with a task and optional context.")
    run.add_argument("task", help="Robot task prompt.")
    run.add_argument("--context", help="Extra scene/task context appended to the prompt.")
    run.add_argument("--context-file", help="File containing extra context appended to the prompt.")
    run.add_argument("--http-url", default=DEFAULT_POLICY_HTTP_URL)
    run.add_argument("--camera-url", default=DEFAULT_CAMERA_URL)
    run.add_argument("--camera-index", default="0")
    run.add_argument("--max-camera-index", type=int, default=9)
    run.add_argument("--ensure-camera", action="store_true", help="Start the camera helper before model control.")
    run.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    run.add_argument("--bitrate", type=int, default=1_000_000)
    run.add_argument("--hz", type=float, default=0.25)
    run.add_argument("--num-steps", type=int, default=5)
    run.add_argument("--max-speed", type=float, default=0.02)
    run.add_argument("--max-target-delta", type=float, default=0.50)
    run.add_argument("--max-temp-mos", type=float, default=55.0)
    run.add_argument("--max-temp-rotor", type=float, default=100.0)
    run.add_argument("--min-gripper-command", type=float, default=0.01)
    run.add_argument("--max-gripper-command", type=float, default=0.59)
    run.add_argument(
        "--cap-gripper-at-current-open",
        action="store_true",
        help="Do not let the policy open the gripper past its startup position.",
    )
    run.add_argument("--max-iterations", type=int, default=10)
    run.add_argument(
        "--execute-action-steps",
        type=int,
        default=1,
        help="Execute this many consecutive trajectory steps from each model response.",
    )
    run.add_argument(
        "--action-step-delay",
        type=float,
        default=0.0,
        help="Delay between local trajectory step commands.",
    )
    run.add_argument(
        "--command-dt",
        type=float,
        default=0.0,
        help="Fixed dt used for speed clipping each command; 0 uses wall-clock dt.",
    )
    run.add_argument(
        "--profile",
        choices=["normal", "fast"],
        default="normal",
        help="Use a preset for rollout length and command speed.",
    )
    run.add_argument("--http-timeout", type=float, default=300.0)
    run.add_argument("--background", action="store_true")
    run.add_argument("--clear-stop", action="store_true")
    run.add_argument("--ignore-hard-stop", action="store_true")
    run.add_argument("--allow-concurrent-owner", action="store_true", help="Bypass the single robot-owner guard.")
    run.add_argument("--allow-modal-bimanual", action="store_true", help="Explicitly allow the legacy bimanual Modal policy path.")
    run.set_defaults(func=cmd_run)

    direct = sub.add_parser("direct", help="Send explicit local joint/gripper controls without Modal.")
    direct.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    direct.add_argument("--bitrate", type=int, default=1_000_000)
    direct.add_argument("--target", help="Seven joint targets as comma- or space-separated numbers.")
    direct.add_argument("--delta", help="Seven relative joint deltas as comma- or space-separated numbers.")
    direct.add_argument("--gripper", type=float, help="Set normalized joint 7/gripper command.")
    direct.add_argument("--duration", type=float, default=1.0)
    direct.add_argument("--steps", type=int, default=20)
    direct.add_argument("--max-delta", type=float, default=0.20)
    direct.add_argument("--max-temp-mos", type=float, default=55.0)
    direct.add_argument("--max-temp-rotor", type=float, default=100.0)
    direct.add_argument("--min-gripper-command", type=float, default=0.01)
    direct.add_argument("--max-gripper-command", type=float, default=0.59)
    direct.add_argument("--read-only", action="store_true")
    direct.add_argument("--ignore-hard-stop", action="store_true")
    direct.add_argument("--allow-concurrent-owner", action="store_true", help="Bypass the single robot-owner guard.")
    direct.set_defaults(func=cmd_direct)

    hybrid = sub.add_parser("hybrid", help="Run one-arm policy actions with local camera/state verification and Codex corrections.")
    hybrid.add_argument("task", help="Robot task prompt.")
    hybrid.add_argument("--context", help="Extra scene/task context appended to the prompt.")
    hybrid.add_argument("--context-file", help="File containing extra context appended to the prompt.")
    hybrid.add_argument("--policy-kind", choices=["lerobot-act", "modal"], default="lerobot-act")
    hybrid.add_argument("--http-url", default=DEFAULT_ONE_ARM_POLICY_HTTP_URL)
    hybrid.add_argument("--camera-url", default=DEFAULT_CAMERA_URL)
    hybrid.add_argument("--camera-index", default="0")
    hybrid.add_argument("--max-camera-index", type=int, default=9)
    hybrid.add_argument("--ensure-camera", action="store_true", help="Start the camera helper before hybrid control.")
    hybrid.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    hybrid.add_argument("--bitrate", type=int, default=1_000_000)
    hybrid.add_argument("--trace-dir", default="logs/hybrid-latest")
    hybrid.add_argument("--hz", type=float, default=0.5)
    hybrid.add_argument("--num-steps", type=int, default=8)
    hybrid.add_argument("--http-timeout", type=float, default=300.0)
    hybrid.add_argument("--max-iterations", type=int, default=8)
    hybrid.add_argument("--max-speed", type=float, default=0.05)
    hybrid.add_argument("--max-target-delta", type=float, default=0.75)
    hybrid.add_argument("--max-temp-mos", type=float, default=55.0)
    hybrid.add_argument("--max-temp-rotor", type=float, default=100.0)
    hybrid.add_argument("--min-gripper-command", type=float, default=0.01)
    hybrid.add_argument("--max-gripper-command", type=float, default=0.59)
    hybrid.add_argument("--arm-slice", choices=["first", "second"], default="first")
    hybrid.add_argument("--action-step", type=int, default=0)
    hybrid.add_argument("--execute-action-steps", type=int, default=3)
    hybrid.add_argument("--action-step-delay", type=float, default=0.05)
    hybrid.add_argument("--command-dt", type=float, default=0.25)
    hybrid.add_argument("--codex-corrections", action="store_true", help="Enable local visual correction moves.")
    hybrid.add_argument("--auto-grasp", action="store_true", help="Allow local close-and-lift correction when aligned.")
    hybrid.add_argument("--align-px", type=float, default=55.0)
    hybrid.add_argument("--descend-px", type=float, default=70.0)
    hybrid.add_argument("--align-joint1-step", type=float, default=0.10)
    hybrid.add_argument("--descend-joint3-step", type=float, default=0.10)
    hybrid.add_argument("--lift-joint3-step", type=float, default=0.22)
    hybrid.add_argument("--correction-duration", type=float, default=0.8)
    hybrid.add_argument("--correction-steps", type=int, default=16)
    hybrid.add_argument("--stop-on-model-warning", action="store_true")
    hybrid.add_argument("--observe-only", action="store_true", help="Call policy and write diagnostics without commanding robot motion.")
    hybrid.add_argument("--background", action="store_true")
    hybrid.add_argument("--clear-stop", action="store_true")
    hybrid.add_argument("--ignore-hard-stop", action="store_true")
    hybrid.add_argument("--allow-concurrent-owner", action="store_true", help="Bypass the single robot-owner guard.")
    hybrid.set_defaults(func=cmd_hybrid)

    dataset = sub.add_parser("dataset", help="Validate/export teleop recordings for LeRobot training.")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    dataset_summary = dataset_sub.add_parser("summary", help="Validate raw teleop recording episodes.")
    dataset_summary.add_argument("--recordings-dir", default="recordings")
    dataset_summary.set_defaults(func=cmd_dataset)

    dataset_export = dataset_sub.add_parser("export", help="Export valid recordings to a LeRobotDataset.")
    dataset_export.add_argument("--recordings-dir", default="recordings")
    dataset_export.add_argument("--output-root", default="lerobot-data")
    dataset_export.add_argument("--repo-id", default="local/yam-bread-toaster")
    dataset_export.add_argument("--task", default="put bread in toaster")
    dataset_export.add_argument("--fps", type=int, default=10)
    dataset_export.add_argument("--image-width", type=int, default=640)
    dataset_export.add_argument("--image-height", type=int, default=360)
    dataset_export.set_defaults(func=cmd_dataset)

    dataset_train = dataset_sub.add_parser("train-act", help="Run or print the ACT training command.")
    dataset_train.add_argument("--repo-id", default="local/yam-bread-toaster")
    dataset_train.add_argument("--dataset-root", default="lerobot-data")
    dataset_train.add_argument("--output-dir", default="outputs/train/yam-bread-toaster-act")
    dataset_train.add_argument("--job-name", default="yam_bread_toaster_act")
    dataset_train.add_argument("--steps", type=int, default=20000)
    dataset_train.add_argument("--batch-size", type=int, default=32)
    dataset_train.add_argument("--chunk-size", type=int, default=50)
    dataset_train.add_argument("--n-action-steps", type=int, default=50)
    dataset_train.add_argument("--device", default="mps")
    dataset_train.add_argument("--dry-run", action="store_true")
    dataset_train.set_defaults(func=cmd_dataset)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
