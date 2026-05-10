"""Command line operator tools for the local YAM control stack."""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from http_camera_fetch import build_urllib_camera_request, fetch_rgb_from_camera_url  # noqa: E402

LOG_DIR = ROOT / "logs"
STOP_FILE = ROOT / "HARD_STOP"
CAN_SOCKET = Path("/tmp/can0.sock")
CAN1_SOCKET = Path("/tmp/can1.sock")

DEFAULT_SERIAL_PORT = os.environ.get("YAM_SERIAL_PORT", "/dev/cu.usbmodem206D338A594E1")
DEFAULT_CAMERA_URL = (
    os.environ.get("YAM_CAMERA_URL")
    or os.environ.get("YAM_FRONT_CAMERA_URL")
    or "http://127.0.0.1:8766/frame.jpg"
)
DEFAULT_ONE_ARM_POLICY_HTTP_URL = os.environ.get("YAM_ONE_ARM_POLICY_HTTP_URL", "http://127.0.0.1:8777/infer")
# Default Modal Molmo HTTP `/infer` when YAM_MODAL_POLICY_HTTP_URL is unset (override per deployment).
_DEFAULT_MODAL_INFER_URL = "https://aacmcgovern--yam-molmoact2-http-bridge-v3-serve.modal.run/infer"
DEFAULT_MODAL_POLICY_HTTP_URL = os.environ.get("YAM_MODAL_POLICY_HTTP_URL", _DEFAULT_MODAL_INFER_URL)
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


_BRIDGE_PS_MARKERS = (
    "slcan_bridge.py",
    "release/can-bridge",
    "debug/can-bridge",
    "start_bimanual_bridges.sh",
)


def _bridge_ps_lines() -> list[str]:
    """ps lines for processes that may own /tmp/can0.sock or /tmp/can1.sock."""

    result = subprocess.run(
        ["ps", "-axo", "pid,command"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    lines: list[str] = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if any(marker in line for marker in _BRIDGE_PS_MARKERS):
            lines.append(line)
    return lines


def _discover_bridge_pid() -> int | None:
    ignored = ("/opt/homebrew/bin/nvim", " rg ", "rg ")
    for line in _bridge_ps_lines():
        if any(ignore in line for ignore in ignored):
            continue
        pid = _pid_from_process_line(line)
        if _pid_running(pid):
            return pid
    return None


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
        "direct_bimanual_parallel.py",
        "local_modal_robot_bridge.py",
        "slcan_bridge.py",
        "release/can-bridge",
        "debug/can-bridge",
        "start_bimanual_bridges.sh",
        "teleop_viewer.py",
        "multi_camera_ws_server.py",
        "yam_control_ws_server.py",
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
    owners = (
        "hybrid_robot_loop.py",
        "local_modal_robot_bridge.py",
        "teleop_viewer.py",
        "direct_robot_control.py",
        "direct_bimanual_parallel.py",
        "yam_control_ws_server.py",
    )
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
        "cameras": ["multi_camera_ws_server.py"],
        "bridge": ["slcan_bridge.py"],
        "control": ["yam_control_ws_server.py"],
        "viewer": ["teleop_viewer.py"],
        "model": ["hybrid_robot_loop.py", "local_modal_robot_bridge.py", "direct_bimanual_parallel.py"],
        "policy": ["yam_lerobot_policy_server.py"],
    }
    if name == "bridge":
        discovered = _discover_bridge_pid()
    else:
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


def _wait_for_bimanual_sockets(timeout_s: float = 8.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if CAN_SOCKET.exists() and CAN1_SOCKET.exists():
            return True
        time.sleep(0.1)
    return CAN_SOCKET.exists() and CAN1_SOCKET.exists()


def _compose_task(task: str, context: str | None, context_file: str | None) -> str:
    parts = [task.strip()]
    if context_file:
        text = Path(context_file).expanduser().read_text().strip()
        if text:
            parts.append(f"Context: {text}")
    if context:
        parts.append(f"Context: {context.strip()}")
    return "\n".join(part for part in parts if part)


def _hybrid_camera_http_index(args: argparse.Namespace) -> int:
    """OpenCV device index for ``camera_http_server`` when ``yamctl hybrid --ensure-camera`` is used."""
    idx = args.camera_index
    if not getattr(args, "right_orbbec", False):
        return idx
    if getattr(args, "top_camera_index", None) is not None:
        return int(args.top_camera_index)
    if idx == 0:
        return int(os.environ.get("YAM_ENSURE_CAMERA_INDEX_WITH_ORBBEC", "1"))
    return idx


def cmd_status(_args: argparse.Namespace) -> int:
    status = {
        "hard_stop": STOP_FILE.exists(),
        "can_socket": CAN_SOCKET.exists(),
        "can1_socket": CAN1_SOCKET.exists(),
        "pid_files": {
            name: {"pid": _read_pid(name), "running": _pid_running(_read_pid(name))}
            for name in ("cameras", "bridge", "viewer", "control", "model", "policy")
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


def cmd_can_check(_args: argparse.Namespace) -> int:
    """Quick check: Unix CAN sockets exist and accept a connection (bridge listening)."""

    def probe(path: Path) -> dict:
        row: dict = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            row["connect_ok"] = False
            row["hint"] = "start can-bridge or can-bridge/start_bimanual_bridges.sh"
            return row
        row["is_socket"] = path.is_socket()
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.75)
            s.connect(str(path))
            s.shutdown(socket.SHUT_RDWR)
            s.close()
            row["connect_ok"] = True
        except OSError as exc:
            row["connect_ok"] = False
            row["connect_error"] = str(exc)
            row["hint"] = "stale socket file or bridge crashed; remove socket only if no bridge process"
        return row

    out = {
        "bridge_process_hints": _bridge_ps_lines()[:5],
        "can0": probe(Path("/tmp/can0.sock")),
        "can1": probe(Path("/tmp/can1.sock")),
    }
    print(json.dumps(out, indent=2))
    return 0


def _http_jpeg_ready(url: str, *, timeout_s: float = 0.75) -> bool:
    """True if ``url`` returns HTTP 200 with a body (JPEG helper is warm)."""

    try:
        req = build_urllib_camera_request(url)
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return int(getattr(response, "status", 200) or 200) == 200 and len(response.read(4096)) > 100
    except OSError:
        return False


def cmd_start_camera(args: argparse.Namespace, *, background_name: str = "camera") -> int:
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
    _start_background(background_name, cmd)
    return 0


def cmd_orbbec_camera_hint(args: argparse.Namespace) -> int:
    """Print the exact ``sudo`` invocation for the Orbbec wrist HTTP server.

    On macOS, the Orbbec SDK can only escape ``VDCAssistant``'s UVC lock when run
    as root (Orbbec issues #9 and #124). We do not run other YAM stack pieces as
    root for safety, so this command just prints the recommended command line
    plus the matching ``--right-camera-url`` for ``yamctl hybrid``.
    """

    venv_python = ROOT / ".venv" / "bin" / "python"
    runner = str(venv_python) if venv_python.exists() else "uv run python"
    cmd_parts = [
        "sudo -E",
        runner,
        "scripts/orbbec_camera_server.py",
        f"--host {args.host}",
        f"--port {args.port}",
        f"--quality {args.quality}",
        f"--max-fps {args.max_fps}",
        f"--flip {args.flip}",
    ]
    print("# Orbbec wrist HTTP server (macOS needs sudo for UVC; this server is the ONLY root piece):")
    print(" ".join(cmd_parts))
    print("# Orientation: JPEGs use --flip (default vertical) inside the server; yamctl hybrid has no wrist flip flags.")
    print()
    print("# Then in another terminal (no sudo) run hybrid against the Orbbec URL:")
    print(
        f'uv run yamctl hybrid "pick up the hat" --policy-kind modal --top-camera-index 1 '
        f"--right-camera-url http://{args.host}:{args.port}/frame.jpg"
    )
    print(
        "# Optional: Molmo \"left\" from iPhone (Continuity Camera). Find OpenCV index via "
        "`uv run python scripts/opencv_camera_probe.py`, then:"
    )
    print(
        "#   ... same hybrid line ... "
        "--ensure-left-camera --left-camera-index N"
    )
    print()
    print("# Verify a snapshot at any time:")
    print(f"curl -s http://{args.host}:{args.port}/frame.jpg -o logs/orbbec-snapshot.jpg && echo logs/orbbec-snapshot.jpg")
    return 0


def cmd_start_cameras(args: argparse.Namespace) -> int:
    cmd = [
        _python(),
        "scripts/multi_camera_ws_server.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-camera-index",
        str(args.max_camera_index),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--quality",
        str(args.quality),
    ]
    if args.camera_specs:
        cmd.extend(["--camera-specs", args.camera_specs])
    else:
        cmd.extend(["--auto-count", str(args.auto_count)])
    _start_background("cameras", cmd)
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
    cam_url = args.camera_url.strip()
    while time.time() < deadline:
        try:
            if cam_url.startswith(("ws://", "wss://")):
                rgb = fetch_rgb_from_camera_url(cam_url, timeout=min(5.0, max(0.5, deadline - time.time())))
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=88)
                body = buf.getvalue()
            else:
                with urllib.request.urlopen(build_urllib_camera_request(cam_url), timeout=1.0) as response:
                    body = response.read()
            output.write_bytes(body)
            print(json.dumps({"camera_url": cam_url, "output": str(output), "bytes": len(body)}))
            return 0
        except Exception as exc:  # noqa: BLE001 - report final camera fetch failure.
            last_error = exc
            time.sleep(0.2)
    print(f"camera snapshot failed from {cam_url}: {last_error}", file=sys.stderr)
    return 1


def _unix_can_bridge_running() -> bool:
    """True if a CAN bridge process is present (Rust release/debug, SLCAN, or startup script)."""

    return bool(_bridge_ps_lines())


def cmd_start_bridge(args: argparse.Namespace) -> int:
    if CAN_SOCKET.exists():
        if _unix_can_bridge_running():
            pid = _discover_bridge_pid()
            if pid is not None:
                _pid_path("bridge").write_text(f"{pid}\n")
            print("CAN bridge already running (socket present); leaving /tmp/can0.sock in place.")
            return 0
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


def cmd_start_control_server(args: argparse.Namespace) -> int:
    if not _ensure_no_robot_owner(allow=args.allow_concurrent_owner):
        return 1
    cmd = [
        _python(),
        "scripts/yam_control_ws_server.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--arm-specs",
        args.arm_specs,
        "--min-gripper",
        str(args.min_gripper),
        "--max-gripper",
        str(args.max_gripper),
        "--reconnect-initial-delay",
        str(args.reconnect_initial_delay),
        "--reconnect-max-delay",
        str(args.reconnect_max_delay),
        "--bridge-startup-timeout",
        str(args.bridge_startup_timeout),
    ]
    if args.background:
        _start_background("control", cmd)
        return 0
    return subprocess.call(cmd, cwd=ROOT)


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
        args.max_iterations = args.max_iterations if args.max_iterations != 10 else 30
        args.execute_action_steps = args.execute_action_steps if args.execute_action_steps != 1 else 3
        args.action_step_delay = args.action_step_delay if args.action_step_delay != 0.0 else 0.05

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
        "--channel",
        args.channel,
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


def _build_direct_both_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        _python(),
        "scripts/direct_bimanual_parallel.py",
        "--left-can",
        args.left_can,
        "--right-can",
        args.right_can,
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
    if getattr(args, "target", None) is not None:
        cmd.extend(["--target", args.target])
    if getattr(args, "delta", None) is not None:
        cmd.extend(["--delta", args.delta])
    if getattr(args, "gripper", None) is not None:
        cmd.extend(["--gripper", str(args.gripper)])
    if getattr(args, "read_only", False):
        cmd.append("--read-only")
    if args.ignore_hard_stop:
        cmd.append("--ignore-hard-stop")
    return cmd


def cmd_direct_both(args: argparse.Namespace) -> int:
    if not _ensure_no_robot_owner(allow=args.allow_concurrent_owner):
        return 3
    bridge_args = argparse.Namespace(serial_port=args.serial_port, bitrate=args.bitrate)
    if cmd_start_bridge(bridge_args) != 0:
        return 1

    return subprocess.call(_build_direct_both_cmd(args), cwd=ROOT)


def _seven_joint_delta_string(joint_index: int, value: float) -> str:
    parts = [0.0] * 7
    parts[joint_index] = value
    return ",".join(str(p) for p in parts)


def cmd_smoke_bimanual(args: argparse.Namespace) -> int:
    """Read both arms, then nudge one joint +/− on both CAN interfaces."""
    if not _ensure_no_robot_owner(allow=args.allow_concurrent_owner):
        return 3
    bridge_args = argparse.Namespace(serial_port=args.serial_port, bitrate=args.bitrate)
    if cmd_start_bridge(bridge_args) != 0:
        return 1

    ji = int(args.joint_index)
    amp = float(args.amplitude)
    base = argparse.Namespace(**vars(args))
    sequence: list[tuple[str, bool, str | None]] = [
        ("read-only (both arms)", True, None),
        (f"joint {ji} +{amp} (both arms)", False, _seven_joint_delta_string(ji, amp)),
        (f"joint {ji} -{amp} (both arms)", False, _seven_joint_delta_string(ji, -amp)),
    ]
    for label, read_only, delta in sequence:
        print(f"smoke-bimanual: {label}", flush=True)
        base.read_only = read_only
        base.delta = delta
        base.target = None
        rc = subprocess.call(_build_direct_both_cmd(base), cwd=ROOT)
        if rc != 0:
            return rc
    return 0


def cmd_hybrid(args: argparse.Namespace) -> int:
    if not _ensure_no_robot_owner(allow=args.allow_concurrent_owner):
        return 3
    if getattr(args, "right_orbbec", False):
        sys.path.insert(0, str(ROOT / "scripts"))
        from orbbec_wrist import release_uvc_interfering_processes

        release_uvc_interfering_processes()
    if args.ensure_camera:
        cam_idx = _hybrid_camera_http_index(args)
        if cam_idx != args.camera_index:
            print(
                f"ensure-camera: OpenCV index {cam_idx} for HTTP helper "
                f"(was {args.camera_index}) — avoid grabbing index 0 while Orbbec wrist uses the SDK.",
                file=sys.stderr,
            )
        camera_args = argparse.Namespace(
            camera_index=str(cam_idx),
            max_camera_index=getattr(args, "max_camera_index", 9),
            host="127.0.0.1",
            port=8766,
        )
        cmd_start_camera(camera_args)

    left_camera_url_final = getattr(args, "left_camera_url", None)
    left_camera_index_final = getattr(args, "left_camera_index", None)
    if getattr(args, "ensure_left_camera", False):
        if left_camera_url_final:
            print(
                "ensure-left-camera: --left-camera-url already set; using it (no extra HTTP helper started).",
                file=sys.stderr,
            )
        elif left_camera_index_final is None:
            print(
                "ensure-left-camera requires --left-camera-index (OpenCV index for iPhone Continuity Camera "
                "or other webcam). Hint: uv run python scripts/opencv_camera_probe.py",
                file=sys.stderr,
            )
            return 2
        else:
            host = getattr(args, "left_camera_http_host", "127.0.0.1")
            port = int(getattr(args, "left_camera_http_port", 8768))
            url = f"http://{host}:{port}/frame.jpg"
            if _http_jpeg_ready(url):
                print(f"left camera HTTP already warm at {url}", file=sys.stderr)
            else:
                cam_left = argparse.Namespace(
                    camera_index=str(int(left_camera_index_final)),
                    max_camera_index=getattr(args, "max_camera_index", 9),
                    host=host,
                    port=port,
                )
                cmd_start_camera(cam_left, background_name="camera_left")
                for _ in range(80):
                    if _http_jpeg_ready(url):
                        break
                    time.sleep(0.05)
                if not _http_jpeg_ready(url):
                    print(
                        f"ensure-left-camera: {url} did not respond; see {_log_path('camera_left')}",
                        file=sys.stderr,
                    )
                    return 1
            left_camera_url_final = url
            left_camera_index_final = None

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
        "--policy-jpeg-quality",
        str(args.policy_jpeg_quality),
        "--http-timeout",
        str(args.http_timeout),
        "--max-iterations",
        str(args.max_iterations),
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
        "--arms",
        args.arms,
        "--left-can",
        args.left_can,
        "--right-can",
        args.right_can,
    ]
    if args.bimanual_io_swap:
        cmd.append("--bimanual-io-swap")
    else:
        cmd.append("--no-bimanual-io-swap")
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
    if getattr(args, "right_orbbec", False):
        cmd.append("--right-orbbec")
    if getattr(args, "right_camera_index", None) is not None:
        cmd.extend(["--right-camera-index", str(args.right_camera_index)])
    if getattr(args, "right_camera_url", None):
        cmd.extend(["--right-camera-url", str(args.right_camera_url)])
    if getattr(args, "top_camera_index", None) is not None:
        cmd.extend(["--top-camera-index", str(args.top_camera_index)])
    if left_camera_index_final is not None:
        cmd.extend(["--left-camera-index", str(left_camera_index_final)])
    if left_camera_url_final:
        cmd.extend(["--left-camera-url", str(left_camera_url_final)])
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

    can_check = sub.add_parser(
        "can-check",
        help="Test /tmp/can0.sock and /tmp/can1.sock (exists + TCP-style connect).",
    )
    can_check.set_defaults(func=cmd_can_check)

    cameras = sub.add_parser("cameras", help="Start one WebSocket endpoint for multiple camera feeds.")
    cameras.add_argument(
        "--camera-specs",
        help=(
            "Comma-separated feeds. Supports id:index, id:opencv:index, "
            "id:avf:device-name, or id:orbbec:color/depth/ir/left_ir/right_ir/dual_ir/all."
        ),
    )
    cameras.add_argument("--auto-count", type=int, default=3)
    cameras.add_argument("--max-camera-index", type=int, default=12)
    cameras.add_argument("--host", default="0.0.0.0")
    cameras.add_argument("--port", type=int, default=8770)
    cameras.add_argument("--width", type=int, default=640)
    cameras.add_argument("--height", type=int, default=360)
    cameras.add_argument("--quality", type=int, default=80)
    cameras.set_defaults(func=cmd_start_cameras)

    bridge = sub.add_parser(
        "bridge",
        help="Start SLCAN -> /tmp/can0.sock (skip if Rust can-bridge or start_bimanual_bridges.sh already provides it).",
    )
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

    control = sub.add_parser("control-server", help="Start the simple YAM WebSocket control server.")
    control.add_argument("--host", default="127.0.0.1")
    control.add_argument("--port", type=int, default=8780)
    control.add_argument("--arm-specs", default="left:can0,right:can1", help="Comma-separated arm_id:channel list. Must contain exactly two arms.")
    control.add_argument("--min-gripper", type=float, default=0.01)
    control.add_argument("--max-gripper", type=float, default=0.59)
    control.add_argument("--reconnect-initial-delay", type=float, default=0.5)
    control.add_argument("--reconnect-max-delay", type=float, default=5.0)
    control.add_argument("--bridge-startup-timeout", type=float, default=10.0)
    control.add_argument("--background", action="store_true")
    control.add_argument("--allow-concurrent-owner", action="store_true", help="Bypass the single robot-owner guard.")
    control.set_defaults(func=cmd_start_control_server)

    stop = sub.add_parser("stop", help="Stop background processes tracked by yamctl.")
    stop.add_argument(
        "targets",
        nargs="*",
        default=["model", "viewer", "control", "policy", "cameras", "bridge"],
        choices=["model", "viewer", "control", "policy", "bridge", "cameras"],
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
    direct.add_argument(
        "--channel",
        default=os.environ.get("YAM_DIRECT_CAN", "can0"),
        help='SocketCAN interface for this arm (default can0 = left with ./start_bimanual_bridges.sh).',
    )
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

    direct_both = sub.add_parser(
        "direct-both",
        help="Move left and right arms together (same sync steps on both CAN interfaces).",
    )
    direct_both.add_argument("--left-can", default=os.environ.get("YAM_BIMANUAL_LEFT_CAN", "can0"))
    direct_both.add_argument("--right-can", default=os.environ.get("YAM_BIMANUAL_RIGHT_CAN", "can1"))
    direct_both.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    direct_both.add_argument("--bitrate", type=int, default=1_000_000)
    direct_both.add_argument("--target", help="Seven joint targets as comma- or space-separated numbers (applied from each arm's current pose semantics).")
    direct_both.add_argument("--delta", help="Seven relative joint deltas for BOTH arms (same delta added per arm).")
    direct_both.add_argument("--gripper", type=float)
    direct_both.add_argument("--duration", type=float, default=1.0)
    direct_both.add_argument("--steps", type=int, default=20)
    direct_both.add_argument("--max-delta", type=float, default=0.20)
    direct_both.add_argument("--max-temp-mos", type=float, default=55.0)
    direct_both.add_argument("--max-temp-rotor", type=float, default=100.0)
    direct_both.add_argument("--min-gripper-command", type=float, default=0.01)
    direct_both.add_argument("--max-gripper-command", type=float, default=0.59)
    direct_both.add_argument("--read-only", action="store_true")
    direct_both.add_argument("--ignore-hard-stop", action="store_true")
    direct_both.add_argument("--allow-concurrent-owner", action="store_true")
    direct_both.set_defaults(func=cmd_direct_both)

    smoke_bimanual = sub.add_parser(
        "smoke-bimanual",
        help="Read both arms, then move one joint +/− by the same delta on left and right CAN (sanity check).",
    )
    smoke_bimanual.add_argument("--left-can", default=os.environ.get("YAM_BIMANUAL_LEFT_CAN", "can0"))
    smoke_bimanual.add_argument("--right-can", default=os.environ.get("YAM_BIMANUAL_RIGHT_CAN", "can1"))
    smoke_bimanual.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    smoke_bimanual.add_argument("--bitrate", type=int, default=1_000_000)
    smoke_bimanual.add_argument(
        "--joint-index",
        type=int,
        choices=list(range(7)),
        default=0,
        help="Which joint (0–6) receives the test delta on both arms (default 0).",
    )
    smoke_bimanual.add_argument(
        "--amplitude",
        type=float,
        default=0.03,
        help="Relative command applied then reversed on that joint (default 0.03; keep small).",
    )
    smoke_bimanual.add_argument("--duration", type=float, default=1.0)
    smoke_bimanual.add_argument("--steps", type=int, default=20)
    smoke_bimanual.add_argument("--max-delta", type=float, default=0.20)
    smoke_bimanual.add_argument("--max-temp-mos", type=float, default=55.0)
    smoke_bimanual.add_argument("--max-temp-rotor", type=float, default=100.0)
    smoke_bimanual.add_argument("--min-gripper-command", type=float, default=0.01)
    smoke_bimanual.add_argument("--max-gripper-command", type=float, default=0.59)
    smoke_bimanual.add_argument("--ignore-hard-stop", action="store_true")
    smoke_bimanual.add_argument("--allow-concurrent-owner", action="store_true")
    smoke_bimanual.set_defaults(func=cmd_smoke_bimanual)

    hybrid = sub.add_parser("hybrid", help="Run one-arm policy actions with local camera/state verification and Codex corrections.")
    hybrid.add_argument(
        "task",
        nargs="?",
        default="pick up the hat",
        help="Robot task prompt (default: pick up the hat).",
    )
    hybrid.add_argument("--context", help="Extra scene/task context appended to the prompt.")
    hybrid.add_argument("--context-file", help="File containing extra context appended to the prompt.")
    hybrid.add_argument("--policy-kind", choices=["lerobot-act", "modal"], default="lerobot-act")
    hybrid.add_argument("--http-url", default=DEFAULT_ONE_ARM_POLICY_HTTP_URL)
    hybrid.add_argument(
        "--camera-url",
        default=DEFAULT_CAMERA_URL,
        help="Molmo 'top' when --top-camera-index omitted: HTTP(S) JPEG or ws/wss camera stream. "
        "Set YAM_CAMERA_URL or YAM_FRONT_CAMERA_URL for a default. Molmo also has left (--left-camera-url) and wrist right.",
    )
    hybrid.add_argument(
        "--right-orbbec",
        action="store_true",
        help="Use Orbbec for Molmo 'right' (wrist). Mutually exclusive with --right-camera-index / --right-camera-url.",
    )
    hybrid.add_argument(
        "--right-camera-index",
        type=int,
        default=None,
        help="OpenCV index for wrist (Molmo 'right'). Mutually exclusive with --right-orbbec / --right-camera-url.",
    )
    hybrid.add_argument(
        "--right-camera-url",
        default=None,
        help="HTTP or ws/wss URL for wrist (Molmo 'right'). Mutually exclusive with --right-orbbec / --right-camera-index.",
    )
    hybrid.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV index for yamctl ``camera`` HTTP helper (--ensure-camera). "
        "With --right-orbbec, defaults to --top-camera-index or 1 so index 0 is not used (Orbbec SDK needs exclusive UVC).",
    )
    hybrid.add_argument(
        "--max-camera-index",
        type=int,
        default=9,
        help="Highest OpenCV index to probe when camera_http_server uses --camera-index=auto.",
    )
    hybrid.add_argument(
        "--top-camera-index",
        type=int,
        default=None,
        help="OpenCV index for overhead Molmo 'top' (separate from wrist).",
    )
    hybrid.add_argument(
        "--left-camera-index",
        type=int,
        default=None,
        help="OpenCV index for Molmo 'left' (omit with --left-camera-url for HTTP JPEG).",
    )
    hybrid.add_argument(
        "--left-camera-url",
        default=os.environ.get("YAM_LEFT_CAMERA_URL"),
        help="HTTP or ws/wss URL for Molmo 'left' (omit with --left-camera-index for OpenCV). "
        "Default: YAM_LEFT_CAMERA_URL.",
    )
    hybrid.add_argument(
        "--ensure-left-camera",
        action="store_true",
        help="Start camera_http_server on --left-camera-http-port for --left-camera-index (e.g. iPhone "
        "Continuity Camera), then pass --left-camera-url so hybrid does not open that device twice.",
    )
    hybrid.add_argument(
        "--left-camera-http-port",
        type=int,
        default=int(os.environ.get("YAM_LEFT_CAMERA_HTTP_PORT", "8768")),
        help="Port for JPEG when using --ensure-left-camera (default 8768).",
    )
    hybrid.add_argument(
        "--left-camera-http-host",
        default=os.environ.get("YAM_LEFT_CAMERA_HTTP_HOST", "127.0.0.1"),
        help="Host bind for --ensure-left-camera (default 127.0.0.1).",
    )
    hybrid.add_argument("--ensure-camera", action="store_true", help="Start the camera helper before hybrid control.")
    hybrid.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    hybrid.add_argument("--bitrate", type=int, default=1_000_000)
    hybrid.add_argument("--trace-dir", default="logs/hybrid-latest")
    hybrid.add_argument("--hz", type=float, default=60.0)
    hybrid.add_argument(
        "--num-steps",
        type=int,
        default=3,
        help="Molmo trajectory length (predict_action decode steps); lower is faster.",
    )
    hybrid.add_argument(
        "--policy-jpeg-quality",
        type=int,
        default=78,
        help="JPEG quality for policy images (lower = smaller uploads to Modal).",
    )
    hybrid.add_argument("--http-timeout", type=float, default=300.0)
    hybrid.add_argument("--max-iterations", type=int, default=300)
    hybrid.add_argument("--max-temp-mos", type=float, default=55.0)
    hybrid.add_argument("--max-temp-rotor", type=float, default=100.0)
    hybrid.add_argument("--min-gripper-command", type=float, default=0.01)
    hybrid.add_argument("--max-gripper-command", type=float, default=0.59)
    hybrid.add_argument("--arm-slice", choices=["first", "second"], default="first")
    hybrid.add_argument("--action-step", type=int, default=0)
    hybrid.add_argument("--execute-action-steps", type=int, default=3)
    hybrid.add_argument("--action-step-delay", type=float, default=0.05)
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
    hybrid.add_argument(
        "--arms",
        choices=["single", "bimanual"],
        default=os.environ.get("YAM_ARMS", "bimanual"),
        help="single: one YAM on --left-can. bimanual: two YAMs, 14D Molmo state (left then right). "
        "Use YAM_ARMS=single or --arms single for one arm.",
    )
    hybrid.add_argument(
        "--left-can",
        default=os.environ.get("YAM_BIMANUAL_LEFT_CAN", "can0"),
        help="CAN iface for left arm (or only arm when arms=single).",
    )
    hybrid.add_argument(
        "--right-can",
        default=os.environ.get("YAM_BIMANUAL_RIGHT_CAN", "can1"),
        help="CAN iface for right arm when arms=bimanual.",
    )
    hybrid.add_argument(
        "--bimanual-io-swap",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("YAM_BIMANUAL_IO_SWAP", "").lower() in ("1", "true", "yes"),
        help="Swap left/right 14D halves for policy state and commands (see hybrid_robot_loop --help).",
    )
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
