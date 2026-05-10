"""Bidirectional YAM teleop: real arm mirrors in sim, sim sliders control real arm."""

import logging
import json
import os
import signal
import sys
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from pathlib import Path

import can
import mujoco
import numpy as np
import viser

_repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root / "can-bridge"))
sys.path.insert(0, str(_repo_root / "scripts"))
from http_camera_fetch import build_urllib_camera_request  # noqa: E402
from can_bridge import CanBridgeBus
from mjviser import ViserMujocoScene

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType


_orig_can_bus = can.interface.Bus


def _mac_can_bus(*args, **kwargs):
    """Route I2RT's Linux socketcan channel through the local Rust bridge."""
    interface = kwargs.get("interface", kwargs.get("bustype"))
    channel = kwargs.get("channel", args[0] if args else None)
    if sys.platform == "darwin" and interface == "socketcan" and channel in {"can0", 0, None}:
        return CanBridgeBus(channel=0)
    return _orig_can_bus(*args, **kwargs)


if os.environ.get("CAN_MAC_PATCH", "1") == "1":
    can.interface.Bus = _mac_can_bus

from i2rt.motor_drivers.dm_driver import DMChainCanInterface


_orig_dm_chain_close = DMChainCanInterface.close


def _patched_dm_chain_close(self):
    _orig_dm_chain_close(self)
    motor_interface = getattr(self, "motor_interface", None)
    if motor_interface is not None:
        motor_interface.close()


if os.environ.get("CAN_MAC_PATCH", "1") == "1":
    DMChainCanInterface.close = _patched_dm_chain_close

# Monkey-patch LINEAR_4310 to skip calibration by providing known limits
_orig_get_limits = GripperType.get_gripper_limits
_orig_get_cal = GripperType.get_gripper_needs_calibration

def _linear_4310_raw_limits() -> tuple[float, float]:
    raw_limits = os.environ.get("YAM_GRIPPER_RAW_LIMITS", "0.0,-4.20")
    closed, open_ = (float(part.strip()) for part in raw_limits.split(",", 1))
    return closed, open_

def _patched_limits(self):
    if self == GripperType.LINEAR_4310:
        # I2RT's JointMapper expects raw gripper motor radians here. The MuJoCo
        # linear jaw range is meters, not the DM4310 motor stroke. The local YAM
        # gripper opens in the negative motor direction.
        return _linear_4310_raw_limits()
    return _orig_get_limits(self)

def _patched_cal(self):
    if self == GripperType.LINEAR_4310:
        return False  # skip calibration
    return _orig_get_cal(self)

GripperType.get_gripper_limits = _patched_limits
GripperType.get_gripper_needs_calibration = _patched_cal

logging.basicConfig(level=logging.WARNING)

# --- Globals ---
robot = None
shutdown_flag = False


REST_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
INTERNAL_CAMERA_WIDTH = 640
INTERNAL_CAMERA_HEIGHT = 400
ENABLE_ORBBEC_SDK = os.environ.get("YAM_ENABLE_ORBBEC_SDK") == "1"
RECORDINGS_DIR = Path(os.environ.get("YAM_RECORDINGS_DIR", "recordings"))
RECORD_CAMERA_URL = os.environ.get("YAM_RECORD_CAMERA_URL", "http://127.0.0.1:8766/frame.jpg")
KEYBOARD_HOST = "127.0.0.1"
KEYBOARD_PORT = int(os.environ.get("YAM_KEYBOARD_PORT", "8765"))
GRIPPER_SIM_OPEN_M = float(os.environ.get("YAM_GRIPPER_SIM_OPEN_M", "0.018"))
GRIPPER_COMMAND_MIN = float(os.environ.get("YAM_GRIPPER_COMMAND_MIN", "0.01"))
GRIPPER_COMMAND_MAX = float(os.environ.get("YAM_GRIPPER_COMMAND_MAX", "0.59"))


def _set_model_qpos_from_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_command: np.ndarray,
    n_arm: int,
    gripper_index: int | None,
    gripper_open_m: float,
) -> None:
    qpos = np.asarray(q_command[:n_arm], dtype=float).copy()
    if gripper_index is not None and gripper_index < len(qpos):
        qpos[gripper_index] = float(np.clip(qpos[gripper_index], GRIPPER_COMMAND_MIN, GRIPPER_COMMAND_MAX)) * gripper_open_m
    data.qpos[:n_arm] = qpos
    mujoco.mj_forward(model, data)


class EpisodeRecorder:
    def __init__(self, root: Path, camera_url: str | None = None):
        self.root = root
        self.camera_url = camera_url
        self.active = False
        self.path = None
        self.episode_dir = None
        self.image_dir = None
        self._file = None
        self.frame_count = 0
        self.started_at = None
        self.image_failures = 0

    def start(self, task: str) -> None:
        if self.active:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in task.strip().lower())[:48]
        if not safe_task:
            safe_task = "yam_episode"
        self.episode_dir = self.root / f"{stamp}_{safe_task}"
        self.image_dir = self.episode_dir / "images" / "front"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.episode_dir / "episode.jsonl"
        self._file = self.path.open("w", encoding="utf-8")
        self.frame_count = 0
        self.image_failures = 0
        self.started_at = time.time()
        self.active = True
        self._file.write(
            json.dumps(
                {
                    "type": "metadata",
                    "task": task,
                    "started_at": self.started_at,
                    "camera_url": self.camera_url,
                    "schema": "yam_raw_episode_v1",
                    "state_key": "observation.state",
                    "image_key": "observation.images.front",
                    "action_key": "action",
                }
            )
            + "\n"
        )
        self._file.flush()

    def stop(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self.active = False

    def _capture_image(self) -> str | None:
        if not self.camera_url or self.image_dir is None:
            return None
        image_rel = Path("images") / "front" / f"frame-{self.frame_count:06d}.jpg"
        image_path = self.episode_dir / image_rel if self.episode_dir is not None else None
        if image_path is None:
            return None
        try:
            with urllib.request.urlopen(build_urllib_camera_request(self.camera_url), timeout=0.5) as response:
                body = response.read()
            image_path.write_bytes(body)
            return image_rel.as_posix()
        except Exception:
            self.image_failures += 1
            return None

    def write_frame(self, *, mode: str, state: np.ndarray, action: np.ndarray, task: str) -> None:
        if not self.active or self._file is None:
            return
        image_path = self._capture_image()
        record = {
            "type": "frame",
            "t": time.time(),
            "dt": time.time() - self.started_at if self.started_at is not None else 0.0,
            "frame_index": self.frame_count,
            "task": task,
            "mode": mode,
            "observation.state": np.asarray(state, dtype=float).tolist(),
            "action": np.asarray(action, dtype=float).tolist(),
            "observation.images.front": image_path,
        }
        self._file.write(json.dumps(record) + "\n")
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            self._file.flush()


class KeyboardTeleopServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._keys = set()
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    def start(self) -> None:
        state = self

        class Handler(BaseHTTPRequestHandler):
            def do_OPTIONS(self):
                self._send_response(204, b"")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body.decode("utf-8"))
                    keys = payload.get("keys", [])
                    with state._lock:
                        state._keys = {str(key) for key in keys}
                    self._send_response(200, b"ok")
                except Exception:
                    self._send_response(400, b"bad request")

            def log_message(self, format, *args):
                return

            def _send_response(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "content-type")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="keyboard-teleop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None
        self._thread = None

    def keys(self) -> set[str]:
        with self._lock:
            return set(self._keys)

    def clear(self) -> None:
        with self._lock:
            self._keys.clear()


def _keyboard_html(port: int) -> str:
    srcdoc = f"""
<!doctype html>
<html>
<head>
  <style>
    html, body {{ margin: 0; background: #161616; color: #eee; font: 13px system-ui, sans-serif; }}
    #box {{ padding: 10px; border: 1px solid #444; border-radius: 6px; outline: none; }}
    #box:focus {{ border-color: #8ab4ff; }}
    #keys {{ margin-top: 8px; color: #8ab4ff; min-height: 18px; }}
    code {{ color: #ddd; }}
  </style>
</head>
<body>
  <div id="box" tabindex="0">
    Click here, then hold <code>W/A/S/D</code> or arrow keys. <code>Q/E</code>, <code>R/F</code> jog wrist. <code>O/P</code> opens/closes the gripper.
    <div id="keys"></div>
  </div>
  <script>
    const box = document.getElementById("box");
    const keysEl = document.getElementById("keys");
    const active = new Set();
    const aliases = {{
      "w": "w", "a": "a", "s": "s", "d": "d",
      "q": "q", "e": "e", "r": "r", "f": "f", "z": "z", "x": "x", "o": "o", "p": "p",
      "arrowup": "arrowup", "arrowdown": "arrowdown", "arrowleft": "arrowleft", "arrowright": "arrowright"
    }};

    function normalize(event) {{ return aliases[event.key.toLowerCase()]; }}
    async function send() {{
      keysEl.textContent = active.size ? "Active: " + [...active].join(", ") : "No keys active";
      try {{
        await fetch("http://127.0.0.1:{port}/keys", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ keys: [...active] }})
        }});
      }} catch (error) {{
        keysEl.textContent = "Keyboard bridge unavailable";
      }}
    }}
    box.addEventListener("keydown", (event) => {{
      const key = normalize(event);
      if (!key) return;
      event.preventDefault();
      active.add(key);
      send();
    }});
    box.addEventListener("keyup", (event) => {{
      const key = normalize(event);
      if (!key) return;
      event.preventDefault();
      active.delete(key);
      send();
    }});
    window.addEventListener("blur", () => {{ active.clear(); send(); }});
    setInterval(send, 150);
    box.focus();
  </script>
</body>
</html>
"""
    return (
        '<iframe title="Keyboard teleop" '
        'style="width:100%; height:105px; border:0; border-radius:6px; background:#161616;" '
        f'srcdoc="{escape(srcdoc, quote=True)}"></iframe>'
    )


def _apply_keyboard_jog(target: np.ndarray, keys: set[str], dt: float, joint_speed: float, gripper_speed: float) -> np.ndarray:
    delta = np.zeros_like(target)
    bindings = {
        "a": (0, -1.0),
        "d": (0, 1.0),
        "w": (1, 1.0),
        "s": (1, -1.0),
        "arrowup": (2, 1.0),
        "arrowdown": (2, -1.0),
        "arrowleft": (3, -1.0),
        "arrowright": (3, 1.0),
        "q": (4, -1.0),
        "e": (4, 1.0),
        "r": (5, 1.0),
        "f": (5, -1.0),
        "z": (6, -1.0),
        "x": (6, 1.0),
        "o": (6, 1.0),
        "p": (6, -1.0),
    }
    for key, (index, sign) in bindings.items():
        if key in keys and index < len(delta):
            speed = gripper_speed if index == 6 else joint_speed
            delta[index] += sign * speed * dt
    return target + delta


def _placeholder_image(message: str, width: int = INTERNAL_CAMERA_WIDTH, height: int = INTERNAL_CAMERA_HEIGHT) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, :] = (18, 18, 18)
    try:
        from PIL import Image, ImageDraw

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        y = 24
        for line in message.splitlines():
            draw.text((24, y), line, fill=(230, 230, 230))
            y += 26
        return np.asarray(pil_image)
    except Exception:
        return image


def _orbbec_video_profile(pipeline, sensor_type):
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    return profile_list.get_default_video_stream_profile()


def _normalize_gray_image(data: np.ndarray) -> np.ndarray:
    import cv2

    image = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def _orbbec_ir_image(frame) -> np.ndarray | None:
    import cv2
    from pyorbbecsdk import OBFormat

    if frame is None:
        return None
    frame = frame.as_video_frame()
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    raw = frame.get_data()

    if fmt == OBFormat.MJPG:
        decoded = cv2.imdecode(np.asanyarray(raw), cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            return None
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    if fmt == OBFormat.Y8:
        data = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
    else:
        data = np.frombuffer(raw, dtype=np.uint16).reshape((height, width))
    return _normalize_gray_image(data)


def _orbbec_depth_image(frame) -> np.ndarray | None:
    import cv2
    from pyorbbecsdk import OBFormat

    if frame is None or frame.get_format() != OBFormat.Y16:
        return None
    width = frame.get_width()
    height = frame.get_height()
    depth = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((height, width))
    depth = np.where((depth > 20) & (depth < 10000), depth, 0).astype(np.uint16)
    normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def _orbbec_color_image(frame) -> np.ndarray | None:
    import cv2
    from pyorbbecsdk import OBFormat

    if frame is None:
        return None
    frame = frame.as_video_frame()
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    raw = frame.get_data()

    if fmt == OBFormat.RGB:
        return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    if fmt == OBFormat.BGR:
        bgr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if fmt == OBFormat.MJPG:
        bgr = cv2.imdecode(np.asanyarray(raw), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    if fmt in (OBFormat.YUYV, OBFormat.YUY2):
        yuyv = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
    return None


class OrbbecInternalStream:
    def __init__(self):
        self.mode = "Off"
        self.status = "Internal stream off"
        self._frame = _placeholder_image("Internal stream off")
        self._pipeline = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._failed_mode = None

    def set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        if mode == self._failed_mode:
            self._set_frame(
                _placeholder_image(f"{mode} unavailable\nSwitch to Off before retrying."),
                f"{mode} unavailable",
            )
            return
        self.stop()
        self.mode = mode
        if mode == "Off":
            self._failed_mode = None
            self._set_frame(_placeholder_image("Internal stream off"), "Internal stream off")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="orbbec-internal-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def frame(self) -> np.ndarray:
        with self._lock:
            return self._frame.copy()

    def _set_frame(self, frame: np.ndarray, status: str) -> None:
        with self._lock:
            self._frame = frame
            self.status = status

    def _run(self) -> None:
        from pyorbbecsdk import Config, OBFrameType, OBSensorType, Pipeline

        try:
            pipeline = Pipeline()
            config = Config()
            frame_types = []

            if self.mode == "Color":
                config.enable_stream(_orbbec_video_profile(pipeline, OBSensorType.COLOR_SENSOR))
                frame_types = [OBFrameType.COLOR_FRAME]
            elif self.mode == "Depth":
                config.enable_stream(_orbbec_video_profile(pipeline, OBSensorType.DEPTH_SENSOR))
                frame_types = [OBFrameType.DEPTH_FRAME]
            elif self.mode == "IR":
                config.enable_stream(_orbbec_video_profile(pipeline, OBSensorType.IR_SENSOR))
                frame_types = [OBFrameType.IR_FRAME]
            elif self.mode == "Left IR":
                config.enable_stream(_orbbec_video_profile(pipeline, OBSensorType.LEFT_IR_SENSOR))
                frame_types = [OBFrameType.LEFT_IR_FRAME]
            elif self.mode == "Right IR":
                config.enable_stream(_orbbec_video_profile(pipeline, OBSensorType.RIGHT_IR_SENSOR))
                frame_types = [OBFrameType.RIGHT_IR_FRAME]
            elif self.mode == "Dual IR":
                config.enable_stream(_orbbec_video_profile(pipeline, OBSensorType.LEFT_IR_SENSOR))
                config.enable_stream(_orbbec_video_profile(pipeline, OBSensorType.RIGHT_IR_SENSOR))
                frame_types = [OBFrameType.LEFT_IR_FRAME, OBFrameType.RIGHT_IR_FRAME]

            pipeline.start(config)
            self._pipeline = pipeline
            self._set_frame(_placeholder_image(f"{self.mode} starting..."), f"{self.mode} starting")

            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(1000)
                if frames is None:
                    continue
                if self.mode == "Color":
                    image = _orbbec_color_image(frames.get_frame(frame_types[0]))
                elif self.mode == "Depth":
                    image = _orbbec_depth_image(frames.get_frame(frame_types[0]))
                elif self.mode == "Dual IR":
                    left = _orbbec_ir_image(frames.get_frame(frame_types[0]))
                    right = _orbbec_ir_image(frames.get_frame(frame_types[1]))
                    image = np.hstack([left, right]) if left is not None and right is not None else None
                else:
                    image = _orbbec_ir_image(frames.get_frame(frame_types[0]))
                if image is not None:
                    self._set_frame(image, self.mode)
        except Exception as exc:
            self._failed_mode = self.mode
            self._set_frame(_placeholder_image(f"Orbbec SDK stream unavailable\n{self.mode}\n{exc}"), f"{self.mode} unavailable")


def shutdown(signum=None, frame=None):
    global robot, shutdown_flag
    shutdown_flag = True
    if robot is not None:
        if os.environ.get("CAN_MAC_REST_ON_SHUTDOWN") == "1":
            print("\nMoving to rest position...", flush=True)
            robot.move_joints(REST_POS, time_interval_s=3.0)
            time.sleep(0.5)
        robot.close()
        robot = None
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


def main():
    global robot

    model_path = "yam_with_gripper.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    n_joints = model.njnt
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}" for i in range(n_joints)]
    print(f"Model has {n_joints} joints: {joint_names}")

    # Connect to real arm
    robot = get_yam_robot(
        channel="can0",
        gripper_type=GripperType.LINEAR_4310,
        zero_gravity_mode=False,
    )

    # Set sim to match real arm's current position
    real_pos = robot.get_joint_pos()
    n_arm = min(len(real_pos), model.nq)
    robot_info = robot.get_robot_info()
    gripper_index = robot_info.get("gripper_index")
    _set_model_qpos_from_command(model, data, real_pos, n_arm, gripper_index, GRIPPER_SIM_OPEN_M)

    # Start viser server
    server = viser.ViserServer()
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.create_scene_gui()
    scene.create_overlay_gui()
    internal_camera = OrbbecInternalStream() if ENABLE_ORBBEC_SDK else None
    recorder = EpisodeRecorder(RECORDINGS_DIR, RECORD_CAMERA_URL)
    keyboard_server = KeyboardTeleopServer(KEYBOARD_HOST, KEYBOARD_PORT)
    keyboard_server.start()
    last_record_time = 0.0
    last_loop_time = time.perf_counter()
    last_action = real_pos[:n_arm].copy()

    # Add mode toggle
    with server.gui.add_folder("Teleop"):
        mode_toggle = server.gui.add_dropdown(
            "Mode",
            options=["Mirror Real Arm", "Control from Sim"],
            initial_value="Mirror Real Arm",
        )
        status_text = server.gui.add_text("Status", initial_value="Mirroring real arm", disabled=True)

        # Joint sliders for sim control
        sliders = []
        with server.gui.add_folder("Joint Commands"):
            for i in range(n_arm):
                if gripper_index is not None and i == gripper_index:
                    jnt_range = [GRIPPER_COMMAND_MIN, GRIPPER_COMMAND_MAX]
                else:
                    jnt_range = model.jnt_range[i] if model.jnt_limited[i] else [-3.14, 3.14]
                lo, hi = float(jnt_range[0]), float(jnt_range[1])
                init = float(np.clip(real_pos[i], lo, hi)) if i < len(real_pos) else 0.0
                s = server.gui.add_slider(
                    joint_names[i] if i < len(joint_names) else f"J{i}",
                    min=lo,
                    max=hi,
                    step=0.01,
                    initial_value=init,
                )
                sliders.append(s)

    with server.gui.add_folder("Camera"):
        if internal_camera is None:
            camera_status = server.gui.add_text("Status", initial_value="SDK camera disabled", disabled=True)
            internal_mode = None
            internal_status = None
            internal_image = None
        else:
            internal_mode = server.gui.add_dropdown(
                "Stream",
                options=["Off", "Color", "Depth", "IR", "Left IR", "Right IR", "Dual IR"],
                initial_value="Off",
            )
            internal_status = server.gui.add_text("Status", initial_value=internal_camera.status, disabled=True)
            internal_image = server.gui.add_image(internal_camera.frame(), label="Gemini 2", format="jpeg", jpeg_quality=75)

    with server.gui.add_folder("Recording"):
        record_toggle = server.gui.add_checkbox("Record", initial_value=False)
        task_text = server.gui.add_text("Task", initial_value="put bread in toaster")
        record_fps = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=10)
        record_status = server.gui.add_text("Status", initial_value="Idle", disabled=True)

    with server.gui.add_folder("Keyboard"):
        keyboard_enabled = server.gui.add_checkbox("Enable", initial_value=False)
        keyboard_joint_speed = server.gui.add_slider("Joint rad/s", min=0.02, max=0.5, step=0.01, initial_value=0.12)
        keyboard_gripper_speed = server.gui.add_slider("Gripper/s", min=0.01, max=1.0, step=0.01, initial_value=0.25)
        gripper_sim_open_m = server.gui.add_slider(
            "Sim full-open m",
            min=0.005,
            max=0.0475,
            step=0.0005,
            initial_value=GRIPPER_SIM_OPEN_M,
        )
        keyboard_status = server.gui.add_text("Status", initial_value="Disabled", disabled=True)
        gripper_status = server.gui.add_text("Joint7 / gripper norm", initial_value="0.00", disabled=True)
        open_gripper_button = server.gui.add_button("Open gripper")
        close_gripper_button = server.gui.add_button("Close gripper")
        server.gui.add_html(_keyboard_html(KEYBOARD_PORT))

    gripper_button_delta = 0.0

    @open_gripper_button.on_hold(callback_hz=20.0)
    def _hold_open_gripper(_event):
        nonlocal gripper_button_delta
        gripper_button_delta += float(keyboard_gripper_speed.value) / 20.0

    @close_gripper_button.on_hold(callback_hz=20.0)
    def _hold_close_gripper(_event):
        nonlocal gripper_button_delta
        gripper_button_delta -= float(keyboard_gripper_speed.value) / 20.0

    print("Teleop viewer running at http://localhost:8080", flush=True)
    print("Modes: 'Mirror Real Arm' reads from robot, 'Control from Sim' sends slider values to robot", flush=True)
    if gripper_index is not None:
        print(f"Joint7 gripper raw limits closed/open: {_linear_4310_raw_limits()}", flush=True)
        print(f"Joint7 sim full-open visual travel: {GRIPPER_SIM_OPEN_M:.4f} m", flush=True)

    try:
        while not shutdown_flag:
            now_perf = time.perf_counter()
            loop_dt = max(now_perf - last_loop_time, 1e-3)
            last_loop_time = now_perf
            mode = mode_toggle.value

            if mode == "Mirror Real Arm":
                # Read real arm, update sim + sliders
                real_pos = robot.get_joint_pos()
                last_action = real_pos[:n_arm].copy()
                _set_model_qpos_from_command(
                    model,
                    data,
                    real_pos,
                    n_arm,
                    gripper_index,
                    float(gripper_sim_open_m.value),
                )
                for i in range(n_arm):
                    if gripper_index is not None and i == gripper_index:
                        sliders[i].value = float(np.clip(real_pos[i], GRIPPER_COMMAND_MIN, GRIPPER_COMMAND_MAX))
                    else:
                        sliders[i].value = float(real_pos[i])
                status_text.value = f"Mirroring | {[f'{p:.2f}' for p in real_pos]}"

            elif mode == "Control from Sim":
                # Read sliders, send to real arm
                target = np.array([s.value for s in sliders])
                keys = keyboard_server.keys() if keyboard_enabled.value else set()
                gripper_button_active = bool(gripper_button_delta)
                if keys:
                    target = _apply_keyboard_jog(
                        target,
                        keys,
                        loop_dt,
                        float(keyboard_joint_speed.value),
                        float(keyboard_gripper_speed.value),
                    )
                if gripper_button_delta and n_arm > 6:
                    target[6] += gripper_button_delta
                    gripper_button_delta = 0.0
                if keys or gripper_button_active:
                    for i in range(n_arm):
                        if gripper_index is not None and i == gripper_index:
                            jnt_range = [GRIPPER_COMMAND_MIN, GRIPPER_COMMAND_MAX]
                        else:
                            jnt_range = model.jnt_range[i] if model.jnt_limited[i] else [-3.14, 3.14]
                        target[i] = float(np.clip(target[i], float(jnt_range[0]), float(jnt_range[1])))
                        sliders[i].value = float(target[i])
                if keys:
                    keyboard_status.value = "Active | " + ", ".join(sorted(keys))
                elif gripper_button_active:
                    keyboard_status.value = "Active | gripper"
                elif keyboard_enabled.value:
                    keyboard_status.value = "Enabled"
                else:
                    keyboard_status.value = "Disabled"
                robot.command_joint_pos(target)
                last_action = target[:n_arm].copy()
                real_pos = robot.get_joint_pos()
                _set_model_qpos_from_command(
                    model,
                    data,
                    target,
                    n_arm,
                    gripper_index,
                    float(gripper_sim_open_m.value),
                )
                status_text.value = f"Controlling | {[f'{p:.2f}' for p in target]}"

            if gripper_index is not None and gripper_index < n_arm:
                obs = robot.get_observations()
                gripper_vel = float(obs.get("gripper_vel", [0.0])[0])
                gripper_eff = float(obs.get("gripper_eff", [0.0])[0])
                gripper_status.value = (
                    f"pos {float(real_pos[gripper_index]):.3f} | "
                    f"target {float(sliders[gripper_index].value):.3f} | "
                    f"vel {gripper_vel:.3f} | eff {gripper_eff:.3f}"
                )

            if record_toggle.value and not recorder.active:
                recorder.start(task_text.value)
            elif not record_toggle.value and recorder.active:
                recorder.stop()

            if recorder.active:
                now = time.time()
                interval = 1.0 / max(float(record_fps.value), 1.0)
                if now - last_record_time >= interval:
                    recorder.write_frame(mode=mode, state=real_pos[:n_arm], action=last_action, task=task_text.value)
                    last_record_time = now
                record_status.value = (
                    f"Recording {recorder.frame_count} frames | "
                    f"image failures {recorder.image_failures} | {recorder.path}"
                )
            else:
                record_status.value = "Idle"

            mujoco.mj_forward(model, data)
            scene.update_from_mjdata(data)
            if internal_camera is not None:
                internal_camera.set_mode(internal_mode.value)
                internal_image.image = internal_camera.frame()
                internal_status.value = internal_camera.status
            time.sleep(0.02)  # 50Hz update

    except Exception as e:
        print(f"Error: {e}", flush=True)
    finally:
        keyboard_server.stop()
        recorder.stop()
        if internal_camera is not None:
            internal_camera.stop()
        shutdown()


if __name__ == "__main__":
    main()
