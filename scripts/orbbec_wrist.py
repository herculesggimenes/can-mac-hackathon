"""Orbbec depth camera color stream for Molmo wrist / ``right`` channel."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import numpy as np


def release_uvc_interfering_processes() -> None:
    """Best-effort teardown of processes that often hold the Gemini UVC stack.

    On macOS only one client may open the Gemini UVC node at a time.

    - ``camera_http_server.py`` (OpenCV index 0) blocks ``pyorbbecsdk``.
    - Browser/Electron ``video_capture.mojom.VideoCaptureService`` can grab the
      same device (Chrome/Granola/Cursor-related helpers); apps may respawn it.

    Disable all of this with ``YAM_ORBBEC_RELEASE_UVC=0``.

    Skip only the VideoCapture ``pkill`` rounds with
    ``YAM_ORBBEC_KILL_VIDEO_CAPTURE=0`` (keep camera_http kill unless release off).
    """

    if os.environ.get("YAM_ORBBEC_RELEASE_UVC", "1").lower() in ("0", "false", "no"):
        return
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "camera_http_server.py"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        proc = None
    if proc is not None and proc.returncode == 0 and proc.stdout.strip():
        for line in proc.stdout.strip().splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                print(
                    "orbbec: stopped camera_http_server "
                    f"pid={pid} (frees UVC for Orbbec SDK; restart HTTP helper on "
                    "`--camera-index` = overhead cam if you need port 8766)",
                    file=sys.stderr,
                )
            except ProcessLookupError:
                pass
        time.sleep(0.5)

    if os.environ.get("YAM_ORBBEC_KILL_VIDEO_CAPTURE", "1").lower() in ("0", "false", "no"):
        time.sleep(0.2)
        return
    for round_i in range(3):
        try:
            subprocess.run(
                ["pkill", "-f", "video_capture.mojom.VideoCaptureService"],
                capture_output=True,
                timeout=8,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            break
        time.sleep(0.35)
    print(
        "orbbec: pkill video_capture.mojom.VideoCaptureService (Chrome/Electron); "
        "if UVC stays busy, quit those apps or unplug/replug the Gemini.",
        file=sys.stderr,
    )
    time.sleep(0.45)


def _orbbec_color_image(frame) -> np.ndarray | None:
    """Decode Orbbec color frame to RGB uint8 (same logic as ``teleop_viewer``)."""

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


class OrbbecColorPipeline:
    """Blocking color grabs from the default Orbbec device (wrist camera)."""

    def __init__(self) -> None:
        self._pipeline = None

    def start(self) -> None:
        from pyorbbecsdk import Config, OBSensorType, Pipeline

        release_uvc_interfering_processes()
        last_exc: BaseException | None = None
        for attempt in range(4):
            try:
                pipeline = Pipeline()
                config = Config()
                profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                config.enable_stream(profile_list.get_default_video_stream_profile())
                pipeline.start(config)
                self._pipeline = pipeline
                return
            except Exception as exc:  # noqa: BLE001 — SDK raises generic RuntimeError
                last_exc = exc
                if attempt < 3:
                    time.sleep(0.45)
                    release_uvc_interfering_processes()
                    continue
                raise last_exc

    def read_rgb(self) -> np.ndarray:
        from pyorbbecsdk import OBFrameType

        if self._pipeline is None:
            raise RuntimeError("Orbbec pipeline not started")
        frames = self._pipeline.wait_for_frames(1000)
        if frames is None:
            raise RuntimeError("Orbbec wait_for_frames returned None")
        frame = frames.get_frame(OBFrameType.COLOR_FRAME)
        image = _orbbec_color_image(frame)
        if image is None:
            raise RuntimeError("Orbbec color decode failed")
        return np.asarray(image, dtype=np.uint8)

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
