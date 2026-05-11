#!/usr/bin/env python3
"""Serve multiple local cameras through one WebSocket endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import multiprocessing as mp
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    driver: str
    value: str

    @property
    def index(self) -> int | None:
        if self.driver != "opencv":
            return None
        return int(self.value)


def _parse_camera_specs(value: str) -> list[CameraSpec]:
    specs: list[CameraSpec] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 1:
            index_text = parts[0]
            specs.append(CameraSpec(camera_id=f"camera_{index_text}", driver="opencv", value=index_text))
            continue
        if len(parts) == 2:
            camera_id, index_text = parts
            driver = "opencv"
            value_text = index_text
        elif len(parts) == 3:
            camera_id, driver, value_text = parts
        else:
            raise argparse.ArgumentTypeError(f"invalid camera spec: {item!r}")
        camera_id = camera_id.strip()
        if not camera_id:
            raise argparse.ArgumentTypeError(f"invalid camera spec: {item!r}")
        driver = driver.strip().lower()
        value_text = value_text.strip()
        if driver == "cv":
            driver = "opencv"
        if driver == "opencv":
            int(value_text)
            specs.append(CameraSpec(camera_id=camera_id, driver=driver, value=value_text))
        elif driver in {"avf", "avfoundation", "name"}:
            specs.append(CameraSpec(camera_id=camera_id, driver="opencv", value=str(_resolve_avfoundation_index(value_text))))
        elif driver == "orbbec":
            value_text = value_text.lower()
            modes = ["color", "depth", "ir"] if value_text == "all" else [value_text]
            valid_modes = {"color", "depth", "ir", "left_ir", "right_ir", "dual_ir"}
            for mode in modes:
                if mode not in valid_modes:
                    raise argparse.ArgumentTypeError(f"invalid Orbbec mode: {mode!r}")
                mode_id = camera_id if value_text != "all" else f"{camera_id}_{mode}"
                specs.append(CameraSpec(camera_id=mode_id, driver=driver, value=mode))
        else:
            raise argparse.ArgumentTypeError(f"invalid camera driver: {driver!r}")
    if not specs:
        raise argparse.ArgumentTypeError("at least one camera spec is required")
    return specs


def _avfoundation_devices() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        check=False,
        text=True,
        capture_output=True,
    )
    devices: list[tuple[int, str]] = []
    in_video = False
    for line in (result.stderr + "\n" + result.stdout).splitlines():
        if "AVFoundation video devices:" in line:
            in_video = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if not in_video:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def _resolve_avfoundation_index(name: str) -> int:
    devices = _avfoundation_devices()
    normalized = name.casefold()
    exact = [index for index, device_name in devices if device_name.casefold() == normalized]
    if len(exact) == 1:
        return exact[0]
    partial = [index for index, device_name in devices if normalized in device_name.casefold()]
    if len(partial) == 1:
        return partial[0]
    listed = ", ".join(f"{index}:{device_name}" for index, device_name in devices) or "none"
    if exact or partial:
        raise argparse.ArgumentTypeError(f"ambiguous AVFoundation camera name {name!r}; devices: {listed}")
    raise argparse.ArgumentTypeError(f"AVFoundation camera name {name!r} not found; devices: {listed}")


def _probe_camera_indexes(max_index: int, width: int, height: int) -> list[int]:
    import cv2

    indexes: list[int] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok = False
        if cap.isOpened():
            ok, frame = cap.read()
            ok = bool(ok and frame is not None)
        cap.release()
        if ok:
            indexes.append(index)
    return indexes


def _auto_camera_specs(count: int, max_index: int, width: int, height: int) -> list[CameraSpec]:
    indexes = _probe_camera_indexes(max_index, width, height)
    if len(indexes) < count:
        raise RuntimeError(f"needed {count} cameras, but only opened indexes {indexes}")
    default_ids = ["front", "top", "wrist"]
    return [
        CameraSpec(camera_id=default_ids[i] if i < len(default_ids) else f"camera_{i}", driver="opencv", value=str(index))
        for i, index in enumerate(indexes[:count])
    ]


def _jpeg_payload(camera_id: str, source: str, jpeg: bytes | None, captured_at: float | None, error: str | None) -> dict[str, Any]:
    if jpeg is None:
        return {
            "type": "warming_up",
            "camera_id": camera_id,
            "camera_index": source,
            "source": source,
            "error": error,
            "sent_at": time.time(),
        }
    return {
        "type": "frame",
        "camera_id": camera_id,
        "camera_index": source,
        "source": source,
        "content_type": "image/jpeg",
        "encoding": "base64",
        "captured_at": captured_at,
        "sent_at": time.time(),
        "data": base64.b64encode(jpeg).decode("ascii"),
    }


def _frame_payload(
    camera_id: str,
    source: str,
    jpeg: bytes | None,
    captured_at: float | None,
    error: str | None,
    *,
    metadata: dict[str, Any] | None = None,
    depth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _jpeg_payload(camera_id, source, jpeg, captured_at, error)
    if payload.get("type") != "frame":
        return payload
    if metadata:
        payload.update(metadata)
    if depth and depth.get("data"):
        depth_payload = dict(depth)
        depth_payload["data"] = base64.b64encode(depth_payload["data"]).decode("ascii")
        payload["depth"] = depth_payload
    return payload


class OpenCvCameraFeed:
    def __init__(
        self,
        spec: CameraSpec,
        *,
        width: int,
        height: int,
        quality: int,
        stale_seconds: float = 10.0,
    ):
        import cv2

        self.cv2 = cv2
        self.spec = spec
        self.width = width
        self.height = height
        self.quality = quality
        self.stale_seconds = stale_seconds
        self.lock = threading.Lock()
        self.reset_lock = threading.Lock()
        self.reset_thread: threading.Thread | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_t: float | None = None
        self.error: str | None = None
        self.stop = threading.Event()
        self._same_jpeg_since: float | None = None
        self._last_encoded: bytes | None = None
        assert spec.index is not None
        self.cap = self._open_capture()

    def _open_capture(self):
        cap = self.cv2.VideoCapture(self.spec.index)
        cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"{self.spec.camera_id}: could not open camera index {self.spec.index}")
        return cap

    def request_reset(self) -> None:
        if self.reset_thread is not None and self.reset_thread.is_alive():
            return
        self.reset_thread = threading.Thread(
            target=self.reset,
            name=f"camera-{self.spec.camera_id}-reset",
            daemon=True,
        )
        self.reset_thread.start()

    def reset(self) -> None:
        if not self.reset_lock.acquire(blocking=False):
            return
        try:
            with self.lock:
                self.error = "resetting"
                self._same_jpeg_since = None
                self._last_encoded = None
            old_cap = self.cap
            try:
                old_cap.release()
            except Exception:
                pass
            try:
                new_cap = self._open_capture()
            except Exception as exc:
                with self.lock:
                    self.error = f"reset_failed: {exc}"
                return
            self.cap = new_cap
            with self.lock:
                self.latest_jpeg = None
                self.latest_t = None
                self.error = None
        finally:
            self.reset_lock.release()

    def _reset_if_stale(self, encoded: bytes) -> None:
        now = time.time()
        if self._last_encoded == encoded:
            if self._same_jpeg_since is None:
                self._same_jpeg_since = now
            elif now - self._same_jpeg_since >= self.stale_seconds:
                self.request_reset()
        else:
            self._last_encoded = encoded
            self._same_jpeg_since = now

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, name=f"camera-{self.spec.camera_id}", daemon=True)
        thread.start()
        return thread

    def run(self) -> None:
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                with self.lock:
                    self.error = "frame_read_failed"
                if self.latest_t is None or time.time() - self.latest_t >= self.stale_seconds:
                    self.request_reset()
                time.sleep(0.05)
                continue
            ok, encoded = self.cv2.imencode(
                ".jpg",
                frame,
                [int(self.cv2.IMWRITE_JPEG_QUALITY), self.quality],
            )
            if ok:
                encoded_bytes = encoded.tobytes()
                with self.lock:
                    self.latest_jpeg = encoded_bytes
                    self.latest_t = time.time()
                    self.error = None
                self._reset_if_stale(encoded_bytes)
            time.sleep(0.001)

    def metadata(self) -> dict[str, Any]:
        return {"camera_id": self.spec.camera_id, "camera_index": self.spec.index, "driver": "opencv"}

    def frame_payload(self) -> dict[str, Any]:
        with self.lock:
            jpeg = self.latest_jpeg
            captured_at = self.latest_t
            error = self.error
        return _jpeg_payload(self.spec.camera_id, str(self.spec.index), jpeg, captured_at, error)

    def close(self) -> None:
        self.stop.set()
        self.cap.release()


def _orbbec_video_profile(pipeline, sensor_type):
    from pyorbbecsdk import OBFormat, OBSensorType

    profile_list = pipeline.get_stream_profile_list(sensor_type)
    preferred = {
        OBSensorType.COLOR_SENSOR: [
            (640, 360, 5, OBFormat.MJPG),
            (640, 480, 5, OBFormat.MJPG),
            (640, 360, 10, OBFormat.MJPG),
            (640, 480, 10, OBFormat.MJPG),
            (640, 360, 5, OBFormat.YUYV),
            (640, 480, 5, OBFormat.YUYV),
        ],
        OBSensorType.DEPTH_SENSOR: [
            (640, 400, 5, OBFormat.Y16),
            (640, 400, 10, OBFormat.Y16),
            (320, 200, 5, OBFormat.Y16),
            (320, 200, 10, OBFormat.Y16),
            (320, 200, 5, OBFormat.Y14),
            (320, 200, 10, OBFormat.Y14),
        ],
        OBSensorType.IR_SENSOR: [
            (640, 400, 5, OBFormat.Y8),
            (640, 400, 10, OBFormat.Y8),
            (320, 200, 5, OBFormat.Y8),
            (320, 200, 10, OBFormat.Y8),
        ],
    }
    profiles = []
    try:
        count = profile_list.get_count()
        for index in range(count):
            profile = profile_list.get_stream_profile_by_index(index).as_video_stream_profile()
            profiles.append(profile)
    except Exception:
        pass
    for width, height, fps, fmt in preferred.get(sensor_type, []):
        for profile in profiles:
            if (
                profile.get_width() == width
                and profile.get_height() == height
                and profile.get_fps() == fps
                and profile.get_format() == fmt
            ):
                return profile
    return profile_list.get_default_video_stream_profile()


def _normalize_gray_image(data):
    import cv2
    import numpy as np

    image = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _orbbec_ir_image(frame):
    import cv2
    import numpy as np
    from pyorbbecsdk import OBFormat

    if frame is None:
        return None
    frame = frame.as_video_frame()
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    raw = frame.get_data()
    if fmt == OBFormat.MJPG:
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR) if decoded is not None else None
    if fmt == OBFormat.Y8:
        data = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
    else:
        data = np.frombuffer(raw, dtype=np.uint16).reshape((height, width))
    return _normalize_gray_image(data)


def _orbbec_depth_image(frame):
    import cv2
    import numpy as np
    from pyorbbecsdk import OBFormat

    if frame is None:
        return None
    frame = frame.as_video_frame()
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    if fmt not in (OBFormat.Y16, OBFormat.Y14):
        return None
    raw = bytes(frame.get_data())
    expected = width * height * 2
    if len(raw) < expected:
        return None
    depth = np.frombuffer(raw[:expected], dtype=np.uint16).reshape((height, width))
    depth = np.where((depth > 20) & (depth < 10000), depth, 0).astype(np.uint16)
    normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def _orbbec_depth_frame(frame) -> tuple[Any | None, dict[str, Any] | None]:
    import cv2
    import numpy as np
    from pyorbbecsdk import OBFormat

    if frame is None:
        return None, None
    video = frame.as_video_frame()
    width = video.get_width()
    height = video.get_height()
    fmt = video.get_format()
    if fmt not in (OBFormat.Y16, OBFormat.Y14):
        return None, None
    raw = bytes(video.get_data())
    expected = width * height * 2
    if len(raw) < expected:
        return None, None
    raw_depth = np.frombuffer(raw[:expected], dtype=np.uint16).reshape((height, width))
    raw_depth = np.ascontiguousarray(np.rot90(raw_depth, 2))
    valid_depth = np.where((raw_depth > 20) & (raw_depth < 10000), raw_depth, 0).astype(np.uint16)
    normalized = cv2.normalize(valid_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return preview, {
        "content_type": "application/octet-stream",
        "encoding": "base64",
        "format": "uint16",
        "byte_order": "little",
        "unit": "meter",
        "scale_to_meters": 0.001,
        "width": width,
        "height": height,
        "data": raw_depth.astype("<u2", copy=False).tobytes(),
    }


def _orbbec_color_image(frame):
    import cv2
    import numpy as np
    from pyorbbecsdk import OBFormat

    if frame is None:
        return None
    frame = frame.as_video_frame()
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    raw = frame.get_data()
    if fmt == OBFormat.RGB:
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if fmt in (OBFormat.YUYV, OBFormat.YUY2):
        yuyv = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    return None


def _orbbec_color_jpeg(frame) -> bytes | None:
    from pyorbbecsdk import OBFormat

    if frame is None:
        return None
    frame = frame.as_video_frame()
    if frame.get_format() != OBFormat.MJPG:
        return None
    return bytes(frame.get_data())


def _rotate_180(image):
    import cv2

    if image is None:
        return None
    return cv2.rotate(image, cv2.ROTATE_180)


def _metadata_rotated_180(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(metadata)
    metadata["orientation"] = "rotate_180"
    intrinsics = metadata.get("intrinsics")
    if isinstance(intrinsics, dict):
        intrinsics = dict(intrinsics)
        width = intrinsics.get("width") or metadata.get("width")
        height = intrinsics.get("height") or metadata.get("height")
        if width is not None and intrinsics.get("cx") is not None:
            intrinsics["cx"] = float(width) - 1.0 - float(intrinsics["cx"])
        if height is not None and intrinsics.get("cy") is not None:
            intrinsics["cy"] = float(height) - 1.0 - float(intrinsics["cy"])
        metadata["intrinsics"] = intrinsics
    return metadata


def _should_rotate_orbbec_180(camera_id: str, value: str) -> bool:
    return value == "color" and camera_id in {"top", "right"}


class OrbbecSdkFeed:
    def __init__(self, specs: list[CameraSpec], *, quality: int):
        self.specs = specs
        self.quality = quality
        self.lock = threading.Lock()
        self.latest_jpegs: dict[str, bytes] = {}
        self.latest_depth: dict[str, dict[str, Any]] = {}
        self.latest_metadata: dict[str, dict[str, Any]] = {}
        self.latest_t: dict[str, float] = {}
        self.errors: dict[str, str] = {}
        self.stop = threading.Event()
        self.process: mp.Process | None = None
        self.queue: mp.Queue | None = None
        self.reset_lock = threading.Lock()
        self._logged_modes: set[str] = set()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, name="camera-orbbec-sdk", daemon=True)
        thread.start()
        return thread

    def metadata(self) -> dict[str, Any]:
        raise RuntimeError("OrbbecSdkFeed contains multiple logical feeds")

    def logical_feeds(self) -> list["OrbbecLogicalFeed"]:
        return [OrbbecLogicalFeed(self, spec) for spec in self.specs]

    def run(self) -> None:
        self.queue = mp.Queue(maxsize=16)
        specs_data = [(spec.camera_id, spec.value) for spec in self.specs]
        self._start_process(specs_data)
        while not self.stop.is_set():
            if self.process is not None and not self.process.is_alive():
                with self.lock:
                    for spec in self.specs:
                        self.errors.setdefault(spec.camera_id, f"orbbec worker exited: {self.process.exitcode}")
                try:
                    self.process.join(timeout=0.2)
                except Exception:
                    pass
                time.sleep(1.0)
                if not self.stop.is_set():
                    self._start_process(specs_data)
                continue
            try:
                message = self.queue.get(timeout=0.25) if self.queue is not None else None
            except Exception:
                continue
            if not message:
                continue
            kind = message.get("type")
            if kind == "frame":
                with self.lock:
                    self.latest_jpegs[message["camera_id"]] = message["jpeg"]
                    if message.get("depth"):
                        self.latest_depth[message["camera_id"]] = message["depth"]
                    if message.get("metadata"):
                        self.latest_metadata[message["camera_id"]] = message["metadata"]
                    self.latest_t[message["camera_id"]] = message["captured_at"]
                    self.errors.pop(message["camera_id"], None)
            elif kind == "error":
                with self.lock:
                    self.errors[message["camera_id"]] = message["error"]

    def _start_process(self, specs_data: list[tuple[str, str]]) -> None:
        if self.queue is None:
            self.queue = mp.Queue(maxsize=16)
        self.process = mp.Process(target=_orbbec_worker, args=(specs_data, self.quality, self.queue), daemon=True)
        self.process.start()

    def request_reset(self, camera_id: str | None = None) -> None:
        threading.Thread(
            target=self._reset_worker,
            args=(camera_id,),
            name="camera-orbbec-sdk-reset",
            daemon=True,
        ).start()

    def _reset_worker(self, camera_id: str | None = None) -> None:
        if not self.reset_lock.acquire(blocking=False):
            return
        try:
            with self.lock:
                affected_ids = [spec.camera_id for spec in self.specs]
                for affected_id in affected_ids:
                    self.errors[affected_id] = "resetting"
                    self.latest_jpegs.pop(affected_id, None)
                    self.latest_depth.pop(affected_id, None)
                    self.latest_metadata.pop(affected_id, None)
                    self.latest_t.pop(affected_id, None)
            if self.process is not None and self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
            if self.process is not None and self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1.0)
        finally:
            self.reset_lock.release()

    def close(self) -> None:
        self.stop.set()
        if self.process is not None and self.process.is_alive():
            self.process.terminate()

    def frame_payload(self, spec: CameraSpec) -> dict[str, Any]:
        with self.lock:
            jpeg = self.latest_jpegs.get(spec.camera_id)
            depth = self.latest_depth.get(spec.camera_id)
            metadata = self.latest_metadata.get(spec.camera_id)
            captured_at = self.latest_t.get(spec.camera_id)
            error = self.errors.get(spec.camera_id) or self.errors.get(spec.value)
        return _frame_payload(
            spec.camera_id,
            f"orbbec:{spec.value}",
            jpeg,
            captured_at,
            error,
            metadata=metadata,
            depth=depth,
        )


def _orbbec_worker(specs_data: list[tuple[str, str]], quality: int, queue: mp.Queue) -> None:
    try:
        _run_orbbec_worker(specs_data, quality, queue)
    except BaseException as exc:
        for camera_id, _mode in specs_data:
            try:
                queue.put_nowait({"type": "error", "camera_id": camera_id, "error": str(exc)})
            except Exception:
                pass


def _run_orbbec_worker(specs_data: list[tuple[str, str]], quality: int, queue: mp.Queue) -> None:
        import cv2
        from pyorbbecsdk import Config, OBFrameType, OBSensorType, Pipeline

        mode_map = {
            "color": (OBSensorType.COLOR_SENSOR, OBFrameType.COLOR_FRAME, _orbbec_color_image),
            "depth": (OBSensorType.DEPTH_SENSOR, OBFrameType.DEPTH_FRAME, _orbbec_depth_image),
            "ir": (OBSensorType.IR_SENSOR, OBFrameType.IR_FRAME, _orbbec_ir_image),
            "left_ir": (OBSensorType.LEFT_IR_SENSOR, OBFrameType.LEFT_IR_FRAME, _orbbec_ir_image),
            "right_ir": (OBSensorType.RIGHT_IR_SENSOR, OBFrameType.RIGHT_IR_FRAME, _orbbec_ir_image),
        }
        enabled: dict[str, tuple[Any, Any]] = {}
        stream_metadata: dict[str, dict[str, Any]] = {}
        logged_modes: set[str] = set()
        pipeline = Pipeline()
        config = Config()
        for _camera_id, value in specs_data:
            modes = ("left_ir", "right_ir") if value == "dual_ir" else (value,)
            for mode in modes:
                if mode in enabled:
                    continue
                sensor_type, frame_type, decoder = mode_map[mode]
                try:
                    profile = _orbbec_video_profile(pipeline, sensor_type)
                    config.enable_stream(profile)
                    enabled[mode] = (frame_type, decoder)
                    stream_metadata[mode] = _orbbec_stream_metadata(profile, mode)
                except Exception as exc:
                    queue.put({"type": "error", "camera_id": _camera_id, "error": str(exc)})
        if not enabled:
            raise RuntimeError("no Orbbec SDK streams could be enabled")
        pipeline.start(config)
        try:
            timeout_count = 0
            while True:
                frames = pipeline.wait_for_frames(5000)
                if frames is None:
                    timeout_count += 1
                    if timeout_count >= 3:
                        raise RuntimeError("Orbbec wait_for_frames timed out 3 times")
                    continue
                timeout_count = 0
                for camera_id, value in specs_data:
                    try:
                        if value not in enabled and value != "dual_ir":
                            continue
                        if value == "dual_ir" and ("left_ir" not in enabled or "right_ir" not in enabled):
                            continue
                        raw_jpeg = None
                        if value == "color":
                            frame_type, _decoder = enabled[value]
                            color_frame = frames.get_frame(frame_type)
                            raw_jpeg = None if _should_rotate_orbbec_180(camera_id, value) else _orbbec_color_jpeg(color_frame)
                        if raw_jpeg is not None:
                            queue.put(
                                {
                                    "type": "frame",
                                    "camera_id": camera_id,
                                    "jpeg": raw_jpeg,
                                    "captured_at": time.time(),
                                }
                            )
                            continue
                        depth_payload = None
                        if value == "depth":
                            frame_type, _decoder = enabled[value]
                            image, depth_payload = _orbbec_depth_frame(frames.get_frame(frame_type))
                        elif value == "dual_ir":
                            left = _decode_orbbec_frame(frames, "left_ir", enabled, logged_modes)
                            right = _decode_orbbec_frame(frames, "right_ir", enabled, logged_modes)
                            image = None if left is None or right is None else cv2.hconcat([left, right])
                        elif value == "color":
                            image = _orbbec_color_image(color_frame)
                            if _should_rotate_orbbec_180(camera_id, value):
                                image = _rotate_180(image)
                        else:
                            image = _decode_orbbec_frame(frames, value, enabled, logged_modes)
                        if image is None:
                            continue
                        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                        if ok:
                            queue.put(
                                {
                                    "type": "frame",
                                    "camera_id": camera_id,
                                    "jpeg": encoded.tobytes(),
                                    "metadata": _metadata_rotated_180(stream_metadata.get(value, {}))
                                    if _should_rotate_orbbec_180(camera_id, value) or camera_id == "depth"
                                    else stream_metadata.get(value, {}),
                                    "depth": depth_payload,
                                    "captured_at": time.time(),
                                }
                            )
                    except Exception as exc:
                        queue.put({"type": "error", "camera_id": camera_id, "error": str(exc)})
        finally:
            pipeline.stop()


def _decode_orbbec_frame(frames, mode: str, enabled: dict[str, tuple[Any, Any]], logged_modes: set[str]):
    if mode not in enabled:
        return None
    frame_type, decoder = enabled[mode]
    frame = frames.get_frame(frame_type)
    if frame is not None and mode not in logged_modes:
        try:
            video = frame.as_video_frame()
            print(
                json.dumps(
                    {
                        "type": "orbbec_frame",
                        "mode": mode,
                        "width": video.get_width(),
                        "height": video.get_height(),
                        "format": str(video.get_format()),
                    }
                ),
                flush=True,
            )
        except Exception:
            pass
        logged_modes.add(mode)
    return decoder(frame)


def _orbbec_stream_metadata(profile, mode: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"mode": mode}
    try:
        intrinsic = profile.get_intrinsic()
        metadata["intrinsics"] = {
            "fx": float(intrinsic.fx),
            "fy": float(intrinsic.fy),
            "cx": float(intrinsic.cx),
            "cy": float(intrinsic.cy),
            "width": int(intrinsic.width),
            "height": int(intrinsic.height),
        }
    except Exception:
        pass
    try:
        metadata["format"] = str(profile.get_format())
        metadata["width"] = int(profile.get_width())
        metadata["height"] = int(profile.get_height())
        metadata["fps"] = int(profile.get_fps())
    except Exception:
        pass
    return metadata

class OrbbecLogicalFeed:
    def __init__(self, source: OrbbecSdkFeed, spec: CameraSpec):
        self.source = source
        self.spec = spec

    def metadata(self) -> dict[str, Any]:
        return {"camera_id": self.spec.camera_id, "camera_index": f"orbbec:{self.spec.value}", "driver": "orbbec"}

    def frame_payload(self) -> dict[str, Any]:
        return self.source.frame_payload(self.spec)

    def request_reset(self) -> None:
        self.source.request_reset(self.spec.camera_id)

    def close(self) -> None:
        pass


class MultiCameraServer:
    def __init__(self, feeds: list[Any]):
        self.feeds = {feed.spec.camera_id: feed for feed in feeds}

    def hello(self) -> dict[str, Any]:
        return {
            "type": "hello",
            "commands": ["get_frame", "get_all_frames", "reset_camera", "subscribe", "stop"],
            "encoding": "base64",
            "cameras": [feed.metadata() for feed in self.feeds.values()],
        }

    def selected_feeds(self, message: dict[str, Any]) -> list[CameraFeed]:
        requested = message.get("cameras") or message.get("camera_ids") or message.get("camera_id") or "all"
        if requested == "all" or requested is None:
            return list(self.feeds.values())
        if isinstance(requested, str):
            requested = [requested]
        selected = []
        for camera_id in requested:
            feed = self.feeds.get(str(camera_id))
            if feed is not None:
                selected.append(feed)
        return selected

    def send_frames(self, ws, message: dict[str, Any]) -> None:
        feeds = self.selected_feeds(message)
        if not feeds:
            ws.send(json.dumps({"type": "error", "error": "no matching cameras"}))
            return
        bundle = bool(message.get("bundle", False))
        frames = [feed.frame_payload() for feed in feeds]
        if bundle:
            ws.send(json.dumps({"type": "frames", "sent_at": time.time(), "frames": frames}))
        else:
            for frame in frames:
                ws.send(json.dumps(frame))

    def reset_cameras(self, message: dict[str, Any]) -> dict[str, Any]:
        feeds = self.selected_feeds(message)
        reset_ids = []
        skipped_ids = []
        for feed in feeds:
            request_reset = getattr(feed, "request_reset", None)
            reset = getattr(feed, "reset", None)
            if callable(request_reset):
                request_reset()
                reset_ids.append(feed.spec.camera_id)
            elif callable(reset):
                threading.Thread(
                    target=reset,
                    name=f"camera-{feed.spec.camera_id}-manual-reset",
                    daemon=True,
                ).start()
                reset_ids.append(feed.spec.camera_id)
            else:
                skipped_ids.append(feed.spec.camera_id)
        return {
            "type": "reset_camera_result",
            "reset": reset_ids,
            "skipped": skipped_ids,
            "sent_at": time.time(),
        }


def _start_ws_server(camera_server: MultiCameraServer, host: str, port: int):
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    def handler(ws) -> None:
        ws.send(json.dumps(camera_server.hello()))
        try:
            while True:
                raw = ws.recv(timeout=None)
                try:
                    message = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                except Exception:
                    ws.send(json.dumps({"type": "error", "error": "expected JSON message"}))
                    continue

                command = message.get("type") or message.get("command") or "get_all_frames"
                if command in {"get_frame", "frame", "get_all_frames"}:
                    if command == "get_all_frames":
                        message["cameras"] = "all"
                    camera_server.send_frames(ws, message)
                elif command == "reset_camera":
                    ws.send(json.dumps(camera_server.reset_cameras(message)))
                elif command == "subscribe":
                    fps = float(message.get("fps") or 5.0)
                    fps = min(max(fps, 0.1), 30.0)
                    interval = 1.0 / fps
                    ws.send(
                        json.dumps(
                            {
                                "type": "subscribed",
                                "fps": fps,
                                "cameras": [feed.spec.camera_id for feed in camera_server.selected_feeds(message)],
                            }
                        )
                    )
                    while True:
                        camera_server.send_frames(ws, message)
                        try:
                            control = ws.recv(timeout=interval)
                        except TimeoutError:
                            continue
                        try:
                            control_message = json.loads(control.decode("utf-8") if isinstance(control, bytes) else control)
                        except Exception:
                            continue
                        if control_message.get("type") == "stop":
                            ws.send(json.dumps({"type": "stopped"}))
                            break
                        if control_message.get("type") in {"get_frame", "get_all_frames"}:
                            camera_server.send_frames(ws, control_message)
                        if control_message.get("type") == "reset_camera":
                            ws.send(json.dumps(camera_server.reset_cameras(control_message)))
                elif command == "stop":
                    ws.send(json.dumps({"type": "stopped"}))
                    return
                else:
                    ws.send(json.dumps({"type": "error", "error": f"unknown command: {command}"}))
        except (ConnectionClosed, EOFError):
            return

    server = serve(handler, host, port, max_size=32 * 1024 * 1024)
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve multiple camera feeds through one WebSocket endpoint.")
    parser.add_argument("--camera-specs", help="Comma-separated id:index list, e.g. front:0,top:1,wrist:2.")
    parser.add_argument("--auto-count", type=int, default=0, help="Probe and use the first N working cameras.")
    parser.add_argument("--max-camera-index", type=int, default=12)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()

    if args.camera_specs:
        specs = _parse_camera_specs(args.camera_specs)
    elif args.auto_count:
        specs = _auto_camera_specs(args.auto_count, args.max_camera_index, args.width, args.height)
    else:
        parser.error("pass --camera-specs or --auto-count")

    feeds: list[Any] = []
    opencv_specs = [spec for spec in specs if spec.driver == "opencv"]
    orbbec_specs = [spec for spec in specs if spec.driver == "orbbec"]
    feeds.extend(OpenCvCameraFeed(spec, width=args.width, height=args.height, quality=args.quality) for spec in opencv_specs)
    orbbec_feed = OrbbecSdkFeed(orbbec_specs, quality=args.quality) if orbbec_specs else None
    if orbbec_feed is not None:
        feeds.extend(orbbec_feed.logical_feeds())
    for feed in feeds:
        if isinstance(feed, OrbbecLogicalFeed):
            continue
        feed.start()
    if orbbec_feed is not None:
        orbbec_feed.start()
    camera_server = MultiCameraServer(feeds)
    print(
        json.dumps(
            {
                "type": "multi_camera_server",
                "url": f"ws://{args.host}:{args.port}/cameras",
                "cameras": camera_server.hello()["cameras"],
            }
        ),
        flush=True,
    )
    try:
        _start_ws_server(camera_server, args.host, args.port)
    finally:
        for feed in feeds:
            feed.close()
        if orbbec_feed is not None:
            orbbec_feed.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
