#!/usr/bin/env python3
"""HTTP frame server for the Orbbec wrist camera.

macOS keeps UVC devices opened exclusively by ``VDCAssistant``/``UVCAssistant``.
The Orbbec SDK can only break that lock when it owns the USB device, which on
macOS requires ``sudo`` (per Orbbec issues #9 and #124). Running the *whole*
hybrid stack as root is risky (CAN, Modal credentials, file writes), so we run
**only this tiny server** as root. Everything else (``yamctl hybrid``,
``camera_http_server`` for the overhead cam, the CAN bridge) stays non-root and
just consumes JPEG frames from ``http://127.0.0.1:8767/frame.jpg``.

JPEG frames are **already orientation-corrected** for an upside-down wrist mount
(default: vertical flip). Override with ``--flip none`` if your mounting differs.

Recommended usage::

    sudo -E uv run python scripts/orbbec_camera_server.py
    # then in another terminal (no sudo):
    uv run yamctl hybrid "pick up the hat" --policy-kind modal \
        --top-camera-index 1 \
        --right-camera-url http://127.0.0.1:8767/frame.jpg
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from orbbec_wrist import OrbbecColorPipeline  # noqa: E402


def _apply_orientation(rgb: np.ndarray, mode: str) -> np.ndarray:
    """Match wrist mounting; default ``vertical`` fixes upside-down sensor."""

    if not mode or mode == "none":
        return np.asarray(rgb, dtype=np.uint8)
    import cv2

    code = {"vertical": 0, "horizontal": 1, "both": -1}.get(str(mode))
    if code is None:
        return np.asarray(rgb, dtype=np.uint8)
    return cv2.flip(np.asarray(rgb, dtype=np.uint8), code)


def _warn_if_not_root() -> None:
    if os.geteuid() == 0:
        return
    print(
        "WARNING: orbbec_camera_server is running without root. macOS UVC "
        "exclusivity will likely fail with 'uvc_open already opened'. "
        "Re-run as: sudo -E uv run python scripts/orbbec_camera_server.py",
        file=sys.stderr,
        flush=True,
    )


class OrbbecState:
    """Background grab loop with a JPEG cache for Molmo wrist consumption."""

    def __init__(self, quality: int, max_fps: float, flip: str) -> None:
        self.flip = flip
        self.quality = max(1, min(95, int(quality)))
        self._min_period_s = max(0.0, 1.0 / max_fps) if max_fps > 0 else 0.0
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_shape: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._pipeline = OrbbecColorPipeline()
        self._frames_served = 0
        self._frames_grabbed = 0
        self._last_error: str | None = None

    def start(self) -> None:
        self._pipeline.start()

    def run(self) -> None:
        next_time = time.monotonic()
        while not self._stop.is_set():
            try:
                rgb = self._pipeline.read_rgb()
                rgb = _apply_orientation(rgb, self.flip)
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=self.quality)
                with self._lock:
                    self._latest_jpeg = buf.getvalue()
                    self._latest_shape = (int(rgb.shape[1]), int(rgb.shape[0]))
                    self._frames_grabbed += 1
                    self._last_error = None
            except Exception as exc:  # noqa: BLE001 - stay alive across transient errors
                with self._lock:
                    self._last_error = repr(exc)
                time.sleep(0.05)
            if self._min_period_s:
                next_time += self._min_period_s
                delay = next_time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_time = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        self._pipeline.stop()

    def jpeg(self) -> bytes | None:
        with self._lock:
            if self._latest_jpeg is not None:
                self._frames_served += 1
            return self._latest_jpeg

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "frames_grabbed": self._frames_grabbed,
                "frames_served": self._frames_served,
                "shape": self._latest_shape,
                "last_error": self._last_error,
                "has_frame": self._latest_jpeg is not None,
                "flip": self.flip,
            }


def _make_handler(state: OrbbecState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict) -> None:
            import json

            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path in {"/healthz", "/health"}:
                self._send_json(200, {"ok": True, **state.stats()})
                return
            if self.path not in {"/", "/frame.jpg", "/frame"}:
                self._send_json(404, {"ok": False, "error": "not found", "paths": ["/frame.jpg", "/healthz"]})
                return
            body = state.jpeg()
            if body is None:
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"orbbec warming up")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - quiet by default
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Orbbec wrist camera HTTP frame server (macOS-friendly via sudo).")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (default: localhost only).")
    parser.add_argument("--port", type=int, default=8767, help="HTTP port (default: 8767).")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-95).")
    parser.add_argument("--max-fps", type=float, default=15.0, help="Max grab rate (0 = unbounded).")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="vertical",
        help="Correct wrist mounting before JPEG encode. Default vertical flips upside-down sensor.",
    )
    args = parser.parse_args()

    _warn_if_not_root()

    state = OrbbecState(quality=args.quality, max_fps=args.max_fps, flip=args.flip)
    print(
        f"orbbec-camera: starting pipeline (sudo={os.geteuid() == 0}, flip={args.flip})",
        flush=True,
    )
    try:
        state.start()
    except Exception as exc:  # noqa: BLE001
        print(
            f"orbbec-camera: failed to start pipeline: {exc}\n"
            "macOS hint: run with `sudo -E uv run python scripts/orbbec_camera_server.py` "
            "and quit Granola/Chrome camera tabs first.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    thread = threading.Thread(target=state.run, name="orbbec-grab", daemon=True)
    thread.start()

    server = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
    print(
        f"orbbec-camera: serving http://{args.host}:{args.port}/frame.jpg "
        f"(stats: http://{args.host}:{args.port}/healthz)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
