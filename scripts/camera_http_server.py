#!/usr/bin/env python3
"""Serve a local webcam frame over HTTP for processes without camera permission."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

def _open_capture(cv2, index: int, width: int, height: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


def _resolve_camera_index(cv2, requested: str, width: int, height: int, max_index: int) -> tuple[int, object]:
    if requested != "auto":
        index = int(requested)
        cap = _open_capture(cv2, index, width, height)
        if cap is None:
            raise RuntimeError(f"Could not open camera index {index}")
        return index, cap

    errors = []
    for index in range(max_index + 1):
        cap = _open_capture(cv2, index, width, height)
        if cap is not None:
            return index, cap
        errors.append(index)
    raise RuntimeError(f"Could not open any camera index from 0..{max_index}; tried {errors}")


class CameraState:
    def __init__(self, index: str, width: int, height: int, quality: int, max_index: int):
        import cv2

        self.cv2 = cv2
        self.quality = quality
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.stop = threading.Event()
        self.index, self.cap = _resolve_camera_index(cv2, index, width, height, max_index)

    def run(self) -> None:
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                ok, encoded = self.cv2.imencode(
                    ".jpg",
                    frame,
                    [int(self.cv2.IMWRITE_JPEG_QUALITY), self.quality],
                )
                if ok:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()
            time.sleep(0.03)

    def close(self) -> None:
        self.stop.set()
        self.cap.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", default="0", help="OpenCV camera index, or 'auto' to probe indexes.")
    parser.add_argument("--max-camera-index", type=int, default=9, help="Highest index to try when --camera-index=auto.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--quality", type=int, default=85)
    args = parser.parse_args()

    state = CameraState(args.camera_index, args.width, args.height, args.quality, args.max_camera_index)
    thread = threading.Thread(target=state.run, daemon=True)
    thread.start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in {"/", "/frame.jpg"}:
                self.send_response(404)
                self.end_headers()
                return
            with state.lock:
                body = state.latest_jpeg
            if body is None:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"camera warming up")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Camera server: http://{args.host}:{args.port}/frame.jpg camera_index={state.index}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
