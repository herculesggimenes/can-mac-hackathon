"""Open ``cv2.VideoCapture`` with a backend that works for virtual webcams on macOS (e.g. iVCam)."""

from __future__ import annotations

import sys

import cv2


def video_capture(device_index: int) -> cv2.VideoCapture:
    """Prefer AVFoundation on macOS; fall back to default backend."""

    idx = int(device_index)
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(idx)
