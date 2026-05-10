#!/usr/bin/env python3
"""Try OpenCV indices (macOS: AVFoundation), save JPEGs, print whether frames look live."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from opencv_util import video_capture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe cv2 camera indices; saves logs/opencv-probe-*.jpg")
    parser.add_argument("--max-index", type=int, default=8, help="Try 0 through this index inclusive.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs",
        help="Directory for probe JPEGs (default: repo logs/).",
    )
    args = parser.parse_args()

    import cv2
    import numpy as np

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "macOS: using AVFoundation first for each index (see scripts/opencv_util.py).\n"
        "Quit Zoom/Chrome tabs using the camera; grant Terminal/Cursor Camera access if prompted.\n",
        flush=True,
    )

    for i in range(args.max_index + 1):
        cap = video_capture(i)
        if not cap.isOpened():
            print(f"index {i}: could not open", flush=True)
            continue
        for _ in range(8):
            cap.read()
        ok, bgr = cap.read()
        cap.release()
        if not ok or bgr is None:
            print(f"index {i}: opened but read failed", flush=True)
            continue
        path = out_dir / f"opencv-probe-index-{i}.jpg"
        cv2.imwrite(str(path), bgr)
        mean = float(np.mean(bgr))
        print(f"index {i}: saved {path} (BGR mean={mean:.1f}; ~0=black)", flush=True)

    print("\nOpen the JPEGs: the one that shows iVCam is your --*-camera-index.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
