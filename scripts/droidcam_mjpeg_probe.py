#!/usr/bin/env python3
"""Fetch first JPEG frame from DroidCam HTTP MJPEG URL (same URL as OBS Browser source).

DroidCam serves multipart MJPEG at URLs like:
  http://PHONE_IP:4747/video/1280x720
  http://PHONE_IP:4747/video/force/1280x720

Run on the same LAN as the phone with the DroidCam app open (Wi‑Fi IP + port shown in app).

Example:
  uv run python scripts/droidcam_mjpeg_probe.py --host 10.104.3.177 --port 4747
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _read_first_jpeg_from_stream(resp, max_bytes: int = 12 * 1024 * 1024) -> bytes:
    """Read raw HTTP body until first complete JPEG (SOI … EOI)."""
    buf = b""
    total = 0
    start_idx: int | None = None
    while total < max_bytes:
        chunk = resp.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if start_idx is None:
            buf += chunk
            i = buf.find(b"\xff\xd8")
            if i == -1:
                if len(buf) > 256 * 1024:
                    buf = buf[-128 * 1024:]
                continue
            start_idx = i
            buf = buf[start_idx:]
            start_idx = 0
        else:
            buf += chunk
        end_rel = buf.find(b"\xff\xd9", 2)
        if end_rel != -1:
            return buf[: end_rel + 2]
    raise ValueError("No complete JPEG in stream (timeout or not MJPEG JPEG)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Save first frame from DroidCam /video/… MJPEG URL.")
    parser.add_argument("--host", default="10.104.3.177", help="Phone Wi‑Fi IP (from DroidCam app).")
    parser.add_argument("--port", type=int, default=4747, help="Port shown in DroidCam app (often 4747).")
    parser.add_argument(
        "--paths",
        default="/video/1280x720,/video/force/1280x720,/video/640x480",
        help="Comma-separated paths to try (first that yields JPEG wins).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JPEG (default: logs/droidcam-mjpeg-probe.jpg).",
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = args.output or (root / "logs/droidcam-mjpeg-probe.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)

    paths = [p.strip() if p.strip().startswith("/") else "/" + p.strip() for p in args.paths.split(",") if p.strip()]

    for path in paths:
        url = f"http://{args.host}:{args.port}{path}"
        print(f"Trying {url} …", flush=True)
        try:
            req = Request(url, headers={"User-Agent": "yam-droidcam-mjpeg-probe/1"})
            with urlopen(req, timeout=args.timeout) as resp:
                jpeg = _read_first_jpeg_from_stream(resp)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"  FAIL ({exc})", flush=True)
            continue
        if len(jpeg) < 100 or jpeg[:2] != b"\xff\xd8":
            print(f"  SKIP (bad JPEG, {len(jpeg)} bytes)", flush=True)
            continue
        out.write_bytes(jpeg)
        print(f"\nOK — first JPEG from:\n  {url}", flush=True)
        print(f"\nSaved: {out.resolve()}", flush=True)
        print("\nExample:\n", flush=True)
        print(f'  export YAM_CAMERA_URL="{url}"', flush=True)
        return 0

    print(
        "\nNo MJPEG JPEG received. Check: phone app running, same Wi‑Fi, firewall, "
        "IP/port match the app; try another port in DroidCam settings.",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
