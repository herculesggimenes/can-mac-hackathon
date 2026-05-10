#!/usr/bin/env python3
"""Try common HTTP snapshot URLs (IP Webcam–style); save first JPEG that works."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _is_jpeg(data: bytes) -> bool:
    return len(data) >= 3 and data[:2] == b"\xff\xd8"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe phone/webcam HTTP JPEG endpoints (e.g. IP Webcam shot.jpg).",
    )
    parser.add_argument("--host", default="10.104.3.177", help="Phone IP on your LAN.")
    parser.add_argument(
        "--ports",
        default="8080,81,80,4747,7777",
        help="Comma-separated ports to try (default: common camera servers).",
    )
    parser.add_argument(
        "--paths",
        default="/shot.jpg,/photo.jpg,/jpeg,/snapshot.jpg,/capture,/current.jpg",
        help="Comma-separated paths (after IP Webcam-style shot.jpg first).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save first good frame here (default: logs/http-camera-probe.jpg).",
    )
    parser.add_argument("--timeout", type=float, default=4.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = args.output or (root / "logs/http-camera-probe.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)

    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    paths = [p.strip() if p.strip().startswith("/") else "/" + p.strip() for p in args.paths.split(",") if p.strip()]

    print(f"Host {args.host!r} — trying {len(ports)} ports × {len(paths)} paths (timeout {args.timeout}s each).\n", flush=True)

    for port in ports:
        for path in paths:
            url = f"http://{args.host}:{port}{path}"
            try:
                req = Request(url, headers={"User-Agent": "yam-http-camera-probe/1"})
                with urlopen(req, timeout=args.timeout) as resp:
                    data = resp.read()
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                print(f"FAIL {url}  ({exc})", flush=True)
                continue
            if not _is_jpeg(data):
                print(f"SKIP {url}  (not JPEG, {len(data)} bytes)", flush=True)
                continue
            out.write_bytes(data)
            print(f"\nOK — JPEG from:\n  {url}", flush=True)
            print(f"\nSaved: {out.resolve()}", flush=True)
            print("\nUse with this repo, e.g.:\n", flush=True)
            print(f'  export YAM_CAMERA_URL="{url}"', flush=True)
            print(f'  export YAM_RECORD_CAMERA_URL="{url}"', flush=True)
            print(
                "  uv run yamctl hybrid \"…\" --policy-kind modal "
                f'--camera-url "{url}" ...',
                flush=True,
            )
            print(
                f'  # wrist Molmo right:  --right-camera-url "{url}"',
                flush=True,
            )
            return 0

    print(
        "\nNo JPEG URL responded. On the phone: start the camera server app, note the "
        "exact URL it shows (port + path), then:\n"
        f'  uv run python scripts/http_camera_probe.py --host {args.host} '
        "--ports YOUR_PORT --paths /YOUR_PATH",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
