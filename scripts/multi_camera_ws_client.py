#!/usr/bin/env python3
"""Debug client for the multi-camera WebSocket endpoint."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from websockets.sync.client import connect


def _save_frame(frame: dict, output_dir: Path) -> None:
    if frame.get("type") != "frame":
        print(json.dumps(frame))
        return
    camera_id = frame["camera_id"]
    path = output_dir / f"{camera_id}.jpg"
    path.write_bytes(base64.b64decode(frame["data"]))
    print(f"saved {camera_id}: {path} ({path.stat().st_size} bytes)")
    depth = frame.get("depth")
    if depth and depth.get("data"):
        raw_path = output_dir / f"{camera_id}.u16"
        meta_path = output_dir / f"{camera_id}.depth.json"
        raw_path.write_bytes(base64.b64decode(depth["data"]))
        metadata = dict(depth)
        metadata.pop("data", None)
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"saved {camera_id} depth: {raw_path} ({raw_path.stat().st_size} bytes), {meta_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Save frames from the multi-camera WebSocket endpoint.")
    parser.add_argument("url", help="ws://.../cameras or wss://.../cameras")
    parser.add_argument("--output-dir", default="logs/multi-camera-check")
    parser.add_argument("--subscribe", action="store_true")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=1, help="Frame batches to save when subscribing.")
    parser.add_argument("--reset-camera", help="Reset one camera id and exit, e.g. top.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with connect(args.url, open_timeout=10, max_size=32 * 1024 * 1024) as ws:
        hello = json.loads(ws.recv())
        print(json.dumps(hello, indent=2))
        if args.reset_camera:
            ws.send(json.dumps({"type": "reset_camera", "camera_id": args.reset_camera}))
            print(json.dumps(json.loads(ws.recv()), indent=2))
            return 0
        if args.subscribe:
            ws.send(json.dumps({"type": "subscribe", "fps": args.fps, "cameras": "all", "bundle": True}))
        else:
            ws.send(json.dumps({"type": "get_all_frames", "bundle": True}))

        batches = 0
        while batches < args.frames:
            payload = json.loads(ws.recv())
            if payload.get("type") == "subscribed":
                print(json.dumps(payload))
                continue
            if payload.get("type") == "frames":
                for frame in payload.get("frames", []):
                    _save_frame(frame, output_dir)
                batches += 1
            else:
                _save_frame(payload, output_dir)
                batches += 1
        if args.subscribe:
            ws.send(json.dumps({"type": "stop"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
