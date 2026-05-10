#!/usr/bin/env python3
"""Small client for the YAM robot websocket proxy."""

from __future__ import annotations

import argparse
import json

from websockets.sync.client import connect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8780/control")
    parser.add_argument("--method", default="get_joint_pos")
    parser.add_argument("--params", default="{}", help="JSON object passed as params.")
    args = parser.parse_args()

    with connect(args.url) as ws:
        print(ws.recv())
        ws.send(json.dumps({"id": "request-1", "method": args.method, "params": json.loads(args.params)}))
        print(ws.recv())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
