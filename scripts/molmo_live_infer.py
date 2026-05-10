#!/usr/bin/env python3
"""POST one Modal Molmo observation using live camera URLs (HTTP or WebSocket).

Use this to verify the ngrok WebSocket ``top`` feed plus Modal without running ``yamctl hybrid``
(the hybrid loop still needs CAN + robots). For full closed-loop control use ``yamctl hybrid`` below.

Examples::

  uv run python scripts/molmo_live_infer.py \\
    --top-url 'wss://YOUR.ngrok-free.app' \\
    --right-url 'http://127.0.0.1:8767/frame.jpg'

  # Top only (right/left sent as black frames matching top resolution)
  uv run python scripts/molmo_live_infer.py --top-url \"\$YAM_CAMERA_URL\"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from http_camera_fetch import fetch_rgb_from_camera_url  # noqa: E402
from local_modal_robot_bridge import (  # noqa: E402
    _extract_bimanual_actions,
    _normalize_molmo_action_matrix,
)


def _encode_rgb(rgb: np.ndarray, quality: int) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="JPEG", quality=max(1, min(95, quality)))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resize_rgb_to_shape(rgb: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    th, tw = hw
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape[0] == th and rgb.shape[1] == tw:
        return rgb
    return cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _black_rgb(hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return np.zeros((h, w, 3), dtype=np.uint8)


def main() -> int:
    default_top = (
        os.environ.get("YAM_CAMERA_URL") or os.environ.get("YAM_FRONT_CAMERA_URL") or ""
    ).strip()
    parser = argparse.ArgumentParser(description="Live Molmo /infer with real camera frames.")
    parser.add_argument(
        "--top-url",
        default=default_top or None,
        required=not bool(default_top),
        help="Molmo top view (HTTP(S) or ws/wss). Default: YAM_CAMERA_URL or YAM_FRONT_CAMERA_URL.",
    )
    parser.add_argument(
        "--right-url",
        default=None,
        help="Molmo right / wrist (HTTP(S) or ws/wss). If omitted, uses black frames sized like top.",
    )
    parser.add_argument(
        "--left-url",
        default=os.environ.get("YAM_LEFT_CAMERA_URL"),
        help="Molmo left (HTTP(S) or ws/wss). If omitted, black. Default: YAM_LEFT_CAMERA_URL.",
    )
    parser.add_argument(
        "--http-url",
        default=os.environ.get(
            "YAM_MODAL_POLICY_HTTP_URL",
            "https://aacmcgovern--yam-molmoact2-http-bridge-v3-serve.modal.run/infer",
        ),
    )
    parser.add_argument("--task", default="pick up the purple hat")
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=78)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save top/left/right JPEGs here for inspection.",
    )
    args = parser.parse_args()

    top_url = (args.top_url or "").strip()
    if not top_url:
        print("molmo_live_infer: pass --top-url or set YAM_CAMERA_URL", file=sys.stderr)
        return 2

    ws_registry: dict = {}
    top_rgb = fetch_rgb_from_camera_url(top_url, timeout=30.0, ws_registry=ws_registry)
    hw = (top_rgb.shape[0], top_rgb.shape[1])

    right_url = (args.right_url or "").strip()
    if right_url:
        right_rgb = fetch_rgb_from_camera_url(right_url, timeout=30.0, ws_registry=ws_registry)
        top_rgb = _resize_rgb_to_shape(top_rgb, (right_rgb.shape[0], right_rgb.shape[1]))
        hw = (top_rgb.shape[0], top_rgb.shape[1])
    else:
        right_rgb = _black_rgb(hw)

    left_url = (args.left_url or "").strip()
    if left_url:
        left_rgb = fetch_rgb_from_camera_url(left_url, timeout=30.0, ws_registry=ws_registry)
        left_rgb = _resize_rgb_to_shape(left_rgb, hw)
    else:
        left_rgb = _black_rgb(hw)

    jq = int(args.jpeg_quality)
    payload = {
        "type": "observation",
        "task": args.task,
        "state": [
            -0.06656748056411743,
            0.014686808921396732,
            0.016594186425209045,
            -0.08602273464202881,
            -0.014686808921396732,
            0.13904783129692078,
            0.9922363758087158,
            0.19512474536895752,
            0.010872052982449532,
            0.010872052982449532,
            -0.06771191209554672,
            -0.07305257022380829,
            -0.08945601433515549,
            0.9888537526130676,
        ],
        "state_format": "yam_bimanual_yam_14d",
        "images": {
            "top": _encode_rgb(top_rgb, jq),
            "left": _encode_rgb(left_rgb, jq),
            "right": _encode_rgb(right_rgb, jq),
        },
        "num_steps": args.num_steps,
    }

    if args.save_dir is not None:
        out = Path(args.save_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for name, arr in (("top", top_rgb), ("left", left_rgb), ("right", right_rgb)):
            Image.fromarray(arr).save(out / f"{name}.jpg", format="JPEG", quality=jq)
        print(json.dumps({"saved_views": str(out.resolve())}))

    print(json.dumps({"POST": args.http_url, "top_hw": list(top_rgb.shape[:2])}))
    r = requests.post(args.http_url, json=payload, timeout=args.timeout)
    print(json.dumps({"status_code": r.status_code}))
    try:
        body = r.json()
    except ValueError:
        print(r.text[:2000])
        return 1

    act = np.asarray(body.get("action", []), dtype=float)
    norm = _normalize_molmo_action_matrix(act, min_cols=14)
    targets = _extract_bimanual_actions(body, action_step=0, execute_action_steps=5)

    report = {
        "source": body.get("source"),
        "input_source": body.get("input_source"),
        "action_shape_json": body.get("action_shape"),
        "raw_asarray_shape": list(act.shape),
        "normalized_shape": None if norm is None else list(norm.shape),
        "extracted_target_count": len(targets),
        "first_target_preview": None if not targets else targets[0][:14].tolist(),
        "error": body.get("error"),
    }
    print(json.dumps(report, indent=2))
    return 0 if body.get("type") != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
