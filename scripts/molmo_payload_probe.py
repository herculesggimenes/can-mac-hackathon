#!/usr/bin/env python3
"""POST a sample observation to Modal Molmo infer and print payload shape + extraction stats."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local_modal_robot_bridge import (  # noqa: E402
    _extract_bimanual_actions,
    _normalize_molmo_action_matrix,
)


def _tiny_jpeg_b64() -> str:
    img = Image.new("RGB", (640, 480), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Modal /infer JSON action field.")
    parser.add_argument(
        "--http-url",
        default=os.environ.get(
            "YAM_MODAL_POLICY_HTTP_URL",
            "https://aacmcgovern--yam-molmoact2-http-bridge-v3-serve.modal.run/infer",
        ),
    )
    parser.add_argument("--task", default="pick up the purple hat")
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    state = [
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
    ]
    b64 = _tiny_jpeg_b64()
    payload = {
        "type": "observation",
        "task": args.task,
        "state": state,
        "state_format": "yam_bimanual_yam_14d",
        "images": {"top": b64, "left": b64, "right": b64},
        "num_steps": args.num_steps,
    }

    print(json.dumps({"POST": args.http_url, "payload_keys": list(payload.keys())}))
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
        "raw_ndim": int(act.ndim),
        "normalized_shape": None if norm is None else list(norm.shape),
        "extracted_target_count": len(targets),
        "first_target_preview": None if not targets else targets[0][:14].tolist(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
