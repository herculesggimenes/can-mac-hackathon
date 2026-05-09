#!/usr/bin/env python3
"""HTTP policy server for a one-arm LeRobot ACT policy.

The server intentionally fails closed when no trained policy is loaded. It
exposes the same minimal /health and /infer shape used by the local hybrid loop,
but returns 7D YAM actions directly instead of bimanual 14D actions.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class PolicyRuntime:
    def __init__(self, policy_path: str | None, *, device: str):
        self.policy_path = policy_path
        self.device = device
        self.policy = None
        self.torch = None
        self.load_error: str | None = None
        if policy_path:
            self._load(policy_path)
        else:
            self.load_error = "no policy path provided"

    @property
    def loaded(self) -> bool:
        return self.policy is not None and self.torch is not None

    def _load(self, policy_path: str) -> None:
        try:
            import torch
            from lerobot.policies.act.modeling_act import ACTPolicy

            policy = ACTPolicy.from_pretrained(policy_path)
            policy.to(self.device)
            policy.eval()
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            self.torch = torch
            self.policy = policy
            self.load_error = None
        except Exception as exc:  # noqa: BLE001 - surfaced through /health.
            self.policy = None
            self.torch = None
            self.load_error = repr(exc)

    def health(self) -> dict[str, Any]:
        return {
            "ok": self.loaded,
            "version": "yam-lerobot-act-one-arm-v1",
            "source": "yam_lerobot_act",
            "policy_path": self.policy_path,
            "policy_loaded": self.loaded,
            "device": self.device,
            "load_error": self.load_error,
            "action_shape": [7],
        }

    def infer(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if not self.loaded:
            return 503, {
                "type": "error",
                "source": "yam_lerobot_act",
                "execute_ok": False,
                "error": self.load_error or "policy not loaded",
            }
        try:
            observation = self._observation_from_payload(payload)
            with self.torch.no_grad():
                action = self.policy.select_action(observation)
            action_array = self._to_numpy(action)
            action_array = np.squeeze(action_array).astype(float)
            if action_array.ndim == 0:
                raise ValueError(f"policy returned scalar action: {action_array}")
            if action_array.ndim > 1:
                action_array = action_array.reshape(-1, action_array.shape[-1])
            if action_array.shape[-1] < 7:
                raise ValueError(f"policy action has fewer than 7 values: shape={list(action_array.shape)}")
            action_array = action_array[..., :7]
            if action_array.ndim == 1:
                action_array = action_array[None, :]
            if not np.isfinite(action_array).all():
                raise ValueError("policy action contains non-finite values")
            return 200, {
                "type": "action",
                "source": "yam_lerobot_act",
                "input_source": "payload",
                "execute_ok": True,
                "state_shape": [7],
                "action_shape": list(action_array.shape),
                "action": action_array.tolist(),
                "policy_path": self.policy_path,
            }
        except Exception as exc:  # noqa: BLE001 - returned as policy error.
            return 500, {
                "type": "error",
                "source": "yam_lerobot_act",
                "execute_ok": False,
                "error": repr(exc),
            }

    def _observation_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(payload.get("state", []), dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f"expected 7D state, got shape={list(state.shape)}")
        rgb = _decode_front_image(payload)
        image = np.asarray(rgb, dtype=np.float32) / 255.0
        image_tensor = self.torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        state_tensor = self.torch.from_numpy(state).unsqueeze(0).to(self.device)
        return {
            "observation.state": state_tensor,
            "observation.images.front": image_tensor,
            "task": [str(payload.get("task", ""))],
        }

    def _to_numpy(self, action: Any) -> np.ndarray:
        if hasattr(action, "detach"):
            return action.detach().cpu().numpy()
        if isinstance(action, dict):
            for key in ("action", "actions"):
                if key in action:
                    return self._to_numpy(action[key])
        return np.asarray(action)


def _decode_front_image(payload: dict[str, Any]) -> Image.Image:
    images = payload.get("images") or {}
    encoded = images.get("front") or images.get("top") or images.get("left") or images.get("right")
    if not encoded:
        raise ValueError("payload missing images.front")
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a one-arm LeRobot ACT policy over HTTP.")
    parser.add_argument("--policy-path", default=None, help="Local path or Hub id for a trained ACT policy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    runtime = PolicyRuntime(args.policy_path, device=args.device)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self._send(404, {"ok": False, "error": "not found"})
                return
            self._send(200 if runtime.loaded else 503, runtime.health())

        def do_POST(self):
            if self.path != "/infer":
                self._send(404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"type": "error", "source": "yam_lerobot_act", "execute_ok": False, "error": repr(exc)})
                return
            status, body = runtime.infer(payload)
            self._send(status, body)

        def log_message(self, format, *args):
            return

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps({"t": time.time(), **body}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"server": f"http://{args.host}:{args.port}", **runtime.health()}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
