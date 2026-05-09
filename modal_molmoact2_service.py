"""Modal HTTP service for MolmoAct2-BimanualYAM inference.

Deploy:
    .venv/bin/modal deploy modal_molmoact2_service.py

The endpoint intentionally returns execute_ok=false. It is for inspecting model
actions before a separate local process is allowed to command hardware.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import modal


APP_NAME = "yam-molmoact2-http-bridge-v3"
REPO_ID = "allenai/MolmoAct2-BimanualYAM"
REPO_REVISION = "1249f2047e509bd3e4abd0b028ab49599b9b7ffa"
NORM_TAG = "yam_dual_molmoact2"
MIN_CONTAINERS = int(os.environ.get("MOLMOACT2_MIN_CONTAINERS", "1"))
SCALEDOWN_WINDOW = int(os.environ.get("MOLMOACT2_SCALEDOWN_WINDOW", "1200"))
PRELOAD_MODEL = os.environ.get("MOLMOACT2_PRELOAD_MODEL", "1") != "0"
SAMPLE_TASK = "Place cups and plate in dishwasher rack, dispose of food waste, and organize remaining items."
SAMPLE_STATE = [
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

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate",
        "huggingface_hub",
        "einops",
        "numpy",
        "pillow",
        "requests",
        "scipy",
        "torch",
        "torchvision",
        "transformers",
    )
)


_MODEL = None
_PROCESSOR = None


def _safe_echo_action(payload: dict[str, Any], note: str) -> dict[str, Any]:
    state = payload.get("state") or []
    if len(state) == 7:
        action = list(state) + [0.0] * 7
    elif len(state) == 14:
        action = list(state)
    else:
        action = [0.0] * 14
    return {
        "type": "action",
        "source": "echo",
        "execute_ok": False,
        "action": action,
        "note": note,
    }


def _load_model():
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype_name = os.environ.get("MOLMOACT2_DTYPE", "bfloat16")
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    norm_stats_path = hf_hub_download(REPO_ID, "norm_stats.json", revision=REPO_REVISION)
    _PROCESSOR = AutoProcessor.from_pretrained(REPO_ID, revision=REPO_REVISION, trust_remote_code=True)
    _MODEL = AutoModelForImageTextToText.from_pretrained(
        REPO_ID,
        revision=REPO_REVISION,
        trust_remote_code=True,
        dtype=dtype,
    ).to("cuda").eval()
    _MODEL.config._name_or_path = str(Path(norm_stats_path).parent)
    return _MODEL, _PROCESSOR


def _ensure_norm_stats_path(model) -> None:
    from huggingface_hub import hf_hub_download

    norm_stats_path = hf_hub_download(REPO_ID, "norm_stats.json", revision=REPO_REVISION)
    model.config._name_or_path = str(Path(norm_stats_path).parent)


def _decode_image(value: Any):
    from PIL import Image

    if isinstance(value, str):
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    if isinstance(value, dict) and "b64" in value:
        return Image.open(io.BytesIO(base64.b64decode(value["b64"]))).convert("RGB")
    raise ValueError("image value must be a base64 string or {'b64': ...}")


def _sample_inputs():
    import numpy as np
    from huggingface_hub import hf_hub_download
    from PIL import Image

    images = [
        Image.open(hf_hub_download(REPO_ID, "assets/sample_top_rgb.png", revision=REPO_REVISION)).convert("RGB"),
        Image.open(hf_hub_download(REPO_ID, "assets/sample_left_rgb.png", revision=REPO_REVISION)).convert("RGB"),
        Image.open(hf_hub_download(REPO_ID, "assets/sample_right_rgb.png", revision=REPO_REVISION)).convert("RGB"),
    ]
    return images, np.asarray(SAMPLE_STATE, dtype=np.float32), SAMPLE_TASK, "official_sample"


def _payload_inputs(payload: dict[str, Any]):
    import numpy as np

    if payload.get("sample") or not payload.get("images"):
        return _sample_inputs()

    image_payload = payload["images"]
    images = [
        _decode_image(image_payload[key])
        for key in ("top", "left", "right")
    ]

    state = np.asarray(payload.get("state") or [], dtype=np.float32)
    if state.shape == (7,):
        state = np.concatenate([state, np.zeros(7, dtype=np.float32)])
    if state.shape != (14,):
        raise ValueError(f"expected 7D or 14D state, got shape {state.shape}")
    return images, state, payload.get("task") or SAMPLE_TASK, "payload"


def _predict(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        return _safe_echo_action(payload, "CUDA is unavailable in this Modal container.")

    model, processor = _load_model()
    _ensure_norm_stats_path(model)
    images, state, task, input_source = _payload_inputs(payload)
    dtype_name = os.environ.get("MOLMOACT2_DTYPE", "bfloat16")
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    num_steps = int(payload.get("num_steps") or os.environ.get("MOLMOACT2_NUM_STEPS", "10"))

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
        out = model.predict_action(
            processor=processor,
            images=images,
            task=task,
            state=state,
            norm_tag=NORM_TAG,
            action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=num_steps,
            normalize_language=True,
            enable_cuda_graph=False,
        )

    def to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            return [to_cpu(item) for item in value]
        if isinstance(value, dict):
            return {key: to_cpu(item) for key, item in value.items()}
        return value

    actions = np.asarray(to_cpu(out.actions), dtype=np.float32)
    return {
        "type": "action",
        "source": "molmoact2_bimanual_yam",
        "input_source": input_source,
        "repo_id": REPO_ID,
        "repo_revision": REPO_REVISION,
        "norm_tag": NORM_TAG,
        "execute_ok": False,
        "state_shape": list(state.shape),
        "action_shape": list(actions.shape),
        "action": actions.tolist(),
        "note": "Model action returned for inspection only. Local bridge must not execute this directly on the single-arm YAM.",
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.asgi_app()
def serve():
    if PRELOAD_MODEL:
        _load_model()

    async def asgi(scope, receive, send):
        if scope["type"] == "http" and scope.get("path") in {"/health", "/infer"}:
            if scope.get("path") == "/health":
                response = {
                    "ok": True,
                    "version": "molmoact2-bimanual-yam-v1",
                    "repo_id": REPO_ID,
                    "repo_revision": REPO_REVISION,
                    "model_loaded": _MODEL is not None,
                    "execute_ok": False,
                }
            else:
                chunks = []
                while True:
                    event = await receive()
                    if event["type"] != "http.request":
                        break
                    chunks.append(event.get("body", b""))
                    if not event.get("more_body", False):
                        break
                try:
                    payload = json.loads(b"".join(chunks).decode("utf-8") or "{}")
                    response = _predict(payload)
                except Exception as exc:
                    response = {"type": "error", "error": repr(exc), "execute_ok": False}

            body = json.dumps(response).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"not found"})

    return asgi
