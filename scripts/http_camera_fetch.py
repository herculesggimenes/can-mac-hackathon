"""HTTP or WebSocket JPEG/PNG fetch; tunneled URLs may need extra headers (e.g. ngrok-free)."""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request

import numpy as np
import requests
from PIL import Image

try:
    import websocket
except ImportError as exc:  # pragma: no cover
    websocket = None  # type: ignore[misc, assignment]
    _websocket_import_error = exc
else:
    _websocket_import_error = None


def camera_request_headers(url: str) -> dict[str, str]:
    """Headers for tunneled camera URLs. ngrok-free returns HTML unless this is set."""

    headers: dict[str, str] = {"User-Agent": "yam-camera-fetch/1"}
    host = (urlparse(url).hostname or "").lower()
    if "ngrok" in host:
        headers["ngrok-skip-browser-warning"] = "69420"
    return headers


def build_urllib_camera_request(url: str) -> Request:
    """Return a ``urllib.request.Request`` suitable for ``urlopen`` on ``url``."""

    return Request(url, headers=camera_request_headers(url))


def _ws_handshake_header_list(url: str) -> list[str]:
    """Headers for ``websocket.create_connection`` (list of ``\"Key: value\"`` strings)."""

    h = camera_request_headers(url)
    return [f"{k}: {v}" for k, v in h.items()]


def _rgb_from_image_bytes(raw: bytes) -> np.ndarray | None:
    if len(raw) < 10:
        return None
    try:
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        return None


def _rgb_from_ws_message(message: Any) -> np.ndarray | None:
    """Decode one WebSocket message to RGB; unknown shape returns None."""

    if isinstance(message, bytes):
        rgb = _rgb_from_image_bytes(message)
        if rgb is not None:
            return rgb
        start = message.find(b"\xff\xd8")
        if start >= 0:
            end = message.find(b"\xff\xd9", start + 2)
            if end > start:
                segment = message[start : end + 2]
                return _rgb_from_image_bytes(segment)
        return None

    if isinstance(message, str):
        try:
            obj = json.loads(message)
        except json.JSONDecodeError:
            try:
                raw = base64.b64decode(message)
            except Exception:
                return None
            return _rgb_from_image_bytes(raw)

        if isinstance(obj, dict):
            for key in ("image", "frame", "jpeg", "jpg", "data", "b64", "payload"):
                val = obj.get(key)
                if isinstance(val, str):
                    try:
                        raw = base64.b64decode(val)
                        rgb = _rgb_from_image_bytes(raw)
                        if rgb is not None:
                            return rgb
                    except Exception:
                        continue
        return None

    return None


def _ws_connect(url: str, *, open_timeout: float) -> Any:
    if websocket is None:
        raise RuntimeError(
            "WebSocket camera URLs require the websocket-client package "
            f"(install failed: {_websocket_import_error})"
        )
    return websocket.create_connection(url, timeout=open_timeout, header=_ws_handshake_header_list(url))


def _close_ws_registry(registry: dict[str, Any]) -> None:
    for _url, ws in list(registry.items()):
        try:
            ws.close()
        except Exception:
            pass
        registry.pop(_url, None)


def _ws_camera_handshake_response(ws: Any, message: str) -> bool:
    """If ``message`` is a ``hello`` JSON from the YAM WS camera protocol, send ``get_frame`` and return True."""

    try:
        obj = json.loads(message)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict) or obj.get("type") != "hello":
        return False
    cmds = obj.get("commands")
    if not isinstance(cmds, list) or "get_frame" not in cmds:
        return False
    setattr(ws, "_yam_ws_use_get_frame_cmd", True)
    ws.send(json.dumps({"command": "get_frame"}))
    return True


def fetch_rgb_from_websocket(
    url: str,
    *,
    registry: dict[str, Any],
    timeout: float = 5.0,
) -> np.ndarray:
    """Receive one image frame from a WebSocket (binary JPEG/PNG or JSON base64).

    The connection is stored in ``registry`` under ``url`` and reused across calls.
    """

    deadline = time.monotonic() + timeout
    last_err: BaseException | None = None

    while time.monotonic() < deadline:
        ws = registry.get(url)
        if ws is None:
            open_budget = min(15.0, max(0.5, deadline - time.monotonic()))
            try:
                ws = _ws_connect(url, open_timeout=open_budget)
                registry[url] = ws
            except BaseException as exc:
                last_err = exc
                time.sleep(0.05)
                continue

        try:
            ws.settimeout(max(0.05, deadline - time.monotonic()))
            # YAM WS camera: hello → send get_frame; later frames need get_frame before each recv.
            if getattr(ws, "_yam_ws_use_get_frame_cmd", False) and getattr(ws, "_yam_need_prefetch_send", False):
                ws.send(json.dumps({"command": "get_frame"}))
                setattr(ws, "_yam_need_prefetch_send", False)
            msg = ws.recv()
        except BaseException as exc:
            last_err = exc
            try:
                ws.close()
            except Exception:
                pass
            registry.pop(url, None)
            continue

        if isinstance(msg, str) and _ws_camera_handshake_response(ws, msg):
            continue

        rgb = _rgb_from_ws_message(msg)
        if rgb is not None:
            setattr(ws, "_yam_need_prefetch_send", True)
            return rgb

    raise TimeoutError(f"no decodable image frame from WebSocket within {timeout}s") from last_err


def fetch_rgb_from_camera_url(
    url: str,
    *,
    timeout: float = 5.0,
    session: requests.Session | None = None,
    ws_registry: dict[str, Any] | None = None,
) -> np.ndarray:
    """Download one frame and return RGB ``uint8`` ``(H, W, 3)``.

    Uses HTTP(S) GET for ``http(s)://`` URLs and WebSocket binary/text frames for ``ws(s)://``.
    """

    u = url.strip()
    if u.startswith(("ws://", "wss://")):
        if ws_registry is None:
            reg: dict[str, Any] = {}
            try:
                return fetch_rgb_from_websocket(u, registry=reg, timeout=timeout)
            finally:
                _close_ws_registry(reg)
        return fetch_rgb_from_websocket(u, registry=ws_registry, timeout=timeout)

    get = session.get if session is not None else requests.get
    response = get(u, timeout=timeout, headers=camera_request_headers(u))
    response.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))
