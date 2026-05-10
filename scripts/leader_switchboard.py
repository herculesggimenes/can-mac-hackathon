#!/usr/bin/env python3
"""Local operator UI for SO leader -> YAM WebSocket teleop."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import time
import threading
from pathlib import Path
from typing import Any

import serial.tools.list_ports
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]
HACKATHON = ROOT.parent
LEROBOT = HACKATHON / "lerobot-MakerMods"
LEADER_BRIDGE = ROOT / "scripts" / "so_leader_ws_bridge.py"

DEFAULT_CONTROL_URL = "wss://2d37-12-125-194-54.ngrok-free.app/control"
DEFAULT_CAMERA_URL = "ws://127.0.0.1:8770/cameras"
DEFAULT_LEADER_PORT = "/dev/cu.usbmodem5B140318401"
LOG_PATH = ROOT / "logs" / "leader_switchboard_bridge.log"
LEADER_STATUS_PATH = ROOT / "logs" / "leader_axes.json"


class TeleopStart(BaseModel):
    arm: str
    control_url: str = DEFAULT_CONTROL_URL
    leader_port: str = DEFAULT_LEADER_PORT
    kind: str = "so100"
    joint_map: str = "0,1,2,3,4"
    joint_signs: str = "-1,-1,-1,-1,-1"
    sixth_joint_source: str = "gripper"
    sixth_joint_sign: float = -1.0
    lock_joints: str = ""
    hz: float = 40.0
    max_step: float = 0.006
    max_gripper_step: float = 0.005
    max_joint_delta: float = 0.15
    sync_samples: int = 5
    fire_and_forget: bool = True
    max_in_flight: int = 3


class ZeroGravityRequest(BaseModel):
    enabled: bool
    control_url: str = DEFAULT_CONTROL_URL


class Runtime:
    def __init__(self) -> None:
        self.proc: subprocess.Popen[str] | None = None
        self.started_at: float | None = None
        self.config: dict[str, Any] | None = None

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
        self.proc = None
        self.started_at = None
        self.config = None

    def status(self) -> dict[str, Any]:
        running = self.proc is not None and self.proc.poll() is None
        return {
            "running": running,
            "pid": self.proc.pid if running and self.proc is not None else None,
            "started_at": self.started_at,
            "config": self.config,
        }


runtime = Runtime()
runtime_lock = threading.Lock()


def _detect_leader_port() -> str:
    candidates = []
    for port in serial.tools.list_ports.comports():
        text = " ".join(str(part or "") for part in (port.device, port.description, port.hwid))
        if "1A86:55D3" in text or "USB Single Serial" in text:
            candidates.append(port.device)
    if candidates:
        return sorted(candidates)[0]
    return DEFAULT_LEADER_PORT


def _robot_rpc(control_url: str, method: str, params: dict[str, Any] | None = None) -> Any:
    with connect(control_url, open_timeout=10, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        ws.send(json.dumps({"id": "switchboard", "method": method, "params": params or {}}))
        response = json.loads(ws.recv())
        if not response.get("ok"):
            raise RuntimeError(response.get("error", response))
        return response.get("result")


def create_app(camera_url: str) -> FastAPI:
    app = FastAPI(title="Leader Switchboard")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/config")
    def api_config() -> dict[str, str]:
        return {
            "control_url": DEFAULT_CONTROL_URL,
            "camera_url": camera_url,
            "leader_port": _detect_leader_port(),
        }

    @app.get("/api/teleop/status")
    def teleop_status() -> dict[str, Any]:
        return runtime.status()

    @app.post("/api/teleop/start")
    def teleop_start(config: TeleopStart) -> dict[str, Any]:
        if config.arm not in {"left", "right", "both"}:
            raise HTTPException(status_code=400, detail="arm must be left, right, or both")
        if config.kind not in {"so100", "so101"}:
            raise HTTPException(status_code=400, detail="kind must be so100 or so101")
        if config.sixth_joint_source not in {"none", "gripper", "wrist_roll"}:
            raise HTTPException(status_code=400, detail="invalid sixth_joint_source")

        with runtime_lock:
            runtime.stop()
            time.sleep(0.25)
            if config.leader_port == DEFAULT_LEADER_PORT and not Path(config.leader_port).exists():
                config.leader_port = _detect_leader_port()
            cmd = [
                str(LEROBOT / ".venv" / "bin" / "python"),
                str(LEADER_BRIDGE),
                "--control-url",
                config.control_url,
                "--port",
                config.leader_port,
                "--kind",
                config.kind,
                "--arm",
                config.arm,
                "--joint-map",
                config.joint_map,
                "--hz",
                str(config.hz),
                "--max-step",
                str(config.max_step),
                "--max-gripper-step",
                str(config.max_gripper_step),
                "--max-joint-delta",
                str(config.max_joint_delta),
                f"--joint-signs={config.joint_signs}",
                "--sixth-joint-source",
                config.sixth_joint_source,
                "--sixth-joint-sign",
                str(config.sixth_joint_sign),
                "--lock-joints",
                config.lock_joints,
                "--sync-samples",
                str(config.sync_samples),
                "--max-in-flight",
                str(config.max_in_flight),
                "--status-path",
                str(LEADER_STATUS_PATH),
                "--execute",
            ]
            if config.fire_and_forget:
                cmd.insert(-1, "--fire-and-forget")
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log = LOG_PATH.open("a")
            log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting {' '.join(cmd)} ---\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=LEROBOT,
                env=env,
                text=True,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            runtime.proc = proc
            runtime.started_at = time.time()
            runtime.config = config.model_dump()
            time.sleep(1.2)
            if proc.poll() is not None:
                runtime.stop()
                detail = "leader bridge exited during startup"
                try:
                    tail = LOG_PATH.read_text(errors="replace").splitlines()[-25:]
                    detail = detail + "\n" + "\n".join(tail)
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail=detail)
            return runtime.status()

    @app.post("/api/teleop/stop")
    def teleop_stop() -> dict[str, Any]:
        with runtime_lock:
            runtime.stop()
            return runtime.status()

    @app.get("/api/robot/status")
    async def robot_status(control_url: str = DEFAULT_CONTROL_URL) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(_robot_rpc, control_url, "get_status", {})
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/leader/axes")
    def leader_axes() -> dict[str, Any]:
        try:
            return json.loads(LEADER_STATUS_PATH.read_text())
        except FileNotFoundError:
            return {"error": "leader axis status is not available yet"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/robot/joints")
    async def robot_joints(control_url: str = DEFAULT_CONTROL_URL) -> list[float]:
        try:
            return await asyncio.to_thread(_robot_rpc, control_url, "get_joint_pos", {})
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/robot/zero-gravity")
    async def set_zero_gravity(request: ZeroGravityRequest) -> dict[str, Any]:
        try:
            await asyncio.to_thread(
                _robot_rpc,
                request.control_url,
                "set_zero_gravity_mode",
                {"enabled": request.enabled},
            )
            return await asyncio.to_thread(_robot_rpc, request.control_url, "get_status", {})
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leader Switchboard</title>
  <style>
    :root { color-scheme: dark; --bg:#111315; --panel:#1b2025; --line:#303740; --text:#eef2f5; --muted:#9ba8b5; --blue:#6db6ff; --red:#ff6b6b; --green:#54d17a; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.4 system-ui,-apple-system,BlinkMacSystemFont,sans-serif; background:var(--bg); color:var(--text); }
    header { height:56px; display:flex; align-items:center; justify-content:space-between; padding:0 18px; border-bottom:1px solid var(--line); }
    h1 { font-size:18px; margin:0; font-weight:650; }
    main { display:grid; grid-template-columns:360px 1fr; gap:16px; padding:16px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
    h2 { margin:0 0 12px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
    label { display:block; margin:10px 0 4px; color:var(--muted); }
    input, select { width:100%; height:34px; border:1px solid var(--line); border-radius:6px; background:#0f1317; color:var(--text); padding:0 9px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .seg { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; }
    button { height:36px; border:0; border-radius:6px; background:#303946; color:var(--text); font-weight:650; cursor:pointer; }
    button.primary { background:var(--blue); color:#071018; }
    button.stop { background:var(--red); color:white; }
    button.active { outline:2px solid var(--blue); }
    .locks { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-top:6px; }
    .locks label { margin:0; display:flex; gap:5px; align-items:center; justify-content:center; height:30px; border:1px solid var(--line); border-radius:6px; color:var(--text); }
    .locks input { width:auto; height:auto; }
    .status { white-space:pre-wrap; color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; max-height:210px; overflow:auto; }
    .cams { display:grid; grid-template-columns:repeat(2,minmax(240px,1fr)); gap:10px; }
    .cam { border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#050607; min-height:180px; }
    .cam-title { height:28px; display:flex; align-items:center; padding:0 10px; color:var(--muted); border-bottom:1px solid var(--line); font-size:12px; }
    .cam img { display:block; width:100%; aspect-ratio:4/3; object-fit:contain; background:#050607; }
    .cam .msg { min-height:180px; display:flex; align-items:center; justify-content:center; color:var(--muted); padding:12px; text-align:center; }
    .pill { padding:4px 8px; border-radius:999px; background:#26303a; color:var(--muted); font-size:12px; }
    .pill.live { color:var(--green); }
  </style>
</head>
<body>
  <header>
    <h1>Leader Switchboard</h1>
    <div id="live" class="pill">loading</div>
  </header>
  <main>
    <div>
      <section>
        <h2>Teleop</h2>
        <label>Arm</label>
        <div class="seg">
          <button id="leftBtn">Left</button>
          <button id="rightBtn" class="active">Right</button>
          <button id="bothBtn">Both</button>
        </div>
        <label>Control WebSocket</label>
        <input id="controlUrl">
        <label>Camera WebSocket</label>
        <input id="cameraUrl">
        <label>Leader port</label>
        <input id="leaderPort">
        <div class="row">
          <div><label>Joint map</label><input id="jointMap" value="0,1,2,3,4"></div>
          <div><label>Joint signs</label><input id="jointSigns" value="-1,-1,-1,-1,-1"></div>
        </div>
        <div class="row">
          <div><label>Sixth source</label><select id="sixthSource"><option>gripper</option><option>wrist_roll</option><option>none</option></select></div>
        </div>
        <div class="row">
          <div><label>Hz</label><input id="hz" type="number" min="1" max="120" step="1" value="40"></div>
          <div><label>Max step</label><input id="maxStep" type="number" min="0.001" max="0.1" step="0.001" value="0.006"></div>
        </div>
        <div class="locks" style="grid-template-columns:1fr">
          <label><input id="fireForget" type="checkbox" checked> Responsive bounded sends</label>
        </div>
        <label>Lock YAM arm joints</label>
        <div class="locks" id="locks"></div>
        <div class="row" style="margin-top:12px">
          <button class="primary" id="startBtn">Start / Sync</button>
          <button class="stop" id="stopBtn">Stop</button>
        </div>
        <div class="row" style="margin-top:8px">
          <button id="zeroOnBtn">Zero-G On</button>
          <button id="zeroOffBtn">Zero-G Off</button>
        </div>
      </section>
      <section style="margin-top:14px">
        <h2>Status</h2>
        <div id="status" class="status"></div>
      </section>
      <section style="margin-top:14px">
        <h2>Leader Axes</h2>
        <div id="axes" class="status"></div>
      </section>
    </div>
    <section>
      <h2>Cameras</h2>
      <div class="cams" id="cams"></div>
    </section>
  </main>
<script>
let activeArm = "right";
let cameraWs = null;
const $ = (id) => document.getElementById(id);
for (let i=0; i<7; i++) {
  const label = document.createElement("label");
  label.innerHTML = `<input type="checkbox" value="${i}"> J${i}`;
  $("locks").appendChild(label);
}
function setArm(arm) {
  activeArm = arm;
  $("leftBtn").classList.toggle("active", arm === "left");
  $("rightBtn").classList.toggle("active", arm === "right");
  $("bothBtn").classList.toggle("active", arm === "both");
}
$("leftBtn").onclick = () => setArm("left");
$("rightBtn").onclick = () => setArm("right");
$("bothBtn").onclick = () => setArm("both");
async function init() {
  const cfg = await fetch("/api/config").then(r => r.json());
  $("controlUrl").value = cfg.control_url;
  $("cameraUrl").value = cfg.camera_url;
  $("leaderPort").value = cfg.leader_port;
  connectCamera();
  await refresh();
}
function lockJoints() {
  return [...document.querySelectorAll("#locks input:checked")].map(i => i.value).join(",");
}
$("startBtn").onclick = async () => {
  $("startBtn").disabled = true;
  const payload = {
    arm: activeArm,
    control_url: $("controlUrl").value,
    leader_port: $("leaderPort").value,
    joint_map: $("jointMap").value,
    joint_signs: $("jointSigns").value,
    sixth_joint_source: $("sixthSource").value,
    lock_joints: lockJoints(),
    hz: Number($("hz").value || 40),
    max_step: Number($("maxStep").value || 0.006),
    fire_and_forget: $("fireForget").checked,
    max_in_flight: 3
  };
  try {
    const res = await fetch("/api/teleop/start", {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify(payload)});
    if (!res.ok) alert(await res.text());
    await refresh();
  } finally {
    $("startBtn").disabled = false;
  }
};
$("stopBtn").onclick = async () => { await fetch("/api/teleop/stop", {method:"POST"}); await refresh(); };
$("zeroOnBtn").onclick = () => setZeroGravity(true);
$("zeroOffBtn").onclick = () => setZeroGravity(false);
async function setZeroGravity(enabled) {
  const res = await fetch("/api/robot/zero-gravity", {
    method:"POST",
    headers:{"content-type":"application/json"},
    body:JSON.stringify({enabled, control_url:$("controlUrl").value})
  });
  if (!res.ok) alert(await res.text());
  await refresh();
}
async function refresh() {
  const [teleop, robot, axes] = await Promise.all([
    fetch("/api/teleop/status").then(r => r.json()),
    fetch(`/api/robot/status?control_url=${encodeURIComponent($("controlUrl").value)}`).then(r => r.json()).catch(e => ({error:String(e)})),
    fetch("/api/leader/axes").then(r => r.json()).catch(e => ({error:String(e)}))
  ]);
  $("live").textContent = teleop.running ? `teleop ${teleop.config.arm}` : "teleop stopped";
  $("live").classList.toggle("live", !!teleop.running);
  $("status").textContent = JSON.stringify({teleop, robot}, null, 2);
  $("axes").textContent = formatAxes(axes);
}
function formatAxes(axes) {
  if (axes.error || !axes.motors) return JSON.stringify(axes, null, 2);
  return axes.motors.map((name, i) => {
    const raw = Number(axes.raw[i]).toFixed(0).padStart(5, " ");
    const delta = Number(axes.delta_raw[i]).toFixed(0).padStart(5, " ");
    const mapped = i < axes.joint_map.length ? ` -> J${axes.joint_map[i]}` : "";
    return `${i}: ${name.padEnd(13)} raw=${raw} delta=${delta}${mapped}`;
  }).join("\n");
}
function connectCamera() {
  if (cameraWs) cameraWs.close();
  const url = $("cameraUrl").value;
  cameraWs = new WebSocket(url);
  cameraWs.onopen = () => cameraWs.send(JSON.stringify({type:"subscribe", fps:6, cameras:"all", bundle:true}));
  cameraWs.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "hello" && Array.isArray(msg.cameras)) {
      for (const cam of msg.cameras) ensureCam(cam.camera_id, "waiting for frame");
      return;
    }
    if (msg.camera_id && msg.type && msg.type !== "frame") {
      ensureCam(msg.camera_id, msg.type.replaceAll("_", " "));
      return;
    }
    const frames = msg.type === "frames" ? msg.frames : (msg.type === "frame" ? [msg] : []);
    for (const frame of frames) {
      if (!frame.camera_id) continue;
      const node = ensureCam(frame.camera_id, frame.type || "waiting for frame");
      if (frame.data) {
        node.innerHTML = `<div class="cam-title">${frame.camera_id}</div><img>`;
        node.querySelector("img").src = `data:image/jpeg;base64,${frame.data}`;
      }
    }
  };
}
function ensureCam(cameraId, message) {
  let node = document.getElementById(`cam-${cameraId}`);
  if (!node) {
    node = document.createElement("div");
    node.id = `cam-${cameraId}`;
    node.className = "cam";
    $("cams").appendChild(node);
  }
  if (!node.querySelector("img")) {
    node.innerHTML = `<div class="cam-title">${cameraId}</div><div class="msg">${message}</div>`;
  }
  return node;
}
$("cameraUrl").addEventListener("change", connectCamera);
setInterval(refresh, 1000);
init();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--camera-url", default=DEFAULT_CAMERA_URL)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.camera_url), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
