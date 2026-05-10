"""Browser dashboard for YAM stack status and CAN / model controls."""

from __future__ import annotations

import argparse
import errno
import io
import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from _yamtest.cli import (
    DEFAULT_SERIAL_PORT,
    ROOT,
    cmd_can_check,
    cmd_clear_stop,
    cmd_hard_stop,
    cmd_start_bridge,
    cmd_status,
    cmd_stop,
)

RESET_SOCKETCAN_SCRIPT = ROOT / "i2rt" / "scripts" / "reset_all_can.sh"


def _capture_json(func, args: argparse.Namespace) -> dict:
    buf = io.StringIO()
    with redirect_stdout(buf):
        func(args)
    text = buf.getvalue().strip()
    # Commands may print a trailing non-JSON line; take last `{...}` block if needed.
    if text.startswith("{"):
        return json.loads(text)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"no JSON in output: {text[:500]}")


def _merge_status() -> dict:
    status = _capture_json(cmd_status, argparse.Namespace())
    can_check = _capture_json(cmd_can_check, argparse.Namespace())
    return {
        **status,
        "can_check": can_check,
        "platform": platform.system(),
        "socketcan_reset_available": RESET_SOCKETCAN_SCRIPT.is_file(),
    }


def create_app() -> FastAPI:
    app = FastAPI(title="YAM Operator Dashboard", version="0.1.0")

    @app.get("/api/status")
    def api_status() -> dict:
        try:
            return _merge_status()
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/restart-bridge")
    def restart_bridge() -> dict:
        serial = os.environ.get("YAM_SERIAL_PORT", DEFAULT_SERIAL_PORT)
        bitrate = int(os.environ.get("YAM_CAN_BITRATE", "1000000"))
        cmd_stop(argparse.Namespace(targets=["bridge"], hard_stop=False))
        rc = cmd_start_bridge(
            argparse.Namespace(serial_port=serial, bitrate=bitrate)
        )
        if rc != 0:
            raise HTTPException(
                status_code=500,
                detail="bridge failed to start; see logs/bridge.log",
            )
        return {"ok": True, "message": f"bridge restarted (serial={serial})"}

    @app.post("/api/actions/reset-socketcan")
    def reset_socketcan() -> dict:
        """Linux only: bounce SocketCAN interfaces (see i2rt docs). macOS uses Unix bridges only."""
        if platform.system() != "Linux":
            raise HTTPException(
                status_code=400,
                detail="SocketCAN reset applies to Linux; on macOS use Restart CAN bridge.",
            )
        if not RESET_SOCKETCAN_SCRIPT.is_file():
            raise HTTPException(status_code=404, detail="reset_all_can.sh not found")
        try:
            proc = subprocess.run(
                ["bash", str(RESET_SOCKETCAN_SCRIPT)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=500, detail="reset timed out") from exc
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    @app.post("/api/actions/restart-model")
    def restart_model() -> dict:
        """Stop inference / policy processes and clear HARD_STOP (does not auto-start a new run)."""
        cmd_stop(argparse.Namespace(targets=["model", "policy"], hard_stop=True))
        cmd_clear_stop(argparse.Namespace())
        return {
            "ok": True,
            "message": "model + policy stopped with hard-stop; HARD_STOP cleared. Start hybrid or policy-server again when ready.",
        }

    @app.post("/api/actions/hard-stop")
    def hard_stop() -> dict:
        cmd_hard_stop(argparse.Namespace())
        return {"ok": True, "message": "HARD_STOP set"}

    @app.post("/api/actions/clear-stop")
    def clear_stop() -> dict:
        cmd_clear_stop(argparse.Namespace())
        return {"ok": True, "message": "HARD_STOP cleared"}

    @app.get("/api/logs/{name}")
    def tail_log(name: str, lines: int = 80) -> dict:
        allowed = {"bridge", "camera", "camera_left", "viewer", "model", "policy"}
        if name not in allowed:
            raise HTTPException(status_code=400, detail=f"name must be one of {sorted(allowed)}")
        path = ROOT / "logs" / f"{name}.log"
        if not path.is_file():
            return {"path": str(path), "lines": [], "exists": False}
        text = path.read_text(errors="replace").splitlines()
        tail = text[-lines:] if lines > 0 else text
        return {"path": str(path), "lines": tail, "exists": True}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    return app


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>YAM operator</title>
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e6edf3;
      --muted: #8b9cb3;
      --ok: #3fb950;
      --warn: #d29922;
      --bad: #f85149;
      --accent: #58a6ff;
      --border: #30363d;
    }
    * { box-sizing: border-box; }
    body {
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 1.25rem;
      line-height: 1.5;
    }
    h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 1rem; }
    .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.1rem;
    }
    .card h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 0.75rem; }
    .pill { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .pill.ok { background: rgba(63, 185, 80, 0.15); color: var(--ok); }
    .pill.bad { background: rgba(248, 81, 73, 0.15); color: var(--bad); }
    .pill.warn { background: rgba(210, 153, 34, 0.15); color: var(--warn); }
    pre {
      margin: 0;
      font-size: 0.72rem;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 220px;
      overflow: auto;
      color: var(--muted);
    }
    .actions { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
    button {
      background: var(--accent);
      color: #0d1117;
      border: none;
      padding: 0.45rem 0.85rem;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.08); }
    button.secondary { background: #30363d; color: var(--text); }
    button.danger { background: var(--bad); color: #fff; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .hint { font-size: 0.8rem; color: var(--muted); margin-top: 0.5rem; }
    .refresh { float: right; font-size: 0.8rem; color: var(--muted); }
  </style>
</head>
<body>
  <h1>YAM operator <span class="refresh" id="updated"></span></h1>
  <div class="grid">
    <div class="card">
      <h2>Stack</h2>
      <div id="flags"></div>
      <div class="actions">
        <button type="button" id="btn-refresh">Refresh</button>
        <button type="button" class="secondary" id="btn-hard-stop">Hard stop</button>
        <button type="button" class="secondary" id="btn-clear-stop">Clear stop</button>
      </div>
    </div>
    <div class="card">
      <h2>CAN bridge</h2>
      <p class="hint">Restarts the tracked SLCAN/Rust bridge (<code>yamctl stop bridge</code> then <code>yamctl bridge</code>). Set <code>YAM_SERIAL_PORT</code> if needed.</p>
      <div class="actions">
        <button type="button" id="btn-restart-bridge">Restart CAN bridge</button>
        <button type="button" class="secondary" id="btn-reset-socketcan">Reset SocketCAN (Linux)</button>
      </div>
    </div>
    <div class="card">
      <h2>Model</h2>
      <p class="hint">Stops hybrid / Modal bridge / policy server, sets hard-stop, then clears the flag. Start a new run from the terminal when ready.</p>
      <div class="actions">
        <button type="button" class="danger" id="btn-restart-model">Stop model &amp; policy</button>
      </div>
    </div>
  </div>
  <div class="card" style="margin-top: 1rem;">
    <h2>Processes <span class="hint" style="font-weight: normal;">(yamctl status + can-check)</span></h2>
    <pre id="raw"></pre>
  </div>
  <div class="grid" style="margin-top: 1rem;">
    <div class="card">
      <h2>bridge.log (tail)</h2>
      <pre id="log-bridge"></pre>
    </div>
    <div class="card">
      <h2>model.log (tail)</h2>
      <pre id="log-model"></pre>
    </div>
  </div>
<script>
async function getJSON(url, opt) {
  const r = await fetch(url, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = data.detail;
    const msg = typeof d === 'string' ? d : (Array.isArray(d) ? d.map(x => x.msg || x).join('; ') : JSON.stringify(d));
    throw new Error(msg || r.statusText || String(r.status));
  }
  return data;
}
function setUpdated() {
  document.getElementById('updated').textContent = 'updated ' + new Date().toLocaleTimeString();
}
async function loadStatus() {
  const s = await getJSON('/api/status');
  const hs = s.hard_stop;
  const can0 = s.can_check && s.can_check.can0 && s.can_check.can0.connect_ok;
  const can1 = s.can_check && s.can_check.can1 && s.can_check.can1.connect_ok;
  let html = '';
  html += '<p><span class="pill ' + (hs ? 'bad' : 'ok') + '">HARD_STOP ' + (hs ? 'set' : 'clear') + '</span></p>';
  html += '<p>/tmp/can0.sock — <span class="pill ' + (can0 ? 'ok' : 'warn') + '">' + (can0 ? 'connect OK' : 'down / stale') + '</span></p>';
  html += '<p>/tmp/can1.sock — <span class="pill ' + (can1 ? 'ok' : 'warn') + '">' + (can1 ? 'connect OK' : 'down / stale') + '</span></p>';
  if (s.pid_files) {
    for (const [k, v] of Object.entries(s.pid_files)) {
      const run = v.running ? 'ok' : 'bad';
      html += '<p style="font-size:0.85rem">' + k + ': pid ' + (v.pid ?? '—') + ' <span class="pill ' + run + '">' + (v.running ? 'running' : 'stopped') + '</span></p>';
    }
  }
  document.getElementById('flags').innerHTML = html;
  document.getElementById('raw').textContent = JSON.stringify(s, null, 2);
  setUpdated();
}
async function loadLogs() {
  try {
    const b = await getJSON('/api/logs/bridge?lines=60');
    document.getElementById('log-bridge').textContent = b.exists ? b.lines.join(String.fromCharCode(10)) : '(no file)';
  } catch (e) { document.getElementById('log-bridge').textContent = String(e); }
  try {
    const m = await getJSON('/api/logs/model?lines=60');
    document.getElementById('log-model').textContent = m.exists ? m.lines.join(String.fromCharCode(10)) : '(no file)';
  } catch (e) { document.getElementById('log-model').textContent = String(e); }
}
document.getElementById('btn-refresh').onclick = () => { loadStatus(); loadLogs(); };
document.getElementById('btn-hard-stop').onclick = async () => {
  await getJSON('/api/actions/hard-stop', { method: 'POST' });
  await loadStatus();
};
document.getElementById('btn-clear-stop').onclick = async () => {
  await getJSON('/api/actions/clear-stop', { method: 'POST' });
  await loadStatus();
};
document.getElementById('btn-restart-bridge').onclick = async () => {
  const btn = document.getElementById('btn-restart-bridge');
  btn.disabled = true;
  try {
    await getJSON('/api/actions/restart-bridge', { method: 'POST' });
    await loadStatus();
    await loadLogs();
  } finally { btn.disabled = false; }
};
document.getElementById('btn-reset-socketcan').onclick = async () => {
  try {
    await getJSON('/api/actions/reset-socketcan', { method: 'POST' });
    alert('SocketCAN reset finished (see response in Network tab if needed).');
  } catch (e) {
    alert(e.message || e);
  }
  await loadStatus();
};
document.getElementById('btn-restart-model').onclick = async () => {
  if (!confirm('Stop model and policy processes and clear HARD_STOP?')) return;
  const btn = document.getElementById('btn-restart-model');
  btn.disabled = true;
  try {
    await getJSON('/api/actions/restart-model', { method: 'POST' });
    await loadStatus();
    await loadLogs();
  } finally { btn.disabled = false; }
};
loadStatus();
loadLogs();
setInterval(loadStatus, 8000);
setInterval(loadLogs, 12000);
</script>
</body>
</html>
"""

app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("YAM_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("YAM_DASHBOARD_PORT", "8890"))

    print(f"YAM dashboard: repo root {ROOT}", flush=True)
    print(f"Listening on http://{host}:{port}/  (change with YAM_DASHBOARD_HOST / YAM_DASHBOARD_PORT)", flush=True)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        in_use = exc.errno == errno.EADDRINUSE or "address already in use" in str(exc).lower()
        if in_use:
            print(
                f"\nPort {port} is already in use (another yam-dashboard or app).\n"
                f"  Try: YAM_DASHBOARD_PORT=8891 uv run yam-dashboard\n"
                f"  Or stop the process holding the port.\n",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1) from exc
        raise
    finally:
        probe.close()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # When run as `python _yamtest/dashboard.py` from repo root, avoid broken import path.
    if __package__ is None or __package__ == "":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
