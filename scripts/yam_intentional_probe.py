#!/usr/bin/env python3
"""Run intentional YAM probe motions and collect snapshots.

This is for local scene understanding: move a hand through known Cartesian
offsets, capture perception at each waypoint, then summarize how depth and
joint state changed. It is deliberately separate from task execution so Codex
can first learn whether "up", "down", "left", or "forward" improves the view.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_URL = "wss://2d37-12-125-194-54.ngrok-free.app/control"
DEFAULT_CAMERA_URL = "ws://127.0.0.1:8770/cameras"


PROBES: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
    "up-down": [
        ("baseline", (0.0, 0.0, 0.0)),
        ("up", (0.0, 0.0, 0.05)),
        ("down", (0.0, 0.0, -0.05)),
        ("return", (0.0, 0.0, 0.0)),
    ],
    "view-height": [
        ("baseline", (0.0, 0.0, 0.0)),
        ("up-small", (0.0, 0.0, 0.04)),
        ("up-large", (0.0, 0.0, 0.08)),
        ("back-to-small", (0.0, 0.0, -0.04)),
    ],
    "approach-retreat": [
        ("baseline", (0.0, 0.0, 0.0)),
        ("retreat", (0.0, 0.0, 0.04)),
        ("approach", (0.0, 0.0, -0.04)),
        ("return", (0.0, 0.0, 0.0)),
    ],
    "left-right": [
        ("baseline", (0.0, 0.0, 0.0)),
        ("left", (-0.04, 0.0, 0.0)),
        ("right", (0.04, 0.0, 0.0)),
    ],
    "forward-back": [
        ("baseline", (0.0, 0.0, 0.0)),
        ("forward", (0.0, 0.04, 0.0)),
        ("back", (0.0, -0.04, 0.0)),
    ],
}


def _run_json(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command did not return JSON: {' '.join(cmd)}\n{proc.stdout}") from exc


def _snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return _run_json(
        [
            "uv",
            "run",
            "python",
            "scripts/yam_codex_snapshot.py",
            "--robot-url",
            args.robot_url,
            "--camera-url",
            args.camera_url,
            "--depth-grid",
            str(args.depth_grid),
        ],
        cwd=ROOT,
    )


def _move(args: argparse.Namespace, delta: tuple[float, float, float]) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/yam_cartesian_control.py",
        "--control-url",
        args.robot_url,
        "--arm",
        args.arm,
        "--frame",
        args.frame,
        f"--delta={','.join(str(value) for value in delta)}",
        "--ik-preset",
        args.ik_preset,
        "--max-segment-m",
        str(args.max_segment_m),
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--steps",
        str(args.steps),
        "--hz",
        str(args.hz),
    ]
    if args.allow_nonconverged_segments:
        cmd.append("--allow-nonconverged-segments")
    if args.execute:
        cmd.append("--execute")
    return _run_json(cmd, cwd=ROOT)


def _depth_centers(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for camera in snapshot.get("perception", {}).get("cameras", []):
        center = camera.get("center_depth") if isinstance(camera, dict) else None
        if not center:
            continue
        entries.append(
            {
                "camera_id": camera.get("camera_id"),
                "depth_m": center.get("depth_m"),
                "pixel": center.get("pixel"),
                "point_camera_m": center.get("point_camera_m"),
            }
        )
    return entries


def _joint_pos(snapshot: dict[str, Any]) -> list[float] | None:
    return snapshot.get("robot", {}).get("joint_summary", {}).get("joint_pos")


def _rpc(ws, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    ws.send(json.dumps({"id": method, "method": method, "params": params or {}}))
    response = json.loads(ws.recv())
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "rpc_failed"))
    return response


def _restore_joint_pos(args: argparse.Namespace, target: list[float]) -> dict[str, Any]:
    target_q = np.asarray(target, dtype=float)
    if target_q.shape != (14,):
        raise RuntimeError(f"baseline restore expected 14D joints, got {target_q.shape}")
    with connect(args.robot_url, max_size=16 * 1024 * 1024) as ws:
        ws.recv()
        current = np.asarray(_rpc(ws, "get_joint_pos")["result"], dtype=float)
        for index in range(1, max(1, args.restore_steps) + 1):
            alpha = index / max(1, args.restore_steps)
            command = current + (target_q - current) * alpha
            _rpc(ws, "command_joint_pos", {"joint_pos": command.tolist()})
            time.sleep(1.0 / max(float(args.hz), 1.0))
        final = _rpc(ws, "get_joint_pos")["result"]
    return {"target_joint_pos": target, "final_joint_pos": final, "steps": args.restore_steps}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run intentional YAM probe motions with snapshots.")
    parser.add_argument("--robot-url", default=DEFAULT_ROBOT_URL)
    parser.add_argument("--camera-url", default=DEFAULT_CAMERA_URL)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--frame", choices=["world", "local"], default="world")
    parser.add_argument("--probe", choices=sorted(PROBES), default="up-down")
    parser.add_argument("--ik-preset", choices=["strict", "live", "loose"], default="loose")
    parser.add_argument("--max-segment-m", type=float, default=0.04)
    parser.add_argument("--max-joint-delta", type=float, default=0.12)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument("--depth-grid", type=int, default=3)
    parser.add_argument("--settle-s", type=float, default=0.25)
    parser.add_argument("--allow-nonconverged-segments", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually move the robot. Omit for dry-run plans only.")
    parser.add_argument(
        "--restore-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After an executed probe, interpolate back to the exact initial 14D joint vector.",
    )
    parser.add_argument("--restore-steps", type=int, default=32)
    parser.add_argument("--output-dir", default="logs/yam-intentional-probes")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    report: dict[str, Any] = {
        "schema": "yam_intentional_probe.v1",
        "probe": args.probe,
        "arm": args.arm,
        "frame": args.frame,
        "execute": bool(args.execute),
        "output_dir": str(output_dir),
        "steps": [],
    }

    baseline_joint_pos: list[float] | None = None
    for label, delta in PROBES[args.probe]:
        step: dict[str, Any] = {"label": label, "delta_m": list(delta)}
        if any(delta):
            step["move"] = _move(args, delta)
            time.sleep(max(0.0, args.settle_s))
        snapshot = _snapshot(args)
        step["snapshot_path"] = snapshot.get("snapshot_path")
        step["contact_sheet_path"] = str(Path(snapshot["snapshot_dir"]) / snapshot.get("perception", {}).get("contact_sheet_path", ""))
        step["depth_centers"] = _depth_centers(snapshot)
        step["joint_pos"] = _joint_pos(snapshot)
        if baseline_joint_pos is None and step["joint_pos"] is not None:
            baseline_joint_pos = step["joint_pos"]
        report["steps"].append(step)

    if args.execute and args.restore_baseline and baseline_joint_pos is not None:
        restore_step: dict[str, Any] = {
            "label": "restore-baseline-joints",
            "delta_m": None,
            "restore": _restore_joint_pos(args, baseline_joint_pos),
        }
        time.sleep(max(0.0, args.settle_s))
        snapshot = _snapshot(args)
        restore_step["snapshot_path"] = snapshot.get("snapshot_path")
        restore_step["contact_sheet_path"] = str(
            Path(snapshot["snapshot_dir"]) / snapshot.get("perception", {}).get("contact_sheet_path", "")
        )
        restore_step["depth_centers"] = _depth_centers(snapshot)
        restore_step["joint_pos"] = _joint_pos(snapshot)
        report["steps"].append(restore_step)

    report_path = output_dir / "probe_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report_path": str(report_path), **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
