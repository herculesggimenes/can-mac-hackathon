#!/usr/bin/env python3
"""Make the bimanual YAM robot do a bounded samba-like dance over WebSocket."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]
STOP_FILE = ROOT / "HARD_STOP"
STOP_REQUESTED = False


def _request_stop(_signum=None, _frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


def _rpc(ws, method: str, params: dict[str, Any] | None = None, *, request_id: str | None = None) -> Any:
    ws.send(json.dumps({"id": request_id or method, "method": method, "params": params or {}}))
    response = json.loads(ws.recv())
    if not response.get("ok"):
        raise RuntimeError(f"{method} failed: {response.get('error', 'rpc_failed')}")
    return response.get("result")


def _require_connected(status: dict[str, Any]) -> None:
    arms = status.get("arms", {})
    disconnected = [arm_id for arm_id, arm_status in arms.items() if not arm_status.get("connected")]
    if disconnected:
        raise RuntimeError(f"robot arm(s) not connected: {', '.join(disconnected)}")


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _ramp_envelope(elapsed: float, duration: float, ramp_s: float) -> float:
    if ramp_s <= 0:
        return 1.0
    return min(_smoothstep(elapsed / ramp_s), _smoothstep((duration - elapsed) / ramp_s))


def _samba_offsets(t: float, *, amplitude: float, wrist_amplitude: float, max_delta: float, tempo: float) -> np.ndarray:
    """Return a 14D relative offset around the observed baseline.

    The pattern uses opposite phase between arms, with slower shoulder/elbow
    sway and light wrist accents. Joint 6 on each arm is the gripper and is
    intentionally left unchanged.
    """
    beat = 2.0 * math.pi * tempo * t
    double = 2.0 * beat
    half = 0.5 * beat

    sway = math.sin(beat)
    bounce = math.sin(double + math.pi / 4.0)
    roll = math.sin(half + math.pi / 3.0)
    accent = math.sin(double + math.pi / 2.0)
    counter = math.sin(beat + math.pi)

    left = np.array(
        [
            0.95 * amplitude * sway,
            0.55 * amplitude * bounce,
            -0.80 * amplitude * counter,
            0.45 * amplitude * roll,
            0.35 * wrist_amplitude * accent,
            0.55 * wrist_amplitude * math.sin(beat + math.pi / 5.0),
            0.0,
        ],
        dtype=float,
    )
    right = np.array(
        [
            -0.95 * amplitude * sway,
            0.55 * amplitude * math.sin(double + 3.0 * math.pi / 4.0),
            0.80 * amplitude * counter,
            -0.45 * amplitude * roll,
            -0.35 * wrist_amplitude * accent,
            0.55 * wrist_amplitude * math.sin(beat + 4.0 * math.pi / 5.0),
            0.0,
        ],
        dtype=float,
    )
    return np.clip(np.concatenate([left, right]), -max_delta, max_delta)


def _interpolate_to(ws, start: np.ndarray, target: np.ndarray, *, steps: int, hz: float) -> None:
    steps = max(1, steps)
    for index in range(1, steps + 1):
        if STOP_REQUESTED or STOP_FILE.exists():
            raise KeyboardInterrupt
        alpha = index / steps
        command = start + alpha * (target - start)
        _rpc(ws, "command_joint_pos", {"joint_pos": command.tolist()}, request_id=f"restore-{index}")
        time.sleep(1.0 / max(hz, 1.0))


def _limit_slew(current: np.ndarray, target: np.ndarray, max_step_delta: float) -> np.ndarray:
    if max_step_delta <= 0:
        return target
    return current + np.clip(target - current, -max_step_delta, max_step_delta)


def run(args: argparse.Namespace) -> int:
    if STOP_FILE.exists() and not args.ignore_hard_stop:
        print(json.dumps({"execution_blocked": "hard_stop_exists", "path": str(STOP_FILE)}), flush=True)
        return 2

    with connect(args.url, max_size=16 * 1024 * 1024) as ws:
        hello = json.loads(ws.recv())
        status = _rpc(ws, "get_status")
        _require_connected(status)
        baseline = np.asarray(_rpc(ws, "get_joint_pos"), dtype=float)
        if baseline.shape != (14,):
            raise RuntimeError(f"expected 14D joint vector, got shape {baseline.shape}")

        print(
            json.dumps(
                {
                    "dance": "samba",
                    "url": args.url,
                    "hello": hello,
                    "baseline_joint_pos": baseline.tolist(),
                    "duration_s": args.duration,
                    "loops": args.loops,
                    "total_duration_s": args.duration * args.loops,
                    "tempo": args.tempo,
                    "hz": args.hz,
                    "amplitude": args.amplitude,
                    "wrist_amplitude": args.wrist_amplitude,
                    "max_delta": args.max_delta,
                    "max_step_delta": args.max_step_delta,
                    "dry_run": args.dry_run,
                }
            ),
            flush=True,
        )

        if args.dry_run:
            preview = [
                (
                    baseline
                    + _samba_offsets(
                        i / args.hz,
                        amplitude=args.amplitude,
                        wrist_amplitude=args.wrist_amplitude,
                        max_delta=args.max_delta,
                        tempo=args.tempo,
                    )
                ).tolist()
                for i in range(min(10, max(1, int(args.duration * args.hz))))
            ]
            print(json.dumps({"preview_targets": preview}), flush=True)
            return 0

        total_duration = args.duration * args.loops
        deadline = time.monotonic() + total_duration
        start = time.monotonic()
        step = 0
        last_status_check = start
        current = baseline.copy()
        while time.monotonic() < deadline:
            if STOP_REQUESTED or STOP_FILE.exists():
                print(json.dumps({"stopped": True, "step": step}), flush=True)
                break

            now = time.monotonic()
            elapsed = now - start
            if now - last_status_check >= args.status_interval_s:
                _require_connected(_rpc(ws, "get_status"))
                last_status_check = now

            phrase_t = elapsed % args.duration
            envelope = _ramp_envelope(elapsed, total_duration, args.ramp_s)
            offset = envelope * _samba_offsets(
                phrase_t,
                amplitude=args.amplitude,
                wrist_amplitude=args.wrist_amplitude,
                max_delta=args.max_delta,
                tempo=args.tempo,
            )
            target = _limit_slew(current, baseline + offset, args.max_step_delta)
            _rpc(ws, "command_joint_pos", {"joint_pos": target.tolist()}, request_id=f"dance-{step}")
            current = target
            if args.print_steps:
                print(json.dumps({"step": step, "target_joint_pos": target.tolist()}), flush=True)
            step += 1
            time.sleep(1.0 / max(args.hz, 1.0))

        final = np.asarray(_rpc(ws, "get_joint_pos"), dtype=float)
        if args.restore:
            _interpolate_to(ws, final if final.shape == (14,) else current, baseline, steps=args.restore_steps, hz=args.hz)
            final = np.asarray(_rpc(ws, "get_joint_pos"), dtype=float)
        print(json.dumps({"dance_done": True, "steps": step, "final_joint_pos": final.tolist()}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small samba-like bimanual YAM dance via the control WebSocket.")
    parser.add_argument("--url", default="ws://127.0.0.1:8780/control")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per samba phrase loop.")
    parser.add_argument("--loops", type=int, default=1, help="Number of bounded phrase loops to run.")
    parser.add_argument("--hz", type=float, default=15.0)
    parser.add_argument("--tempo", type=float, default=0.55, help="Samba phrase tempo in cycles per second.")
    parser.add_argument("--amplitude", type=float, default=0.065, help="Base joint sway amplitude in radians.")
    parser.add_argument("--wrist-amplitude", type=float, default=0.018, help="Wrist accent amplitude in radians.")
    parser.add_argument("--max-delta", type=float, default=0.09, help="Absolute per-joint offset cap from baseline.")
    parser.add_argument("--max-step-delta", type=float, default=0.025, help="Per-command joint delta cap.")
    parser.add_argument("--ramp-s", type=float, default=2.0, help="Fade in/out time to avoid abrupt motion.")
    parser.add_argument("--status-interval-s", type=float, default=1.0)
    parser.add_argument("--restore-steps", type=int, default=30)
    parser.add_argument("--restore", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-steps", action="store_true")
    parser.add_argument("--allow-large-motion", action="store_true")
    parser.add_argument("--ignore-hard-stop", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.hz <= 0:
        parser.error("--hz must be positive")
    if args.loops <= 0:
        parser.error("--loops must be positive")
    if args.loops > 6:
        parser.error("--loops must stay <= 6 for bounded robot motion")
    if args.tempo <= 0:
        parser.error("--tempo must be positive")
    if args.max_delta > 0.10 and not args.allow_large_motion:
        parser.error("--max-delta above 0.10 requires --allow-large-motion")
    if args.max_delta > 1.0:
        parser.error("--max-delta must stay <= 1.0")
    if args.max_step_delta > 0.05:
        parser.error("--max-step-delta must stay <= 0.05")
    if args.amplitude > args.max_delta or args.wrist_amplitude > args.max_delta:
        parser.error("--amplitude and --wrist-amplitude must be <= --max-delta")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
