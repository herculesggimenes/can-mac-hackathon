#!/usr/bin/env python3
"""Validate and export YAM teleop recordings for LeRobot training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDINGS_DIR = ROOT / "recordings"


@dataclass
class Episode:
    path: Path
    metadata: dict[str, Any]
    frames: list[dict[str, Any]]


def _episode_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = sorted(root.glob("*/episode.jsonl"))
    files.extend(sorted(path for path in root.glob("*.jsonl") if path.name != "trace.jsonl"))
    return files


def _load_episode(path: Path) -> Episode:
    metadata: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            record_type = record.get("type")
            if record_type == "metadata":
                metadata = record
            elif record_type == "frame":
                frames.append(record)
    return Episode(path=path, metadata=metadata, frames=frames)


def _image_path(episode: Episode, frame: dict[str, Any]) -> Path | None:
    value = frame.get("observation.images.front")
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return episode.path.parent / path


def _validate_episode(episode: Episode) -> list[str]:
    errors: list[str] = []
    if not episode.metadata:
        errors.append("missing metadata record")
    if not episode.frames:
        errors.append("contains no frame records")
        return errors
    for index, frame in enumerate(episode.frames):
        state = frame.get("observation.state")
        action = frame.get("action")
        if not isinstance(state, list) or len(state) != 7:
            errors.append(f"frame {index}: observation.state must have 7 values")
        if not isinstance(action, list) or len(action) != 7:
            errors.append(f"frame {index}: action must have 7 values")
        image_path = _image_path(episode, frame)
        if image_path is None:
            errors.append(f"frame {index}: missing observation.images.front")
        elif not image_path.exists():
            errors.append(f"frame {index}: missing image {image_path}")
    return errors


def _load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _iter_valid_episodes(root: Path) -> list[Episode]:
    episodes = [_load_episode(path) for path in _episode_files(root)]
    return [episode for episode in episodes if not _validate_episode(episode)]


def cmd_summary(args: argparse.Namespace) -> int:
    root = Path(args.recordings_dir).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    episodes = [_load_episode(path) for path in _episode_files(root)]
    payload: dict[str, Any] = {
        "recordings_dir": str(root),
        "episodes": len(episodes),
        "frames": sum(len(episode.frames) for episode in episodes),
        "valid_episodes": 0,
        "items": [],
    }
    for episode in episodes:
        errors = _validate_episode(episode)
        payload["valid_episodes"] += 0 if errors else 1
        task = episode.metadata.get("task") or (episode.frames[0].get("task") if episode.frames else None)
        payload["items"].append(
            {
                "path": str(episode.path),
                "task": task,
                "frames": len(episode.frames),
                "valid": not errors,
                "errors": errors[:10],
            }
        )
    print(json.dumps(payload, indent=2))
    return 0 if payload["valid_episodes"] == payload["episodes"] else 1


def cmd_export(args: argparse.Namespace) -> int:
    try:
        from lerobot.datasets import LeRobotDataset
    except ModuleNotFoundError:
        print(
            "lerobot is not installed. Install it in this environment first, for example:\n"
            "  uv pip install 'lerobot[pi0,smolvla]'\n"
            "Then rerun this export command.",
            file=sys.stderr,
        )
        return 2

    root = Path(args.recordings_dir).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    episodes = _iter_valid_episodes(root)
    if not episodes:
        print(f"no valid episodes found under {root}", file=sys.stderr)
        return 1

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=output_root,
        robot_type="yam_single_arm",
        features=features,
        use_videos=True,
    )

    for episode in episodes:
        task = episode.metadata.get("task") or args.task
        for frame in episode.frames:
            image_path = _image_path(episode, frame)
            assert image_path is not None
            dataset.add_frame(
                {
                    "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
                    "action": np.asarray(frame["action"], dtype=np.float32),
                    "observation.images.front": _load_image_rgb(image_path),
                    "task": frame.get("task") or task,
                }
            )
        try:
            dataset.save_episode(task=task)
        except TypeError:
            dataset.save_episode()

    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "root": str(output_root),
                "episodes": len(episodes),
                "frames": sum(len(episode.frames) for episode in episodes),
            },
            indent=2,
        )
    )
    return 0


def cmd_train_act(args: argparse.Namespace) -> int:
    command = [
        "lerobot-train",
        f"--dataset.repo_id={args.repo_id}",
        "--policy.type=act",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]
    if args.dataset_root:
        command.append(f"--dataset.root={args.dataset_root}")
    if args.steps is not None:
        command.append(f"--steps={args.steps}")
    if args.batch_size is not None:
        command.append(f"--batch_size={args.batch_size}")
    if args.chunk_size is not None:
        command.append(f"--policy.chunk_size={args.chunk_size}")
    if args.n_action_steps is not None:
        command.append(f"--policy.n_action_steps={args.n_action_steps}")
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.call(command, cwd=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YAM recording validation and LeRobot export helpers.")
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summary", help="Validate raw teleop recordings.")
    summary.add_argument("--recordings-dir", default=str(DEFAULT_RECORDINGS_DIR))
    summary.set_defaults(func=cmd_summary)

    export = sub.add_parser("export", help="Export valid recordings to a LeRobotDataset.")
    export.add_argument("--recordings-dir", default=str(DEFAULT_RECORDINGS_DIR))
    export.add_argument("--output-root", default="lerobot-data")
    export.add_argument("--repo-id", default="local/yam-bread-toaster")
    export.add_argument("--task", default="put bread in toaster")
    export.add_argument("--fps", type=int, default=10)
    export.add_argument("--image-width", type=int, default=640)
    export.add_argument("--image-height", type=int, default=360)
    export.set_defaults(func=cmd_export)

    train = sub.add_parser("train-act", help="Run or print the ACT training command.")
    train.add_argument("--repo-id", default="local/yam-bread-toaster")
    train.add_argument("--dataset-root", default="lerobot-data")
    train.add_argument("--output-dir", default="outputs/train/yam-bread-toaster-act")
    train.add_argument("--job-name", default="yam_bread_toaster_act")
    train.add_argument("--steps", type=int, default=20000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--chunk-size", type=int, default=50)
    train.add_argument("--n-action-steps", type=int, default=50)
    train.add_argument("--device", default="mps")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=cmd_train_act)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
