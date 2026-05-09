#!/usr/bin/env python3
"""MolmoAct2-BimanualYAM sample inference preflight.

This does not command hardware. It runs the Hugging Face sample from
allenai/MolmoAct2-BimanualYAM and prints the predicted action shape.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--run", action="store_true", help="Actually load model weights and run inference.")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
        from PIL import Image
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        print("Install with: pip install pillow numpy huggingface_hub", file=sys.stderr)
        return 2

    if args.run:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            print(f"Missing dependency: {exc}", file=sys.stderr)
            print("Install on a CUDA machine with: pip install torch transformers pillow numpy huggingface_hub", file=sys.stderr)
            return 2

        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Not loading the 5B model on this machine.", file=sys.stderr)
            return 2
        if args.device == "mps" and not torch.backends.mps.is_available():
            print("MPS is not available.", file=sys.stderr)
            return 2
        if args.device != "cuda":
            print("Refusing non-CUDA run by default. The model card documents CUDA inference and large memory needs.", file=sys.stderr)
            print("Use this script on a CUDA GPU box, then adapt outputs offline before touching hardware.", file=sys.stderr)
            return 2

    top_rgb = Image.open(hf_hub_download(REPO_ID, "assets/sample_top_rgb.png")).convert("RGB")
    left_rgb = Image.open(hf_hub_download(REPO_ID, "assets/sample_left_rgb.png")).convert("RGB")
    right_rgb = Image.open(hf_hub_download(REPO_ID, "assets/sample_right_rgb.png")).convert("RGB")
    task = "Place cups and plate in dishwasher rack, dispose of food waste, and organize remaining items."
    robot_state = np.array(
        [
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
        dtype=np.float32,
    )

    print(f"Loaded sample assets. State shape: {robot_state.shape}; images: {[im.size for im in [top_rgb, left_rgb, right_rgb]]}")
    if not args.run:
        print("Preflight complete. Pass --run on a CUDA machine to load model weights and predict actions.")
        return 0

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    processor = AutoProcessor.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO_ID,
        trust_remote_code=True,
        dtype=dtype,
    ).to(args.device).eval()

    with torch.inference_mode(), torch.autocast(args.device, dtype=dtype, enabled=dtype == torch.bfloat16):
        out = model.predict_action(
            processor=processor,
            images=[top_rgb, left_rgb, right_rgb],
            task=task,
            state=robot_state,
            norm_tag=NORM_TAG,
            action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=args.num_steps,
            normalize_language=True,
            enable_cuda_graph=False,
        )

    actions = np.asarray(out.actions)
    print(f"Predicted actions shape: {actions.shape}")
    print(actions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
