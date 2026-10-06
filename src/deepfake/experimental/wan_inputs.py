"""Validated local Wan replacement controls; import as a ComfyUI custom node.

Controls must come from real pose/face/segmentation preprocessing. This does not
infer unseen clothing, download models, or enable a live head/body mode.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np


def load_controls(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in ("reference", "background", "pose", "face", "mask")}
    ref = data["reference"]
    if ref.ndim != 3 or ref.shape[2] != 3 or ref.dtype != np.uint8:
        raise ValueError("reference must be an RGB uint8 HWC image")
    frames = data["background"]
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or not len(frames):
        raise ValueError("background must be a nonempty RGB uint8 NHWC batch")
    for name in ("pose", "face"):
        arr = data[name]
        if arr.ndim != 4 or arr.shape[0] != len(frames) or arr.shape[-1] != 3 or arr.dtype != np.uint8:
            raise ValueError(f"{name} must be an RGB uint8 NHWC batch matching frame count")
    mask = data["mask"]
    if mask.shape != frames.shape[:3] or not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError("mask must be finite NHW person coverage in [0,1], matching background")
    return data


def prepare_controls(source: Path, output: Path, width: int = 512, height: int = 288) -> None:
    """Match author letterboxing and native template's blockified black background."""
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("output dimensions must be positive multiples of 32")
    if source.resolve() == output.resolve():
        raise ValueError("preserve the original controls in a separate file")
    data = load_controls(source)
    ref = data["reference"]
    scale = min(width / ref.shape[1], height / ref.shape[0])
    w, h = max(1, int(ref.shape[1] * scale)), max(1, int(ref.shape[0] * scale))
    padded = np.zeros((height, width, 3), np.uint8)
    padded[(height - h) // 2 : (height - h) // 2 + h, (width - w) // 2 : (width - w) // 2 + w] = cv2.resize(
        ref, (w, h), interpolation=cv2.INTER_AREA
    )
    masks = []
    for mask in data["mask"]:
        expanded = cv2.dilate((mask > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
        blocks = cv2.resize(expanded.astype(np.float32), (width // 32, height // 32), interpolation=cv2.INTER_AREA) > 0
        masks.append(
            cv2.resize(blocks.astype(np.float32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        )
    data["reference"] = padded
    data["mask"] = np.stack(masks)
    data["background"] = (data["background"] * (1 - data["mask"][..., None])).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)


class LocalReplacementInputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input_file": ("STRING", {"default": "inputs.npz"})}}

    @classmethod
    def IS_CHANGED(cls, input_file):
        with Path(input_file).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("reference", "background", "pose", "face", "mask")
    FUNCTION = "load"
    CATEGORY = "deepfake/experimental"

    def load(self, input_file):
        import torch

        data = load_controls(Path(input_file))
        images = [torch.from_numpy(data[name]).float() / 255 for name in ("reference", "background", "pose", "face")]
        images[0] = images[0].unsqueeze(0)
        return tuple(images + [torch.from_numpy(data["mask"]).float()])


NODE_CLASS_MAPPINGS = {"LocalReplacementInputs": LocalReplacementInputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="raw RGB controls NPZ, including original background and person mask")
    parser.add_argument("output", type=Path, help="prepared NPZ; keep separate from raw controls")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    args = parser.parse_args()
    prepare_controls(args.source, args.output, args.width, args.height)


if __name__ == "__main__":
    main()
