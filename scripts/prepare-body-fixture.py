"""Encode a tiny synthetic driving video from the generated full-body fixture.

This is a translated still, deliberately not a motion-quality benchmark.
The original generated PNGs remain unmodified.
"""
from pathlib import Path

import cv2
import numpy as np

source = cv2.imread("docs/demos/fullbody-source.png")
assert source is not None
image = cv2.resize(source, (384, 576))
canvas = np.full((672, 384, 3), 220, np.uint8)
canvas[48:624] = image
path = Path("/tmp/deepfake-integration-artifacts/body-input.mp4")
path.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16, (384, 672))
assert writer.isOpened()
for i in range(21):
    shift = int(3 * np.sin(i / 20 * np.pi * 2))
    frame = cv2.warpAffine(canvas, np.float32([[1, 0, shift], [0, 1, 0]]), (384, 672), borderMode=cv2.BORDER_REPLICATE)
    writer.write(frame)
writer.release()
print(path)
