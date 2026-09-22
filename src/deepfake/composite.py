"""Paste swapped face back into frame with optional blend / color match."""

from __future__ import annotations

import numpy as np

from .detect import FaceBox


def _feather_mask(h: int, w: int, feather: int) -> np.ndarray:
    import cv2

    mask = np.zeros((h, w), dtype=np.float32)
    inset = max(1, feather)
    mask[inset : h - inset, inset : w - inset] = 1.0
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def color_match_lab(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match mean/std of src to ref in LAB (simple)."""
    import cv2

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src_lab.copy()
    for c in range(3):
        s = src_lab[:, :, c]
        r = ref_lab[:, :, c]
        s_std = s.std() + 1e-6
        r_std = r.std() + 1e-6
        out[:, :, c] = (s - s.mean()) * (r_std / s_std) + r.mean()
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def paste_face(
    frame_bgr: np.ndarray,
    face_bgr: np.ndarray,
    box: FaceBox,
    *,
    feather: int = 16,
    color_match: bool = True,
    temporal_prev: np.ndarray | None = None,
    temporal_smooth: float = 0.0,
) -> np.ndarray:
    import cv2

    x, y, w, h = box.x, box.y, box.w, box.h
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = min(fw, x + w), min(fh, y + h)
    x, y = max(0, x), max(0, y)
    w, h = x1 - x, y1 - y
    if w <= 1 or h <= 1:
        return frame_bgr

    resized = cv2.resize(face_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    roi = frame_bgr[y:y1, x:x1]
    if color_match and roi.size and resized.shape == roi.shape:
        resized = color_match_lab(resized, roi)

    if temporal_prev is not None and temporal_smooth > 0 and temporal_prev.shape == resized.shape:
        a = float(np.clip(temporal_smooth, 0.0, 1.0))
        resized = cv2.addWeighted(resized, 1.0 - a, temporal_prev, a, 0)

    mask = _feather_mask(h, w, feather)[:, :, None]
    blended = (resized.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)).astype(
        np.uint8
    )
    out = frame_bgr.copy()
    out[y:y1, x:x1] = blended
    return out
