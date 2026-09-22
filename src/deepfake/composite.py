"""Paste swapped face back into frame with oval soft mask / color match."""

from __future__ import annotations

import numpy as np

from .detect import FaceBox


def _oval_soft_mask(h: int, w: int, feather: int) -> np.ndarray:
    """Face-shaped soft mask: filled ellipse with Gaussian feather at edges."""
    import cv2

    mask = np.zeros((h, w), dtype=np.float32)
    # Inset so hairline/jaw edges stay from the original frame
    margin = max(2, int(min(h, w) * 0.06))
    axes = (max(1, w // 2 - margin), max(1, h // 2 - margin))
    center = (w // 2, int(h * 0.48))  # slightly higher = more forehead, less neck
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    # Feather: scale blur with face size; CLI feather is a hint
    blur = max(feather, int(min(h, w) * 0.08))
    k = blur * 2 + 1
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mmax = float(mask.max()) or 1.0
    return mask / mmax


def color_match_lab(src: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Match mean/std of src to ref in LAB, optionally only inside mask."""
    import cv2

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src_lab.copy()
    if mask is not None:
        m = mask.astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        # Use high-confidence interior for stats
        sel = m > 0.55
        if int(sel.sum()) < 64:
            sel = m > 0.2
    else:
        sel = np.ones(src_lab.shape[:2], dtype=bool)

    for c in range(3):
        s = src_lab[:, :, c][sel]
        r = ref_lab[:, :, c][sel]
        if s.size == 0 or r.size == 0:
            continue
        s_std = float(s.std()) + 1e-6
        r_std = float(r.std()) + 1e-6
        out[:, :, c] = (src_lab[:, :, c] - float(s.mean())) * (r_std / s_std) + float(r.mean())
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def paste_face(
    frame_bgr: np.ndarray,
    face_bgr: np.ndarray,
    box: FaceBox,
    *,
    feather: int = 24,
    color_match: bool = True,
    temporal_prev: np.ndarray | None = None,
    temporal_smooth: float = 0.0,
) -> np.ndarray:
    import cv2

    x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = min(fw, x + w), min(fh, y + h)
    x, y = max(0, x), max(0, y)
    w, h = x1 - x, y1 - y
    if w <= 1 or h <= 1:
        return frame_bgr

    resized = cv2.resize(face_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    roi = frame_bgr[y:y1, x:x1]
    mask2d = _oval_soft_mask(h, w, feather)

    if color_match and roi.size and resized.shape == roi.shape:
        resized = color_match_lab(resized, roi, mask=mask2d)

    if temporal_prev is not None and temporal_smooth > 0:
        prev = temporal_prev
        if prev.shape[:2] != (h, w):
            prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_LINEAR)
        if prev.shape == resized.shape:
            a = float(np.clip(temporal_smooth, 0.0, 1.0))
            resized = cv2.addWeighted(resized, 1.0 - a, prev, a, 0)

    mask = mask2d[:, :, None]
    blended = (
        resized.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)
    ).astype(np.uint8)
    out = frame_bgr
    # Avoid full-frame copy when caller already gave us a writable buffer;
    # still copy-on-write safe if frame is shared.
    if not out.flags.writeable:
        out = frame_bgr.copy()
    else:
        # process_frame starts from original frame each time; copy once
        out = frame_bgr.copy()
    out[y:y1, x:x1] = blended
    return out
