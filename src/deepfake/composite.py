"""Paste swapped face back into frame with oval soft mask / color match."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class ColorState:
    gain: np.ndarray | None = None
    offset: np.ndarray | None = None


def color_match_lab(
    src: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray | None = None,
    state: ColorState | None = None,
    smooth: float = 0.0,
) -> np.ndarray:
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

    if int(sel.sum()) < 64:
        return src
    gain = np.clip(ref_lab[sel].std(0) / (src_lab[sel].std(0) + 1e-6), 0.5, 2.0)
    offset = ref_lab[sel].mean(0) - src_lab[sel].mean(0) * gain
    if state is not None:
        weight = float(np.clip(smooth, 0, 0.8))
        if state.gain is not None:
            gain = gain * (1 - weight) + state.gain * weight
            offset = offset * (1 - weight) + state.offset * weight
        state.gain, state.offset = gain, offset
    out = src_lab * gain + offset
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
    visible_mask: np.ndarray | None = None,
    color_state: ColorState | None = None,
) -> np.ndarray:
    import cv2

    x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = min(fw, x + w), min(fh, y + h)
    x, y = max(0, x), max(0, y)
    w, h = x1 - x, y1 - y
    if w <= 1 or h <= 1:
        return frame_bgr

    # Resize into the ORIGINAL box, then clip; never squash an offscreen face.
    original_w, original_h = int(box.w), int(box.h)
    ox, oy = x - int(box.x), y - int(box.y)
    resized = cv2.resize(face_bgr, (original_w, original_h))[oy : oy + h, ox : ox + w]
    roi = frame_bgr[y:y1, x:x1]
    full_mask = (
        _oval_soft_mask(original_h, original_w, feather)
        if visible_mask is None
        else cv2.resize(visible_mask.astype(np.float32), (original_w, original_h))
    )
    mask2d = np.clip(full_mask[oy : oy + h, ox : ox + w], 0, 1)
    if visible_mask is not None:
        # Bilinear upsampling can reveal an occluder boundary. Intersect nearest
        # support to keep pixels declared hidden at exactly zero opacity.
        support = cv2.resize(
            (visible_mask > 0).astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST
        )[oy : oy + h, ox : ox + w]
        mask2d *= support
    if color_match and roi.size and resized.shape == roi.shape:
        # Estimate gains from the canonical crop, then apply to the full ROI.
        sample_size = (min(w, 128), min(h, 128))
        sample_src, sample_ref = cv2.resize(resized, sample_size), cv2.resize(roi, sample_size)
        sample_mask = cv2.resize(mask2d, sample_size, interpolation=cv2.INTER_NEAREST)
        state = color_state if color_state is not None else ColorState()
        color_match_lab(sample_src, sample_ref, mask=sample_mask, state=state, smooth=temporal_smooth)
        if state.gain is not None:
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab = np.clip(lab * state.gain + state.offset, 0, 255)
            # Retain scene chroma on partly saturated profiles where global
            # statistics alone introduce cyan/magenta casts.
            target_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
            interior = mask2d > 0.55
            if np.count_nonzero(interior) >= 64:
                target_lab[~interior] = target_lab[interior].mean(0)
                radius = max(1, round(min(w, h) / 40))
                local = cv2.GaussianBlur(target_lab, (2 * radius + 1, 2 * radius + 1), 0)
                lab[:, :, 1:] = 0.3 * lab[:, :, 1:] + 0.7 * local[:, :, 1:]
            lab = lab.astype(np.uint8)
            resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if temporal_prev is not None and temporal_smooth > 0:
        prev = temporal_prev
        if prev.shape[:2] != (h, w):
            prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_LINEAR)
        if prev.shape == resized.shape:
            a = float(np.clip(temporal_smooth, 0.0, 1.0))
            resized = cv2.addWeighted(resized, 1.0 - a, prev, a, 0)

    mask = mask2d[:, :, None]
    blended = (resized.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
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
