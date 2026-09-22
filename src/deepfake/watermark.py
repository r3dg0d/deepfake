"""Optional synthetic-media disclosure overlay."""

from __future__ import annotations

import numpy as np

# ASCII only: OpenCV's Hershey fonts render non-ASCII (e.g. an em dash) as "???".
DEFAULT_LABEL = "SYNTHETIC MEDIA - disclosed demo"


def apply_watermark(
    frame_bgr: np.ndarray,
    text: str = DEFAULT_LABEL,
    *,
    enabled: bool = True,
) -> np.ndarray:
    if not enabled:
        return frame_bgr
    import cv2

    out = frame_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, w / 1280 * 0.55)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    x0, y0 = 8, h - th - pad * 2 - 8
    cv2.rectangle(out, (x0, y0), (x0 + tw + pad * 2, y0 + th + pad * 2), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (x0 + pad, y0 + th + pad // 2),
        font,
        scale,
        (220, 220, 220),
        thickness,
        cv2.LINE_AA,
    )
    return out
