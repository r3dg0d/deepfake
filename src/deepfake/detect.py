"""Face detection / alignment backends (OpenCV first; optional insightface)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    landmarks: Any = None  # optional 5-point landmarks

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


class FaceDetector:
    """Thin interface; implementations may use OpenCV Haar/DNN or InsightFace."""

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        raise NotImplementedError


class OpenCVHaarDetector(FaceDetector):
    def __init__(self) -> None:
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade at {path}")

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(64, 64),
            flags=getattr(__import__("cv2"), "CASCADE_SCALE_IMAGE", 0),
        )
        boxes = [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]
        boxes.sort(key=lambda b: b.w * b.h, reverse=True)
        return boxes


class OpenCVYuNetDetector(FaceDetector):
    """Optional YuNet if opencv face module + model file available; else raises."""

    def __init__(self, model_path: str | None = None) -> None:
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV FaceDetectorYN not available in this build")
        # Without a model file, fall back is caller's job.
        if not model_path:
            raise RuntimeError("YuNet requires a model path")
        self._det = cv2.FaceDetectorYN.create(model_path, "", (320, 320))

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        h, w = frame_bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame_bgr)
        out: list[FaceBox] = []
        if faces is None:
            return out
        for f in faces:
            x, y, bw, bh, score = f[:5]
            out.append(FaceBox(int(x), int(y), int(bw), int(bh), float(score)))
        return out


def create_detector(prefer: str = "auto") -> FaceDetector:
    if prefer in ("opencv", "haar", "auto"):
        try:
            return OpenCVHaarDetector()
        except Exception:
            if prefer != "auto":
                raise
    raise RuntimeError(
        "No face detector available. Install opencv-python-headless "
        "(Haar cascade is built-in)."
    )


def face_square_box(box: FaceBox, frame_h: int, frame_w: int, pad: float = 0.35) -> FaceBox:
    """Expand detection box to a square paste/crop region (clamped to frame)."""
    cx = box.x + box.w / 2.0
    cy = box.y + box.h / 2.0
    # Bias slightly upward so forehead is included (Haar boxes sit low).
    cy = cy - box.h * 0.05
    side = max(box.w, box.h) * (1.0 + pad)
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = int(round(cx + side / 2.0))
    y1 = int(round(cy + side / 2.0))
    # Keep square while clamping: shift then shrink if needed.
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > frame_w:
        x0 -= x1 - frame_w
        x1 = frame_w
    if y1 > frame_h:
        y0 -= y1 - frame_h
        y1 = frame_h
    x0 = max(0, x0)
    y0 = max(0, y0)
    side_x = x1 - x0
    side_y = y1 - y0
    side_i = min(side_x, side_y)
    # Center the square inside the clamped rect
    x0 = x0 + (side_x - side_i) // 2
    y0 = y0 + (side_y - side_i) // 2
    return FaceBox(x0, y0, side_i, side_i, score=box.score, landmarks=box.landmarks)


def align_crop(
    frame_bgr: np.ndarray, box: FaceBox, size: int = 256, pad: float = 0.35
) -> tuple[np.ndarray, FaceBox]:
    """Square crop around face with padding; resize to size×size.

    Returns (crop_bgr, paste_box) where paste_box is the exact frame rectangle
    the swapped face must be composited back into (same region as the crop).
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    paste = face_square_box(box, h, w, pad=pad)
    if paste.w <= 1 or paste.h <= 1:
        return np.zeros((size, size, 3), dtype=np.uint8), paste
    crop = frame_bgr[paste.y : paste.y + paste.h, paste.x : paste.x + paste.w]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8), paste
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized, paste
