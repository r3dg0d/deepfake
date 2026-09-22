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
        rects = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]


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


def align_crop(frame_bgr: np.ndarray, box: FaceBox, size: int = 256, pad: float = 0.25) -> np.ndarray:
    """Simple square crop around face with padding; resize to size×size."""
    import cv2

    h, w = frame_bgr.shape[:2]
    cx = box.x + box.w / 2
    cy = box.y + box.h / 2
    side = max(box.w, box.h) * (1.0 + pad)
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(w, cx + side / 2))
    y1 = int(min(h, cy + side / 2))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
