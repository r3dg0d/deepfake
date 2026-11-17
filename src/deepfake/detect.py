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
    # Haar cost grows with pixel count; faces in webcam framing stay far above
    # the minimum size after downscaling, so detect at <= this width and map
    # boxes back (measured 33 ms → a few ms per call at 1280x720).
    DETECT_WIDTH = 480

    def __init__(self, detect_width: int | None = None) -> None:
        import cv2

        path = self._cascade_path()
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade at {path}")
        self.detect_width = detect_width or self.DETECT_WIDTH

    @staticmethod
    def _cascade_path() -> str:
        """Locate haarcascade_frontalface_default.xml on pip and Nix installs.

        PyPI's opencv-python(-headless) bundles the cascade XMLs inside the
        package and exposes them via `cv2.data.haarcascades`. nixpkgs' opencv4
        ships the same files under <store-path>/share/opencv4/haarcascades/
        instead and has no `cv2.data` module, so fall back to walking up from
        the installed cv2 package to find that sibling share/ directory.
        """
        import cv2

        name = "haarcascade_frontalface_default.xml"
        if hasattr(cv2, "data"):
            return cv2.data.haarcascades + name

        import pathlib

        for parent in pathlib.Path(cv2.__file__).resolve().parents:
            candidate = parent / "share" / "opencv4" / "haarcascades" / name
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError(
            "could not locate haarcascade_frontalface_default.xml "
            "(no cv2.data and no share/opencv4/haarcascades found)"
        )

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        import cv2

        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.detect_width / float(w))
        small = frame_bgr if scale >= 1.0 else cv2.resize(
            frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        min_side = max(24, int(64 * scale))
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(min_side, min_side),
            flags=getattr(cv2, "CASCADE_SCALE_IMAGE", 0),
        )
        inv = 1.0 / scale
        boxes = [
            FaceBox(int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)) for (x, y, bw, bh) in rects
        ]
        boxes.sort(key=lambda b: b.w * b.h, reverse=True)
        return boxes


class YuNetDetector(FaceDetector):
    """OpenCV's YuNet (cv2.FaceDetectorYN): boxes plus 5 landmarks, robust to pose and lighting where Haar is not.

    Default detector. Runs at <= DETECT_WIDTH like the Haar path and maps boxes back. Weights are the pinned
    OpenCV Zoo model installed by ``deepfake models install yunet --yes``.
    """

    DETECT_WIDTH = 640
    REFINE_SIZE = 320  # side of the face region the landmark pass runs on
    REFINE_MAX_FACES = 4

    def __init__(self, model_path: str | None = None, *, score_threshold: float = 0.7, detect_width: int | None = None,
                 refine: bool = True) -> None:
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV FaceDetectorYN not available in this build")
        if model_path is None:
            from .paths import models_dir

            p = models_dir() / "yunet" / "face_detection_yunet_2023mar.onnx"
            if not p.is_file():
                raise RuntimeError("YuNet model not installed: deepfake models install yunet --yes")
            model_path = str(p)
        self._det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)
        self.detect_width = detect_width or self.DETECT_WIDTH
        self._size = (0, 0)
        # Landmarks from the downscaled full frame are too coarse to align with: on a face ~90 px wide
        # rolled 16 degrees they report about 2 degrees. A second pass on the face region at a fixed
        # size gives usable points; boxes are left exactly as the first pass found them.
        self.refine = refine
        self._refiner = (cv2.FaceDetectorYN.create(model_path, "", (self.REFINE_SIZE, self.REFINE_SIZE),
                                                   min(score_threshold, 0.5), 0.3, 50) if refine else None)

    def refresh_landmarks(self, frame_bgr: np.ndarray, box: FaceBox) -> None:
        """Re-measure the landmarks of a box found on an earlier frame against the current frame."""
        if self._refiner is not None:
            self._refine_landmarks(frame_bgr, box)

    def _refine_landmarks(self, frame_bgr: np.ndarray, box: FaceBox) -> None:
        import cv2

        side = max(box.w, box.h) * 1.5
        if side < 8:
            return
        cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
        n = self.REFINE_SIZE
        s = n / side
        # frame -> region: uniform scale + translation (no clamping, so the mapping stays exact at frame edges)
        m = np.array([[s, 0.0, n / 2.0 - s * cx], [0.0, s, n / 2.0 - s * cy]], dtype=np.float64)
        region = cv2.warpAffine(frame_bgr, m, (n, n), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        _, faces = self._refiner.detect(region)
        if faces is None or len(faces) == 0:
            return
        # the face this box belongs to is the one nearest the region centre
        best = min(faces, key=lambda f: (f[0] + f[2] / 2.0 - n / 2.0) ** 2 + (f[1] + f[3] / 2.0 - n / 2.0) ** 2)
        lm = np.asarray(best[4:14], dtype=np.float64).reshape(5, 2)
        box.landmarks = ((lm - m[:, 2]) / s).astype(np.float32)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        import cv2

        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.detect_width / float(w))
        small = frame_bgr if scale >= 1.0 else cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        if (sw, sh) != self._size:
            self._det.setInputSize((sw, sh))
            self._size = (sw, sh)
        _, faces = self._det.detect(small)
        out: list[FaceBox] = []
        if faces is None:
            return out
        inv = 1.0 / scale
        for f in faces:
            x, y, bw, bh = (float(v) * inv for v in f[:4])
            lm = np.asarray(f[4:14], dtype=np.float32).reshape(5, 2) * inv
            out.append(FaceBox(int(x), int(y), int(bw), int(bh), float(f[14]), landmarks=lm))
        out.sort(key=lambda b: b.w * b.h, reverse=True)
        if self._refiner is not None:
            for box in out[: self.REFINE_MAX_FACES]:
                self._refine_landmarks(frame_bgr, box)
        return out


OpenCVYuNetDetector = YuNetDetector  # previous name


class MediaPipeLandmarker:
    """Five alignment points from the MediaPipe Face Landmarker (478-point mesh), run on each face region.

    Optional: needs the ``mediapipe`` package and ``deepfake models install face-landmarker --yes``.
    Boxes still come from the detector; only ``FaceBox.landmarks`` is replaced.
    """

    REGION = 256
    # mesh indices: outer/inner corners of each eye (averaged to an eye centre), nose tip, mouth corners
    LEFT_EYE, RIGHT_EYE, NOSE, MOUTH_L, MOUTH_R = (33, 133), (362, 263), 1, 61, 291

    def __init__(self, model_path: str | None = None) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as e:  # noqa: BLE001 - import errors include missing system libraries
            raise RuntimeError(f"mediapipe is not usable ({e}). Install: pip install 'deepfake[landmarks]'") from e
        if model_path is None:
            from .paths import models_dir

            p = models_dir() / "face-landmarker" / "face_landmarker.task"
            if not p.is_file():
                raise RuntimeError("face landmarker not installed: deepfake models install face-landmarker --yes")
            model_path = str(p)
        self._mp = mp
        self._lmk = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path), num_faces=1))

    def landmarks(self, frame_bgr: np.ndarray, box: FaceBox) -> np.ndarray | None:
        import cv2

        side = max(box.w, box.h) * 1.6
        if side < 8:
            return None
        n = self.REGION
        s = n / side
        cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
        m = np.array([[s, 0.0, n / 2.0 - s * cx], [0.0, s, n / 2.0 - s * cy]], dtype=np.float64)
        region = cv2.warpAffine(frame_bgr, m, (n, n), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        rgb = np.ascontiguousarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        res = self._lmk.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None
        pts = np.array([[p.x * n, p.y * n] for p in res.face_landmarks[0]], dtype=np.float64)
        five = np.stack([pts[list(self.LEFT_EYE)].mean(axis=0), pts[list(self.RIGHT_EYE)].mean(axis=0),
                         pts[self.NOSE], pts[self.MOUTH_L], pts[self.MOUTH_R]])
        return ((five - m[:, 2]) / s).astype(np.float32)


class LandmarkRefinedDetector(FaceDetector):
    """A detector whose boxes get their landmarks from a dedicated landmarker."""

    MAX_FACES = 4

    def __init__(self, detector: FaceDetector, landmarker: MediaPipeLandmarker) -> None:
        self.detector, self.landmarker = detector, landmarker
        self.landmark_source = "mediapipe"

    def refresh_landmarks(self, frame_bgr: np.ndarray, box: FaceBox) -> None:
        lm = self.landmarker.landmarks(frame_bgr, box)
        if lm is not None:
            box.landmarks = lm

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        boxes = self.detector.detect(frame_bgr)
        for box in boxes[: self.MAX_FACES]:
            lm = self.landmarker.landmarks(frame_bgr, box)
            if lm is not None:
                box.landmarks = lm
        return boxes


def create_detector(prefer: str = "auto", landmarker: str = "auto") -> FaceDetector:
    """``landmarker``: auto (MediaPipe when installed, else refined YuNet points) | mediapipe | yunet."""
    det = _create_box_detector(prefer)
    if landmarker in ("auto", "mediapipe"):
        try:
            return LandmarkRefinedDetector(det, MediaPipeLandmarker())
        except (RuntimeError, OSError, ImportError):
            if landmarker == "mediapipe":
                raise
    return det


def landmark_source(detector: FaceDetector) -> str:
    if isinstance(detector, LandmarkRefinedDetector):
        return detector.landmark_source
    if isinstance(detector, YuNetDetector):
        return "yunet-refined" if detector.refine else "yunet"
    return "none"


def _create_box_detector(prefer: str = "auto") -> FaceDetector:
    errors = []
    if prefer in ("yunet", "auto"):
        try:
            return YuNetDetector()
        except Exception as e:  # noqa: BLE001
            errors.append(f"YuNet: {e}")
            if prefer != "auto":
                raise
    if prefer in ("opencv", "haar", "auto"):
        try:
            return OpenCVHaarDetector()
        except Exception as e:  # noqa: BLE001
            errors.append(f"Haar: {e}")
            if prefer != "auto":
                raise
    raise RuntimeError(
        "No face detector available (" + "; ".join(errors) + "). "
        "Install the default detector: deepfake models install yunet --yes "
        "(OpenCV 5 removed the Haar cascades)."
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


class AsyncDetector:
    """Runs a detector on its own thread against the newest submitted frame.

    Live mode uses this so detection never sits on the swap's critical path;
    the swap uses the most recent boxes (typically from the previous frame).
    """

    def __init__(self, detector: FaceDetector) -> None:
        import threading

        self._det = detector
        self._lock = threading.Lock()  # detectors are not thread-safe (YuNet's DNN graph asserts under concurrent use)
        self._cv = threading.Condition()
        self._pending: np.ndarray | None = None
        self._boxes: list[FaceBox] | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="deepfake-detect", daemon=True)
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> None:
        with self._cv:
            self._pending = frame_bgr
            self._cv.notify()

    def detect_now(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        """Synchronous detection on the caller's thread, serialized with the worker (first frame)."""
        with self._lock:
            return self._det.detect(frame_bgr)

    def latest(self) -> list[FaceBox] | None:
        with self._cv:
            return self._boxes

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                frame, self._pending = self._pending, None
            with self._lock:
                boxes = self._det.detect(frame)
            with self._cv:
                self._boxes = boxes

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify()
