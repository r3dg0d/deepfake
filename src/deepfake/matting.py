"""Real-time rotoscoping and background replacement for the LivePortrait head engine.

The head engine generates a square crop that contains the reference character *and the reference
image's own background*. Pasting that rectangle over the camera frame overwrites whatever is behind
the head. This module cuts the character out instead:

* ``PersonSegmenter`` runs PP-HumanSeg (OpenCV Zoo, Apache-2.0) through ``cv2.dnn`` on the CPU, so it
  never competes with LivePortrait for the GPU. The pinned weights are installed with
  ``deepfake models install pphumanseg --yes``.
* ``HeadMatte`` turns the raw 192x192 probabilities into a usable matte (edge-aware refinement against
  the image, contrast shaping, temporal smoothing), composites the generated head through it, hides
  the parts of the real head the character does not cover with a learned clean plate, and optionally
  replaces the camera background.

Everything is a matte estimate from a small segmentation network: fine hair, motion blur and hands in
front of the face are approximations, not a studio key.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .detect import FaceBox

MODEL_NAME = "pphumanseg"
MODEL_FILE = "human_segmentation_pphumanseg_2023mar.onnx"
INSTALL_HINT = f"deepfake models install {MODEL_NAME} --yes"

NAMED_COLORS: dict[str, tuple[int, int, int]] = {  # BGR
    "green": (0, 255, 0),  # chroma key for OBS
    "blue": (255, 0, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}
BACKGROUND_CHOICES = ("keep", "blur", *NAMED_COLORS)
SCOPES = ("person", "head")


class MatteUnavailable(RuntimeError):
    pass


def model_path() -> Path:
    from .paths import models_dir

    return models_dir() / MODEL_NAME / MODEL_FILE


def model_installed() -> bool:
    return model_path().is_file()


@dataclass(frozen=True)
class BackgroundSpec:
    kind: str = "keep"  # keep | blur | color | image
    color: tuple[int, int, int] = (0, 0, 0)  # BGR, for kind == "color"
    path: Path | None = None  # for kind == "image"

    @property
    def replaces(self) -> bool:
        return self.kind != "keep"

    def describe(self) -> str:
        if self.kind == "color":
            return "#{:02x}{:02x}{:02x}".format(*reversed(self.color))
        return str(self.path) if self.kind == "image" else self.kind


def parse_background(value: str | None) -> BackgroundSpec:
    """``keep`` | ``blur`` | a colour name | ``#rrggbb`` | path to an image file."""
    raw = (value or "keep").strip()
    name = raw.lower()
    if name in ("keep", "none", "off", ""):
        return BackgroundSpec("keep")
    if name == "blur":
        return BackgroundSpec("blur")
    if name in ("remove", "transparent"):  # video sinks carry no alpha channel: black is the removed background
        return BackgroundSpec("color", NAMED_COLORS["black"])
    if name in NAMED_COLORS:
        return BackgroundSpec("color", NAMED_COLORS[name])
    if m := re.fullmatch(r"#?([0-9a-f]{6})", name):
        r, g, b = (int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
        return BackgroundSpec("color", (b, g, r))
    path = Path(raw).expanduser()
    if path.is_file():
        return BackgroundSpec("image", path=path)
    raise ValueError(f"Unknown background {raw!r}: use {', '.join(BACKGROUND_CHOICES)}, #rrggbb or an image file")


def resolve_rotoscope(choice: bool | None, background: BackgroundSpec) -> tuple[bool, str | None]:
    """Decide whether matting runs. ``choice`` None means auto: on when the weights are installed.

    Returns ``(enabled, note)``. Raises when the request cannot be met rather than silently showing
    the camera background the user asked to remove.
    """
    installed = model_installed()
    if choice is False:
        if background.replaces:
            raise ValueError("Background replacement needs the matte: drop --no-rotoscope or use --background keep")
        return False, None
    if installed:
        return True, None
    if choice is True or background.replaces:
        raise MatteUnavailable(f"Rotoscoping needs the PP-HumanSeg weights (6 MB, Apache-2.0): {INSTALL_HINT}")
    return False, f"rotoscoping is off: the head is pasted as a soft rectangle. Enable it with: {INSTALL_HINT}"


# --------------------------------------------------------------------------- segmentation


class PersonSegmenter:
    """PP-HumanSeg person probability map. One instance per thread of use (``cv2.dnn`` nets are not reentrant)."""

    SIZE = 192

    def __init__(self, path: Path | str | None = None) -> None:
        import cv2

        path = Path(path) if path is not None else model_path()
        if not path.is_file():
            raise MatteUnavailable(f"PP-HumanSeg weights not found at {path}: {INSTALL_HINT}")
        # OpenCV 5's new graph engine mis-registers this network (the matte comes out shifted by ~8% of the
        # image and is not mirror-consistent; measured on 5.0.0). The classic engine matches OpenCV 4.
        classic = getattr(cv2.dnn, "ENGINE_CLASSIC", None)
        self._net = cv2.dnn.readNet(str(path)) if classic is None else cv2.dnn.readNetFromONNX(str(path), engine=classic)

    def alpha(self, bgr: np.ndarray) -> np.ndarray:
        """float32 ``(SIZE, SIZE)`` person probability for the whole image (not resized back)."""
        import cv2

        rgb = cv2.cvtColor(cv2.resize(bgr, (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._net.setInput(cv2.dnn.blobFromImage(rgb.astype(np.float32) / 127.5 - 1.0))
        out = self._net.forward()[0]
        if out.min() < -1e-3 or out.max() > 1.0 + 1e-3:  # an export without the softmax head
            e = np.exp(out - out.max(axis=0, keepdims=True))
            out = e / e.sum(axis=0, keepdims=True)
        return np.ascontiguousarray(out[1], dtype=np.float32)


# --------------------------------------------------------------------------- matte shaping


def guided_refine(alpha: np.ndarray, guide_bgr: np.ndarray, radius: int = 6, eps: float = 2e-3) -> np.ndarray:
    """Edge-aware upsampling: snap a soft low-resolution matte to the edges of ``guide_bgr`` (guided filter)."""
    import cv2

    h, w = guide_bgr.shape[:2]
    if alpha.shape != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    guide = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i, mean_p = cv2.blur(guide, k), cv2.blur(alpha, k)
    var = cv2.blur(guide * guide, k) - mean_i * mean_i
    cov = cv2.blur(guide * alpha, k) - mean_i * mean_p
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return np.clip(cv2.blur(a, k) * guide + cv2.blur(b, k), 0.0, 1.0)


def sharpen(alpha: np.ndarray, lo: float = 0.2, hi: float = 0.8) -> np.ndarray:
    """Smoothstep contrast: probabilities below ``lo`` become background, above ``hi`` solid."""
    t = np.clip((alpha - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@lru_cache(maxsize=8)
def edge_falloff(h: int, w: int) -> np.ndarray:
    """Soft rectangular window so a matte that touches the crop border never ends in a hard seam (read-only)."""
    import cv2

    margin = max(1, min(w, h) // 16)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[margin:h - margin, margin:w - margin] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, margin / 2))
    mask.flags.writeable = False
    return mask


class TemporalMatte:
    """Exponential smoothing that follows fast changes: flicker is damped, real motion is not trailed."""

    def __init__(self, keep: float = 0.5, motion: float = 0.35) -> None:
        self.keep, self.motion = keep, motion
        self._prev: np.ndarray | None = None

    def __call__(self, alpha: np.ndarray) -> np.ndarray:
        prev = self._prev
        if prev is None or prev.shape != alpha.shape:
            self._prev = alpha
            return alpha
        # weight of the previous matte falls to zero where the matte changed by more than ``motion``
        w = self.keep * np.clip(1.0 - np.abs(alpha - prev) / self.motion, 0.0, 1.0)
        out = alpha + (prev - alpha) * w
        self._prev = out
        return out

    def reset(self) -> None:
        self._prev = None


# --------------------------------------------------------------------------- compositor


def _clip_box(box: FaceBox, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    x0, y0 = max(0, int(box.x)), max(0, int(box.y))
    x1, y1 = min(frame_w, int(box.x + box.w)), min(frame_h, int(box.y + box.h))
    return x0, y0, x1, y1


class HeadMatte:
    """Composites a generated head crop into the camera frame through a person matte.

    Per frame: ``begin(frame, driver_crop)`` starts the camera-side segmentations on worker threads
    (they overlap the GPU inference), then ``composite(frame, generated, paste_box, face_box)`` builds
    the output. ``passthrough(frame)`` handles frames without a face.
    """

    CROP = 256
    PLATE_RATE = 0.12  # clean-plate learning rate per frame where the background is visible

    def __init__(self, background: BackgroundSpec | None = None, scope: str = "person", *,
                 segmenter_factory=PersonSegmenter) -> None:
        if scope not in SCOPES:
            raise ValueError(f"matte scope must be one of {', '.join(SCOPES)}")
        self.background = background or BackgroundSpec()
        self.scope = scope
        self._seg_generated = segmenter_factory()
        self._seg_driver = segmenter_factory()
        self._needs_frame = self.background.replaces and scope == "person"
        self._seg_frame = segmenter_factory() if self._needs_frame else None
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="deepfake-matte")
        self._driver: Future | None = None
        self._frame: Future | None = None
        self._smooth_generated = TemporalMatte()
        self._smooth_frame = TemporalMatte(keep=0.35)
        self._plate: np.ndarray | None = None  # float32 HxWx3 background estimate
        self._known: np.ndarray | None = None  # float32 HxW, 1 where the plate has been observed
        self._bg_image: np.ndarray | None = None
        self._ms: float | None = None
        if self.background.kind == "image":
            import cv2

            self._bg_source = cv2.imread(str(self.background.path), cv2.IMREAD_COLOR)
            if self._bg_source is None:
                raise ValueError(f"Cannot read background image {self.background.path}")

    # -- stages

    def begin(self, frame: np.ndarray, driver_crop: np.ndarray | None) -> None:
        if driver_crop is not None and not self.background.replaces:
            self._driver = self._pool.submit(self._seg_driver.alpha, driver_crop)
        if self._seg_frame is not None:
            self._frame = self._pool.submit(self._frame_matte, frame)

    def _frame_matte(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        h, w = frame.shape[:2]
        # refine at a bounded working size, then scale: the matte has no detail beyond it
        scale = min(1.0, 384.0 / max(h, w))
        small = frame if scale >= 1.0 else cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        alpha = sharpen(guided_refine(self._seg_frame.alpha(frame), small, radius=5))
        return alpha if scale >= 1.0 else cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    def _take(self, name: str):
        future: Future | None = getattr(self, name)
        setattr(self, name, None)
        return None if future is None else future.result()

    def composite(self, frame: np.ndarray, generated_bgr: np.ndarray, paste_box: FaceBox,
                  face_box: FaceBox | None = None) -> np.ndarray:
        """Returns a new frame: generated head matted over ``frame`` (and the background replaced)."""
        import cv2

        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]
        x0, y0, x1, y1 = _clip_box(paste_box, fh, fw)
        w, h = x1 - x0, y1 - y0
        driver = self._take("_driver")
        frame_alpha = self._take("_frame")
        if w <= 1 or h <= 1:
            return self._finish(frame.copy(), frame_alpha, None, (x0, y0, x1, y1), None, t0)

        # 1. matte of the generated character, in crop space where it is temporally stable
        if generated_bgr.shape[:2] != (self.CROP, self.CROP):
            generated_bgr = cv2.resize(generated_bgr, (self.CROP, self.CROP), interpolation=cv2.INTER_LINEAR)
        matte = sharpen(guided_refine(self._seg_generated.alpha(generated_bgr), generated_bgr))
        matte = self._smooth_generated(matte)
        head = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR) * edge_falloff(h, w)
        generated = cv2.resize(generated_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        base = frame[y0:y1, x0:x1]
        zone = self._head_zone(h, w, y0, face_box)

        # 2. keep-background mode: hide the real head where the character does not cover it
        if driver is not None:
            real = sharpen(cv2.resize(driver, (w, h), interpolation=cv2.INTER_LINEAR))
            plate, known = self._learn_plate(frame, real, (x0, y0, x1, y1))
            stick = np.clip(real - head, 0.0, 1.0) * zone * known
            base = cv2.blendLinear(base, cv2.convertScaleAbs(plate), 1.0 - stick, stick)

        out = frame.copy()
        out[y0:y1, x0:x1] = cv2.blendLinear(np.ascontiguousarray(base), generated, 1.0 - head, head)
        return self._finish(out, frame_alpha, head, (x0, y0, x1, y1), zone, t0)

    def passthrough(self, frame: np.ndarray) -> np.ndarray:
        """A frame without a detected face: only the background treatment applies."""
        t0 = time.perf_counter()
        self._take("_driver")
        self._smooth_generated.reset()
        return self._finish(frame.copy(), self._take("_frame"), None, None, None, t0)

    # -- pieces

    @staticmethod
    def _head_zone(h: int, w: int, y0: int, face_box: FaceBox | None) -> np.ndarray:
        """1 above the chin, 0 below it: where the character's matte alone decides what is foreground."""
        if face_box is None:
            return np.ones((h, w), dtype=np.float32)
        chin = float(face_box.y + face_box.h) - y0
        ramp = max(2.0, 0.15 * float(face_box.h))
        rows = np.clip((chin + ramp - np.arange(h, dtype=np.float32)) / ramp, 0.0, 1.0)
        return np.repeat(rows[:, None], w, axis=1)

    def _learn_plate(self, frame: np.ndarray, real: np.ndarray, rect: tuple[int, int, int, int]):
        """Update the clean plate where the real person is absent; return the plate and its validity in ``rect``."""
        x0, y0, x1, y1 = rect
        if self._plate is None or self._plate.shape[:2] != frame.shape[:2]:
            self._plate = frame.astype(np.float32)
            self._known = np.zeros(frame.shape[:2], dtype=np.float32)
        import cv2

        plate, known = self._plate[y0:y1, x0:x1], self._known[y0:y1, x0:x1]
        roi = frame[y0:y1, x0:x1]
        visible = real < 0.08
        seen = known > 0
        # first sight of a pixel takes it directly; afterwards follow lighting changes slowly
        cv2.accumulateWeighted(roi, plate, self.PLATE_RATE, mask=(visible & seen).astype(np.uint8))
        fresh = visible & ~seen
        if fresh.any():
            plate[fresh] = roi[fresh]
            known[fresh] = 1.0
        return plate, known

    def _backdrop(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        h, w = frame.shape[:2]
        spec = self.background
        if spec.kind == "blur":
            small = cv2.resize(frame, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
            return cv2.resize(cv2.GaussianBlur(small, (0, 0), 3.0), (w, h), interpolation=cv2.INTER_LINEAR)
        if self._bg_image is None or self._bg_image.shape[:2] != (h, w):
            if spec.kind == "image":
                sh, sw = self._bg_source.shape[:2]
                s = max(w / sw, h / sh)  # cover, centred
                fit = cv2.resize(self._bg_source, (max(w, round(sw * s)), max(h, round(sh * s))), interpolation=cv2.INTER_AREA)
                oy, ox = (fit.shape[0] - h) // 2, (fit.shape[1] - w) // 2
                self._bg_image = np.ascontiguousarray(fit[oy:oy + h, ox:ox + w])
            else:
                self._bg_image = np.full((h, w, 3), spec.color, dtype=np.uint8)
        return self._bg_image

    def _finish(self, out, frame_alpha, head, rect, zone, t0) -> np.ndarray:
        if self.background.replaces:
            import cv2

            fh, fw = out.shape[:2]
            if frame_alpha is not None:
                fg = self._smooth_frame(frame_alpha).copy()
            else:
                fg = np.zeros((fh, fw), dtype=np.float32)
            if head is not None:
                x0, y0, x1, y1 = rect
                # inside the head window the real person only counts below the chin; the window fades out
                # towards the crop border so nothing ends on a straight edge
                body = fg[y0:y1, x0:x1] * (1.0 - zone * edge_falloff(y1 - y0, x1 - x0))
                fg[y0:y1, x0:x1] = np.maximum(head, body)
            out = cv2.blendLinear(out, self._backdrop(out), fg, 1.0 - fg)
        ms = (time.perf_counter() - t0) * 1000
        self._ms = ms if self._ms is None else self._ms + 0.1 * (ms - self._ms)
        return out

    @property
    def matte_ms(self) -> float | None:
        """Smoothed cost of the compositing stage on the inference thread (camera-side segmentation overlaps the GPU)."""
        return None if self._ms is None else round(self._ms, 2)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


__all__ = ["BACKGROUND_CHOICES", "BackgroundSpec", "HeadMatte", "INSTALL_HINT", "MODEL_FILE", "MODEL_NAME",
           "MatteUnavailable", "PersonSegmenter", "SCOPES", "TemporalMatte", "edge_falloff", "guided_refine",
           "model_installed", "model_path", "parse_background", "resolve_rotoscope", "sharpen"]
