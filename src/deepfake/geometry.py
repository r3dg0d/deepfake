"""Face geometry: landmark alignment, the transform back into the frame, masks and smoothing.

Everything here works in two coordinate spaces and keeps them explicit:

* **frame space**: pixels of the full output frame.
* **crop space**: pixels of the ``size`` x ``size`` image the swap model sees.

``AlignedFace.matrix`` maps frame -> crop (a similarity: rotation, uniform scale,
translation; never a non-uniform scale, so facial proportions cannot change).
The paste uses the exact inverse of the same matrix, so crop and paste cannot
disagree about where the face is.

The legacy path (``detect.align_crop`` + ``composite.paste_face``) is an
axis-aligned padded box around the detector rectangle. It stays available as
``alignment="bbox"`` for comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Five-point templates in unit crop coordinates (x, y in 0..1), in detector order:
# image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner.
#
# "arcface": the InsightFace ArcFace 112x112 reference points / 112. This is the
#   alignment the ArcFace identity encoders (and inswapper-family generators) are
#   trained on; eye distance is 0.315 of the crop.
# "ffhq": the FFHQ dataset alignment (crop side = 4 x eye distance, centre 0.1 x
#   eye-to-mouth below the eye midpoint); eye distance is 0.25 of the crop. Its
#   framing is close to the legacy padded box on an upright frontal face.
TEMPLATES: dict[str, np.ndarray] = {
    "arcface": np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float64,
    ) / 112.0,
    "ffhq": np.array(
        [[0.375, 0.470], [0.625, 0.470], [0.500, 0.610], [0.396, 0.742], [0.604, 0.742]],
        dtype=np.float64,
    ),
}

ALIGNMENTS = ("landmark", "bbox")
MASKS = ("landmark", "oval")
FITS = ("cover", "contain", "stretch")


def template_points(name: str, size: int) -> np.ndarray:
    if name not in TEMPLATES:
        raise KeyError(f"unknown alignment template {name!r}; choose from {sorted(TEMPLATES)}")
    return TEMPLATES[name] * float(size)


def estimate_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (Umeyama, no reflection) mapping ``src`` points onto ``dst``.

    Returns a 2x3 matrix ``[sR | t]``. The scale is uniform by construction.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape or src.shape[0] < 2:
        raise ValueError("need at least two matching point pairs")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = float((s0**2).sum() / src.shape[0])
    if var_s < 1e-12:
        raise ValueError("degenerate landmarks (all points coincide)")
    cov = d0.T @ s0 / src.shape[0]
    u, sing, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    scale = float((sing * d).sum() / var_s)
    m = np.zeros((2, 3), dtype=np.float64)
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - scale * rot @ mu_s
    return m


def invert_affine(m: np.ndarray) -> np.ndarray:
    a = np.vstack([np.asarray(m, dtype=np.float64), [0.0, 0.0, 1.0]])
    return np.linalg.inv(a)[:2]


def transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ np.asarray(m, dtype=np.float64)[:, :2].T + np.asarray(m, dtype=np.float64)[:, 2]


def similarity_params(m: np.ndarray) -> tuple[float, float]:
    """(uniform scale, rotation in degrees) of a similarity matrix."""
    return float(math.hypot(m[0, 0], m[1, 0])), float(math.degrees(math.atan2(m[1, 0], m[0, 0])))


def anisotropy(m: np.ndarray) -> float:
    """Ratio of the two singular values of the linear part; 1.0 means proportions are preserved."""
    s = np.linalg.svd(np.asarray(m, dtype=np.float64)[:, :2], compute_uv=False)
    return float(s[0] / max(s[1], 1e-12))


@dataclass
class AlignedFace:
    crop: np.ndarray  # size x size BGR, crop space
    matrix: np.ndarray  # 2x3, frame -> crop
    size: int
    template: str
    landmarks: np.ndarray  # 5x2, frame space (the points the matrix was fitted to)

    @property
    def inverse(self) -> np.ndarray:
        return invert_affine(self.matrix)


def align_face(frame_bgr: np.ndarray, landmarks: np.ndarray, *, size: int = 256, template: str = "ffhq") -> AlignedFace:
    """Warp the face so its landmarks sit on the template. Roll is removed; scale is uniform."""
    import cv2

    lm = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    m = estimate_similarity(lm, template_points(template, size))
    crop = cv2.warpAffine(frame_bgr, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return AlignedFace(crop=crop, matrix=m, size=size, template=template, landmarks=lm)


def landmark_mask(size: int, template: str = "ffhq", *, feather: int = 24) -> np.ndarray:
    """Soft face mask in crop space, placed from the template landmarks.

    An ellipse centred between the eyes and the mouth, sized from the eye
    distance, so it follows the face and not the padded crop. It is zero at the
    crop border, which keeps the paste free of a rectangular edge.
    """
    import cv2

    pts = template_points(template, size)
    eye_mid = (pts[0] + pts[1]) / 2.0
    mouth_mid = (pts[3] + pts[4]) / 2.0
    eye_dist = float(np.linalg.norm(pts[1] - pts[0]))
    eye_mouth = float(np.linalg.norm(mouth_mid - eye_mid))
    centre = eye_mid + (mouth_mid - eye_mid) * 0.45
    # Cheek to cheek is about 1.9 eye distances; brow to chin about 2.5 eye-to-mouth distances.
    ax = eye_dist * 0.98
    ay = eye_mouth * 1.28
    blur = max(3, int(round(min(feather, size * 0.12))))
    limit = size / 2.0 - blur - 2  # the feather must finish inside the crop
    ax, ay = min(ax, limit), min(ay, limit)
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(mask, (int(round(centre[0])), int(round(centre[1]))), (int(round(ax)), int(round(ay))), 0, 0, 360, 1.0, -1)
    k = blur * 2 + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    edge = max(1, blur // 2)
    mask[:edge] = 0
    mask[-edge:] = 0
    mask[:, :edge] = 0
    mask[:, -edge:] = 0
    return mask / (float(mask.max()) or 1.0)


def paste_aligned(
    frame_bgr: np.ndarray,
    face_bgr: np.ndarray,
    aligned: AlignedFace,
    *,
    mask: np.ndarray | None = None,
    feather: int = 24,
    color_match: bool = True,
) -> np.ndarray:
    """Composite a crop-space face into the frame through the inverse of the crop transform."""
    import cv2

    from .composite import color_match_lab

    size = aligned.size
    if face_bgr.shape[:2] != (size, size):
        face_bgr = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    if mask is None:
        mask = landmark_mask(size, aligned.template, feather=feather)
    if color_match:
        face_bgr = color_match_lab(face_bgr, aligned.crop, mask=mask)

    fh, fw = frame_bgr.shape[:2]
    inv = aligned.inverse
    corners = transform_points(inv, np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64))
    x0 = max(0, int(math.floor(corners[:, 0].min())))
    y0 = max(0, int(math.floor(corners[:, 1].min())))
    x1 = min(fw, int(math.ceil(corners[:, 0].max())) + 1)
    y1 = min(fh, int(math.ceil(corners[:, 1].max())) + 1)
    if x1 - x0 <= 1 or y1 - y0 <= 1:
        return frame_bgr
    # Same transform as the crop, expressed for the region of interest only.
    roi_inv = inv.copy()
    roi_inv[:, 2] -= (x0, y0)
    roi_size = (x1 - x0, y1 - y0)
    scale, _ = similarity_params(inv)
    interp = cv2.INTER_AREA if scale < 0.75 else cv2.INTER_LINEAR
    warped = cv2.warpAffine(face_bgr, roi_inv, roi_size, flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.warpAffine(mask.astype(np.float32), roi_inv, roi_size, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)[:, :, None]
    out = frame_bgr.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(warped.astype(np.float32) * alpha + roi * (1.0 - alpha), 0, 255).astype(np.uint8)
    return out


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012) over an array of coordinates.

    Low jitter while the face is still, low lag while it moves: the cutoff rises
    with the measured speed. Speeds are in units per second of ``scale`` (the eye
    distance), so the same parameters work at any resolution and face size.

    Defaults were chosen on the bundled fixture at 24 fps against known landmark
    positions: still-face jitter 0.079 -> 0.023 px, error while moving 0.57 px
    (0.33 px unfiltered). The textbook starting point (1.5, 0.6) lagged by 4.8 px.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 20.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None
        self._dx = None

    def __call__(self, x: np.ndarray, dt: float, scale: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._x.shape != x.shape or dt <= 0:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            return x.copy()
        scale = max(float(scale), 1e-6)
        dx = (x - self._x) / dt / scale
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        speed = np.linalg.norm(self._dx.reshape(-1, 2), axis=1, keepdims=True) if x.ndim == 2 else np.abs(self._dx)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()


def fit_frame(frame_bgr: np.ndarray, size: tuple[int, int], mode: str = "cover") -> np.ndarray:
    """Bring a frame to ``size`` (width, height).

    * ``cover``: uniform scale, centre crop the overflow (default; no distortion, no bars).
    * ``contain``: uniform scale, pad with black.
    * ``stretch``: independent x/y scale. This is the legacy behaviour and distorts
      faces whenever the camera does not deliver the requested aspect ratio.
    """
    import cv2

    w, h = int(size[0]), int(size[1])
    fh, fw = frame_bgr.shape[:2]
    if (fw, fh) == (w, h):
        return frame_bgr
    if mode == "stretch":
        return cv2.resize(frame_bgr, (w, h))
    if mode not in ("cover", "contain"):
        raise ValueError(f"unknown fit mode {mode!r}; choose from {FITS}")
    s = max(w / fw, h / fh) if mode == "cover" else min(w / fw, h / fh)
    nw, nh = max(1, int(round(fw * s))), max(1, int(round(fh * s)))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if mode == "cover":
        x0, y0 = max(0, (nw - w) // 2), max(0, (nh - h) // 2)
        out = resized[y0:y0 + h, x0:x0 + w]
        if out.shape[:2] != (h, w):  # rounding left it a pixel short
            out = cv2.resize(out, (w, h))
        return out
    canvas = np.zeros((h, w, 3), dtype=frame_bgr.dtype)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized[:h, :w]
    return canvas


def identity_crop(source_bgr: np.ndarray, detector=None, *, mode: str = "arcface", size: int = 112) -> np.ndarray:
    """Crop of the identity image for the ArcFace encoder.

    ``arcface`` aligns the five landmarks to the ArcFace reference points, which is
    the input the encoder was trained on. ``bbox`` is the legacy padded box.
    Falls back to the padded box, then to the whole image, when no landmarks are found.
    """
    import cv2

    from .detect import align_crop, create_detector

    try:
        det = detector or create_detector("auto")
        boxes = det.detect(source_bgr)
    except Exception:  # noqa: BLE001 - a missing detector must not block loading an identity
        boxes = []
    if not boxes:
        return cv2.resize(source_bgr, (size, size), interpolation=cv2.INTER_AREA)
    box = max(boxes, key=lambda b: b.w * b.h)
    if mode == "arcface" and box.landmarks is not None:
        return align_face(source_bgr, box.landmarks, size=size, template="arcface").crop
    crop, _ = align_crop(source_bgr, box, size=256, pad=0.25)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def eye_metrics(landmarks: np.ndarray) -> dict[str, float]:
    """Proportion measurements from five landmarks (any coordinate space)."""
    lm = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    eye_dist = float(np.linalg.norm(lm[1] - lm[0]))
    eye_mid = (lm[0] + lm[1]) / 2.0
    mouth_mid = (lm[3] + lm[4]) / 2.0
    eye_mouth = float(np.linalg.norm(mouth_mid - eye_mid))
    return {
        "eye_distance": eye_dist,
        "eye_to_mouth": eye_mouth,
        "eye_to_mouth_ratio": eye_dist / max(eye_mouth, 1e-6),
        "roll_deg": float(math.degrees(math.atan2(lm[1, 1] - lm[0, 1], lm[1, 0] - lm[0, 0]))),
    }
