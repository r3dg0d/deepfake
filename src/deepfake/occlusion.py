"""Occlusion-aware face mask. Visible face = face region minus occluder.

This slice does the mask math and a deterministic test double. It does not
download weights. A face-parsing file under the deepfake cache is noted, but
logits are not applied until a runtime can execute them and the class map is
tested. Without that, ``estimate`` returns the same ellipse ``paste_face``
already uses and reports the parser unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .composite import _oval_soft_mask
from .detect import FaceBox
from .paths import cache_home, models_dir

# CelebAMask-HQ ids, for a later parser. Not applied in this slice.
FACE_PART_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13)  # skin through lips
OCCLUDER_PART_IDS = (9, 14, 16, 17, 18)  # ear_r, neck, cloth, hair, hat

_WEIGHT_NAMES = ("bisenet_resnet_18.onnx",)


class OccluderSource(Protocol):
    """Something that marks occluding pixels. The test double implements this."""

    def occluder_mask(self, frame_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        """Float32 mask, 1 where the swap must not paint. Same HxW as the frame."""


@dataclass
class OcclusionResult:
    face_mask: np.ndarray
    occluder_mask: np.ndarray
    visible_face_mask: np.ndarray
    confidence: float
    temporal_state: Any
    parser_available: bool
    detail: str


def combine_visible_mask(face_mask: np.ndarray, occluder_mask: np.ndarray) -> np.ndarray:
    """Face region with occluder pixels removed. Both masks are float HxW in 0..1."""
    face = np.clip(np.asarray(face_mask, dtype=np.float32), 0.0, 1.0)
    occ = np.clip(np.asarray(occluder_mask, dtype=np.float32), 0.0, 1.0)
    if face.shape != occ.shape:
        raise ValueError(f"mask shape mismatch: face {face.shape} vs occluder {occ.shape}")
    return np.clip(face * (1.0 - occ), 0.0, 1.0)


def ellipse_face_mask(frame_bgr: np.ndarray, box: FaceBox, *, feather: int = 24) -> np.ndarray:
    """The current paste ellipse, placed in the full frame (zeros outside the box)."""
    fh, fw = frame_bgr.shape[:2]
    mask = np.zeros((fh, fw), dtype=np.float32)
    x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    x, y = max(0, x), max(0, y)
    w, h = x1 - x, y1 - y
    if w <= 1 or h <= 1:
        return mask
    roi = _oval_soft_mask(h, w, feather)
    mask[y:y1, x:x1] = roi[:h, :w]
    return mask


class ColorKeyOccluder:
    """Deterministic test double: pixels near ``color_bgr`` are occluders.

    This is not a face parser and does not load weights. Tests use it for a
    synthetic hand (or hair) swatch inside the ellipse.
    """

    def __init__(self, color_bgr: tuple[int, int, int], *, tolerance: int = 8) -> None:
        self.color = np.asarray(color_bgr, dtype=np.int16)
        self.tolerance = int(tolerance)

    def occluder_mask(self, frame_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        frame = np.asarray(frame_bgr)
        diff = np.abs(frame.astype(np.int16) - self.color.reshape(1, 1, 3))
        hit = np.all(diff <= self.tolerance, axis=2)
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
        fh, fw = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        if x1 <= x0 or y1 <= y0:
            return mask
        mask[y0:y1, x0:x1] = hit[y0:y1, x0:x1].astype(np.float32)
        return mask


def _weight_candidates() -> list[Path]:
    roots = [
        models_dir() / "vision",
        models_dir(),
        cache_home() / "vision",
        Path(__file__).resolve().parents[2] / "models" / "vision",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for name in _WEIGHT_NAMES:
            path = root / name
            if path in seen:
                continue
            seen.add(path)
            if path.is_file():
                found.append(path)
    return found


def _dnn_can_execute(path: Path) -> bool:
    """True only if OpenCV DNN can both load and forward the graph.

    The checked-in local BiSeNet export fails shape inference (dynamic
    AveragePool). This probe must not download anything and must not guess
    class ids from a tensor we have not tested.
    """
    try:
        import cv2
    except ImportError:
        return False
    try:
        net = cv2.dnn.readNetFromONNX(str(path))
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(blank, scalefactor=1 / 255.0, size=(64, 64), swapRB=True)
        net.setInput(blob)
        out = net.forward()
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.size > 0


class OcclusionEngine:
    """Estimate the visible-face mask for one tracked face.

    ``temporal_state`` is accepted and returned unchanged. This slice does not
    filter masks across frames, so tests may pass ``None``.
    """

    def __init__(self, parser: OccluderSource | None = None, *, feather: int = 24) -> None:
        self._parser = parser
        self.feather = feather
        self._probe_detail: str | None = None

    def estimate(
        self,
        frame: np.ndarray,
        face_track: FaceBox,
        landmarks: Any = None,
        face_mask: np.ndarray | None = None,
        temporal_state: Any = None,
    ) -> OcclusionResult:
        del landmarks  # detectors do not fill these yet; accepted for the later pipeline
        box = face_track
        if face_mask is None:
            face = ellipse_face_mask(frame, box, feather=self.feather)
        else:
            face = np.clip(np.asarray(face_mask, dtype=np.float32), 0.0, 1.0)
            if face.ndim == 3:
                face = face[:, :, 0]
            if face.shape != frame.shape[:2]:
                raise ValueError(f"face_mask shape {face.shape} != frame {frame.shape[:2]}")

        if self._parser is not None:
            occ = np.clip(np.asarray(self._parser.occluder_mask(frame, box), dtype=np.float32), 0.0, 1.0)
            if occ.shape != face.shape:
                raise ValueError(f"occluder shape {occ.shape} != face {face.shape}")
            visible = combine_visible_mask(face, occ)
            return OcclusionResult(
                face_mask=face,
                occluder_mask=occ,
                visible_face_mask=visible,
                confidence=1.0,
                temporal_state=temporal_state,
                parser_available=False,
                detail="deterministic test double; face parser unavailable (no model executed)",
            )

        detail = self._parser_status()
        occ = np.zeros(face.shape, dtype=np.float32)
        # A file on disk is not a mask. Until labels are tested, keep the ellipse.
        visible = combine_visible_mask(face, occ)
        return OcclusionResult(
            face_mask=face,
            occluder_mask=occ,
            visible_face_mask=visible,
            confidence=0.0,
            temporal_state=temporal_state,
            parser_available=False,
            detail=detail,
        )

    def _parser_status(self) -> str:
        if self._probe_detail is not None:
            return self._probe_detail
        found = _weight_candidates()
        if not found:
            self._probe_detail = "parser unavailable: no face-parsing weights installed"
            return self._probe_detail
        path = found[0]
        if _dnn_can_execute(path):
            # Reachable only if a future runtime forwards the graph. Class ids
            # stay unwired on purpose: an untested map would punch the face.
            self._probe_detail = (
                f"parser unavailable: {path.name} loaded but face-part labels are not wired"
            )
        else:
            self._probe_detail = (
                f"parser unavailable: {path.name} is installed but could not be executed"
                " (no download attempted)"
            )
        return self._probe_detail
