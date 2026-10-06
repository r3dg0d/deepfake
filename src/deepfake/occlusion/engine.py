"""Current-frame occluders always win; smoothing never reveals hidden pixels."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MaskState:
    gray: np.ndarray
    visible: np.ndarray
    face: np.ndarray
    age: int = 0
    confidence: float = 0.0


@dataclass
class MaskEstimate:
    face_mask: np.ndarray
    occluder_mask: np.ndarray
    visible_mask: np.ndarray
    confidence: float
    temporal_state: MaskState | None
    status: str
    latency_ms: float


class OcclusionEngine:
    def __init__(self, device: str = "cpu", *, models: Any = None, parser_interval: int = 2) -> None:
        self.models = models
        self.error = ""
        self.parser_interval = max(1, parser_interval)
        if self.models is None:
            try:
                from .models import SemanticMasks

                self.models = SemanticMasks(device)
            except (ImportError, RuntimeError, OSError) as e:
                self.error = str(e)
        self.backend = getattr(self.models, "backend", "preserve-original (models unavailable)")

    def estimate(
        self,
        frame: np.ndarray,
        face_track=None,
        landmarks=None,
        face_mask=None,
        temporal_state: MaskState | None = None,
    ) -> MaskEstimate:
        """Estimate masks in the supplied aligned face crop's coordinates.

        ``face_track``/``landmarks`` are reserved for optional conditioning. They
        are not invented when the detector cannot measure them. XSeg is refreshed
        on EVERY crop; semantic-only adapters can propagate parsing between refreshes.
        Mouth-aware adapters refresh parsing every frame to follow articulation.
        """
        import cv2

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        zeros = np.zeros((h, w), np.float32)
        if self.models is None:
            return MaskEstimate(
                zeros,
                np.ones_like(zeros),
                zeros,
                0.0,
                None,
                "preserve-original: " + self.error,
                (time.perf_counter() - t0) * 1000,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev = temporal_state
        face, confidence = None, 0.0
        mouth = zeros.copy()
        detailed = hasattr(self.models, "parse_details")
        warped_visible = None
        age = 0 if prev is None else prev.age + 1
        if prev is not None and prev.gray.shape == gray.shape:
            # Backward flow samples the old crop in CURRENT coordinates. Unlike a
            # plain EMA it follows motion; appearance/cut mismatch forces refresh.
            delta = float(np.mean(cv2.absdiff(gray, prev.gray)))
            if delta < 35:
                small = cv2.resize(gray, (128, 128))
                old_small = cv2.resize(prev.gray, (128, 128))
                flow = cv2.calcOpticalFlowFarneback(small, old_small, None, 0.5, 2, 13, 2, 5, 1.1, 0)
                flow = cv2.resize(flow, (w, h)) * np.array([w / 128, h / 128], np.float32)
                yy, xx = np.indices((h, w), dtype=np.float32)
                mx, my = xx + flow[:, :, 0], yy + flow[:, :, 1]
                warped_visible = cv2.remap(prev.visible, mx, my, cv2.INTER_LINEAR)
                if age % self.parser_interval and not detailed:
                    face = cv2.remap(prev.face, mx, my, cv2.INTER_LINEAR)
            else:
                prev = None
                age = 0
        try:
            if face is None:
                if detailed:
                    # Speech changes locally even when whole-crop motion is small.
                    # Refresh oral detail every frame; never propagate an old mouth.
                    face, confidence, mouth = self.models.parse_details(frame)
                else:
                    face, confidence = self.models.parse(frame)
            else:
                confidence = prev.confidence * 0.98
            current = self.models.visible(frame)
            if (
                face.shape != (h, w)
                or current.shape != (h, w)
                or not np.isfinite(face).all()
                or not np.isfinite(current).all()
                or mouth.shape != (h, w)
                or not np.isfinite(mouth).all()
                or not np.isfinite(confidence)
            ):
                raise ValueError("invalid mask output")
        except Exception as e:
            return MaskEstimate(
                zeros,
                np.ones_like(zeros),
                zeros,
                0.0,
                None,
                "preserve-original: mask inference failed: " + str(e),
                (time.perf_counter() - t0) * 1000,
            )
        face = np.nan_to_num(np.clip(face, 0, 1)).astype(np.float32)
        current = np.nan_to_num(np.clip(current, 0, 1)).astype(np.float32)
        if face_mask is not None:
            face = np.minimum(face, np.clip(face_mask, 0, 1))
        # Keep uncertain pixels and a small boundary margin from the TARGET.
        support = ((face > 0.75) & (current > 0.85)).astype(np.uint8)
        support = cv2.erode(support, np.ones((5, 5), np.uint8))
        # A clipped Gaussian starts near 0.7 at a hard boundary. An inward
        # distance ramp starts near zero and cannot spill onto hidden pixels.
        padded = np.pad(support, 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        width = max(3.0, min(h, w) * 8.0 / 256)
        ramp = np.clip(distance / width, 0, 1)
        alpha = ramp * ramp * (3 - 2 * ramp)
        # Preserve measured target articulation (including teeth/tongue) rather
        # than combining two differently shaped mouths. This only removes alpha;
        # it never fills XSeg holes or reveals an occluder.
        if np.any(mouth > 0.5):
            oral = cv2.dilate((mouth > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
            outside = cv2.distanceTransform(1 - oral, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            oral_width = max(3.0, min(h, w) * 6.0 / 256)
            oral_ramp = np.clip(outside / oral_width, 0, 1)
            alpha *= oral_ramp * oral_ramp * (3 - 2 * oral_ramp)
        if warped_visible is not None:
            # Immediate disappearance, short recovery only. Never blend old face
            # over a newly observed occluder, including one entering this frame.
            alpha = np.minimum(alpha, alpha * 0.85 + warped_visible * 0.15)
        confidence = float(min(confidence, np.mean(np.maximum(current, 1 - current))))
        coverage = float(np.mean(support))
        if confidence < 0.50 or coverage < 0.025:
            alpha[:] = 0
            status = "preserve-original (uncertain/full occlusion)"
        else:
            status = "occlusion active" if np.any((face > 0.5) & (current < 0.8)) else "visible face"
        state = MaskState(gray, alpha.copy(), face.copy(), age, confidence)
        return MaskEstimate(
            face, np.clip(face - alpha, 0, 1), alpha, confidence, state, status, (time.perf_counter() - t0) * 1000
        )
