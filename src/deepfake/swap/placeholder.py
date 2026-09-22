"""Labeled passthrough that errors clearly when models missing."""

from __future__ import annotations

import time

import numpy as np

from .base import SwapResult


class MissingModelSwapper:
    """Used when no AlphaFace/inswapper weights are installed."""

    name = "missing"

    def __init__(self, hint: str | None = None) -> None:
        self.hint = hint or (
            "No face-swap model installed. Run:\n"
            "  deepfake models list\n"
            "  deepfake models install alphaface --yes   # Drive weights; undocumented license\n"
            "  # or fallback (NON-COMMERCIAL research):\n"
            "  deepfake models install inswapper --yes"
        )

    def set_source(self, source_bgr: np.ndarray) -> None:
        _ = source_bgr
        raise RuntimeError(self.hint)

    def swap(self, target_face_bgr: np.ndarray) -> SwapResult:
        _ = target_face_bgr
        raise RuntimeError(self.hint)


class PassthroughSwapper:
    """Debug: returns target unchanged (labeled). Requires explicit opt-in."""

    name = "passthrough"

    def __init__(self) -> None:
        self._source: np.ndarray | None = None

    def set_source(self, source_bgr: np.ndarray) -> None:
        self._source = source_bgr

    def swap(self, target_face_bgr: np.ndarray) -> SwapResult:
        t0 = time.perf_counter()
        out = target_face_bgr.copy()
        ms = (time.perf_counter() - t0) * 1000
        return SwapResult(face_bgr=out, inference_ms=ms, backend=self.name)
