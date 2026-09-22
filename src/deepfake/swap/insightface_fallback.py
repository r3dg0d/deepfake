"""InsightFace inswapper fallback (NON-COMMERCIAL model terms)."""

from __future__ import annotations

import time

import numpy as np

from .base import SwapResult


class InswapperSwapper:
    name = "inswapper"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._source_face = None
        self._app = None
        self._swapper = None
        self._load()

    def _load(self) -> None:
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "insightface not installed. "
                "pip install insightface onnxruntime  # NON-COMMERCIAL research models"
            ) from e
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=0 if self.device != "cpu" else -1, det_size=(640, 640))
        try:
            self._swapper = insightface.model_zoo.get_model(
                "inswapper_128.onnx", download=False, download_zip=False
            )
        except Exception:
            # Allow download only if user already ran models install --yes
            self._swapper = insightface.model_zoo.get_model("inswapper_128.onnx")

    def set_source(self, source_bgr: np.ndarray) -> None:
        faces = self._app.get(source_bgr)
        if not faces:
            raise RuntimeError("no face found in source image for inswapper")
        self._source_face = faces[0]

    def swap(self, target_face_bgr: np.ndarray) -> SwapResult:
        if self._source_face is None:
            raise RuntimeError("call set_source() first")
        t0 = time.perf_counter()
        faces = self._app.get(target_face_bgr)
        if not faces:
            ms = (time.perf_counter() - t0) * 1000
            return SwapResult(face_bgr=target_face_bgr, inference_ms=ms, backend=self.name)
        out = target_face_bgr.copy()
        for face in faces:
            out = self._swapper.get(out, face, self._source_face, paste_back=True)
        ms = (time.perf_counter() - t0) * 1000
        return SwapResult(face_bgr=out, inference_ms=ms, backend=self.name)
