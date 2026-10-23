"""InsightFace inswapper fallback (NON-COMMERCIAL model terms)."""

from __future__ import annotations

import time

import numpy as np

from .base import SwapResult


class InswapperSwapper:
    name = "inswapper"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.on_gpu = False
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
                "pip install insightface onnxruntime-gpu  # NON-COMMERCIAL research models"
            ) from e
        from ..ort_util import providers, session_on_gpu
        from ..paths import models_dir

        root = models_dir() / "inswapper"
        onnx = root / "inswapper_128.onnx"
        if not onnx.is_file() or not (root / "models" / "buffalo_l" / "det_10g.onnx").is_file():
            raise RuntimeError("inswapper weights missing. Run: deepfake models install inswapper --yes")
        prov = providers(self.device)
        # Files are the verified copies under deepfake's model dir; insightface must not download anything.
        self._app = FaceAnalysis(name="buffalo_l", root=str(root), providers=prov,
                                 allowed_modules=["detection", "recognition", "landmark_2d_106"])
        self._app.prepare(ctx_id=0 if self.device != "cpu" else -1, det_size=(640, 640))
        self._swapper = insightface.model_zoo.get_model(str(onnx), providers=prov)
        self.on_gpu = session_on_gpu(self._swapper.session)
        if self.device != "cpu" and not self.on_gpu:
            import sys

            print("deepfake: warning: inswapper is running on the CPU (onnxruntime CUDA provider unavailable; "
                  "install an onnxruntime-gpu build that matches your CUDA runtime)", file=sys.stderr)

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
