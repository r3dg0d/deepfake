"""Optional learned eye restoration on generated faces, gated by current visibility."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .detect import arcface_transform
from .neural_assets import asset_path, verify_asset

FFHQ = (
    np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.725411],
            [0.61150205, 0.72490465],
        ],
        dtype=np.float32,
    )
    * 512
)


class EyeRestorer:
    def __init__(self, device: str = "cpu", strength: float = 0.65):
        import onnxruntime as ort

        if not verify_asset("gfpgan"):
            raise RuntimeError("Eye restoration needs a verified model: deepfake models install gfpgan --yes")
        self.strength = float(strength)
        if not 0 <= self.strength <= 1:
            raise ValueError("eye restoration strength must be between zero and one")
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            import torch  # load the same CUDA/cuDNN libraries as the swapper

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA eye restoration requested but CUDA is unavailable")
            # Limit arena reservations while sharing GPU memory with the swapper.
            providers.insert(0, ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        self.session = ort.InferenceSession(str(asset_path("gfpgan")), providers=providers)
        self.latency_ms = 0.0

    def restore(self, face_bgr, landmarks, eye_mask, visible_mask):
        self.latency_ms = 0.0
        if eye_mask is None or landmarks is None or self.strength == 0:
            return face_bgr
        if arcface_transform(landmarks) is None:
            return face_bgr
        shape = face_bgr.shape[:2]
        eyes = np.asarray(eye_mask, np.float32)
        visible = np.asarray(visible_mask, np.float32)
        if not np.isfinite(eyes).all() or not np.isfinite(visible).all():
            return face_bgr
        eyes = cv2.resize(eyes, shape[::-1], interpolation=cv2.INTER_NEAREST)
        visible = cv2.resize(visible, shape[::-1], interpolation=cv2.INTER_NEAREST)
        # Soften the detail transition inward; no restoration outside a current eye mask.
        gate = cv2.GaussianBlur(np.clip(eyes, 0, 1), (0, 0), 0.8)
        gate *= (eyes > 0) & (visible >= 0.8)
        gate *= self.strength
        if not np.any(gate):
            return face_bgr
        matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32), FFHQ, method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():
            return face_bgr
        start = time.perf_counter()
        crop = cv2.warpAffine(face_bgr, matrix, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        tensor = np.ascontiguousarray((crop[..., ::-1].astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None])
        try:
            output = self.session.run(None, {"input": tensor})[0]
        except Exception as error:
            message = str(error).lower()
            memory_failure = any(
                s in message for s in ("failed to allocate memory", "out of memory", "cuda_error_out_of_memory")
            )
            if not memory_failure or "CUDAExecutionProvider" not in self.session.get_providers():
                raise
            import sys

            import onnxruntime as ort

            print(
                "deepfake: eye restoration switched to CPU after GPU memory exhaustion; inference will be slower",
                file=sys.stderr,
            )
            del self.session
            self.session = ort.InferenceSession(str(asset_path("gfpgan")), providers=["CPUExecutionProvider"])
            output = self.session.run(None, {"input": tensor})[0]
        if output.shape != (1, 3, 512, 512) or not np.isfinite(output).all():
            return face_bgr
        restored = np.clip((output[0].transpose(1, 2, 0)[..., ::-1] + 1) * 127.5, 0, 255)
        restored = cv2.warpAffine(restored, cv2.invertAffineTransform(matrix), shape[::-1], flags=cv2.INTER_CUBIC)
        delta = np.clip(restored - face_bgr.astype(np.float32), -32, 32)
        result = np.clip(face_bgr.astype(np.float32) + delta * gate[..., None], 0, 255).round().astype(np.uint8)
        self.latency_ms = (time.perf_counter() - start) * 1000
        return result
