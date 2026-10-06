"""Independent ONNX inference adapters, no upstream Python code execution."""

from __future__ import annotations

import numpy as np

from ..neural_assets import asset_path, verify_asset

FACE_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)


class SemanticMasks:
    def __init__(self, device: str = "cpu") -> None:
        import onnxruntime as ort

        for name in ("bisenet", "xseg"):
            if not verify_asset(name):
                raise RuntimeError(f"missing/unverified {name}; deepfake models install {name} --yes")
        self.device_id = int(device.split(":")[-1]) if ":" in device else 0
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            # Preloading PyTorch makes its matching CUDA/cuDNN shared libraries available.
            import torch  # noqa: F401

            providers.insert(0, ("CUDAExecutionProvider", {"device_id": self.device_id}))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3
        self.parser = ort.InferenceSession(str(asset_path("bisenet")), opts, providers=providers)
        xseg_source = str(asset_path("xseg"))
        if device.startswith("cuda"):
            try:
                from .onnx_opt import cuda_xseg_bytes

                xseg_source = cuda_xseg_bytes(asset_path("xseg"))
            except ImportError:
                pass  # correctness unchanged; doctor reports runtime provider
        self.occluder = ort.InferenceSession(xseg_source, opts, providers=providers)
        self.backend = "bisenet+xseg / " + ",".join(self.parser.get_providers())
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def parse(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        face, confidence, _ = self.parse_details(crop)
        return face, confidence

    def parse_details(self, crop: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        face, confidence, mouth, _ = self.parse_regions(crop)
        return face, confidence, mouth

    def parse_regions(self, crop: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Face, mouth and eye regions from one current-frame inference."""
        import cv2

        rgb = cv2.resize(crop, (512, 512))[:, :, ::-1].astype(np.float32) / 255
        inp = np.ascontiguousarray(((rgb - self.mean) / self.std).transpose(2, 0, 1)[None])
        if "CUDAExecutionProvider" in self.parser.get_providers():
            import torch
            import torch.nn.functional as F

            if not hasattr(self, "_logits"):
                self._logits = torch.empty((1, 19, 512, 512), device=f"cuda:{self.device_id}", dtype=torch.float32)
            binding = self.parser.io_binding()
            binding.bind_cpu_input(self.parser.get_inputs()[0].name, inp)
            binding.bind_output(
                self.parser.get_outputs()[0].name,
                "cuda",
                self.device_id,
                np.float32,
                tuple(self._logits.shape),
                self._logits.data_ptr(),
            )
            self.parser.run_with_iobinding(binding)
            binding.synchronize_outputs()
            labels = self._logits.argmax(1)
            face = torch.zeros_like(labels, dtype=torch.bool)
            for cls in FACE_CLASSES:
                face |= labels == cls
            confidence_map = self._logits.softmax(1).amax(1)
            confidence = float(confidence_map[face].mean().item()) if bool(face.any()) else 0.0
            mask = F.interpolate(face[:, None].float(), size=crop.shape[:2], mode="nearest")
            mouth = (labels == 11) | (labels == 12) | (labels == 13)
            mouth = F.interpolate(mouth[:, None].float(), size=crop.shape[:2], mode="nearest")
            eyes = (labels == 4) | (labels == 5)
            eyes = F.interpolate(eyes[:, None].float(), size=crop.shape[:2], mode="nearest")
            return mask[0, 0].cpu().numpy(), confidence, mouth[0, 0].cpu().numpy(), eyes[0, 0].cpu().numpy()
        logits = self.parser.run([self.parser.get_outputs()[0].name], {self.parser.get_inputs()[0].name: inp})[0][0]
        labels = logits.argmax(0)
        shifted = logits - logits.max(0)
        certainty = 1 / np.exp(shifted).sum(0)
        face = np.isin(labels, FACE_CLASSES)
        confidence = float(certainty[face].mean()) if face.any() else 0.0
        mouth = np.isin(labels, (11, 12, 13))
        eyes = np.isin(labels, (4, 5))
        return (
            cv2.resize(face.astype(np.float32), crop.shape[1::-1], interpolation=cv2.INTER_NEAREST),
            confidence,
            cv2.resize(mouth.astype(np.float32), crop.shape[1::-1], interpolation=cv2.INTER_NEAREST),
            cv2.resize(eyes.astype(np.float32), crop.shape[1::-1], interpolation=cv2.INTER_NEAREST),
        )

    def visible(self, crop: np.ndarray) -> np.ndarray:
        import cv2

        inp = cv2.resize(crop, (256, 256)).astype(np.float32)[None] / 255
        result = self.occluder.run([self.occluder.get_outputs()[0].name], {self.occluder.get_inputs()[0].name: inp})[0][
            0, :, :, 0
        ]
        return cv2.resize(np.clip(result, 0, 1), crop.shape[1::-1])
