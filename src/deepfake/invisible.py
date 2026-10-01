"""Optional Adobe TrustMark Q frame watermark, SHA-256 verified and weights-only loaded.

This is a public disclosure signal, NOT a signature, NOT SynthID, and not
assumed to survive arbitrary transformations. Verify the encoded output.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .neural_assets import asset_path, verify_asset

MESSAGE = "DFv1AI"


def ready() -> bool:
    return importlib.util.find_spec("trustmark") is not None and all(
        verify_asset(k) for k in ("trustmark-config", "trustmark-encoder", "trustmark-decoder")
    )


class InvisibleMarker:
    def __init__(self, device: str = "cuda") -> None:
        if not ready():
            raise RuntimeError("Install [watermark] and run deepfake models install trustmark --yes")
        import torch
        from omegaconf import OmegaConf
        from trustmark import TrustMark
        from trustmark.model import instantiate_from_config

        class PinnedTrustMark(TrustMark):
            def load_model(self, config_path, weight_path, device, secret_len, part="all"):
                if part not in ("encoder", "decoder"):
                    raise RuntimeError("only disclosure embedding and detection are enabled")
                # Ignore vendor-computed package paths. Verified config/weights are
                # in the controlled model cache; downloads never occur here.
                config = OmegaConf.load(asset_path("trustmark-config")).model
                other = "decoder" if part == "encoder" else "encoder"
                config.params[f"secret_{other}_config"].target = "trustmark.model.Identity"
                for key in ("discriminator_config", "loss_config", "noise_config"):
                    config.params[key].target = "trustmark.model.Identity"
                model = instantiate_from_config(config)
                state = torch.load(asset_path("trustmark-" + part), map_location="cpu", weights_only=True)
                model.load_state_dict(state.get("state_dict", state), strict=False)
                return model.to(device).eval()

        self.tm = PinnedTrustMark(
            verbose=False, model_type="Q", device=device, loadRemover=False, loadBBoxDetector=False
        )

    def encode(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        encoded = self.tm.encode(Image.fromarray(bgr[:, :, ::-1]), MESSAGE, MODE="text")
        return np.ascontiguousarray(np.asarray(encoded)[:, :, ::-1])

    def decode(self, bgr: np.ndarray) -> bool:
        from PIL import Image

        message, present, _schema = self.tm.decode(Image.fromarray(bgr[:, :, ::-1]), MODE="text")
        return bool(present and message.rstrip("\x00") == MESSAGE)

    def verify_video(self, path: Path, samples: int = 10) -> dict:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            points = np.linspace(0, max(0, total - 1), min(samples, max(1, total)), dtype=int)
            checked = found = 0
            for point in points:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(point))
                ok, frame = cap.read()
                if ok:
                    checked += 1
                    found += int(self.decode(frame))
            return {
                "implementation": "Adobe TrustMark Q",
                "schema": "deepfake-disclosure-v1",
                "payload": MESSAGE,
                "samples": checked,
                "detected_samples": found,
                "present": checked > 0 and found >= max(1, checked // 2),
            }
        finally:
            cap.release()


class MarkedSink:
    def __init__(self, sink, marker: InvisibleMarker) -> None:
        self.sink, self.marker = sink, marker

    def write(self, frame: np.ndarray) -> None:
        self.sink.write(self.marker.encode(frame))

    def close(self) -> None:
        self.sink.close()
