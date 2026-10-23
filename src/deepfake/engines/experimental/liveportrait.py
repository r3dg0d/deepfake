"""LivePortrait head reenactment through official neural model classes.

Single-subject, Haar-aligned experimental integration. Hair inside the head crop
is generated; hand occlusions and background reconstruction are not segmented.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

from ...detect import OpenCVHaarDetector, align_crop
from ...ipc import read_json
from ...metrics import MetricsTracker
from ...paths import models_dir
from ..base import EngineCapabilities, EngineUnavailable, TransformationEngine
from .adapters import SOURCES, settings_path


class LivePortraitEngine(TransformationEngine):
    def load(self, config):
        from ...models import verify_model

        self.config = config
        settings = read_json(settings_path()).get("liveportrait", {})
        checkout = Path(settings.get("checkout", "/nonexistent"))
        if not checkout.is_dir():
            raise EngineUnavailable("Configure LivePortrait first: deepfake experimental configure --backend liveportrait "
                                    "--checkout /path/to/LivePortrait --python /path/to/python")
        import subprocess

        revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if revision != SOURCES["liveportrait"][1]:
            raise EngineUnavailable("LivePortrait checkout does not match the pinned revision")
        if not verify_model("liveportrait")["ok"]:
            raise EngineUnavailable("LivePortrait weights missing or corrupt: deepfake models install liveportrait --yes")
        try:
            import torch

            # Alias the upstream src package: never shadow Deepfake or another
            # optional repository's generic `src` module in this interpreter.
            name = "_deepfake_liveportrait"
            package = types.ModuleType(name)
            package.__path__ = [str(checkout / "src")]
            sys.modules[name] = package
            wrapper = importlib.import_module(name + ".live_portrait_wrapper")
            config_module = importlib.import_module(name + ".config.inference_config")
            camera = importlib.import_module(name + ".utils.camera")
            self.rotation = camera.get_rotation_matrix
            root = models_dir() / "liveportrait" / "liveportrait"
            kwargs = {f"checkpoint_{key}": str(root / subdir / filename) for key, subdir, filename in (
                ("F", "base_models", "appearance_feature_extractor.pth"),
                ("M", "base_models", "motion_extractor.pth"),
                ("W", "base_models", "warping_module.pth"),
                ("G", "base_models", "spade_generator.pth"),
                ("S", "retargeting_models", "stitching_retargeting_module.pth"))}
            # All upstream torch.load calls must retain restricted deserialization.
            os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
            cpu = str(config.device) == "cpu"
            device_id = int(str(config.device).split(":")[1]) if ":" in str(config.device) else 0
            self.model = wrapper.LivePortraitWrapper(config_module.InferenceConfig(
                **kwargs, flag_force_cpu=cpu, device_id=device_id, flag_use_half_precision=not cpu,
                flag_do_torch_compile=False))
            self.torch = torch
        except ImportError as e:
            raise EngineUnavailable(f"LivePortrait dependency unavailable ({e}). Install the experimental-head extra.") from e
        self.detector = OpenCVHaarDetector()
        self.metrics = MetricsTracker()
        self.first_driver = None
        self.last_box = None
        self.tracking_lost = False
        self.sequence = 0

    def _crop(self, image, *, reference=False):
        boxes = self.detector.detect(image)
        if len(boxes) > 1:
            raise EngineUnavailable("LivePortrait currently requires exactly one visible person; stopping to avoid subject switching")
        if not boxes:
            if reference:
                raise EngineUnavailable("No frontal face detected in the reference image")
            if self.last_box is not None:
                self.tracking_lost = True
            return None, None
        box = boxes[0]
        if not reference and self.tracking_lost:
            raise EngineUnavailable("Tracking was lost. Restart to select a subject explicitly; automatic reassignment is disabled")
        if self.last_box is not None and not reference:
            old = self.last_box
            displacement = np.hypot(box.x - old.x, box.y - old.y) / max(old.w, 1)
            if displacement > 1.5:
                raise EngineUnavailable("Subject left the tracked region; restart the session to select a subject explicitly")
        if not reference:
            self.last_box = box
        return align_crop(image, box, size=256, pad=0.8)

    def prepare_reference(self, reference):
        import cv2

        crop, _ = self._crop(reference, reference=True)
        with self.torch.inference_mode():
            source = self.model.prepare_source(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            self.source_info = self.model.get_kp_info(source)
            self.source_kp = self.model.transform_keypoint(self.source_info)
            self.feature = self.model.extract_feature_3d(source)
            self.source_rotation = self.rotation(self.source_info["pitch"], self.source_info["yaw"], self.source_info["roll"])
        self.first_driver = None

    def process_frame(self, frame, context):
        return self.swap_frame(frame)

    def swap_frame(self, frame):
        import cv2

        start = time.perf_counter()
        crop, box = self._crop(frame)
        self.sequence += 1
        if crop is None:
            return frame, self.metrics.record(inference_ms=0, total_ms=(time.perf_counter() - start) * 1000)
        infer_start = time.perf_counter()
        with self.torch.inference_mode():
            driver = self.model.get_kp_info(self.model.prepare_source(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            rotation = self.rotation(driver["pitch"], driver["yaw"], driver["roll"])
            if self.first_driver is None:
                self.first_driver = driver
                self.first_rotation = rotation
            source, initial = self.source_info, self.first_driver
            relative_rotation = (rotation @ self.first_rotation.permute(0, 2, 1)) @ self.source_rotation
            expression = source["exp"] + driver["exp"] - initial["exp"]
            scale = source["scale"] * driver["scale"] / initial["scale"]
            translation = source["t"] + driver["t"] - initial["t"]
            keypoints = scale[..., None] * (source["kp"] @ relative_rotation + expression)
            keypoints[:, :, :2] += translation[:, None, :2]
            keypoints = self.model.stitching(self.source_kp, keypoints)
            generated = self.model.warp_decode(self.feature, self.source_kp, keypoints)
            output = self.model.parse_output(generated["out"])[0]
        infer_ms = (time.perf_counter() - infer_start) * 1000
        # Blend the complete generated head crop. This is not an occlusion mask.
        x, y, w, h = box.x, box.y, box.w, box.h
        generated_bgr = cv2.resize(cv2.cvtColor(output, cv2.COLOR_RGB2BGR), (w, h))
        mask = np.zeros((h, w), dtype=np.float32)
        margin = max(1, min(w, h) // 16)
        mask[margin:-margin, margin:-margin] = 1
        mask = cv2.GaussianBlur(mask, (0, 0), max(1, margin / 2))[..., None]
        out = frame.copy()
        out[y:y+h, x:x+w] = np.clip(generated_bgr * mask + frame[y:y+h, x:x+w] * (1-mask), 0, 255).astype(np.uint8)
        return out, self.metrics.record(inference_ms=infer_ms, total_ms=(time.perf_counter() - start) * 1000)

    def capabilities(self):
        return EngineCapabilities(supports_head=True, supports_streaming=True, supports_cuda=True)

    def unload(self):
        for attr in ("feature", "source_info", "source_kp", "first_driver", "model"):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    set_source = prepare_reference
    close = unload
