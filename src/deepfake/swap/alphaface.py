"""AlphaFace backend — vendor checkout + Drive weights."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..paths import models_dir, vendor_dir
from .base import SwapResult


def _normalize_by_127_5(img):
    # Match upstream eval.py (expects float tensor in [0,1] then maps to [-1,1])
    img = (img * 255.0).int()
    return (img / 127.5) - 1.0


class AlphaFaceSwapper:
    """Realtime-ish wrapper around upstream build_AlphaFace().Swapper."""

    name = "alphaface"

    def __init__(self, device: str = "cuda", precision: str = "fp32") -> None:
        self.device = device if device != "auto" else ("cuda" if self._cuda_ok() else "cpu")
        if precision not in ("fp32", "bf16"):
            # fp16 is rejected on purpose: measured mean abs error 0.11-0.13 vs fp32
            # on this model (AdaIN statistics overflow), bf16 stays at ~0.016.
            raise ValueError("AlphaFace precision must be fp32 or bf16")
        self.precision = precision
        self._source_tensor = None
        self._id_code = None
        self._model = None
        self._vendor = vendor_dir()
        self._weights = models_dir() / "alphaface"
        self._load()

    @staticmethod
    def _cuda_ok() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _ensure_layout(self) -> Path:
        vendor = self._vendor
        if not vendor.is_dir():
            raise RuntimeError(
                f"AlphaFace vendor missing at {vendor}. "
                "Run: deepfake models install alphaface --yes"
            )
        demo = self._weights / "alphaface_demo.pt"
        arc = self._weights / "arcface.pt"
        if not demo.is_file() or demo.stat().st_size < 10_000_000:
            raise RuntimeError(
                f"AlphaFace weights missing/incomplete at {demo}. "
                "Re-run download or: deepfake models install alphaface --yes"
            )
        if not arc.is_file() or arc.stat().st_size < 10_000_000:
            raise RuntimeError(
                f"ArcFace weights missing/incomplete at {arc}. "
                "Re-run download or: deepfake models install alphaface --yes"
            )

        # Upstream build_AlphaFace hardcodes relative paths under the vendor root.
        models = vendor / "Models"
        models.mkdir(parents=True, exist_ok=True)
        link_arc = models / "arcface_w600k_r50_pytorch.pt"
        if link_arc.is_symlink() or link_arc.exists():
            if link_arc.resolve() != arc.resolve():
                link_arc.unlink()
                link_arc.symlink_to(arc)
        else:
            link_arc.symlink_to(arc)
        link_demo = vendor / "alphaface_demo.pt"
        if link_demo.is_symlink() or link_demo.exists():
            if link_demo.resolve() != demo.resolve():
                link_demo.unlink()
                link_demo.symlink_to(demo)
        else:
            link_demo.symlink_to(demo)
        return vendor

    def _load(self) -> None:
        import torch
        from easydict import EasyDict as edict

        vendor = self._ensure_layout()
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

        # Upstream uses relative ./Models/... paths
        prev = Path.cwd()
        os.chdir(vendor)
        try:
            from Models.Swapper_AlphaFace import build_AlphaFace

            cfg = edict()
            cfg.model_path = str(vendor / "alphaface_demo.pt")
            cfg.id_network_path = ""
            cfg.id_network = "vit_b"

            # Skip discriminator/VGG (training-only) for inference. Upstream print()s
            # progress; keep stdout clean for --json consumers.
            with contextlib.redirect_stdout(sys.stderr):
                model = build_AlphaFace(config=cfg, fine_tune=False, adv_train=False, new_id_model=False)
            ckpt = torch.load(cfg.model_path, map_location="cpu")
            if isinstance(ckpt, dict) and "swapper" in ckpt:
                model.Swapper.load_state_dict(ckpt["swapper"])
            else:
                model.Swapper.load_state_dict(ckpt)

            if self.device.startswith("cuda") and torch.cuda.is_available():
                model = model.cuda()
                torch_device = torch.device("cuda")
            else:
                # Upstream AlphaFace() forces .cuda() in __init__; if that worked we are on GPU.
                torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if torch_device.type == "cpu":
                    raise RuntimeError(
                        "AlphaFace upstream requires CUDA (model constructors call .cuda()). "
                        "Use a CUDA torch build on this 4090 host."
                    )

            model.Swapper.eval()
            model.Id_encoder.eval()
            self._model = model
            self._torch = torch
            self._device = torch_device
        finally:
            os.chdir(prev)

    def set_source(self, source_bgr: np.ndarray) -> None:
        torch = self._torch
        # Prefer a face crop of the identity image so ArcFace gets face, not background.
        face = source_bgr
        try:
            from ..detect import align_crop, create_detector

            boxes = create_detector("auto").detect(source_bgr)
            if boxes:
                boxes = sorted(boxes, key=lambda b: b.w * b.h, reverse=True)
                face, _ = align_crop(source_bgr, boxes[0], size=256, pad=0.25)
        except Exception:
            face = source_bgr
        # Source identity path: 112×112, normalized like eval.py s_transform
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        ten = _normalize_by_127_5(ten).unsqueeze(0).to(self._device)
        self._source_tensor = ten
        # The ResNet-50 identity encoder only depends on the source face, so
        # run it once here instead of on every frame (upstream forward() does).
        with torch.inference_mode():
            self._id_code = self._model.get_id_code(ten)

    def swap(self, target_face_bgr: np.ndarray) -> SwapResult:
        if self._source_tensor is None:
            raise RuntimeError("call set_source() before swap()")
        torch = self._torch
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(target_face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        target = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        target = target.unsqueeze(0).to(self._device)

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16"
        ):
            out = self._model.Swapper(target, self._id_code)
            if out.dim() == 4:
                out = out[0]
            if out.shape[0] in {1, 3, 4}:
                out = out.permute(1, 2, 0)
            if float(out.max()) <= 1.5:
                out = out * 255.0
            out = out.float().clamp(0, 255).byte().detach().cpu().numpy()

        if out.ndim == 2:
            face = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        else:
            face = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # Match pipeline crop size (usually 256 already)
        if face.shape[0] != target_face_bgr.shape[0] or face.shape[1] != target_face_bgr.shape[1]:
            face = cv2.resize(
                face,
                (target_face_bgr.shape[1], target_face_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        ms = (time.perf_counter() - t0) * 1000
        return SwapResult(face_bgr=face, inference_ms=ms, backend=self.name)
