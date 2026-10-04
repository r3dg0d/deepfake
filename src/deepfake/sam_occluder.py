"""Optional SAM 2.1 Hiera-Tiny occluder.

The checkpoint is the official Meta release asset. The ``sam2`` package is not
imported unless it is already installed. A missing checkpoint or package leaves
the BiSeNet + XSeg mask unchanged and does not stop the webcam.

A box prompt returns the main object, which should be the face. Pixels inside
the face crop that are BiSeNet skin and outside that object are an occluder
only when the object still covers most of the crop and the hole is a minority
of the skin. Anything else is ignored. A mask that would delete the face is
worse than no SAM.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .paths import models_dir

SAM21_TINY_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM21_TINY_BYTES = 156_008_466
SAM21_TINY_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SAM21_TINY_NAME = "sam2.1_hiera_tiny.pt"
# Official README speed is GPU. On CPU, do not run every frame.
SAM_INTERVAL = 8
# The box prompt must still cover most of the face crop.
SAM_MIN_COVERAGE = 0.55
# A hole larger than this fraction of BiSeNet skin is treated as "eating the face".
SAM_MAX_HOLE_FRACTION = 0.35
SAM_MIN_HOLE_PX = 32.0


def checkpoint_path() -> Path:
    return models_dir() / "vision" / SAM21_TINY_NAME


def sam2_package_available() -> bool:
    return importlib.util.find_spec("sam2") is not None


def sam2_runtime_status() -> tuple[bool, str]:
    """(runnable, doctor line). False does not mean the webcam should stop."""
    path = checkpoint_path()
    if not path.is_file():
        return (
            False,
            "SAM 2.1 Hiera-Tiny checkpoint not installed; occlusion stays BiSeNet+XSeg (webcam still runs)",
        )
    if path.stat().st_size != SAM21_TINY_BYTES:
        return (
            False,
            f"SAM 2.1 checkpoint {path.name} is {path.stat().st_size} bytes, expected {SAM21_TINY_BYTES}; "
            "not using it. Occlusion stays BiSeNet+XSeg",
        )
    if not sam2_package_available():
        return (
            False,
            f"SAM 2.1 Hiera-Tiny checkpoint present ({SAM21_TINY_BYTES} bytes) but the sam2 package "
            "is not installed; occlusion stays BiSeNet+XSeg (webcam still runs). "
            "The optional extra pulls torch>=2.5.1 and was not installed in this environment.",
        )
    return (
        True,
        f"SAM 2.1 Hiera-Tiny {path.name} importable; box-prompt hole check every {SAM_INTERVAL} frames. "
        "A mask that does not cover the face is discarded.",
    )


def sam_hole_occluder(sam_keep: np.ndarray, skin: np.ndarray) -> np.ndarray | None:
    """Skin pixels the box prompt left out, or None if that would eat the face.

    ``sam_keep`` is 1 on the main object inside the face crop. ``skin`` is 1 on
    BiSeNet skin. Both are HxW.
    """
    keep = np.clip(np.asarray(sam_keep, dtype=np.float32), 0.0, 1.0)
    skin_m = np.clip(np.asarray(skin, dtype=np.float32), 0.0, 1.0)
    if keep.shape != skin_m.shape or keep.ndim != 2:
        return None
    if keep.size == 0 or float(keep.mean()) < SAM_MIN_COVERAGE:
        return None
    skin_on = skin_m > 0.5
    skin_area = float(skin_on.sum())
    if skin_area < SAM_MIN_HOLE_PX:
        return None
    hole = skin_on & (keep < 0.5)
    hole_area = float(hole.sum())
    if hole_area < SAM_MIN_HOLE_PX or hole_area > SAM_MAX_HOLE_FRACTION * skin_area:
        return None
    out = np.zeros(keep.shape, dtype=np.float32)
    out[hole] = 1.0
    return out


class Sam2BoxOccluder:
    """Lazy SAM 2.1 image predictor. Failures disable it for this process."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._predictor = None
        self._device = ""
        self.error = ""

    def keep_mask(self, frame_bgr: np.ndarray, face_xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
        """Face-crop matte for an expanded box prompt, or None if SAM cannot be trusted."""
        if self.error:
            return None
        try:
            predictor = self._load()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._predictor = None
            return None
        if predictor is None:
            return None
        import cv2

        x, y, x1, y1 = face_xyxy
        fh, fw = frame_bgr.shape[:2]
        cx, cy = (x + x1) / 2.0, (y + y1) / 2.0
        bw, bh = max(2.0, (x1 - x) * 1.25), max(2.0, (y1 - y) * 1.25)
        ex0, ey0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
        ex1, ey1 = min(fw, int(cx + bw / 2)), min(fh, int(cy + bh / 2))
        if ex1 - ex0 < 2 or ey1 - ey0 < 2:
            return None
        crop = frame_bgr[ey0:ey1, ex0:ex1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        box = np.array([x - ex0, y - ey0, x1 - ex0, y1 - ey0], dtype=np.float32)
        try:
            predictor.set_image(rgb)
            masks, scores, _logits = predictor.predict(box=box, multimask_output=True)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return None
        if masks is None or len(masks) == 0:
            return None
        chosen = np.asarray(masks[int(np.argmax(scores))], dtype=np.float32)
        if chosen.ndim == 3:
            chosen = chosen[0]
        # Map back onto the original face crop. Corners of the expanded box are not occluders.
        fy0, fx0 = y - ey0, x - ex0
        fy1, fx1 = fy0 + (y1 - y), fx0 + (x1 - x)
        face = chosen[fy0:fy1, fx0:fx1]
        if face.shape != (y1 - y, x1 - x):
            return None
        return np.clip(face, 0.0, 1.0)

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(self.path), device=self._device)
        self._predictor = SAM2ImagePredictor(model)
        return self._predictor
