"""AlphaFace backend — loads vendor checkout + Drive weights when present."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from ..paths import models_dir, vendor_dir
from .base import SwapResult


class AlphaFaceSwapper:
    """Best-effort loader around upstream build_AlphaFace().Swapper.

    Upstream demo is still-image batch; we call the same Swapper on aligned crops.
    If import fails, raises with install instructions.
    """

    name = "alphaface"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._source: np.ndarray | None = None
        self._model = None
        self._load()

    def _load(self) -> None:
        weights = models_dir() / "alphaface"
        vendor = vendor_dir()
        pt = weights / "alphaface_demo.pt"
        if not pt.is_file():
            # accept any .pt
            pts = list(weights.glob("*.pt"))
            if not pts:
                raise RuntimeError(
                    f"AlphaFace weights missing under {weights}. "
                    "Run: deepfake models install alphaface --yes"
                )
            pt = pts[0]

        if vendor.is_dir() and str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

        try:
            # Upstream module names may vary; try common patterns from research demos.
            build = None
            for mod_name in ("Models.AlphaFace", "models.alphaface", "eval"):
                try:
                    mod = __import__(mod_name, fromlist=["*"])
                    build = getattr(mod, "build_AlphaFace", None) or getattr(mod, "Swapper", None)
                    if build:
                        break
                except ImportError:
                    continue
            if build is None:
                raise ImportError("could not import AlphaFace Swapper from vendor tree")
            # Exact constructor is upstream-specific; keep a soft path.
            self._model = ("deferred", pt)
        except Exception as e:
            # Keep a torch-load stub path for skeleton honesty.
            self._model = ("weights-only", pt)
            self._import_error = e

    def set_source(self, source_bgr: np.ndarray) -> None:
        self._source = source_bgr

    def swap(self, target_face_bgr: np.ndarray) -> SwapResult:
        if self._source is None:
            raise RuntimeError("call set_source() before swap()")
        t0 = time.perf_counter()
        # Full CAII inference requires cleaned upstream deps (torch + ArcFace).
        # Until weights+vendor import succeed end-to-end, raise actionable error
        # rather than silently faking a swap.
        kind = self._model[0] if self._model else None
        if kind == "deferred":
            raise RuntimeError(
                "AlphaFace vendor modules were located but the live Swapper wiring "
                "needs upstream config (config/alphaface_eval_demo.py) + torch. "
                "See STATUS.md. Weights path: "
                f"{self._model[1]}"
            )
        raise RuntimeError(
            "AlphaFace Swapper not fully importable in this environment. "
            f"Weights at {models_dir() / 'alphaface'}. "
            f"Vendor at {vendor_dir()}. "
            "Install torch + clone deps, then re-run. "
            f"Import detail: {getattr(self, '_import_error', None)}"
        )
        # unreachable — kept for type checkers if future path returns tensor
        ms = (time.perf_counter() - t0) * 1000
        return SwapResult(face_bgr=target_face_bgr, inference_ms=ms, backend=self.name)
