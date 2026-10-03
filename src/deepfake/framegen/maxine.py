"""Optional NVIDIA Maxine Video Frame Generation backend.

Activates only when the Maxine VFX SDK + nvvfxvideoframegeneration feature
are installed (NGC). Otherwise selection falls through to NvOF.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import FrameGenCapabilities, FrameGenerationBackend, FrameGenUnavailable


def maxine_available() -> tuple[bool, str]:
    roots = []
    env = os.environ.get("DEEPFAKE_MAXINE_ROOT") or os.environ.get("NV_VIDEO_EFFECTS_PATH")
    if env:
        roots.append(Path(env))
    roots += [
        Path("/usr/local/VideoFX"),
        Path("/opt/nvidia/VideoFX"),
        Path.home() / ".local/share/maxine/VideoFX",
    ]
    for root in roots:
        so = root / "lib" / "libNVVideoEffects.so"
        if so.is_file():
            feat = root / "features" / "nvvfxvideoframegeneration"
            if feat.is_dir() or list(root.glob("**/nvvfxvideoframegeneration*")):
                return True, str(root)
            return False, f"VideoFX at {root} but VFG feature missing (install nvvfxvideoframegeneration)"
    return False, "Maxine VFX SDK not installed (set DEEPFAKE_MAXINE_ROOT or install from NGC)"


class MaxineVFGBackend(FrameGenerationBackend):
    """Placeholder that raises until Maxine libs are present — keeps auto-detect honest."""

    name = "maxine"

    def __init__(self, **kw) -> None:
        self._kw = kw

    @property
    def capabilities(self) -> FrameGenCapabilities:
        return FrameGenCapabilities(
            name=self.name,
            variant="vfg",
            max_factor=8,
            arbitrary_timestep=True,
            device="cuda",
            precision="fp16",
            license="NVIDIA Maxine VFX SDK / Open Model License for weights",
            paper="NVIDIA Maxine Video Frame Generation",
            source="https://catalog.ngc.nvidia.com/orgs/nvidia/maxine/collections/nvvfxvideoframegeneration",
        )

    @property
    def ready(self) -> bool:
        return False

    def initialize(self, width: int, height: int) -> None:
        ok, detail = maxine_available()
        if not ok:
            raise FrameGenUnavailable(detail)
        # Full ctypes/C++ Maxine VFG path lands when SDK is on the machine.
        raise FrameGenUnavailable(
            f"Maxine VFG SDK found ({detail}) but Python bindings are not yet wired; "
            "using NvOF instead. Set --frame-gen-backend nvof."
        )

    def push(self, frame_bgr) -> None:  # pragma: no cover
        raise FrameGenUnavailable("maxine not initialized")

    def interpolate(self, timesteps):  # pragma: no cover
        raise FrameGenUnavailable("maxine not initialized")

    def benchmark(self, width, height, factor, iterations=60):  # pragma: no cover
        raise FrameGenUnavailable("maxine not initialized")

    def shutdown(self) -> None:
        return
