"""Frame-generation (temporal interpolation) backend interface.

A backend receives consecutive *swapped* frames and synthesises frames at
fractional timesteps between them. It never duplicates frames: when it
cannot run, callers must emit fewer frames instead of padding the count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class FrameGenUnavailable(RuntimeError):
    """Backend cannot run here (missing weights, no CUDA, …)."""


@dataclass(frozen=True)
class FrameGenCapabilities:
    name: str
    variant: str
    max_factor: int  # largest supported multiplier per source interval
    arbitrary_timestep: bool  # can synthesise any t in (0, 1), not only 0.5
    device: str  # "cuda" or "cpu"
    precision: str
    license: str
    paper: str
    source: str


@dataclass
class FrameGenBenchmark:
    backend: str
    variant: str
    width: int
    height: int
    factor: int
    ms_per_generated_frame: float
    ms_per_source_interval: float
    generated_fps_capacity: float
    peak_vram_mb: float | None
    samples: int
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "variant": self.variant,
            "resolution": f"{self.width}x{self.height}",
            "factor": self.factor,
            "ms_per_generated_frame": round(self.ms_per_generated_frame, 3),
            "ms_per_source_interval": round(self.ms_per_source_interval, 3),
            "generated_fps_capacity": round(self.generated_fps_capacity, 1),
            "peak_vram_mb": None if self.peak_vram_mb is None else round(self.peak_vram_mb, 1),
            "samples": self.samples,
            **{k: round(v, 3) for k, v in self.extra.items()},
        }


class FrameGenerationBackend(ABC):
    """Swappable interpolation backend.

    Lifecycle: ``initialize(w, h)`` → many ``push`` / ``interpolate`` calls →
    ``shutdown()``. ``push`` hands over the newest source frame so per-frame
    work (upload, feature encoding) happens once per source frame, not once
    per generated frame.
    """

    name: str = "base"

    @property
    @abstractmethod
    def capabilities(self) -> FrameGenCapabilities: ...

    @abstractmethod
    def initialize(self, width: int, height: int) -> None: ...

    @abstractmethod
    def push(self, frame_bgr: np.ndarray) -> None:
        """Make ``frame_bgr`` the newest keyframe (previous newest becomes the older one)."""

    @abstractmethod
    def interpolate(self, timesteps: list[float]) -> list[np.ndarray]:
        """Frames at each t in (0, 1) between the two most recent pushed keyframes."""

    @abstractmethod
    def benchmark(self, width: int, height: int, factor: int, iterations: int = 60) -> FrameGenBenchmark: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @property
    @abstractmethod
    def ready(self) -> bool:
        """True once two keyframes have been pushed."""


def timesteps_for(factor: int) -> list[float]:
    """Intermediate timesteps for an integer multiplier (2x → [0.5], 3x → [1/3, 2/3])."""
    if factor < 2:
        return []
    return [k / factor for k in range(1, factor)]
