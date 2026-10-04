"""Real-time frame generation (NVIDIA Optical Flow / Maxine / passthrough)."""

from .base import (
    FrameGenBenchmark,
    FrameGenCapabilities,
    FrameGenerationBackend,
    FrameGenUnavailable,
    timesteps_for,
)
from .registry import BACKENDS, create_backend, select_backend_name

__all__ = [
    "BACKENDS",
    "FrameGenBenchmark",
    "FrameGenCapabilities",
    "FrameGenUnavailable",
    "FrameGenerationBackend",
    "create_backend",
    "select_backend_name",
    "timesteps_for",
]
