"""Real-time frame generation (temporal interpolation) for deepfake output."""

from .base import (
    FrameGenBenchmark,
    FrameGenCapabilities,
    FrameGenerationBackend,
    FrameGenUnavailable,
    timesteps_for,
)
from .registry import BACKENDS, create_backend

__all__ = [
    "BACKENDS",
    "FrameGenBenchmark",
    "FrameGenCapabilities",
    "FrameGenUnavailable",
    "FrameGenerationBackend",
    "create_backend",
    "timesteps_for",
]
