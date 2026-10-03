"""Passthrough frame-generation backend (no interpolation)."""

from __future__ import annotations

import time

import numpy as np

from .base import FrameGenBenchmark, FrameGenCapabilities, FrameGenerationBackend, timesteps_for


class PassthroughBackend(FrameGenerationBackend):
    name = "passthrough"

    def __init__(self, **_kw) -> None:
        self._prev = None
        self._cur = None
        self._w = self._h = 0

    @property
    def capabilities(self) -> FrameGenCapabilities:
        return FrameGenCapabilities(
            name=self.name,
            variant="none",
            max_factor=1,
            arbitrary_timestep=False,
            device="cpu",
            precision="n/a",
            license="MIT",
            paper="",
            source="",
        )

    @property
    def ready(self) -> bool:
        return self._prev is not None and self._cur is not None

    def initialize(self, width: int, height: int) -> None:
        self._w, self._h = width, height

    def push(self, frame_bgr: np.ndarray) -> None:
        self._prev, self._cur = self._cur, frame_bgr

    def interpolate(self, timesteps: list[float]) -> list[np.ndarray]:
        # Duplicate newest keyframe — honest about not generating motion
        assert self._cur is not None
        return [self._cur.copy() for _ in timesteps]

    def benchmark(self, width: int, height: int, factor: int, iterations: int = 60) -> FrameGenBenchmark:
        self.initialize(width, height)
        a = np.zeros((height, width, 3), np.uint8)
        self.push(a)
        self.push(a)
        ts = timesteps_for(factor)
        t0 = time.perf_counter()
        for _ in range(iterations):
            self.interpolate(ts)
        ms = (time.perf_counter() - t0) / max(1, iterations * max(1, len(ts))) * 1000
        return FrameGenBenchmark(self.name, "none", width, height, factor, ms, ms, 1000 / ms if ms else 0, None, iterations)

    def shutdown(self) -> None:
        self._prev = self._cur = None
