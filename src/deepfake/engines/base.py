from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class EngineUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineCapabilities:
    supports_face: bool = False
    supports_head: bool = False
    supports_fullbody: bool = False
    supports_streaming: bool = False
    supports_offline: bool = True
    supports_cuda: bool = False
    minimum_vram_gb: float | None = None


@dataclass(frozen=True)
class FrameContext:
    frame_id: int
    capture_timestamp: float
    tracking_id: str | None = None


class TransformationEngine(ABC):
    @abstractmethod
    def load(self, config: Any) -> None: ...

    @abstractmethod
    def prepare_reference(self, reference: np.ndarray) -> None: ...

    @abstractmethod
    def process_frame(self, frame: np.ndarray, context: FrameContext) -> tuple[np.ndarray, Any]: ...

    @abstractmethod
    def capabilities(self) -> EngineCapabilities: ...

    @abstractmethod
    def unload(self) -> None: ...
