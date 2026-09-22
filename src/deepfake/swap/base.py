from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class SwapResult:
    face_bgr: np.ndarray
    inference_ms: float
    backend: str


class Swapper(Protocol):
    name: str

    def set_source(self, source_bgr: np.ndarray) -> None: ...

    def swap(self, target_face_bgr: np.ndarray) -> SwapResult: ...
