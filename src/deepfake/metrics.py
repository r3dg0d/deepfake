"""Live metrics: FPS, latency, drops, VRAM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class FrameMetrics:
    fps: float = 0.0
    inference_ms: float = 0.0
    total_frame_ms: float = 0.0
    dropped_frames: int = 0
    vram_mb: float | None = None
    frames: int = 0


@dataclass
class MetricsTracker:
    window: float = 1.0
    dropped_frames: int = 0
    _times: list[float] = field(default_factory=list)
    _infer_ms: list[float] = field(default_factory=list)
    _total_ms: list[float] = field(default_factory=list)
    frames: int = 0

    def mark_drop(self, n: int = 1) -> None:
        self.dropped_frames += n

    def record(self, *, inference_ms: float, total_ms: float) -> FrameMetrics:
        now = time.monotonic()
        self._times.append(now)
        self._infer_ms.append(inference_ms)
        self._total_ms.append(total_ms)
        self.frames += 1
        cutoff = now - self.window
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)
            self._infer_ms.pop(0)
            self._total_ms.pop(0)
        fps = len(self._times) / self.window if self._times else 0.0
        avg_infer = sum(self._infer_ms) / len(self._infer_ms) if self._infer_ms else 0.0
        avg_total = sum(self._total_ms) / len(self._total_ms) if self._total_ms else 0.0
        return FrameMetrics(
            fps=fps,
            inference_ms=avg_infer,
            total_frame_ms=avg_total,
            dropped_frames=self.dropped_frames,
            vram_mb=_vram_mb(),
            frames=self.frames,
        )


def _vram_mb() -> float | None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        pass
    return None


def format_metrics(m: FrameMetrics) -> str:
    vram = f"{m.vram_mb:.0f}MB" if m.vram_mb is not None else "n/a"
    return (
        f"fps={m.fps:.1f} infer={m.inference_ms:.1f}ms "
        f"frame={m.total_frame_ms:.1f}ms drop={m.dropped_frames} vram={vram}"
    )
