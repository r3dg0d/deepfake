"""User-facing frame-generation settings and CLI value parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameGenSettings:
    enabled: bool = False
    factor: int | None = None  # 2 → "2x"; None → derive from output_fps / auto
    output_fps: float | None = None
    backend: str = "rife"
    variant: str = "4.25"
    precision: str = "fp16"
    max_latency_ms: float = 250.0
    flow_scale: float | None = None  # None → automatic (0.5 at ≥1080p)

    def resolve_output_fps(self, source_fps: float) -> float:
        """Output rate: explicit --output-fps wins, else factor × source rate."""
        if self.output_fps:
            return float(self.output_fps)
        return float(source_fps) * float(self.factor or 2)

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        parts = [f"{self.backend} {self.variant}"]
        if self.factor:
            parts.append(f"{self.factor}x")
        if self.output_fps:
            parts.append(f"→ {self.output_fps:g} fps")
        return " ".join(parts)


def parse_frame_gen(value: str | None) -> tuple[bool, int | None]:
    """Parse ``--frame-gen [VALUE]``.

    Accepts ``2x``/``3x``/``4x``, bare ``2``, ``auto`` (pick multiplier),
    ``on`` (2x), ``off``/``none``. Returns (enabled, factor);
    factor None with enabled means "auto".

    ``None`` means unset at the parser layer (callers / config default to auto).
    """
    if value is None:
        return False, None
    v = value.strip().lower()
    if v in ("off", "none", "no", "false", "0"):
        return False, None
    if v in ("", "on", "yes", "true"):
        return True, 2
    if v == "auto":
        return True, None
    if v.endswith("x"):
        v = v[:-1]
    try:
        n = int(v)
    except ValueError:
        raise ValueError(f"invalid --frame-gen value {value!r} (use 2x, 3x, 4x, auto or off)") from None
    if not 2 <= n <= 4:
        raise ValueError("--frame-gen multiplier must be 2x, 3x or 4x")
    return True, n


def factor_for(output_fps: float, source_fps: float) -> int:
    """Smallest integer multiplier that reaches ``output_fps`` from ``source_fps``."""
    if source_fps <= 0:
        raise ValueError("source_fps must be > 0")
    return max(1, min(4, math.ceil(output_fps / source_fps - 1e-6)))


# Frame-generation behaviour per quality preset.
PRESET_FRAME_GEN: dict[str, dict[str, object]] = {
    "latency": {"variant": "4.25.lite", "max_latency_ms": 120.0, "swap_precision": "bf16"},
    "balanced": {"variant": "4.25", "max_latency_ms": 200.0, "swap_precision": "bf16"},
    "quality": {"variant": "4.26", "max_latency_ms": 300.0, "swap_precision": "fp32", "flow_scale": 1.0},
}
