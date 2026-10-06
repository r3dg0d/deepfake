"""User-facing frame-generation settings and CLI value parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameGenSettings:
    enabled: bool = True
    factor: int | None = None
    output_fps: float | None = None
    backend: str = "auto"
    variant: str = "balanced"
    precision: str = "fp16"
    max_latency_ms: float = 200.0
    flow_scale: float | None = None
    mode: str = "balanced"

    def resolve_output_fps(self, source_fps: float) -> float:
        if self.output_fps:
            return float(self.output_fps)
        return float(source_fps) * float(self.factor or 2)

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        parts = [self.backend or "auto"]
        if self.factor:
            parts.append(f"{self.factor}x")
        if self.output_fps:
            parts.append(f"→ {self.output_fps:g} fps")
        parts.append(f"[{self.mode}]")
        return " ".join(parts)


def parse_frame_gen(value: str | None) -> tuple[bool, int | None]:
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
    if source_fps <= 0:
        raise ValueError("source_fps must be > 0")
    ratio = output_fps / source_fps
    nearest = round(ratio)
    # Container timing and NTSC rates can make 60/29.999 or 60/29.97 just
    # exceed two. Do not select an entire extra interpolation pass for that.
    if nearest >= 1 and math.isclose(ratio, nearest, rel_tol=0.002):
        ratio = float(nearest)
    return max(1, min(4, math.ceil(ratio)))


PRESET_FRAME_GEN: dict[str, dict[str, object]] = {
    "latency": {"variant": "latency", "mode": "latency", "max_latency_ms": 80.0, "swap_precision": "bf16"},
    "balanced": {"variant": "balanced", "mode": "balanced", "max_latency_ms": 160.0, "swap_precision": "bf16"},
    "quality": {"variant": "quality", "mode": "quality", "max_latency_ms": 250.0, "swap_precision": "fp32"},
}
