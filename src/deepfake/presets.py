"""Latency / quality presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PresetName = Literal["low-latency", "balanced", "high-quality"]


@dataclass(frozen=True)
class Preset:
    name: PresetName
    width: int
    height: int
    fps: int
    detect_interval: int  # run detector every N frames
    temporal_smooth: float  # 0..1 exponential blend
    blend_feather: int
    color_match: bool
    multi_face: bool


PRESETS: dict[str, Preset] = {
    "low-latency": Preset(
        name="low-latency",
        width=640,
        height=480,
        fps=30,
        detect_interval=3,
        temporal_smooth=0.15,
        blend_feather=8,
        color_match=False,
        multi_face=False,
    ),
    "balanced": Preset(
        name="balanced",
        width=960,
        height=540,
        fps=24,
        detect_interval=2,
        temporal_smooth=0.4,
        blend_feather=28,
        color_match=True,
        multi_face=False,
    ),
    "high-quality": Preset(
        name="high-quality",
        width=1280,
        height=720,
        fps=24,
        detect_interval=1,
        temporal_smooth=0.55,
        blend_feather=36,
        color_match=True,
        multi_face=True,
    ),
}


# Short names used by the frame-generation presets (--preset latency|balanced|quality).
PRESET_ALIASES: dict[str, str] = {"latency": "low-latency", "quality": "high-quality"}


def canonical_preset(name: str) -> str:
    return PRESET_ALIASES.get(name, name)


def framegen_preset_name(name: str) -> str:
    """Map a swap preset to its frame-generation preset (latency/balanced/quality)."""
    inverse = {v: k for k, v in PRESET_ALIASES.items()}
    return inverse.get(canonical_preset(name), "balanced")


def get_preset(name: str) -> Preset:
    name = canonical_preset(name)
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
