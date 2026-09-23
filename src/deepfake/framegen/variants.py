"""RIFE variant table (torch-free so the CLI can list choices without importing torch)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RifeVariant:
    name: str
    last_block_width: int
    pyramid: tuple[float, ...]
    modulo: int  # input H/W must be padded to a multiple of this (at scale 1.0)


VARIANTS: dict[str, RifeVariant] = {
    "4.25": RifeVariant("4.25", 32, (16, 8, 4, 2, 1), 64),
    # lite uses a narrower final IFBlock (c=24); matches flownet_v4.25.lite.safetensors
    "4.25.lite": RifeVariant("4.25.lite", 24, (16, 8, 4, 2, 1), 128),
    # installed 4.26 weights use the same final width as 4.25 (c=32)
    "4.26": RifeVariant("4.26", 32, (32, 16, 8, 4, 1), 64),
}
