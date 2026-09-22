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
    "4.25.lite": RifeVariant("4.25.lite", 32, (16, 8, 4, 2, 1), 128),
    "4.26": RifeVariant("4.26", 24, (32, 16, 8, 4, 1), 64),
}
