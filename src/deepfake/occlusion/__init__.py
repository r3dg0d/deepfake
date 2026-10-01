"""Occlusion-aware compositing with conservative temporal mask stabilization."""

from .engine import MaskEstimate, MaskState, OcclusionEngine

__all__ = ["MaskEstimate", "MaskState", "OcclusionEngine"]
