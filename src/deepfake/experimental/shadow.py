"""Partial reimplementation of arXiv:2609.17740, equations 9–12.

Requires an externally computed shading field and skin mask. Dense mesh,
light estimation and ray marching are NOT implemented. No automatic activation:
applying additional shadows to an already shaded donor can double-shadow it.
"""

from __future__ import annotations

import numpy as np


def shadow_gain(shading, skin, *, threshold=0.90, strength=0.45, floor=0.82):
    import cv2

    shading = np.asarray(shading, np.float32)
    skin = np.asarray(skin, bool)
    if shading.ndim != 2 or skin.shape != shading.shape or not np.isfinite(shading).all():
        raise ValueError("finite shading and matching skin mask required")
    if not 0 < threshold <= 1 or not 0 <= strength <= 1 or not 0 <= floor <= 1:
        raise ValueError("invalid gain parameters")
    if not skin.any():
        return np.ones_like(shading)
    normalizer = float(np.percentile(shading[skin], 75))
    if normalizer <= 0:
        return np.ones_like(shading)  # invalid geometry: preserve input
    field = np.clip(shading / normalizer, 0, 1)
    gain = np.clip(1 + strength * (np.clip(field / threshold, 0, 1) - 1), floor, 1)
    # Normalized edge-stopping blur guided by shading, independently implemented.
    # Smooth only similar shadow values so nose-shadow edges remain separated.
    accum, weights = np.zeros_like(gain), np.zeros_like(gain)
    for center in np.linspace(0, 1, 8):
        weight = np.exp(-0.5 * ((field - center) / 0.10) ** 2).astype(np.float32) * skin
        accum += weight * cv2.GaussianBlur(gain * weight, (0, 0), 2)
        weights += weight * cv2.GaussianBlur(weight, (0, 0), 2)
    gain = np.divide(accum, weights, out=gain.copy(), where=weights > 1e-6)
    return np.where(skin, np.clip(gain, floor, 1), 1).astype(np.float32)


def apply_linear_gain(image, gain, alpha):
    """Apply bounded channel-uniform gain in linear RGB, then encode sRGB."""
    image = np.asarray(image, np.float32)
    gain, alpha = np.asarray(gain, np.float32), np.asarray(alpha, np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or gain.shape != image.shape[:2] or alpha.shape != gain.shape:
        raise ValueError("matching RGB image, gain and alpha required")
    if not all(np.isfinite(a).all() for a in (image, gain, alpha)):
        raise ValueError("non-finite inputs")
    if np.any((image < 0) | (image > 1)) or np.any((gain < 0) | (gain > 1)):
        raise ValueError("image and gain must be in [0,1]")
    linear = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    linear *= (1 - np.clip(alpha, 0, 1) * (1 - gain))[..., None]
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    # Preserve untouched pixels bitwise, including foreground and background.
    return np.where(((alpha <= 0) | (gain == 1))[..., None], image, encoded)
