"""XDG paths for config, cache (models), and data."""

from __future__ import annotations

import os
from pathlib import Path


def data_home() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "deepfake"
    return Path.home() / ".local" / "share" / "deepfake"


def cache_home() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "deepfake"
    return Path.home() / ".cache" / "deepfake"


def config_home() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "deepfake"
    return Path.home() / ".config" / "deepfake"


def models_dir() -> Path:
    return cache_home() / "models"


def vendor_dir() -> Path:
    """Optional git checkout of AlphaFace_Official (after models install)."""
    return cache_home() / "vendor" / "Alphaface_Official"


def ensure_dirs() -> None:
    data_home().mkdir(parents=True, exist_ok=True)
    models_dir().mkdir(parents=True, exist_ok=True)
    config_home().mkdir(parents=True, exist_ok=True)
    (cache_home() / "vendor").mkdir(parents=True, exist_ok=True)
