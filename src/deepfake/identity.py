"""Optional synthetic identity path from fakeperson identity dirs."""

from __future__ import annotations

import json
import os
from pathlib import Path


def fakeperson_identities_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "fakeperson" / "identities"
    return Path.home() / ".local" / "share" / "fakeperson" / "identities"


def list_fakeperson_identities() -> list[str]:
    root = fakeperson_identities_root()
    if not root.is_dir():
        return []
    names: list[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "identity.json").is_file():
            names.append(p.name)
    return names


def resolve_source_image(source: str | None, identity: str | None) -> Path | None:
    """Resolve --source path or --identity name to an image file."""
    if source:
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"source image not found: {p}")
        return p
    if identity:
        root = fakeperson_identities_root() / identity
        if not root.is_dir():
            raise FileNotFoundError(
                f"fakeperson identity '{identity}' not found under {fakeperson_identities_root()}"
            )
        # Prefer common render names, else any png/jpg
        for cand in ("source.png", "face.png", "identity.png", "render.png"):
            if (root / cand).is_file():
                return root / cand
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            hits = sorted(root.glob(ext))
            if hits:
                return hits[0]
        # identity.json may point at a path
        meta = json.loads((root / "identity.json").read_text(encoding="utf-8"))
        for key in ("source", "face", "image", "last_render"):
            if key in meta and Path(meta[key]).is_file():
                return Path(meta[key])
        raise FileNotFoundError(
            f"identity '{identity}' has no image files; render one with fakeperson first"
        )
    return None
