"""Which settings and model revision produced an output.

A JSON sidecar (``<output>.deepfake.json``) is written next to every rendered
video, and the same record is published to the widget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import __version__


def model_record(name: str | None) -> dict[str, Any]:
    """Installed-model record (revision / hashes as recorded at install time) without loading weights."""
    if not name:
        return {"name": None}
    from .models import CATALOG, _load_installed

    rec = dict(_load_installed().get(name) or {})
    cat = CATALOG.get(name) or {}
    out: dict[str, Any] = {"name": name, "source": cat.get("source"), "weights_license": cat.get("weights_license")}
    for key in ("manifest_revision", "revision", "sha256", "files", "installed_at", "load_check"):
        if key in rec:
            out[key] = rec[key]
    return out


def build_provenance(cfg, *, backend: str | None = None, frame_gen=None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from .models import active_backend

    name = backend or cfg.backend or (cfg.model_backend if cfg.engine == "experimental" else None) or active_backend()
    p = cfg.preset
    data: dict[str, Any] = {
        "tool": "deepfake",
        "version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": cfg.engine,
        "mode": cfg.mode,
        "backend": name,
        "model": model_record(name),
        "detector": model_record("yunet"),
        "swap_precision": cfg.swap_precision,
        "preset": {"name": p.name, "width": p.width, "height": p.height, "fps": p.fps,
                   "detect_interval": p.detect_interval},
        "geometry": cfg.geometry_settings() if hasattr(cfg, "geometry_settings") else {},
        "composite": {
            "blend_feather": cfg.blend_feather if cfg.blend_feather is not None else p.blend_feather,
            "color_match": cfg.color_match if cfg.color_match is not None else p.color_match,
            "temporal_smooth": cfg.temporal_smooth if cfg.temporal_smooth is not None else p.temporal_smooth,
        },
        "watermark": bool(cfg.watermark),
    }
    if frame_gen is not None:
        data["frame_generation"] = {k: getattr(frame_gen, k, None) for k in ("enabled", "factor", "backend", "variant")}
    if extra:
        data.update(extra)
    return data


def write_sidecar(output: Path, cfg, *, input_video: Path | None = None, source: Path | None = None,
                  frame_gen=None, summary: dict[str, Any] | None = None) -> Path:
    extra: dict[str, Any] = {"output": str(output)}
    if input_video is not None:
        extra["input"] = str(input_video)
    if source is not None:
        extra["identity_image"] = str(source)
    if summary:
        extra["run"] = {k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool, type(None)))}
    path = output.with_name(output.name + ".deepfake.json")
    path.write_text(json.dumps(build_provenance(cfg, frame_gen=frame_gen, extra=extra), indent=2, default=str) + "\n",
                    encoding="utf-8")
    return path
