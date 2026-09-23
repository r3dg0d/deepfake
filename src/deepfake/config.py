"""User preferences (~/.config/deepfake/config.toml) with sensible defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .paths import config_home, ensure_dirs

DEFAULTS = {
    "frame_generation": "auto",  # auto | 2x | 3x | 4x | off
    "quickshell": "auto",  # auto | on | off
    "gpu": "auto",
    "preset": "balanced",
    "encoder": "auto",
    "preview": True,
    "watermark": True,
    "show_metrics": True,
}


@dataclass
class DeepfakeConfig:
    frame_generation: str = "auto"
    quickshell: str = "auto"
    gpu: str = "auto"
    preset: str = "balanced"
    encoder: str = "auto"
    preview: bool = True
    watermark: bool = True
    show_metrics: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_path() -> Path:
    return config_home() / "config.toml"


def load_config() -> DeepfakeConfig:
    ensure_dirs()
    path = config_path()
    data = dict(DEFAULTS)
    if path.is_file():
        try:
            raw = _parse_simple_toml(path.read_text(encoding="utf-8"))
            data.update({k: raw[k] for k in DEFAULTS if k in raw})
        except (OSError, ValueError):
            pass
    known = {f.name for f in fields(DeepfakeConfig)}
    return DeepfakeConfig(**{k: v for k, v in data.items() if k in known})


def save_config(cfg: DeepfakeConfig) -> Path:
    ensure_dirs()
    path = config_path()
    lines = ["# deepfake defaults — edit freely; CLI flags still override\n"]
    for k, v in cfg.to_dict().items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}\n")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"\n')
        else:
            lines.append(f"{k} = {v}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def ensure_default_config() -> DeepfakeConfig:
    """Create config.toml on first use if missing."""
    path = config_path()
    cfg = load_config()
    if not path.is_file():
        save_config(cfg)
    return cfg


def _parse_simple_toml(text: str) -> dict[str, Any]:
    """Tiny TOML subset (keys we ship) so we do not depend on tomllib features/version."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            out[key] = val[1:-1]
        elif val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        else:
            try:
                out[key] = int(val)
            except ValueError:
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
    return out
