"""Backend registry — NVIDIA Maxine → NvOF → passthrough."""

from __future__ import annotations

from collections.abc import Callable

from .base import FrameGenerationBackend


def _maxine(**kw) -> FrameGenerationBackend:
    from .maxine import MaxineVFGBackend

    return MaxineVFGBackend(**kw)


def _nvof(**kw) -> FrameGenerationBackend:
    from .nvof import NvidiaOpticalFlowBackend

    return NvidiaOpticalFlowBackend(**kw)


def _passthrough(**kw) -> FrameGenerationBackend:
    from .passthrough_fg import PassthroughBackend

    return PassthroughBackend(**kw)


BACKENDS: dict[str, Callable[..., FrameGenerationBackend]] = {
    "maxine": _maxine,
    "nvof": _nvof,
    "nvfruc": _nvof,
    "passthrough": _passthrough,
    "rife": _nvof,  # legacy alias → NvOF
}


def select_backend_name(requested: str | None = None) -> str:
    """Resolve requested name to a concrete backend id."""
    name = (requested or "auto").lower().strip()
    if name in ("rife", ""):
        name = "auto"
    if name == "nvfruc":
        name = "nvof"
    if name != "auto":
        return name

    from .maxine import maxine_available
    from .nvof_api import optical_flow_available

    ok_m, _ = maxine_available()
    if ok_m:
        # SDK present — still prefer NvOF until Maxine Python path is complete;
        # auto will try Maxine only when explicitly requested or DEEPFAKE_PREFER_MAXINE=1.
        import os

        if os.environ.get("DEEPFAKE_PREFER_MAXINE", "").strip() in ("1", "true", "yes"):
            return "maxine"
    ok, _ = optical_flow_available()
    return "nvof" if ok else "passthrough"


def create_backend(name: str = "auto", **kwargs) -> FrameGenerationBackend:
    resolved = select_backend_name(name)
    kwargs.pop("flow_scale", None)
    try:
        factory = BACKENDS[resolved]
    except KeyError as e:
        raise KeyError(
            f"unknown frame-generation backend {name!r}; available: "
            f"{sorted({'maxine', 'nvof', 'nvfruc', 'passthrough', 'auto'})}"
        ) from e
    return factory(**kwargs)
