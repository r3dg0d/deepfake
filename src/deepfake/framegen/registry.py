"""Backend registry so the interpolation model can be swapped later."""

from __future__ import annotations

from collections.abc import Callable

from .base import FrameGenerationBackend


def _rife(**kw) -> FrameGenerationBackend:
    from .rife import RifeBackend

    return RifeBackend(**kw)


BACKENDS: dict[str, Callable[..., FrameGenerationBackend]] = {"rife": _rife}


def create_backend(name: str = "rife", **kwargs) -> FrameGenerationBackend:
    try:
        factory = BACKENDS[name]
    except KeyError:
        raise KeyError(f"unknown frame-generation backend {name!r}; available: {sorted(BACKENDS)}") from None
    return factory(**kwargs)
