from __future__ import annotations

from ..models import active_backend, is_model_ready
from .base import Swapper
from .placeholder import MissingModelSwapper, PassthroughSwapper


def create_swapper(
    backend: str | None = None,
    *,
    device: str = "cpu",
    allow_passthrough: bool = False,
    precision: str = "fp32",
) -> Swapper:
    name = backend or active_backend()
    if name is None:
        if allow_passthrough:
            return PassthroughSwapper()
        return MissingModelSwapper()
    if name == "passthrough":
        return PassthroughSwapper()
    if name == "alphaface":
        if not is_model_ready("alphaface"):
            return MissingModelSwapper("AlphaFace not installed. deepfake models install alphaface --yes")
        from .alphaface import AlphaFaceSwapper

        return AlphaFaceSwapper(device=device, precision=precision)
    if name in ("inswapper", "insightface"):
        if not is_model_ready("inswapper"):
            return MissingModelSwapper(
                "inswapper not installed. deepfake models install inswapper --yes "
                "(NON-COMMERCIAL research)"
            )
        from .insightface_fallback import InswapperSwapper

        return InswapperSwapper(device=device)
    raise KeyError(f"unknown swap backend {name!r}")
