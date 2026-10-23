from __future__ import annotations

from dataclasses import asdict

from .base import EngineCapabilities, EngineUnavailable

CAPABILITIES = {
    "liveportrait": EngineCapabilities(supports_head=True, supports_streaming=True, supports_cuda=True),
    "alphaface": EngineCapabilities(supports_face=True, supports_streaming=True, supports_cuda=True),
    # These adapters deliberately expose file processing only until live integration is validated.
    "personalive": EngineCapabilities(supports_head=True, supports_cuda=True),
    "editalive": EngineCapabilities(supports_fullbody=True, supports_cuda=True),
}


def engine_status():
    from .experimental.adapters import backend_status

    return [
        {"name": name, "engine": "alphaface" if name == "alphaface" else "experimental",
         "capabilities": asdict(caps), **(backend_status(name) if name != "alphaface" else {})}
        for name, caps in CAPABILITIES.items()
    ]


def create_engine(config):
    if config.engine == "experimental" and config.model_backend == "liveportrait" and config.mode == "head":
        from .experimental.liveportrait import LivePortraitEngine

        engine = LivePortraitEngine()
        engine.load(config)
        return engine
    if config.engine == "alphaface":
        if config.mode != "face":
            raise EngineUnavailable("AlphaFace supports --mode face. Select --engine experimental for head/body.")
        from .alphaface import AlphaFaceEngine

        engine = AlphaFaceEngine()
        engine.load(config)
        return engine
    raise EngineUnavailable(
        "Experimental backends currently support offline video only. "
        "Use deepfake video --engine experimental --mode head|fullbody --reference IMAGE. "
        "Run deepfake experimental doctor for installation status. Hybrid is not implemented."
    )
