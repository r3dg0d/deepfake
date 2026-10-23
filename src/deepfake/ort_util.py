"""ONNX Runtime provider setup that does not silently fall back to the CPU.

onnxruntime-gpu needs the CUDA/cuDNN runtime its wheel was built against. In the deepfake venv
those come from the nvidia-* wheels that PyTorch installs; ``onnxruntime.preload_dlls()`` loads
them from site-packages, so LD_LIBRARY_PATH does not have to know about them. A CUDA-13 build of
onnxruntime next to a CUDA-12 PyTorch finds nothing and quietly runs on the CPU (~20x slower):
:func:`session_on_gpu` lets callers detect that.
"""

from __future__ import annotations

_prepared = False


def providers(device: str = "cuda") -> list[str]:
    global _prepared
    import onnxruntime as ort

    if device == "cpu":
        return ["CPUExecutionProvider"]
    if not _prepared:
        try:
            import torch  # noqa: F401  (loads the CUDA 12 runtime libraries first)
        except ImportError:
            pass
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                pass
        _prepared = True
    avail = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]


def session_on_gpu(session) -> bool:
    return "CUDAExecutionProvider" in session.get_providers()
