"""Real load + sample-inference checks for installed core models (GPU, no mocks).

Each check loads the model the same way a live session does, runs one sanctioned sample
inference on synthetic frames or the InsightFace package's own test photo (``t1.jpg``, a
group photo shipped for exactly this purpose), and reports device placement, timing and VRAM.
The face-swap checks only move one face of that photo onto another face of the same photo.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _vram() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {"torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                    "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _device_used_mib() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        pids = {p.pid: p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)}
        import os

        used = pids.get(os.getpid())
        return round(used / 2**20, 1) if used else None
    except Exception:  # noqa: BLE001
        return None


def sample_photo() -> np.ndarray:
    from insightface.data import get_image

    return get_image("t1")  # BGR


def check_rife() -> dict[str, Any]:
    import torch

    from .framegen.rife import RifeBackend

    out: dict[str, Any] = {"variants": {}}
    yy, xx = np.mgrid[0:360, 0:640]
    a = np.dstack([(xx % 256), (yy % 256), ((xx + yy) % 256)]).astype(np.uint8)
    b = np.roll(a, 8, axis=1)  # 8 px horizontal motion
    ok = True
    for variant in ("4.25", "4.25.lite", "4.26"):
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        be = RifeBackend(variant=variant)
        try:
            be.initialize(640, 360)
            be.push(a)
            be.push(b)
            t0 = time.perf_counter()
            mids = be.interpolate([0.5])
            ms = (time.perf_counter() - t0) * 1000
            mid = mids[0]
            # the midpoint should sit between the two inputs, not equal either of them
            err_mid = float(np.abs(mid.astype(np.int16) - np.roll(a, 4, axis=1).astype(np.int16))[:, 16:-16].mean())
            good = mid.shape == a.shape and np.isfinite(mid).all() and err_mid < 20
            bench = be.benchmark(640, 360, 2, iterations=20)
            out["variants"][variant] = {"ok": bool(good), "first_call_ms": round(ms, 2), "mean_abs_err_vs_shift4": round(err_mid, 2),
                                        "ms_per_generated_frame_640x360": round(bench.ms_per_generated_frame, 3),
                                        "device": be.capabilities.device, **_vram()}
            ok &= bool(good)
        finally:
            be.shutdown()
    out["ok"] = ok
    return out


def check_inswapper() -> dict[str, Any]:
    import insightface
    from insightface.app import FaceAnalysis

    from .ort_util import providers as ort_providers
    from .ort_util import session_on_gpu
    from .paths import models_dir

    root = models_dir() / "inswapper"
    providers = ort_providers("cuda")
    t0 = time.perf_counter()
    app = FaceAnalysis(name="buffalo_l", root=str(root), providers=providers,
                       allowed_modules=["detection", "recognition", "landmark_2d_106"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(str(root / "inswapper_128.onnx"), providers=providers)
    load_ms = (time.perf_counter() - t0) * 1000
    img = sample_photo()
    t1 = time.perf_counter()
    faces = app.get(img)
    det_ms = (time.perf_counter() - t1) * 1000
    if len(faces) < 2:
        return {"ok": False, "error": f"expected several faces in the test photo, found {len(faces)}"}
    faces.sort(key=lambda f: f.bbox[0])
    src, dst = faces[0], faces[1]
    swapper.get(img.copy(), dst, src, paste_back=True)  # warm-up
    t2 = time.perf_counter()
    out = swapper.get(img.copy(), dst, src, paste_back=True)
    swap_ms = (time.perf_counter() - t2) * 1000
    x0, y0, x1, y1 = (int(v) for v in dst.bbox)
    region_change = float(np.abs(out[y0:y1, x0:x1].astype(np.int16) - img[y0:y1, x0:x1].astype(np.int16)).mean())
    # identity check with the recognition model: the swapped face should move toward the source identity
    swapped = [f for f in app.get(out) if abs(f.bbox[0] - dst.bbox[0]) < 20]
    sim_before = float(np.dot(dst.normed_embedding, src.normed_embedding))
    sim_after = float(np.dot(swapped[0].normed_embedding, src.normed_embedding)) if swapped else None
    sess = swapper.session
    det_sess = app.models["detection"].session
    import torch

    gpu_expected = torch.cuda.is_available()
    on_gpu = session_on_gpu(sess) and session_on_gpu(det_sess)
    ok = region_change > 2.0 and sim_after is not None and sim_after > sim_before + 0.2 and (on_gpu or not gpu_expected)
    return {"ok": ok, "on_gpu": on_gpu, "error": None if ok or on_gpu or not gpu_expected else
            "ran on CPUExecutionProvider although CUDA is available (onnxruntime-gpu/CUDA runtime mismatch)",
            "faces": len(faces), "load_ms": round(load_ms, 1), "detect_ms": round(det_ms, 2),
            "swap_ms": round(swap_ms, 2), "region_mean_abs_change": round(region_change, 2),
            "identity_cos_before": round(sim_before, 3), "identity_cos_after": None if sim_after is None else round(sim_after, 3),
            "swapper_providers": sess.get_providers(), "detector_providers": det_sess.get_providers(),
            "process_vram_mib": _device_used_mib(), "insightface": insightface.__version__}


def check_alphaface() -> dict[str, Any]:
    import torch

    from .detect import align_crop
    from .swap.alphaface import AlphaFaceSwapper

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    sw = AlphaFaceSwapper(device="cuda")
    load_ms = (time.perf_counter() - t0) * 1000
    img = sample_photo()
    boxes = _faces_for_check(img)
    if len(boxes) < 2:
        return {"ok": False, "error": f"expected several faces in the test photo, found {len(boxes)}"}
    src_crop, _ = align_crop(img, boxes[0], size=256, pad=0.25)
    dst_crop, _ = align_crop(img, boxes[1], size=256, pad=0.35)
    sw.set_source(src_crop)
    sw.swap(dst_crop)  # warm-up
    times = []
    for _ in range(10):
        r = sw.swap(dst_crop)
        times.append(r.inference_ms)
    face = r.face_bgr
    change = float(np.abs(face.astype(np.int16) - dst_crop.astype(np.int16)).mean())
    ok = face.shape == dst_crop.shape and face.std() > 5 and change > 2.0
    param = next(sw._model.Swapper.parameters())
    return {"ok": bool(ok), "load_ms": round(load_ms, 1), "swap_ms_median": round(float(np.median(times)), 2),
            "swap_ms_p95": round(float(np.percentile(times, 95)), 2), "crop_mean_abs_change": round(change, 2),
            "output_std": round(float(face.std()), 2), "device": str(param.device), "dtype": str(param.dtype), **_vram()}


def _faces_for_check(img):
    """Face boxes for the check photo: InsightFace SCRFD when installed (better boxes), else Haar."""
    try:
        from insightface.app import FaceAnalysis

        from .detect import FaceBox
        from .paths import models_dir

        root = models_dir() / "inswapper"
        if (root / "models" / "buffalo_l" / "det_10g.onnx").is_file():
            from .ort_util import providers

            app = FaceAnalysis(name="buffalo_l", root=str(root), allowed_modules=["detection"], providers=providers("cuda"))
            app.prepare(ctx_id=0, det_size=(640, 640))
            fs = sorted(app.get(img), key=lambda f: f.bbox[0])
            return [FaceBox(int(f.bbox[0]), int(f.bbox[1]), int(f.bbox[2] - f.bbox[0]), int(f.bbox[3] - f.bbox[1]), float(f.det_score))
                    for f in fs]
    except Exception:  # noqa: BLE001
        pass
    from .detect import create_detector

    return sorted(create_detector("auto").detect(img), key=lambda b: b.x)


def check_yunet() -> dict[str, Any]:
    from .detect import YuNetDetector

    det = YuNetDetector()
    img = sample_photo()
    det.detect(img)  # warm-up
    t0 = time.perf_counter()
    boxes = det.detect(img)
    ms = (time.perf_counter() - t0) * 1000
    with_lm = sum(1 for b in boxes if b.landmarks is not None)
    return {"ok": len(boxes) >= 5 and with_lm == len(boxes), "faces": len(boxes), "with_landmarks": with_lm,
            "detect_ms": round(ms, 2), "input": f"{img.shape[1]}x{img.shape[0]}"}


CHECKS = {"rife": check_rife, "inswapper": check_inswapper, "alphaface": check_alphaface, "yunet": check_yunet}


def run_check(name: str) -> dict[str, Any]:
    if name not in CHECKS:
        return {"ok": True, "skipped": "no load check defined"}
    t0 = time.perf_counter()
    try:
        r = CHECKS[name]()
    except ImportError as e:
        return {"ok": False, "error": f"missing dependency: {e}"}
    r["check_seconds"] = round(time.perf_counter() - t0, 2)
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return r
