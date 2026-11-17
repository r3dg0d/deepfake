"""A/B harness for face geometry: the same frames through named setting variants.

``deepfake compare`` uses it. Measurements that do not depend on the swap model
(where the landmarks land in the crop the model sees, how much they drift, how
far the composite reaches outside the face) are valid with any backend,
including ``passthrough`` and the labelled ``stand-in`` test double. Measurements
of the swapped result (eye proportions of the output) only say something about a
real model when a real backend is used; the report records which one ran.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import eye_metrics, fit_frame, template_points

LEGACY = {"alignment": "bbox", "align_template": "ffhq", "mask_mode": "oval", "landmark_smooth": False,
          "identity_align": "bbox", "fit": "stretch", "landmarker": "yunet"}
CURRENT = {"alignment": "landmark", "align_template": "ffhq", "mask_mode": "landmark", "landmark_smooth": True,
           "identity_align": "arcface", "fit": "cover", "landmarker": "auto"}
NAMED = {"legacy": LEGACY, "current": CURRENT,
         "arcface-template": {**CURRENT, "align_template": "arcface"},
         "landmark-oval-mask": {**CURRENT, "mask_mode": "oval"},
         "landmark-unsmoothed": {**CURRENT, "landmark_smooth": False},
         "yunet-landmarks": {**CURRENT, "landmarker": "yunet"}}


class StandInSwapper:
    """Test double, not a face swap: returns the crop tinted and slightly blurred.

    A real generator changes colour and texture over the whole crop; this makes the
    same kind of whole-crop change so the composite mask and its edge can be
    measured without weights. Results are labelled ``stand-in`` in the report.
    """

    name = "stand-in"

    def set_source(self, source_bgr: np.ndarray) -> None:  # noqa: ARG002
        return None

    def swap(self, target_face_bgr: np.ndarray):
        import cv2

        from .swap.base import SwapResult

        t0 = time.perf_counter()
        out = cv2.GaussianBlur(target_face_bgr, (5, 5), 0).astype(np.float32)
        out = np.clip(out * np.array([0.82, 0.95, 1.12], dtype=np.float32) + 10.0, 0, 255).astype(np.uint8)
        return SwapResult(face_bgr=out, inference_ms=(time.perf_counter() - t0) * 1000, backend=self.name)


def parse_variant(spec: str) -> tuple[str, dict[str, Any]]:
    name, _, rest = spec.partition(":")
    settings = dict(NAMED.get(name, CURRENT))
    if name not in NAMED and not rest:
        raise ValueError(f"unknown variant {name!r}; named variants: {', '.join(NAMED)}")
    for item in filter(None, rest.split(",")):
        key, _, val = item.partition("=")
        if key not in CURRENT:
            raise ValueError(f"unknown geometry setting {key!r}; choose from {', '.join(CURRENT)}")
        settings[key] = (val.lower() in ("1", "true", "on", "yes")) if isinstance(CURRENT[key], bool) else val
    return name, settings


def synthetic_clip(frames: int = 48, size: tuple[int, int] = (960, 540), *, seed: int = 7) -> list[np.ndarray]:
    """Bundled fixture: the benchmark face moving, rolling (+-18 deg) and scaling over a textured background."""
    import cv2

    face = cv2.imread(str(Path(__file__).parent / "assets" / "bench_face.jpg"), cv2.IMREAD_COLOR)
    if face is None:
        raise RuntimeError("bundled fixture face is missing")
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([60 + 50 * xx / w, 70 + 40 * yy / h, 90 + 30 * (xx + yy) / (w + h)], axis=-1)
    bg += cv2.resize(rng.normal(0, 14, (h // 8, w // 8, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    fh, fw = face.shape[:2]
    feather = np.zeros((fh, fw), np.float32)
    cv2.ellipse(feather, (fw // 2, fh // 2), (int(fw * 0.47), int(fh * 0.48)), 0, 0, 360, 1.0, -1)
    feather = cv2.GaussianBlur(feather, (41, 41), 0)
    out = []
    for i in range(frames):
        t = i / max(1, frames - 1)
        roll = 18.0 * np.sin(2 * np.pi * t)
        scale = (h * 0.62 / fh) * (1.0 + 0.10 * np.sin(2 * np.pi * t + 1.0))
        cx = w * (0.5 + 0.12 * np.sin(2 * np.pi * t * 0.5))
        cy = h * (0.5 + 0.03 * np.cos(2 * np.pi * t))
        m = cv2.getRotationMatrix2D((fw / 2, fh / 2), roll, scale)
        m[:, 2] += (cx - fw / 2, cy - fh / 2)
        warped = cv2.warpAffine(face, m, (w, h), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(feather, m, (w, h), flags=cv2.INTER_LINEAR)[:, :, None]
        frame = warped.astype(np.float32) * alpha + bg * (1 - alpha)
        frame += rng.normal(0, 2.0, frame.shape)  # sensor noise: this is what makes detections jitter
        out.append(np.clip(frame, 0, 255).astype(np.uint8))
    return out


def load_frames(path: Path, limit: int) -> tuple[list[np.ndarray], float]:
    import cv2

    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read image: {path}")
        return [img], 24.0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, float(fps)


def _first_landmarks(detector, image: np.ndarray) -> np.ndarray | None:
    boxes = detector.detect(image)
    if not boxes or boxes[0].landmarks is None:
        return None
    return np.asarray(boxes[0].landmarks, dtype=np.float64)


def _face_region(shape: tuple[int, int], lm: np.ndarray, grow: float) -> np.ndarray:
    """Boolean face ellipse in frame space from five landmarks (same construction as landmark_mask)."""
    import cv2

    eye_mid, mouth_mid = (lm[0] + lm[1]) / 2, (lm[3] + lm[4]) / 2
    eye_dist = float(np.linalg.norm(lm[1] - lm[0]))
    eye_mouth = float(np.linalg.norm(mouth_mid - eye_mid))
    centre = eye_mid + (mouth_mid - eye_mid) * 0.45
    angle = float(np.degrees(np.arctan2(lm[1, 1] - lm[0, 1], lm[1, 0] - lm[0, 0])))
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (int(round(centre[0])), int(round(centre[1]))),
                (int(round(eye_dist * 0.98 * grow)), int(round(eye_mouth * 1.28 * grow))), angle, 0, 360, 1, -1)
    return mask.astype(bool)


def _peak_vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def run_variant(name: str, settings: dict[str, Any], frames: list[np.ndarray], *, fps: float, backend: str,
                source: np.ndarray | None, preset: str, device: str, detector) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    from .pipeline import FaceSwapPipeline, build_config

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass
    stand_in = backend == "stand-in"
    cfg = build_config(preset, device=device, backend="passthrough" if stand_in else (None if backend == "auto" else backend),
                       allow_passthrough=backend in ("passthrough", "stand-in"), watermark=False, show_metrics=False,
                       **settings)
    pipe = FaceSwapPipeline(cfg)
    if stand_in:
        pipe.swapper = StandInSwapper()
    pipe.frame_dt = 1.0 / fps
    crops: list[np.ndarray] = []
    inner_swap = pipe.swapper.swap

    def recording_swap(crop: np.ndarray):
        crops.append(crop.copy())
        return inner_swap(crop)

    pipe.swapper.swap = recording_swap  # type: ignore[method-assign]
    if source is not None:
        pipe.set_source(source)
    elif pipe.swapper.name not in ("passthrough", "stand-in"):
        raise RuntimeError(f"backend {pipe.swapper.name} needs an identity image: pass -f face.png")

    outputs: list[np.ndarray] = []
    ms: list[float] = []
    try:
        for frame in frames:
            t0 = time.perf_counter()
            out, _ = pipe.swap_frame(frame)
            ms.append((time.perf_counter() - t0) * 1000)
            outputs.append(out)
    finally:
        pipe.close()

    # Crop space: what the swap model is given.
    size = 256
    tmpl = template_points(settings["align_template"], size)
    crop_lm = [lm for lm in (_first_landmarks(detector, c) for c in crops) if lm is not None]
    crop_stats: dict[str, Any] = {"faces_found_in_crops": len(crop_lm), "crops": len(crops)}
    if crop_lm:
        arr = np.stack(crop_lm)  # n x 5 x 2
        eye_frac = np.linalg.norm(arr[:, 1] - arr[:, 0], axis=1) / size
        rolls = np.array([eye_metrics(lm)["roll_deg"] for lm in crop_lm])
        crop_stats.update({
            "eye_distance_frac_mean": round(float(eye_frac.mean()), 4),
            "eye_distance_frac_std": round(float(eye_frac.std()), 4),
            "eye_distance_frac_min_max": [round(float(eye_frac.min()), 4), round(float(eye_frac.max()), 4)],
            "roll_abs_mean_deg": round(float(np.abs(rolls).mean()), 2),
            "roll_abs_max_deg": round(float(np.abs(rolls).max()), 2),
            "landmark_drift_px": round(float(arr.std(axis=0).mean()), 3),
            "template_error_px": round(float(np.linalg.norm(arr.mean(axis=0) - tmpl, axis=1).mean()), 2),
        })

    # Frame space: the composited output against the input.
    ratios, widths, outside_frac, outside_mean, flicker = [], [], [], [], []
    prev_resid = None
    for frame, out in zip(frames, outputs, strict=True):
        lm_in = _first_landmarks(detector, frame)
        if lm_in is None:
            continue
        lm_out = _first_landmarks(detector, out)
        if lm_out is not None:
            a, b = eye_metrics(lm_in), eye_metrics(lm_out)
            ratios.append(b["eye_to_mouth_ratio"] / a["eye_to_mouth_ratio"])
            widths.append(b["eye_distance"] / a["eye_distance"])
        diff = np.abs(out.astype(np.int16) - frame.astype(np.int16)).max(axis=2)
        outside = ~_face_region(diff.shape, lm_in, 1.3)
        changed = (diff > 8) & outside
        face_px = max(1, int(_face_region(diff.shape, lm_in, 1.0).sum()))
        outside_frac.append(float(changed.sum()) / face_px)
        outside_mean.append(float(diff[changed].mean()) if changed.any() else 0.0)
        resid = out.astype(np.float32) - frame.astype(np.float32)
        if prev_resid is not None:
            region = _face_region(diff.shape, lm_in, 1.6)
            flicker.append(float(np.abs(resid - prev_resid)[region].mean()))
        prev_resid = resid

    def _stat(v: list[float], nd: int = 4) -> dict[str, float] | None:
        return {"mean": round(float(np.mean(v)), nd), "max": round(float(np.max(v)), nd)} if v else None

    from .detect import landmark_source

    result = {
        "variant": name,
        "settings": settings,
        "landmark_source": landmark_source(pipe.detector) if settings["alignment"] == "landmark" else "unused",
        "backend": pipe.swapper.name,
        "frames": len(frames),
        "crop_space": crop_stats,
        "output": {
            "eye_to_mouth_ratio_vs_input": _stat(ratios),
            "eye_distance_vs_input": _stat(widths),
            "changed_outside_face_frac": _stat(outside_frac),
            "changed_outside_face_level": _stat(outside_mean, 2),
            "flicker_residual": _stat(flicker, 3),
        },
        "runtime_ms_per_frame": {"mean": round(float(np.mean(ms)), 2), "p95": round(float(np.percentile(ms, 95)), 2)},
        "peak_vram_mb": _peak_vram_mb(),
    }
    return result, outputs, crops


def aspect_check(detector, *, preset_size: tuple[int, int] = (960, 540)) -> dict[str, Any]:
    """A 4:3 camera frame brought to a 16:9 preset: stretch (legacy) vs cover."""
    frame = synthetic_clip(1, (640, 480))[0]
    base = _first_landmarks(detector, frame)
    out: dict[str, Any] = {"capture": [640, 480], "preset": list(preset_size)}
    if base is None:
        return {**out, "error": "no face found in the fixture"}
    ref = eye_metrics(base)["eye_to_mouth_ratio"]
    for mode in ("stretch", "cover", "contain"):
        lm = _first_landmarks(detector, fit_frame(frame, preset_size, mode))
        out[mode] = None if lm is None else round(eye_metrics(lm)["eye_to_mouth_ratio"] / ref, 3)
    out["note"] = "eye-distance / eye-to-mouth relative to the camera frame; 1.0 = proportions preserved"
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (20, 10, 28), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 180), 1, cv2.LINE_AA)
    return out


def run_compare(*, input_path: Path | None, source_path: Path | None, out_dir: Path, backend: str = "auto",
                frames: int = 48, variants: list[str] | None = None, preset: str = "balanced",
                device: str = "cpu") -> dict[str, Any]:
    import cv2

    from . import __version__
    from .detect import create_detector, landmark_source
    from .models import active_backend
    from .provenance import model_record

    frames = max(1, min(int(frames), 600))
    detector = create_detector("auto")
    if input_path is None:
        clip, fps, fixture = synthetic_clip(frames), 24.0, "synthetic (bundled bench_face.jpg, roll +-18 deg)"
    else:
        clip, fps = load_frames(Path(input_path), frames)
        fixture = str(input_path)
    if backend == "auto" and active_backend() is None:
        backend = "stand-in"  # no swap model installed: still measure geometry, clearly labelled
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR) if source_path else None
    specs = [parse_variant(v) for v in (variants or ["legacy", "current"])]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, columns, crop_rows = [], [], []
    picks = sorted({0, len(clip) // 4, len(clip) // 2, (3 * len(clip)) // 4, len(clip) - 1})
    for name, settings in specs:
        res, outputs, crops = run_variant(name, settings, clip, fps=fps, backend=backend, source=source,
                                          preset=preset, device=device, detector=detector)
        for i in picks:
            cv2.imwrite(str(out_dir / f"{name}-frame{i:04d}.png"), outputs[i])
        columns.append([_label(outputs[i], f"{name} ({res['backend']}) frame {i}") for i in picks])
        if crops:
            step = max(1, len(crops) // 6)
            crop_rows.append(_label(np.hstack(crops[::step][:6]), f"{name}: crops given to the swap model"))
        results.append(res)
    rows = [np.hstack([_label(clip[i], f"input frame {i}")] + [col[k] for col in columns]) for k, i in enumerate(picks)]
    sheet = np.vstack(rows)
    if sheet.shape[1] > 2880:
        s = 2880 / sheet.shape[1]
        sheet = cv2.resize(sheet, (2880, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if crop_rows:
        width = max(r.shape[1] for r in crop_rows)
        crop_rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in crop_rows]
        cv2.imwrite(str(out_dir / "crops.jpg"), np.vstack(crop_rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

    used = results[0]["backend"] if results else backend
    report = {
        "tool": "deepfake compare", "version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixture": fixture, "frames": len(clip), "fps": fps, "preset": preset, "device": device,
        "backend": used, "model": model_record(used if used not in ("passthrough", "stand-in") else None),
        "measured_with": landmark_source(detector) + " landmarks (same instrument for every variant)",
        "real_swap_model": used not in ("passthrough", "stand-in"),
        "aspect": aspect_check(detector),
        "variants": results,
        "files": {"sheet": str(out_dir / "sheet.jpg"), "crops": str(out_dir / "crops.jpg"), "dir": str(out_dir)},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"fixture: {report['fixture']}  ({report['frames']} frames)",
             f"backend: {report['backend']}" + ("" if report["real_swap_model"] else
                                                "  [not a real swap model: output proportions are not model evidence]"),
             f"aspect 4:3 camera -> 16:9 preset (1.0 = undistorted): {json.dumps({k: report['aspect'].get(k) for k in ('stretch', 'cover', 'contain')})}",
             ""]
    keys = [("crop eye distance (frac of crop) mean / std", lambda r: f"{r['crop_space'].get('eye_distance_frac_mean')} / {r['crop_space'].get('eye_distance_frac_std')}"),
            ("crop roll, mean abs / max (deg)", lambda r: f"{r['crop_space'].get('roll_abs_mean_deg')} / {r['crop_space'].get('roll_abs_max_deg')}"),
            ("crop landmark drift (px)", lambda r: str(r["crop_space"].get("landmark_drift_px"))),
            ("output eye/mouth ratio vs input", lambda r: str((r["output"]["eye_to_mouth_ratio_vs_input"] or {}).get("mean"))),
            ("changed area outside face (x face area)", lambda r: str((r["output"]["changed_outside_face_frac"] or {}).get("mean"))),
            ("flicker of the overlay residual", lambda r: str((r["output"]["flicker_residual"] or {}).get("mean"))),
            ("ms / frame (mean)", lambda r: str(r["runtime_ms_per_frame"]["mean"])),
            ("peak VRAM (MB)", lambda r: str(r["peak_vram_mb"]))]
    names = [r["variant"] for r in report["variants"]]
    lines.append(f"{'':44}" + "".join(f"{n:>22}" for n in names))
    for label, fn in keys:
        lines.append(f"{label:44}" + "".join(f"{fn(r):>22}" for r in report["variants"]))
    lines += ["", f"previews: {report['files']['sheet']}", f"report:   {report['files']['dir']}/report.json"]
    return "\n".join(lines)
