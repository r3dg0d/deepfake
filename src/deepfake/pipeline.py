"""Detect → swap → composite → watermark → sink loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .composite import ColorState, paste_face
from .detect import FaceBox, FaceDetector, align_crop, create_detector
from .metrics import MetricsTracker, format_metrics
from .occlusion import OcclusionEngine
from .presets import Preset, get_preset
from .swap.base import Swapper
from .swap.factory import create_swapper
from .tracking import FaceTracker
from .watermark import apply_watermark

if TYPE_CHECKING:
    from .framegen.settings import FrameGenSettings


@dataclass
class PipelineConfig:
    preset: Preset
    device: str = "cpu"
    backend: str | None = None
    face_index: int = 0
    multi_face: bool | None = None
    blend_feather: int | None = None
    color_match: bool | None = None
    temporal_smooth: float | None = None
    watermark: bool = True
    show_metrics: bool = True
    max_frames: int | None = None
    allow_passthrough: bool = False
    swap_precision: str = "fp32"
    async_detect: bool = False  # legacy option; tracked detection uses the current frame
    frame_gen: FrameGenSettings | None = None
    debug_overlay: bool = False
    show_mask: str | None = None


class FaceSwapPipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.detector: FaceDetector = create_detector("auto")
        self.swapper: Swapper = create_swapper(
            cfg.backend,
            device=cfg.device,
            allow_passthrough=cfg.allow_passthrough,
            precision=cfg.swap_precision,
        )
        self.metrics = MetricsTracker()
        self._async = None
        self.tracker = FaceTracker()
        self.occlusion = OcclusionEngine(cfg.device, parser_interval=1 if cfg.preset.name == "high-quality" else 2)
        if self.occlusion.models is None:
            import sys

            print(
                "deepfake: preserving original faces; occlusion unavailable: " + self.occlusion.error, file=sys.stderr
            )
        self._mask_states = {}
        self._colors: dict[int, ColorState] = {}
        self.stage_stats = {}
        self.last_masks = []
        self._frame_i = 0

    def set_source(self, source_bgr: np.ndarray) -> None:
        self.swapper.set_source(source_bgr)

    def reset_tracks(self) -> None:
        self.tracker = FaceTracker()
        self._mask_states.clear()
        self._colors.clear()
        self.last_masks = []
        self._frame_i = 0

    def close(self) -> None:
        if self._async is not None:
            self._async.close()

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, Any]:
        out, m = self.swap_frame(frame_bgr)
        return apply_watermark(out, enabled=self.cfg.watermark), m

    def swap_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, Any]:
        """Detect → swap → composite, *without* the disclosure watermark.

        Frame generation interpolates these clean frames; the watermark is
        stamped on every presented frame afterwards so it stays crisp.
        """
        t0 = time.perf_counter()
        preset = self.cfg.preset
        multi = self.cfg.multi_face if self.cfg.multi_face is not None else preset.multi_face
        feather = self.cfg.blend_feather if self.cfg.blend_feather is not None else preset.blend_feather
        color = self.cfg.color_match if self.cfg.color_match is not None else preset.color_match
        smooth = self.cfg.temporal_smooth if self.cfg.temporal_smooth is not None else preset.temporal_smooth

        infer_ms = 0.0
        detect_ms = track_ms = occlusion_ms = composite_ms = 0.0
        run_detect = (self._frame_i % max(1, preset.detect_interval)) == 0 or not self.tracker.tracks
        td = time.perf_counter()
        boxes = self.detector.detect(frame_bgr) if run_detect else None
        detect_ms = (time.perf_counter() - td) * 1000
        tt = time.perf_counter()
        tracks = self.tracker.update(frame_bgr, boxes)
        track_ms = (time.perf_counter() - tt) * 1000
        alive = {t.track_id for t in tracks}
        self._mask_states = {i: state for i, state in self._mask_states.items() if i in alive}
        self._colors = {i: state for i, state in self._colors.items() if i in alive}
        out = frame_bgr
        self.last_masks = []
        confidences, statuses = [], []
        selected = tracks if multi else sorted(tracks, key=lambda t: t.box.w * t.box.h, reverse=True)
        if not multi and selected:
            selected = [selected[min(self.cfg.face_index, len(selected) - 1)]]
        for track in selected:
            box, tid = track.box, track.track_id
            crop, paste_box = align_crop(frame_bgr, box, size=256, pad=0.35)
            pose_proxy = None
            if box.landmarks is not None:
                points = np.asarray(box.landmarks)
                eye_mid = (points[0] + points[1]) / 2
                eye_distance = float(np.linalg.norm(points[1] - points[0]))
                pose_proxy = abs(float(points[2, 0] - eye_mid[0])) / max(1.0, eye_distance)
            estimate = self.occlusion.estimate(crop, track, box.landmarks, temporal_state=self._mask_states.get(tid))
            if pose_proxy is not None and pose_proxy > 0.65:
                estimate.visible_mask[:] = 0
                estimate.status = "preserve-original (extreme yaw proxy)"
                estimate.temporal_state = None
            if track.confidence < 0.35:
                estimate.visible_mask[:] = 0
                estimate.status = "preserve-original (uncertain track)"
                estimate.temporal_state = None
            self._mask_states[tid] = estimate.temporal_state
            self.last_masks.append((paste_box, estimate, track))
            occlusion_ms += estimate.latency_ms
            confidences.append(estimate.confidence)
            statuses.append(estimate.status)
            if not np.any(estimate.visible_mask) or track.confidence < 0.35:
                continue
            result = self.swapper.swap(crop)
            infer_ms += result.inference_ms
            tc = time.perf_counter()
            out = paste_face(
                out,
                result.face_bgr,
                paste_box,
                feather=feather,
                color_match=color,
                temporal_smooth=smooth,
                visible_mask=estimate.visible_mask,
                color_state=self._colors.setdefault(tid, ColorState()),
            )
            composite_ms += (time.perf_counter() - tc) * 1000
        if self.cfg.debug_overlay or self.cfg.show_mask:
            out = self._debug(out)
        self.stage_stats = {
            "swap_backend": self.swapper.name,
            "detect_ms": round(detect_ms, 2),
            "track_ms": round(track_ms, 2),
            "occlusion_ms": round(occlusion_ms, 2),
            "composite_ms": round(composite_ms, 2),
            "occlusion_backend": self.occlusion.backend,
            "occlusion_confidence": round(min(confidences), 3) if confidences else None,
            "occlusion_status": "; ".join(statuses) if statuses else "no face",
            "tracking_status": "locked" if selected else "searching",
            "track_ids": [t.track_id for t in selected],
            "parser_refresh_interval": (
                1 if hasattr(self.occlusion.models, "parse_details") else self.occlusion.parser_interval
            ),
            "pose_measurement": "5-point nose/eye yaw proxy; not degrees"
            if selected and selected[0].box.landmarks is not None
            else "unavailable",
        }

        total_ms = (time.perf_counter() - t0) * 1000
        m = self.metrics.record(inference_ms=infer_ms, total_ms=total_ms)
        self._frame_i += 1
        return out, m

    def framegen_context(self, original: np.ndarray):
        """Full-frame masks and target pixels for coordinated FrameGen warping."""
        import cv2

        if not self.last_masks:
            return None
        h, w = original.shape[:2]
        visible, region = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        for box, estimate, _track in self.last_masks:
            x, y, bw, bh = box.x, box.y, box.w, box.h
            visible[y : y + bh, x : x + bw] = np.maximum(
                visible[y : y + bh, x : x + bw], cv2.resize(estimate.visible_mask, (bw, bh))
            )
            # The whole crop includes hair/glasses/foreground excluded by parsing.
            region[y : y + bh, x : x + bw] = 1
        return original, visible, region

    def _debug(self, out: np.ndarray) -> np.ndarray:
        import cv2

        out = out.copy()
        for box, estimate, track in self.last_masks:
            x, y, w, h = box.x, box.y, box.w, box.h
            if self.cfg.show_mask:
                key = {"face": "face_mask", "occlusion": "occluder_mask", "visible": "visible_mask"}[self.cfg.show_mask]
                mask = cv2.resize(getattr(estimate, key), (w, h))
                out[y : y + h, x : x + w] = np.repeat((mask * 255).astype(np.uint8)[:, :, None], 3, 2)
            if self.cfg.debug_overlay:
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 1)
                cv2.putText(
                    out,
                    f"ID {track.track_id} mask confidence {estimate.confidence:.2f}",
                    (x, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 220, 0),
                    1,
                )
                if track.box.landmarks is not None:
                    for lx, ly in track.box.landmarks:
                        cv2.circle(out, (int(lx), int(ly)), 2, (0, 255, 255), -1)
        return out


def motion_scaled_smoothing(smooth: float, prev: FaceBox | None, cur: FaceBox, full_at: float = 0.04) -> float:
    """Temporal blend weight that fades out as the face moves.

    Blending the previous swapped crop is only valid while the face is (almost)
    still inside the crop; with head motion it produced a visible double face.
    Displacement is measured relative to face size; beyond ``full_at`` of the
    box width per frame no blending happens.
    """
    if prev is None or smooth <= 0:
        return 0.0
    dx = (cur.x + cur.w / 2) - (prev.x + prev.w / 2)
    dy = (cur.y + cur.h / 2) - (prev.y + prev.h / 2)
    ds = abs(cur.w - prev.w)
    motion = (float(np.hypot(dx, dy)) + ds) / max(1.0, float(cur.w))
    return float(smooth * max(0.0, 1.0 - motion / full_at))


def open_capture(source: int | str):
    import cv2

    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video source: {source}")
    return cap


def load_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img


def run_loop(
    capture_source: int | str,
    source_image: Path,
    sink,
    cfg: PipelineConfig,
) -> None:
    import cv2

    pipe = FaceSwapPipeline(cfg)
    pipe.set_source(load_bgr(source_image))
    cap = open_capture(capture_source)
    try:
        # apply resolution hint
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.preset.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.preset.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.preset.fps)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != cfg.preset.width or frame.shape[0] != cfg.preset.height:
                frame = cv2.resize(frame, (cfg.preset.width, cfg.preset.height))
            out, m = pipe.process_frame(frame)
            sink.write(out)
            if cfg.show_metrics and m.frames % 15 == 0:
                print(format_metrics(m), flush=True)
            if cfg.max_frames is not None and m.frames >= cfg.max_frames:
                break
    finally:
        cap.release()
        pipe.close()
        sink.close()


def build_config(
    preset_name: str = "balanced",
    **overrides: Any,
) -> PipelineConfig:
    preset = get_preset(preset_name)
    return PipelineConfig(preset=preset, **overrides)
