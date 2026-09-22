"""Detect → swap → composite → watermark → sink loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .composite import paste_face
from .detect import FaceBox, FaceDetector, align_crop, create_detector
from .metrics import MetricsTracker, format_metrics
from .presets import Preset, get_preset
from .swap.base import Swapper
from .swap.factory import create_swapper
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
    async_detect: bool = False  # live mode: detection on its own thread
    frame_gen: FrameGenSettings | None = None


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
        if cfg.async_detect:
            from .detect import AsyncDetector

            self._async = AsyncDetector(self.detector)
        self._prev_faces: dict[int, np.ndarray] = {}
        self._prev_boxes: dict[int, FaceBox] = {}
        self._alpha: dict[int, float] = {}
        self._frame_i = 0

    def set_source(self, source_bgr: np.ndarray) -> None:
        self.swapper.set_source(source_bgr)

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
        smooth = (
            self.cfg.temporal_smooth
            if self.cfg.temporal_smooth is not None
            else preset.temporal_smooth
        )

        infer_ms = 0.0
        if self._async is not None:
            self._async.submit(frame_bgr)
            boxes = self._async.latest()
            if boxes is None:  # first frame: nothing detected yet
                boxes = self.detector.detect(frame_bgr)
            self._last_boxes = boxes
        else:
            run_detect = (self._frame_i % max(1, preset.detect_interval)) == 0
            if run_detect or not hasattr(self, "_last_boxes"):
                boxes = self.detector.detect(frame_bgr)
                self._last_boxes = boxes
            else:
                boxes = self._last_boxes
        fresh = self._async is not None or boxes is not getattr(self, "_boxes_prev_frame", None)
        self._boxes_prev_frame = boxes

        out = frame_bgr
        if not boxes:
            self.metrics.mark_drop(0)
        else:
            selected: list[FaceBox]
            if multi:
                selected = boxes
            else:
                idx = min(self.cfg.face_index, len(boxes) - 1)
                selected = [boxes[idx]]

            for i, box in enumerate(selected):
                crop, paste_box = align_crop(frame_bgr, box, size=256, pad=0.35)
                result = self.swapper.swap(crop)
                infer_ms += result.inference_ms
                prev = self._prev_faces.get(i)
                if fresh:
                    alpha = motion_scaled_smoothing(smooth, self._prev_boxes.get(i), box)
                    self._alpha[i] = alpha
                else:  # reused box: it can't show motion, keep the last measured weight
                    alpha = self._alpha.get(i, 0.0)
                out = paste_face(
                    out,
                    result.face_bgr,
                    paste_box,
                    feather=feather,
                    color_match=color,
                    temporal_prev=prev,
                    temporal_smooth=alpha,
                )
                self._prev_faces[i] = result.face_bgr
                self._prev_boxes[i] = box

        total_ms = (time.perf_counter() - t0) * 1000
        m = self.metrics.record(inference_ms=infer_ms, total_ms=total_ms)
        self._frame_i += 1
        return out, m


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
        sink.close()


def build_config(
    preset_name: str = "balanced",
    **overrides: Any,
) -> PipelineConfig:
    preset = get_preset(preset_name)
    return PipelineConfig(preset=preset, **overrides)
