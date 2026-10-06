"""Local development comparison of three clips and recomputed mask diagnostics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detect import align_crop, create_detector
from .devices import resolve_cuda
from .occlusion import OcclusionEngine
from .outputs.sinks import FFmpegPipeSink
from .tracking import FaceTracker
from .watermark import apply_watermark


def render_comparison(original: Path, before: Path, after: Path, output: Path):
    if output.resolve() in {p.resolve() for p in (original, before, after)}:
        raise ValueError("comparison must not overwrite an input")
    caps = [cv2.VideoCapture(str(p)) for p in (original, before, after)]
    if not all(c.isOpened() for c in caps):
        for cap in caps:
            cap.release()
        raise RuntimeError("cannot open comparison inputs")
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30
    sink = None
    tracker, engine, detector = FaceTracker(), OcclusionEngine(resolve_cuda("auto")), create_detector()
    states = {}
    count = 0
    try:
        sink = FFmpegPipeSink(
            [
                "-loglevel",
                "error",
                "-i",
                str(original),
                "-map",
                "0:v",
                "-map",
                "1:a?",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                str(output),
            ],
            1600,
            180,
            fps,
        )
        while True:
            ok, frame = caps[0].read()
            if not ok:
                break
            panels = [cv2.resize(frame, (320, 180))]
            for cap in caps[1:]:
                cap.set(cv2.CAP_PROP_POS_MSEC, count * 1000 / fps)
                ok, rendered = cap.read()
                if not ok:
                    raise RuntimeError("comparison clip ends before original")
                panels.append(cv2.resize(rendered, (320, 180)))
            tracks = tracker.update(frame, detector.detect(frame))
            for ident in set(states) - {track.track_id for track in tracks}:
                states.pop(ident, None)
            face_mask, occ_mask = np.zeros(frame.shape[:2], np.float32), np.zeros(frame.shape[:2], np.float32)
            if tracks:
                track = max(tracks, key=lambda t: t.box.w * t.box.h)
                ident, box = track.track_id, track.box
                crop, paste_box = align_crop(frame, box)
                result = engine.estimate(crop, track, box.landmarks, temporal_state=states.get(ident))
                states[ident] = result.temporal_state
                x, y, width, height = paste_box.x, paste_box.y, paste_box.w, paste_box.h
                for canvas, mask in ((face_mask, result.visible_mask), (occ_mask, result.occluder_mask)):
                    canvas[y : y + height, x : x + width] = cv2.resize(mask, (width, height))
            for mask in (face_mask, occ_mask):
                panels.append(cv2.cvtColor(cv2.resize((mask * 255).astype(np.uint8), (320, 180)), cv2.COLOR_GRAY2BGR))
            panel = np.hstack(panels)
            for index, title in enumerate(("Original", "Before", "After", "Recomputed visible", "Recomputed occluder")):
                cv2.putText(panel, title, (index * 320 + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            sink.write(apply_watermark(panel, enabled=True))
            count += 1
    finally:
        for cap in caps:
            cap.release()
        if sink is not None:
            sink.close()
    return {"frames": count, "fps": fps, "masks": "recomputed source-frame diagnostics"}
