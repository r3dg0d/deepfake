"""Reproducible stylized foreground tests with known masks (not real-hand footage)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def foreground(frame: np.ndarray, kind: str, index: int, frames: int = 18):
    import cv2

    h, w = frame.shape[:2]
    gt = np.zeros((h, w), np.uint8)
    dx = round((index / max(1, frames - 1) - 0.5) * w * 0.6)
    c = w // 2 + dx
    if kind == "glasses":
        cv2.ellipse(gt, (c - 34, round(h * 0.42)), (25, 14), 0, 0, 360, 1, 5)
        cv2.ellipse(gt, (c + 34, round(h * 0.42)), (25, 14), 0, 0, 360, 1, 5)
        cv2.line(gt, (c - 10, round(h * 0.42)), (c + 10, round(h * 0.42)), 1, 4)
    elif kind == "hair":
        for off in range(-35, 36, 8):
            cv2.line(gt, (c + off - 40, 15), (c + off + 15, 112), 1, 5)
    elif kind == "microphone":
        cv2.ellipse(gt, (c, round(h * 0.72)), (25, 34), 0, 0, 360, 1, -1)
        cv2.rectangle(gt, (c - 12, round(h * 0.72)), (c + 12, h), 1, -1)
    elif kind == "cup":
        cv2.rectangle(gt, (c - 38, 100), (c + 38, 198), 1, -1)
        cv2.ellipse(gt, (c + 43, 145), (17, 24), 0, 0, 360, 1, 9)
    elif kind == "phone":
        cv2.rectangle(gt, (c - 34, 40), (c + 34, 220), 1, -1)
    elif kind == "full":
        gt[:] = 1
    else:
        cv2.ellipse(gt, (c, 166), (38, 45), 0, 0, 360, 1, -1)
        for off in (-27, -9, 9, 27):
            cv2.rectangle(gt, (c + off - 7, 65 + abs(off) // 2), (c + off + 7, 161), 1, -1)
    color = {
        "hand": (140, 178, 212),
        "hair": (20, 25, 30),
        "glasses": (12, 12, 12),
        "microphone": (35, 35, 35),
        "cup": (220, 185, 40),
        "phone": (24, 30, 35),
        "full": (40, 45, 50),
    }[kind]
    target = frame.copy()
    target[gt > 0] = color
    return target, gt


def run_occlusion_benchmark(device: str = "cuda", comparison: Path | None = None) -> dict:
    import cv2

    from ..benchmark import bench_face_path
    from ..composite import paste_face
    from ..detect import FaceBox, align_crop, create_detector
    from .engine import OcclusionEngine

    image = cv2.imread(str(bench_face_path()))
    boxes = create_detector().detect(image)
    if not boxes:
        raise RuntimeError("benchmark fixture face not detected")
    crop, _ = align_crop(image, boxes[0], 256)
    engine = OcclusionEngine(device, parser_interval=2)
    if engine.models is None:
        raise RuntimeError(engine.error)
    baseline = engine.estimate(crop)
    support = baseline.visible_mask > 0.5
    # A contrasting substitute makes even one leaking foreground pixel observable.
    substitute = np.full_like(crop, (220, 50, 180))
    writer = None
    if comparison:
        comparison.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(comparison), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 280))
        if not writer.isOpened():
            raise RuntimeError("comparison video writer could not open")
    rows = {}
    try:
        for kind in ("hand", "hair", "glasses", "microphone", "cup", "phone", "full"):
            state = baseline.temporal_state
            scores, leaks, retained, latencies, jitter = [], [], [], [], []
            previous = None
            old_leaks, boundary_scores = [], []
            for i in range(18):
                target, gt = foreground(crop, kind, i)
                result = engine.estimate(target, temporal_state=state)
                state = result.temporal_state
                old = paste_face(target, substitute, FaceBox(0, 0, 256, 256), color_match=False)
                new = paste_face(
                    target, substitute, FaceBox(0, 0, 256, 256), color_match=False, visible_mask=result.visible_mask
                )
                truth = (gt > 0) & support
                predicted = (result.visible_mask < 0.05) & support
                union = np.count_nonzero(truth | predicted)
                scores.append(np.count_nonzero(truth & predicted) / max(1, union))
                changed = np.any(new != target, axis=2)
                old_changed = np.any(old != target, axis=2)
                old_leaks.append(np.count_nonzero(old_changed & truth) / max(1, np.count_nonzero(truth)))

                def boundary(mask):
                    u8 = mask.astype(np.uint8)
                    return (u8 - cv2.erode(u8, np.ones((3, 3), np.uint8))) > 0

                tb, pb = boundary(truth), boundary(predicted)
                near_truth = cv2.dilate(tb.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                near_prediction = cv2.dilate(pb.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                precision = np.count_nonzero(pb & near_truth) / max(1, pb.sum())
                recall = np.count_nonzero(tb & near_prediction) / max(1, tb.sum())
                boundary_scores.append(2 * precision * recall / max(1e-8, precision + recall))
                leaks.append(np.count_nonzero(changed & truth) / max(1, np.count_nonzero(truth)))
                available = support & ~truth
                retained.append(np.count_nonzero((result.visible_mask > 0.5) & available) / max(1, available.sum()))
                latencies.append(result.latency_ms)
                if previous is not None:
                    # Raw frame-to-frame alpha delta includes true object movement;
                    # do not label this a motion-compensated flicker metric.
                    jitter.append(float(np.abs(result.visible_mask - previous).mean()))
                previous = result.visible_mask.copy()
                if writer:
                    masks = [result.face_mask, result.occluder_mask]
                    panels = [
                        target,
                        old,
                        new,
                        *[np.repeat((m * 255).astype(np.uint8)[:, :, None], 3, 2) for m in masks],
                    ]
                    labels = ["Original", "Oval (old)", "Visible composite", "Face mask", "Occlusion mask"]
                    row = np.zeros((280, 1280, 3), np.uint8)
                    for j, panel in enumerate(panels):
                        row[24:, j * 256 : (j + 1) * 256] = panel
                        cv2.putText(
                            row,
                            labels[j] + " / " + kind,
                            (j * 256 + 4, 17),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                        )
                    writer.write(row)
            static_state, static_previous, static_deltas = None, None, []
            static_frame, _ = foreground(crop, kind, 9)
            for _ in range(12):
                static = engine.estimate(static_frame, temporal_state=static_state)
                static_state = static.temporal_state
                if static_previous is not None:
                    static_deltas.append(float(np.abs(static.visible_mask - static_previous).mean()))
                static_previous = static.visible_mask.copy()
            rows[kind] = {
                "old_foreground_leak_fraction": round(float(np.mean(old_leaks)), 4),
                "boundary_f1_tolerance_2px": round(float(np.mean(boundary_scores)), 4),
                "static_alpha_delta_mean": round(float(np.mean(static_deltas)), 6),
                "mask_iou_mean": round(float(np.mean(scores)), 4),
                "foreground_leak_fraction": round(float(np.mean(leaks)), 4),
                "visible_face_retained_fraction": round(float(np.mean(retained)), 4),
                "alpha_delta_mean": round(float(np.mean(jitter)), 4),
                "occlusion_ms_p50": round(float(np.median(latencies)), 2),
                "frames": 18,
            }
    finally:
        if writer:
            writer.release()
    return {
        "fixture": "fictional face + stylized foreground shapes; NOT real-object validation",
        "backend": engine.backend,
        "rows": rows,
        "comparison": str(comparison) if comparison else None,
        "measured_at_unix": time.time(),
    }
