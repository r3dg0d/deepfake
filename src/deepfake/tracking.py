"""Stable face IDs with LK motion prediction and periodic detector correction."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .detect import FaceBox


@dataclass
class FaceTrack:
    track_id: int
    box: FaceBox
    missed: int = 0
    confidence: float = 1.0


def iou(a: FaceBox, b: FaceBox) -> float:
    x, y = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    intersection = max(0, x1 - x) * max(0, y1 - y)
    return intersection / max(1, a.w * a.h + b.w * b.h - intersection)


class FaceTracker:
    def __init__(self, max_missed: int = 3) -> None:
        self.tracks: list[FaceTrack] = []
        self.gray = None
        self.next_id = 1
        self.max_missed = max_missed

    def update(self, frame: np.ndarray, detections: list[FaceBox] | None) -> list[FaceTrack]:
        import cv2

        h, w = frame.shape[:2]
        scale = min(1.0, 640 / w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.gray is not None and self.gray.shape == gray.shape:
            if np.mean(cv2.absdiff(gray, self.gray)) > 55:
                self.tracks = []
            for track in self.tracks:
                b = track.box
                mask = np.zeros_like(gray)
                x0, y0 = max(0, round(b.x * scale)), max(0, round(b.y * scale))
                x1, y1 = min(gray.shape[1], round((b.x + b.w) * scale)), min(gray.shape[0], round((b.y + b.h) * scale))
                mask[y0:y1, x0:x1] = 255
                pts = cv2.goodFeaturesToTrack(self.gray, 40, 0.02, 4, mask=mask)
                if pts is None or len(pts) < 5:
                    track.confidence *= 0.5
                    continue
                cur, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, pts, None)
                back, valid, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, cur, None)
                keep = (
                    ok.ravel().astype(bool)
                    & valid.ravel().astype(bool)
                    & (np.linalg.norm(back - pts, axis=2).ravel() < 1.5)
                )
                if keep.sum() < 5:
                    track.confidence *= 0.5
                    continue
                matrix, inliers = cv2.estimateAffinePartial2D(pts[keep], cur[keep], method=cv2.RANSAC)
                if matrix is None:
                    continue
                corners = np.array([[b.x, b.y], [b.x + b.w, b.y + b.h]], np.float32) * scale
                moved = cv2.transform(corners[None], matrix)[0] / scale
                nw, nh = moved[1] - moved[0]
                if 0.8 * b.w < nw < 1.25 * b.w and 0.8 * b.h < nh < 1.25 * b.h:
                    lm = b.landmarks
                    if lm is not None:
                        lm = cv2.transform((np.asarray(lm, np.float32) * scale)[None], matrix)[0] / scale
                    track.box = replace(
                        b, x=round(moved[0, 0]), y=round(moved[0, 1]), w=round(nw), h=round(nh), landmarks=lm
                    )
                track.confidence = float(keep.mean())
        if detections is not None:
            unmatched = set(range(len(detections)))
            pairs = sorted(
                ((iou(t.box, b), ti, bi) for ti, t in enumerate(self.tracks) for bi, b in enumerate(detections)),
                reverse=True,
            )
            matched = set()
            for overlap, ti, bi in pairs:
                if overlap < 0.25 or ti in matched or bi not in unmatched:
                    continue
                t = self.tracks[ti]
                observed = detections[bi]
                predicted = t.box
                # LK predicts current motion; smooth only detector correction,
                # rather than lagging the entire box behind physical motion.
                t.box = replace(
                    observed,
                    x=round(0.7 * observed.x + 0.3 * predicted.x),
                    y=round(0.7 * observed.y + 0.3 * predicted.y),
                    w=round(0.7 * observed.w + 0.3 * predicted.w),
                    h=round(0.7 * observed.h + 0.3 * predicted.h),
                )
                t.missed, t.confidence = 0, observed.score
                matched.add(ti)
                unmatched.remove(bi)
            for ti, t in enumerate(self.tracks):
                if ti not in matched:
                    t.missed += 1
                    t.confidence *= 0.7
            self.tracks = [t for t in self.tracks if t.missed <= self.max_missed and t.confidence > 0.25]
            for bi in sorted(unmatched):
                self.tracks.append(FaceTrack(self.next_id, detections[bi], confidence=detections[bi].score))
                self.next_id += 1
        self.gray = gray
        return sorted(self.tracks, key=lambda t: t.track_id)
