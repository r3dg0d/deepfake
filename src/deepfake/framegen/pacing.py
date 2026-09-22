"""Output timeline + frame pacer for frame-generated real-time output.

Timeline model
--------------
Output frames live on a constant-rate grid ``T_j = t0 + j / output_fps``
expressed in the *capture* clock. When swapped keyframe ``k`` (captured at
``t_k``) becomes available, every grid point in ``(t_{k-1}, t_k]`` is filled:

* a grid point within ``snap`` of ``t_k`` is served by the real keyframe;
* every other grid point gets a frame interpolated at
  ``s = (T_j - t_{k-1}) / (t_k - t_{k-1})``.

So the number of generated frames per interval follows the measured source
cadence instead of a fixed count, timestamps are strictly increasing, and no
frame is ever a copy presented as "generated".

Pacing
------
The pacer presents slot ``j`` at wall time ``T_j + delay``. ``delay`` starts
at an estimate and only grows when frames arrive late (bounded by
``max_delay``), so latency cannot accumulate without limit. A slot whose
frame has not arrived at its deadline re-sends the last frame so constant-rate
sinks (files, v4l2loopback, OBS) keep valid timestamps; those are counted as
*held*, never as generated. Frames that arrive after their slot was already
presented are discarded and counted as *late*.
"""

from __future__ import annotations

import heapq
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class FrameKind(StrEnum):
    KEY = "key"  # real swapped frame
    GENERATED = "generated"  # interpolated frame
    HELD = "held"  # previous frame re-sent because the slot's frame was late


@dataclass(frozen=True)
class GridPoint:
    index: int
    time: float
    s: float | None  # None → use the keyframe itself


class OutputTimeline:
    """Maps source keyframe timestamps onto a constant-rate output grid."""

    def __init__(self, output_fps: float, snap_fraction: float = 0.5) -> None:
        if output_fps <= 0:
            raise ValueError("output_fps must be > 0")
        self.output_fps = float(output_fps)
        self.period = 1.0 / self.output_fps
        self.snap = snap_fraction * self.period
        self.t0: float | None = None
        self._next_index = 0

    def time_of(self, index: int) -> float:
        assert self.t0 is not None
        return self.t0 + index * self.period

    def start(self, t_first: float) -> GridPoint:
        """Anchor the grid on the first keyframe; slot 0 is that keyframe."""
        self.t0 = t_first
        self._next_index = 1
        return GridPoint(0, t_first, None)

    def points_between(self, t_prev: float, t_cur: float) -> list[GridPoint]:
        """Grid points in ``(t_prev, t_cur]`` (plus a snapped keyframe slot), in order."""
        if self.t0 is None:
            raise RuntimeError("start() first")
        if t_cur <= t_prev:
            return []
        dt = t_cur - t_prev
        points: list[GridPoint] = []
        # Skip grid slots that fell behind t_prev (e.g. after a stall).
        first_possible = math.floor((t_prev - self.t0) / self.period) + 1
        j = max(self._next_index, first_possible)
        key_used = False
        while True:
            tj = self.time_of(j)
            if tj > t_cur + self.snap:
                break
            if abs(tj - t_cur) <= self.snap and not key_used:
                points.append(GridPoint(j, tj, None))
                key_used = True
            elif tj < t_cur:
                s = (tj - t_prev) / dt
                if s > 0.0:
                    points.append(GridPoint(j, tj, min(s, 0.999)))
            else:
                break
            j += 1
        self._next_index = j
        return points


@dataclass
class PacerStats:
    presented: int = 0
    keyframes: int = 0
    generated: int = 0
    held: int = 0
    late_dropped: int = 0
    queue_depth: int = 0
    max_queue_depth: int = 0
    delay_ms: float = 0.0
    latency_ms_samples: list[float] = field(default_factory=list)

    def latency_p50(self) -> float | None:
        if not self.latency_ms_samples:
            return None
        s = sorted(self.latency_ms_samples[-240:])
        return s[len(s) // 2]


@dataclass(order=True)
class _Slot:
    index: int
    frame: np.ndarray = field(compare=False)
    kind: FrameKind = field(compare=False)
    capture_time: float = field(compare=False)  # capture time of the newest source frame used


class FramePacer:
    """Presents grid slots at a steady cadence on its own thread."""

    def __init__(
        self,
        timeline: OutputTimeline,
        write: Callable[[np.ndarray, int, FrameKind], None],
        *,
        initial_delay: float = 0.05,
        max_delay: float = 0.25,
        min_delay: float = 0.0,
        slack_margin: float = 0.006,
        max_queue: int = 24,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.timeline = timeline
        self._write = write
        self.delay = initial_delay
        self.max_delay = max_delay
        self.min_delay = min_delay
        self.slack_margin = slack_margin
        self._slack_window: list[float] = []
        self.max_queue = max_queue
        self._clock = clock
        self._sleep = sleep
        self._heap: list[_Slot] = []
        self._cv = threading.Condition()
        self._next_present = 0
        self._last_frame: np.ndarray | None = None
        self._stop = False
        self._drain = True
        self._thread: threading.Thread | None = None
        self.stats = PacerStats(delay_ms=initial_delay * 1000)

    # -- producer side -------------------------------------------------
    def submit(self, index: int, frame: np.ndarray, kind: FrameKind, capture_time: float) -> bool:
        """Queue slot ``index``. Returns False if it was already presented (late)."""
        with self._cv:
            if index < self._next_present:
                self.stats.late_dropped += 1
                # Arrived after its deadline: raise the delay to what this frame needed.
                self._raise_delay_to(self._clock() - self.timeline.time_of(index) + self.slack_margin)
                return False
            if len(self._heap) >= self.max_queue:
                # Runaway producer: shed the oldest queued slot rather than let latency grow.
                heapq.heappop(self._heap)
                self.stats.late_dropped += 1
            heapq.heappush(self._heap, _Slot(index, frame, kind, capture_time))
            self._observe_slack(self.timeline.time_of(index) + self.delay - self._clock())
            self.stats.queue_depth = len(self._heap)
            self.stats.max_queue_depth = max(self.stats.max_queue_depth, len(self._heap))
            self._cv.notify()
            return True

    def _observe_slack(self, slack: float) -> None:
        """Shrink the delay when every recent frame arrived with time to spare.

        Uses the minimum slack over a ~0.5 s window, keeps ``slack_margin`` in
        reserve and moves at most 20 ms per ~0.5 s window (half the spare), so latency converges down
        to what the pipeline actually needs without oscillating.
        """
        self._slack_window.append(slack)
        if len(self._slack_window) < max(8, int(self.timeline.output_fps / 2)):
            return
        spare = min(self._slack_window) - self.slack_margin
        self._slack_window.clear()
        if spare > 0:
            self.delay = max(self.min_delay, self.delay - min(spare / 2, 0.02))
            self.stats.delay_ms = self.delay * 1000

    def _raise_delay_to(self, needed: float) -> None:
        # Idempotent: several late frames from one hiccup raise it once, not cumulatively.
        # Beyond max_delay it's an outage (stall, warm-up), not a steady-state
        # need: drop those frames instead of pinning latency at the cap.
        if self.delay < needed <= self.max_delay:
            self.delay = needed
            self.stats.delay_ms = self.delay * 1000

    # -- consumer side -------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="deepfake-pacer", daemon=True)
        self._thread.start()

    def stop(self, drain: bool = True) -> None:
        with self._cv:
            self._stop = True
            self._drain = drain
            self._cv.notify()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stop and (not self._drain or not self._heap):
                    return
                if self.timeline.t0 is None or (not self._heap and self._last_frame is None):
                    self._cv.wait(timeout=0.01)
                    continue
                idx = self._next_present
            deadline = self.timeline.time_of(idx) + self.delay
            wait = deadline - self._clock()
            if wait > 0 and not self._stop:
                self._sleep(min(wait, 0.005))
                continue
            self.present_due(idx)

    def present_due(self, idx: int) -> None:
        """Present slot ``idx`` now (public for deterministic tests)."""
        with self._cv:
            # discard queued slots older than idx (already superseded)
            while self._heap and self._heap[0].index < idx:
                heapq.heappop(self._heap)
                self.stats.late_dropped += 1
            slot = None
            if self._heap and self._heap[0].index == idx:
                slot = heapq.heappop(self._heap)
            self.stats.queue_depth = len(self._heap)
            self._next_present = idx + 1
        if slot is not None:
            self._last_frame = slot.frame
            kind = slot.kind
            if kind is FrameKind.KEY:
                self.stats.keyframes += 1
            else:
                self.stats.generated += 1
            self.stats.latency_ms_samples.append((self._clock() - slot.capture_time) * 1000)
            if len(self.stats.latency_ms_samples) > 2000:
                del self.stats.latency_ms_samples[:1000]
            frame = slot.frame
        else:
            if self._last_frame is None:
                return
            kind = FrameKind.HELD
            self.stats.held += 1
            frame = self._last_frame
        self.stats.presented += 1
        self._write(frame, idx, kind)
