"""Threaded capture → swap → frame generation → paced output loop.

Stages run concurrently so the expensive face swap never waits on the
camera or the sink:

    capture thread ──latest frame──▶ swap (main thread) ──bounded queue──▶
    frame-gen thread ──timeline slots──▶ pacer thread ──▶ watermark ──▶ sink

The camera slot holds only the newest frame (older ones are counted as
dropped source frames), the swap→frame-gen queue holds two keyframes, and the
pacer bounds output latency (see framegen/pacing.py). Nothing grows without
limit when the GPU falls behind.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .framegen import create_backend
from .framegen.pacing import FrameKind, FramePacer, OutputTimeline
from .framegen.settings import FrameGenSettings
from .pipeline import FaceSwapPipeline, PipelineConfig, load_bgr, open_capture
from .watermark import apply_watermark


class RateMeter:
    """Events per second over a sliding window."""

    def __init__(self, window: float = 2.0) -> None:
        self.window = window
        self._t: deque[float] = deque()

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._t.append(now)
        cutoff = now - self.window
        while self._t and self._t[0] < cutoff:
            self._t.popleft()

    def rate(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


@dataclass
class LiveStats:
    capture: RateMeter = field(default_factory=RateMeter)
    swapped: RateMeter = field(default_factory=RateMeter)
    presented: RateMeter = field(default_factory=RateMeter)  # every frame handed to the sink
    fresh: RateMeter = field(default_factory=RateMeter)  # excludes held (repeated) frames
    swap_ms: deque = field(default_factory=lambda: deque(maxlen=120))
    framegen_ms: deque = field(default_factory=lambda: deque(maxlen=120))
    direct_latency_ms: deque = field(default_factory=lambda: deque(maxlen=240))
    source_dropped: int = 0
    swap_queue_dropped: int = 0
    shed_level: int = 0
    gpu_util: float | None = None
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None

    @staticmethod
    def _p50(d: deque) -> float | None:
        if not d:
            return None
        s = sorted(d)
        return s[len(s) // 2]

    def snapshot(self, pacer: FramePacer | None) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "input_fps": round(self.capture.rate(), 1),
            "swap_fps": round(self.swapped.rate(), 1),
            # output_fps counts only new frames (keyframes + generated); sink_fps
            # also counts held repeats that keep constant-rate sinks fed.
            "output_fps": round(self.fresh.rate(), 1),
            "sink_fps": round(self.presented.rate(), 1),
            "swap_latency_ms": _r(self._p50(self.swap_ms)),
            "framegen_ms_per_frame": _r(self._p50(self.framegen_ms)),
            "dropped_source_frames": self.source_dropped + self.swap_queue_dropped,
            "gpu_util_pct": self.gpu_util,
            "vram_used_mb": self.vram_used_mb,
        }
        if pacer is None:
            snap["total_latency_ms"] = _r(self._p50(self.direct_latency_ms))
        else:
            st = pacer.stats
            snap.update(
                {
                    "total_latency_ms": _r(st.latency_p50()),
                    "generated_frames": st.generated,
                    "keyframes": st.keyframes,
                    "held_frames": st.held,
                    "late_dropped": st.late_dropped,
                    "queue_depth": st.queue_depth,
                    "pacer_delay_ms": round(pacer.delay * 1000, 1),
                    "framegen_shed_level": self.shed_level,
                }
            )
        return snap


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


def format_live(s: dict[str, Any]) -> str:
    def f(key: str, unit: str = "") -> str:
        v = s.get(key)
        return "–" if v is None else f"{v}{unit}"

    line = (
        f"in {f('input_fps')} fps │ swap {f('swap_fps')} fps {f('swap_latency_ms', 'ms')} │ out {f('output_fps')} fps"
    )
    if "generated_frames" in s:
        line += (
            f" │ gen {f('framegen_ms_per_frame', 'ms')}/f {s['generated_frames']} │ "
            f"latency {f('total_latency_ms', 'ms')} │ q {s['queue_depth']} │ "
            f"held {s['held_frames']} late {s['late_dropped']}"
        )
    line += f" │ drop {s['dropped_source_frames']}"
    if s.get("gpu_util_pct") is not None:
        line += f" │ gpu {s['gpu_util_pct']:.0f}% {s['vram_used_mb']:.0f}MB"
    return line


class GpuPoller(threading.Thread):
    """Low-rate GPU utilisation / VRAM sampling (NVML if present, else nvidia-smi)."""

    def __init__(self, stats: LiveStats, interval: float = 1.0) -> None:
        super().__init__(name="deepfake-gpu", daemon=True)
        self.stats, self.interval = stats, interval
        self._halt = threading.Event()
        self._nvml = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:
            self._smi = shutil.which("nvidia-smi")

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            try:
                if self._nvml:
                    nv, h = self._nvml
                    self.stats.gpu_util = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
                    mem = nv.nvmlDeviceGetMemoryInfo(h)
                    self.stats.vram_used_mb = mem.used / 2**20
                    self.stats.vram_total_mb = mem.total / 2**20
                elif self._smi:
                    out = (
                        subprocess.run(
                            [
                                self._smi,
                                "--query-gpu=utilization.gpu,memory.used,memory.total",
                                "--format=csv,noheader,nounits",
                                "-i",
                                "0",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=3,
                            check=False,
                        )
                        .stdout.strip()
                        .split(",")
                    )
                    if len(out) == 3:
                        self.stats.gpu_util = float(out[0])
                        self.stats.vram_used_mb = float(out[1])
                        self.stats.vram_total_mb = float(out[2])
            except Exception:
                pass

    def stop(self) -> None:
        self._halt.set()


class CaptureThread(threading.Thread):
    """Reads the camera continuously; keeps only the newest frame."""

    def __init__(self, cap, size: tuple[int, int], stats: LiveStats) -> None:
        super().__init__(name="deepfake-capture", daemon=True)
        self.cap, self.size, self.stats = cap, size, stats
        self._cv = threading.Condition()
        self._latest: tuple[float, np.ndarray] | None = None
        self._seq = 0
        self._taken = 0
        self.eof = False
        self._halt = threading.Event()

    def run(self) -> None:
        import cv2

        while not self._halt.is_set():
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok:
                with self._cv:
                    self.eof = True
                    self._cv.notify_all()
                return
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size)
            self.stats.capture.tick(t)
            with self._cv:
                if self._latest is not None and self._taken < self._seq:
                    self.stats.source_dropped += 1  # overwritten before the swap stage took it
                self._latest = (t, frame)
                self._seq += 1
                self._cv.notify_all()

    def take(self, timeout: float = 1.0) -> tuple[float, np.ndarray] | None:
        with self._cv:
            if not self._cv.wait_for(lambda: self._taken < self._seq or self.eof, timeout=timeout):
                return None
            if self._taken >= self._seq:
                return None
            self._taken = self._seq
            return self._latest

    def stop(self) -> None:
        self._halt.set()


class ShedController:
    """Degrades frame generation instead of letting the whole pipeline stall.

    Level 0 generates every slot; level 1 keeps only the slot nearest each
    interval's midpoint; level 2 generates nothing (keyframes only, the pacer
    holds). The level rises when more than ``high`` of the slots presented in
    the last window arrived late, and falls after ``cooldown`` seconds
    below ``low``, so the swap stage keeps camera rate under GPU contention.
    """

    def __init__(self, window: float = 1.0, high: float = 0.15, low: float = 0.03, cooldown: float = 3.0) -> None:
        self.window, self.high, self.low, self.cooldown = window, high, low, cooldown
        self.level = 0
        self._t0: float | None = None
        self._bad0 = self._total0 = 0
        self._calm_since: float | None = None

    def update(self, now: float, bad: int, total: int) -> int:
        if self._t0 is None:
            self._t0, self._bad0, self._total0 = now, bad, total
            return self.level
        if now - self._t0 < self.window:
            return self.level
        slots = max(1, total - self._total0)
        ratio = (bad - self._bad0) / slots
        self._t0, self._bad0, self._total0 = now, bad, total
        if ratio > self.high and self.level < 2:
            self.level += 1
            self._calm_since = None
        elif ratio < self.low and self.level > 0:
            self._calm_since = self._calm_since or now
            if now - self._calm_since >= self.cooldown:
                self.level -= 1
                self._calm_since = None
        else:
            self._calm_since = None
        return self.level

    @staticmethod
    def select(points: list, level: int) -> list:
        if level <= 0 or not points:
            return points
        if level >= 2:
            return []
        return [min(points, key=lambda p: abs(p.s - 0.5))]


class FrameGenWorker(threading.Thread):
    """Turns swapped keyframes into timeline slots (key + interpolated)."""

    def __init__(self, backend, timeline: OutputTimeline, pacer: FramePacer, stats: LiveStats) -> None:
        super().__init__(name="deepfake-framegen", daemon=True)
        self.backend, self.timeline, self.pacer, self.stats = backend, timeline, pacer, stats
        self.inbox: queue.Queue[tuple[float, np.ndarray] | None] = queue.Queue(maxsize=2)
        self._prev_t: float | None = None
        self.error: BaseException | None = None
        self.shed = ShedController()

    def offer(self, item: tuple[float, np.ndarray]) -> None:
        """Non-blocking hand-off; if the worker is behind, the oldest keyframe is dropped."""
        while True:
            try:
                self.inbox.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.inbox.get_nowait()
                    self.stats.swap_queue_dropped += 1
                except queue.Empty:
                    pass

    def run(self) -> None:
        try:
            while True:
                item = self.inbox.get()
                if item is None:
                    return
                self.process(*item)
        except BaseException as e:  # surfaced by the main thread
            self.error = e

    def process(self, t: float, frame: np.ndarray) -> None:
        if self._prev_t is None:
            self.backend.push(frame)
            slot = self.timeline.start(t)
            self.pacer.submit(slot.index, frame, FrameKind.KEY, t)
            self._prev_t = t
            return
        points = self.timeline.points_between(self._prev_t, t)
        self.backend.push(frame)
        st = self.pacer.stats
        # Late arrivals (not holds: at level 2 holding is the point) drive the level.
        level = self.shed.update(time.monotonic(), st.late_dropped, st.presented)
        self.stats.shed_level = level
        interp = self.shed.select([p for p in points if p.s is not None], level)
        if interp:
            t0 = time.perf_counter()
            frames = self.backend.interpolate([p.s for p in interp])
            self.stats.framegen_ms.append((time.perf_counter() - t0) * 1000 / len(interp))
            for p, f in zip(interp, frames, strict=True):
                self.pacer.submit(p.index, f, FrameKind.GENERATED, t)
        for p in points:
            if p.s is None:
                self.pacer.submit(p.index, frame, FrameKind.KEY, t)
        self._prev_t = t


def run_realtime(
    capture_source: int | str | None,
    source_image: Path | None,
    sink,
    cfg: PipelineConfig,
    fg: FrameGenSettings,
    *,
    source_fps: float,
    print_stats: bool = True,
    duration_s: float | None = None,
    capture=None,
    on_stats=None,
    source_bgr: np.ndarray | None = None,
) -> dict[str, Any]:
    """Live loop with frame generation. Returns the final stats snapshot.

    ``capture`` may be any object with OpenCV's ``read()/set()/release()``
    (used by the benchmark's synthetic camera); default opens ``capture_source``.
    """
    from dataclasses import replace as _replace

    import cv2

    pipe = FaceSwapPipeline(_replace(cfg, async_detect=True))
    pipe.set_source(source_bgr if source_bgr is not None else load_bgr(source_image))
    size = (cfg.preset.width, cfg.preset.height)
    stats = LiveStats()
    backend = pacer = worker = None
    # Warm the swap path (cuDNN autotune, allocator) before the clock starts.
    for _ in range(3):
        pipe.swap_frame(np.full((size[1], size[0], 3), 96, dtype=np.uint8))

    if fg.enabled:
        out_fps = fg.resolve_output_fps(source_fps)
        backend = create_backend(fg.backend, variant=fg.variant, precision=fg.precision, flow_scale=fg.flow_scale)
        backend.initialize(*size)  # raises FrameGenUnavailable with an actionable message

        def write(frame: np.ndarray, index: int, kind: FrameKind) -> None:
            sink.write(apply_watermark(frame, enabled=cfg.watermark))
            stats.presented.tick()
            if kind is not FrameKind.HELD:
                stats.fresh.tick()

        timeline = OutputTimeline(out_fps)
        pacer = FramePacer(
            timeline,
            write,
            initial_delay=min(0.06, fg.max_latency_ms / 1000 / 2),
            max_delay=fg.max_latency_ms / 1000,
            max_queue=int(out_fps * fg.max_latency_ms / 1000) + 4,
        )
        worker = FrameGenWorker(backend, timeline, pacer, stats)
    cap = capture if capture is not None else open_capture(capture_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, source_fps)
    capture = CaptureThread(cap, size, stats)
    gpu = GpuPoller(stats)
    for th in (capture, worker, gpu):
        if th is not None:
            th.start()
    if pacer is not None:
        pacer.start()
    started = time.monotonic()
    last_print = started
    swapped = 0
    try:
        while True:
            if worker is not None and worker.error is not None:
                raise worker.error
            item = capture.take(timeout=2.0)
            if item is None:
                if capture.eof:
                    break
                continue
            t_cap, frame = item
            t0 = time.perf_counter()
            out, _ = pipe.swap_frame(frame)
            stats.swap_ms.append((time.perf_counter() - t0) * 1000)
            stats.swapped.tick()
            if worker is not None:
                worker.offer((t_cap, out))
            else:
                sink.write(apply_watermark(out, enabled=cfg.watermark))
                stats.presented.tick()
                stats.fresh.tick()
                stats.direct_latency_ms.append((time.monotonic() - t_cap) * 1000)
            swapped += 1
            now = time.monotonic()
            if now - last_print >= 1.0:
                snap = stats.snapshot(pacer)
                if print_stats:
                    print(format_live(snap), flush=True)
                if on_stats is not None:
                    on_stats(snap)
                last_print = now
            if cfg.max_frames is not None and swapped >= cfg.max_frames:
                break
            if duration_s is not None and now - started >= duration_s:
                break
    finally:
        capture.stop()
        if worker is not None:
            worker.inbox.put(None)
            worker.join(timeout=5)
        if pacer is not None:
            pacer.stop(drain=True)
        gpu.stop()
        cap.release()
        pipe.close()
        final = stats.snapshot(pacer)
        if backend is not None:
            backend.shutdown()
        sink.close()
    return final


def run_offline(
    input_video: str,
    source_image: Path,
    sink_factory,
    cfg: PipelineConfig,
    fg: FrameGenSettings | None,
    *,
    print_stats: bool = True,
) -> dict[str, Any]:
    """Deterministic file → file processing (optionally frame-generated).

    Source timestamps come from the file's frame rate, so the output timeline
    is exact: ``output_fps = source_fps × factor`` (or ``--output-fps``).
    """
    import cv2

    pipe = FaceSwapPipeline(cfg)
    pipe.set_source(load_bgr(source_image))
    cap = open_capture(input_video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.preset.width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.preset.height
    out_fps = fg.resolve_output_fps(src_fps) if fg and fg.enabled else src_fps
    sink = sink_factory(w, h, out_fps)
    counts = {"source_frames": 0, "keyframes": 0, "generated_frames": 0}
    swap_ms: list[float] = []
    gen_ms: list[float] = []
    backend = None
    timeline = OutputTimeline(out_fps) if fg and fg.enabled else None
    if timeline is not None:
        backend = create_backend(fg.backend, variant=fg.variant, precision=fg.precision, flow_scale=fg.flow_scale)
        backend.initialize(w, h)

    def emit(frame: np.ndarray) -> None:
        sink.write(apply_watermark(frame, enabled=cfg.watermark))

    t_start = time.perf_counter()
    prev_t: float | None = None
    k = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            out, _ = pipe.swap_frame(frame)
            swap_ms.append((time.perf_counter() - t0) * 1000)
            counts["source_frames"] += 1
            t = k / src_fps
            k += 1
            if timeline is None:
                emit(out)
                counts["keyframes"] += 1
            elif prev_t is None:
                backend.push(out)
                timeline.start(t)
                emit(out)
                counts["keyframes"] += 1
            else:
                points = timeline.points_between(prev_t, t)
                backend.push(out)
                interp = [p for p in points if p.s is not None]
                gen = []
                if interp:
                    t1 = time.perf_counter()
                    gen = backend.interpolate([p.s for p in interp])
                    gen_ms.append((time.perf_counter() - t1) * 1000 / len(interp))
                gi = iter(gen)
                for p in points:
                    if p.s is None:
                        emit(out)
                        counts["keyframes"] += 1
                    else:
                        emit(next(gi))
                        counts["generated_frames"] += 1
            prev_t = t
            if cfg.max_frames is not None and counts["source_frames"] >= cfg.max_frames:
                break
            if print_stats and counts["source_frames"] % 30 == 0:
                print(
                    f"{counts['source_frames']} src → {counts['keyframes'] + counts['generated_frames']} out frames",
                    flush=True,
                )
    finally:
        cap.release()
        sink.close()
        if backend is not None:
            backend.shutdown()
    elapsed = time.perf_counter() - t_start
    med = lambda xs: None if not xs else round(sorted(xs)[len(xs) // 2], 2)  # noqa: E731
    return {
        **counts,
        "source_fps": round(src_fps, 3),
        "output_fps": round(out_fps, 3),
        "swap_ms_p50": med(swap_ms),
        "framegen_ms_per_frame_p50": med(gen_ms),
        "wall_s": round(elapsed, 2),
    }
