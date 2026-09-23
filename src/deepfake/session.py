"""Shared DeepfakeSession — FrameGen, metrics IPC, and Quickshell lifecycle.

Processing modes (webcam / virtualcam / video) build on this so FrameGen and
the desktop widget start automatically without per-command special flags.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .config import DeepfakeConfig, ensure_default_config
from .framegen import FrameGenUnavailable, create_backend
from .framegen.settings import PRESET_FRAME_GEN, FrameGenSettings, factor_for, parse_frame_gen
from .ipc import SessionBus
from .paths import config_home
from .presets import framegen_preset_name
from .widget import WidgetHandle, cleanup_orphans, quickshell_available


def _framegen_cache_path() -> Path:
    return config_home() / "framegen_cache.json"


def load_framegen_cache() -> dict[str, Any]:
    from .ipc import read_json

    return read_json(_framegen_cache_path())


def save_framegen_cache(data: dict[str, Any]) -> None:
    from .ipc import write_json_atomic

    write_json_atomic(_framegen_cache_path(), data)


def heuristic_factor(source_fps: float, target_fps: float = 60.0) -> int:
    """Pick a multiplier that reaches target_fps without a long benchmark."""
    if source_fps <= 0:
        return 2
    return max(2, min(3, factor_for(target_fps, source_fps)))


def resolve_frame_gen(
    *,
    cli_value: str | None,
    no_frame_gen: bool,
    output_fps: float | None,
    source_fps: float,
    preset: str,
    backend: str,
    variant: str | None,
    force_bench: bool = False,
    progress: Callable[[str], None] | None = None,
) -> FrameGenSettings:
    """Default is auto-on. Explicit --no-frame-gen / --frame-gen off disables."""
    cfg = ensure_default_config()
    raw = cli_value
    if raw is None:
        raw = cfg.frame_generation or "auto"
    if no_frame_gen:
        return FrameGenSettings(enabled=False)
    try:
        enabled, factor = parse_frame_gen(raw)
    except ValueError:
        # treat unknown config strings as auto
        enabled, factor = True, None
    if output_fps is not None:
        if output_fps <= source_fps:
            raise ValueError(f"--output-fps {output_fps:g} must exceed source rate ({source_fps:g})")
        enabled = True
        factor = factor or factor_for(output_fps, source_fps)
    if not enabled:
        return FrameGenSettings(enabled=False)

    pre = PRESET_FRAME_GEN[framegen_preset_name(preset)]
    settings = FrameGenSettings(
        enabled=True,
        factor=factor,
        output_fps=output_fps,
        backend=backend or "rife",
        variant=variant or str(pre["variant"]),
        max_latency_ms=float(pre["max_latency_ms"]),
        flow_scale=pre.get("flow_scale"),  # type: ignore[arg-type]
    )
    if factor is None:  # auto
        settings = _resolve_auto(settings, source_fps=source_fps, force_bench=force_bench, progress=progress)
    return settings


def _resolve_auto(
    settings: FrameGenSettings,
    *,
    source_fps: float,
    force_bench: bool,
    progress: Callable[[str], None] | None,
) -> FrameGenSettings:
    cache = load_framegen_cache()
    age = time.time() - float(cache.get("ts") or 0)
    if not force_bench and cache.get("mode") and age < 7 * 86400:
        mode = str(cache["mode"])
        if progress:
            progress(f"frame-gen auto → {mode} (cached)")
        if mode == "off":
            return FrameGenSettings(enabled=False)
        return replace(settings, factor=int(mode.rstrip("x")))

    # Fast path: no multi-second GPU bench on every launch.
    factor = heuristic_factor(source_fps, settings.output_fps or 60.0)
    if progress:
        progress(f"frame-gen auto → {factor}x (heuristic for {source_fps:g}→{settings.output_fps or 60:g} fps)")
    save_framegen_cache({"mode": f"{factor}x", "reason": "heuristic", "source_fps": source_fps, "ts": time.time()})
    return replace(settings, factor=factor)


def try_init_framegen(fg: FrameGenSettings, width: int, height: int) -> tuple[FrameGenSettings, Any | None, str | None]:
    """Initialize the interpolation backend; on failure return disabled settings + warning."""
    if not fg.enabled:
        return fg, None, None
    try:
        backend = create_backend(fg.backend, variant=fg.variant, precision=fg.precision, flow_scale=fg.flow_scale)
        backend.initialize(width, height)
        return fg, backend, None
    except FrameGenUnavailable as e:
        return FrameGenSettings(enabled=False), None, str(e)
    except Exception as e:  # noqa: BLE001
        return FrameGenSettings(enabled=False), None, f"{type(e).__name__}: {e}"


class DeepfakeSession:
    """Owns config, IPC, Quickshell, and FrameGen policy for one processing run."""

    def __init__(
        self,
        mode: str,
        *,
        source_face: Path | None = None,
        input_device: str | None = None,
        output_device: str | None = None,
        resolution: tuple[int, int] | None = None,
        widget: bool | None = None,
        user_config: DeepfakeConfig | None = None,
    ) -> None:
        self.mode = mode
        self.source_face = source_face
        self.input_device = input_device
        self.output_device = output_device
        self.resolution = resolution
        self.config = user_config or ensure_default_config()
        self.session_id = uuid.uuid4().hex[:12]
        self.bus = SessionBus(self.session_id)
        self.widget = WidgetHandle()
        self.frame_gen = FrameGenSettings(enabled=False)
        self.frame_gen_warning: str | None = None
        self._t0 = time.monotonic()
        self._want_widget = widget
        self._stop_requested = False

    def should_start_widget(self) -> bool:
        if self._want_widget is False:
            return False
        if self._want_widget is True:
            return True
        qs = (self.config.quickshell or "auto").lower()
        if qs in ("off", "false", "0", "no"):
            return False
        if os.environ.get("DEEPFAKE_QS_PREVIEW", "1").strip() in ("0", "false", "no"):
            return False
        return True

    def start(self) -> None:
        cleanup_orphans(current_session=self.session_id)
        face_name = self.source_face.name if self.source_face else None
        w, h = self.resolution or (0, 0)
        self.bus.update(
            mode=self.mode,
            state="starting",
            source_face=face_name,
            source_face_path=str(self.source_face) if self.source_face else None,
            input_device=self.input_device,
            output_device=self.output_device,
            resolution=f"{w}x{h}" if w and h else None,
            width=w,
            height=h,
            framegen_enabled=False,
            uptime_s=0.0,
        )
        if self.should_start_widget():
            ok, detail = quickshell_available()
            if ok:
                if self.widget.start():
                    self.bus.update(widget="running", widget_message=self.widget.message)
                else:
                    print(f"⚠ Quickshell unavailable\n{self.widget.message}\nContinuing without desktop widget.", flush=True)
                    self.bus.update(widget="failed", widget_message=self.widget.message)
            else:
                print(f"⚠ Quickshell unavailable\n{detail}\nContinuing without desktop widget.", flush=True)
                self.bus.update(widget="unavailable", widget_message=detail)
        else:
            self.bus.update(widget="disabled")

    def apply_frame_gen(self, fg: FrameGenSettings, *, warning: str | None = None) -> None:
        self.frame_gen = fg
        self.frame_gen_warning = warning
        self.bus.update(
            framegen_enabled=bool(fg.enabled),
            framegen_backend=fg.backend if fg.enabled else None,
            framegen_multiplier=fg.factor if fg.enabled else None,
            framegen_variant=fg.variant if fg.enabled else None,
            framegen_warning=warning,
        )
        if warning:
            print(
                f"⚠ FrameGen unavailable\n\n{warning}\n\nFalling back:\nnative AlphaFace output",
                flush=True,
            )

    def mark_running(self) -> None:
        self.bus.update(state="running" if self.mode != "video" else "processing")

    def publish_stats(self, snap: dict[str, Any]) -> None:
        """Map realtime snapshot keys into the widget schema."""
        for cmd in self.bus.drain_controls():
            self._handle_control(cmd)
        gen = snap.get("generated_frames")
        factor = self.frame_gen.factor if self.frame_gen.enabled else None
        payload = {
            "state": "running" if self.mode != "video" else "processing",
            "source_fps": snap.get("input_fps"),
            "alphaface_fps": snap.get("swap_fps"),
            "output_fps": snap.get("output_fps") or snap.get("sink_fps"),
            "latency_ms": snap.get("total_latency_ms") or snap.get("swap_latency_ms"),
            "vram_mb": snap.get("vram_used_mb"),
            "vram_total_mb": snap.get("vram_total_mb"),
            "gpu_util_pct": snap.get("gpu_util_pct"),
            "generated_frames": gen,
            "keyframes": snap.get("keyframes"),
            "framegen_ms_per_frame": snap.get("framegen_ms_per_frame"),
            "framegen_enabled": bool(self.frame_gen.enabled),
            "framegen_multiplier": factor,
            "dropped_source_frames": snap.get("dropped_source_frames"),
            "uptime_s": round(time.monotonic() - self._t0, 1),
            "frames": (snap.get("keyframes") or 0) + (gen or 0) if gen is not None else snap.get("frames"),
        }
        # resolution from snapshot if present
        if snap.get("width"):
            payload["width"] = snap["width"]
            payload["height"] = snap.get("height")
            payload["resolution"] = f"{snap['width']}x{snap.get('height')}"
        self.bus.update(**payload)

    def publish_video_progress(self, *, pct: float, eta_s: float | None, **extra: Any) -> None:
        self.bus.update(
            state="processing",
            progress_pct=round(pct, 1),
            eta_s=None if eta_s is None else round(eta_s, 1),
            uptime_s=round(time.monotonic() - self._t0, 1),
            **extra,
        )

    def stop_requested(self) -> bool:
        for cmd in self.bus.drain_controls():
            self._handle_control(cmd)
        return self._stop_requested

    def _handle_control(self, cmd: dict[str, Any]) -> None:
        name = str(cmd.get("cmd") or "").lower()
        if name in ("stop", "quit", "exit"):
            self._stop_requested = True
            self.bus.update(state="stopping")
        elif name == "set_face" and cmd.get("path"):
            self.bus.update(pending_face=str(cmd["path"]))
        elif name == "set_frame_gen":
            self.bus.update(pending_frame_gen=str(cmd.get("value") or "auto"))

    def close(self) -> None:
        try:
            self.widget.stop()
        finally:
            self.bus.close(state="stopped")
