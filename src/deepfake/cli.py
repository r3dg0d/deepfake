"""Click CLI for deepfake."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import click

from . import __version__
from .config import DeepfakeConfig, ensure_default_config, load_config, save_config
from .consent import require_consent
from .devices import format_devices, list_v4l2_devices, resolve_cuda
from .doctor import render_doctor, run_doctor
from .framegen.registry import BACKENDS as FRAMEGEN_BACKENDS
from .framegen.settings import FrameGenSettings
from .framegen.variants import VARIANTS as RIFE_VARIANTS
from .identity import list_fakeperson_identities, resolve_source_image
from .models import CATALOG, install_model, list_models
from .outputs.sinks import create_sink
from .paths import ensure_dirs
from .pipeline import build_config
from .presets import PRESET_ALIASES, PRESETS
from .session import DeepfakeSession, resolve_frame_gen, try_init_framegen

PRESET_CHOICES = sorted([*PRESETS, *PRESET_ALIASES])


def _common_io_options(fn):
    opts = [
        click.option(
            "-f",
            "--source",
            "source_path",
            type=click.Path(path_type=Path, exists=False),
            default=None,
            help="Face image to swap in (remembered for next time).",
        ),
        click.option(
            "--identity",
            default=None,
            help="Optional fakeperson identity name (uses XDG identities dir)",
        ),
        click.option(
            "--preset",
            type=click.Choice(PRESET_CHOICES),
            default=None,
            help="latency | balanced | quality (default from config.toml)",
        ),
        click.option("--device", default=None, help="cuda|cuda:0|cpu|auto (default from config)"),
        click.option("--input-device", "input_device", default=None, help="Capture index or /dev/videoN"),
        click.option("--output-device", "output_device", default=None, help="v4l2loopback path for virtualcam"),
        click.option("--width", type=int, default=None),
        click.option("--height", type=int, default=None),
        click.option("--fps", type=int, default=None),
        click.option("--face-index", type=int, default=0, show_default=True),
        click.option("--multi-face/--single-face", default=None),
        click.option("--blend-feather", type=int, default=None),
        click.option("--color-match/--no-color-match", default=None),
        click.option("--temporal-smooth", type=float, default=None),
        click.option("--backend", type=click.Choice(["auto", "alphaface", "inswapper", "passthrough"]), default="auto"),
        click.option("--engine", type=click.Choice(["alphaface", "experimental"]), default="alphaface"),
        click.option("--mode", type=click.Choice(["face", "head", "fullbody"]), default=None),
        click.option("--model-backend", type=click.Choice(["liveportrait", "personalive", "editalive"]), default=None),
        click.option("--reference", type=click.Path(path_type=Path, exists=True), default=None),
        click.option("--prompt", default="", help="Appearance instruction for EditaLive offline rendering."),
        click.option("--encoder", type=click.Choice(["auto", "libx264", "h264_nvenc"]), default=None),
        click.option("--watermark/--no-watermark", default=None),
        click.option("--consent-ack", is_flag=True, hidden=True),
        click.option("--preview/--no-preview", default=None, help="OpenCV/file preview frames (default: on)."),
        click.option("--widget/--no-widget", default=None, help="Desktop Quickshell widget (default: auto/on)."),
        click.option("-o", "--output", "output_video", type=click.Path(path_type=Path), default=None),
        click.option("--ffmpeg-out", multiple=True, help="Extra ffmpeg argv after raw input (repeatable)"),
        click.option("--gstreamer", default=None, help="GStreamer pipeline after videoconvert (optional)"),
        click.option("--max-frames", type=int, default=None),
        click.option("--show-metrics/--hide-metrics", default=None),
        click.option(
            "--frame-gen",
            "frame_gen",
            default=None,
            is_flag=False,
            flag_value="auto",
            metavar="[auto|2x|3x|4x|off]",
            help="AI frame generation. Default: auto (on). Use --frame-gen off or --no-frame-gen to disable.",
        ),
        click.option("--no-frame-gen", "no_frame_gen", is_flag=True, help="Force frame generation off."),
        click.option(
            "--output-fps",
            type=float,
            default=None,
            help="Target presented fps (implies frame-gen). E.g. 60 or 120.",
        ),
        click.option(
            "--frame-gen-backend",
            type=click.Choice(sorted(FRAMEGEN_BACKENDS)),
            default="rife",
            show_default=True,
        ),
        click.option(
            "--frame-gen-model",
            type=click.Choice(sorted(RIFE_VARIANTS)),
            default=None,
            help="RIFE variant (default from --preset).",
        ),
        click.option(
            "--swap-precision",
            type=click.Choice(["auto", "fp32", "bf16"]),
            default="auto",
            show_default=True,
        ),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _settings_path() -> Path:
    from .paths import config_home

    return config_home() / "settings.json"


def _load_settings() -> dict:
    try:
        return json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_settings(data: dict) -> None:
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _resolve_source(source_path: Path | None, identity: str | None) -> Path:
    """--source / -f / --identity, else the last face used, else ask (terminal only)."""
    settings = _load_settings()
    try:
        p = resolve_source_image(str(source_path) if source_path else None, identity)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    if p is None and settings.get("source") and Path(settings["source"]).is_file():
        p = Path(settings["source"])
        click.echo(f"deepfake: using last face {p} (change with -f)", err=True)
    if p is None:
        ids = list_fakeperson_identities()
        if not sys.stdin.isatty():
            raise click.ClickException(
                "No face image yet: pass -f face.jpg once (it is remembered)"
                + (f" or --identity one of: {', '.join(ids)}" if ids else "")
            )
        hint = f" or a fakeperson identity ({', '.join(ids)})" if ids else ""
        answer = click.prompt(f"Face image to swap in{hint}", type=str).strip()
        try:
            p = resolve_source_image(None, answer) if answer in ids else resolve_source_image(
                os.path.expanduser(answer), None
            )
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e
    p = p.resolve()
    if settings.get("source") != str(p):
        settings["source"] = str(p)
        _save_settings(settings)
    return p


def _source_from_kwargs(kwargs):
    if kwargs.get("reference") and kwargs.get("source_path"):
        raise click.UsageError("Choose --reference or -f/--source, not both")
    return _resolve_source(kwargs.get("reference") or kwargs.get("source_path"), kwargs.get("identity"))


def _apply_config_defaults(kwargs: dict) -> dict:
    cfg = ensure_default_config()
    if kwargs.get("preset") is None:
        kwargs["preset"] = cfg.preset or "balanced"
    if kwargs.get("device") is None:
        kwargs["device"] = cfg.gpu or "auto"
    if kwargs.get("watermark") is None:
        kwargs["watermark"] = cfg.watermark
    if kwargs.get("preview") is None:
        kwargs["preview"] = cfg.preview
    if kwargs.get("show_metrics") is None:
        kwargs["show_metrics"] = cfg.show_metrics
    if kwargs.get("frame_gen") is None and not kwargs.get("no_frame_gen"):
        kwargs["frame_gen"] = cfg.frame_generation or "auto"
    return kwargs


def _swap_precision(kwargs: dict) -> str:
    from .framegen.settings import PRESET_FRAME_GEN
    from .presets import framegen_preset_name

    choice = kwargs.get("swap_precision") or "auto"
    if choice != "auto":
        return choice
    return str(PRESET_FRAME_GEN[framegen_preset_name(kwargs["preset"])]["swap_precision"])


def _fg_from_kwargs(kwargs: dict, source_fps: float) -> FrameGenSettings:
    try:
        return resolve_frame_gen(
            cli_value=kwargs.get("frame_gen"),
            no_frame_gen=bool(kwargs.get("no_frame_gen")),
            output_fps=kwargs.get("output_fps"),
            source_fps=source_fps,
            preset=kwargs["preset"],
            backend=kwargs.get("frame_gen_backend") or "rife",
            variant=kwargs.get("frame_gen_model"),
            progress=lambda m: click.echo(m, err=True),
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


def _cfg_from_kwargs(kwargs: dict):
    from .presets import Preset

    preset = kwargs["preset"]
    device = resolve_cuda(kwargs.get("device") or "auto")
    backend = kwargs.get("backend")
    if backend == "auto":
        backend = None
    cfg = build_config(
        preset,
        device=device,
        engine=kwargs.get("engine", "alphaface"),
        mode=kwargs.get("mode") or ("head" if kwargs.get("engine") == "experimental" else "face"),
        model_backend=kwargs.get("model_backend") or ("liveportrait" if kwargs.get("engine") == "experimental" and kwargs.get("mode") != "fullbody" else None),
        preview_original=bool(kwargs.get("preview", True)),
        backend=backend,
        face_index=kwargs.get("face_index") or 0,
        multi_face=kwargs.get("multi_face"),
        blend_feather=kwargs.get("blend_feather"),
        color_match=kwargs.get("color_match"),
        temporal_smooth=kwargs.get("temporal_smooth"),
        watermark=bool(kwargs.get("watermark", True)),
        show_metrics=bool(kwargs.get("show_metrics", True)),
        max_frames=kwargs.get("max_frames"),
        allow_passthrough=(backend == "passthrough"),
        swap_precision=_swap_precision(kwargs),
    )
    w, h, fps = kwargs.get("width"), kwargs.get("height"), kwargs.get("fps")
    if w or h or fps:
        p = cfg.preset
        cfg.preset = Preset(
            name=p.name,
            width=w or p.width,
            height=h or p.height,
            fps=fps or p.fps,
            detect_interval=p.detect_interval,
            temporal_smooth=p.temporal_smooth,
            blend_feather=p.blend_feather,
            color_match=p.color_match,
            multi_face=p.multi_face,
        )
    return cfg


def _make_sink(kwargs: dict, cfg, fps: float | None = None, size: tuple[int, int] | None = None):
    ffmpeg_args = list(kwargs.get("ffmpeg_out") or ())
    width, height = size or (cfg.preset.width, cfg.preset.height)
    return create_sink(
        preview=bool(kwargs.get("preview", True)),
        output_video=kwargs.get("output_video"),
        ffmpeg_args=ffmpeg_args or None,
        gstreamer_pipeline=kwargs.get("gstreamer"),
        v4l2_device=kwargs.get("output_device"),
        width=width,
        height=height,
        fps=float(fps or cfg.preset.fps),
        audio_from=kwargs.get("audio_from"),
        encoder=kwargs.get("encoder") or "auto",
        realtime_recording=bool(kwargs.get("_realtime_recording")),
    )


def _widget_flag(kwargs: dict) -> bool | None:
    return kwargs.get("widget")


def _run_live(kwargs: dict, source: Path, cfg, *, mode: str) -> None:
    """Webcam / virtualcam: session + FrameGen auto + Quickshell."""

    session = DeepfakeSession(
        mode,
        source_face=source,
        input_device=str(kwargs.get("input_device") or ""),
        output_device=kwargs.get("output_device"),
        resolution=(cfg.preset.width, cfg.preset.height),
        widget=_widget_flag(kwargs),
    )
    session.start()
    try:
        return _run_live_started(kwargs, source, cfg, mode=mode, session=session)
    except Exception as e:
        session.bus.update(state="error", error=str(e))
        raise
    finally:
        session.close()


def _run_live_started(kwargs, source, cfg, *, mode, session):
    from .realtime import run_realtime

    kwargs["_realtime_recording"] = True

    session.bus.update(engine=cfg.engine, backend=cfg.model_backend or cfg.backend or "alphaface",
                       transformation_mode=cfg.mode, output_video=str(kwargs.get("output_video") or ""))
    file_fps = _probe_file_fps(kwargs.get("input_device"))
    if file_fps:  # a video file as the camera: its real rate, not the preset's request, drives FrameGen
        cfg.preset = dataclasses.replace(cfg.preset, fps=int(round(file_fps)))
    fg = _fg_from_kwargs(kwargs, float(cfg.preset.fps))
    # Probe FrameGen early so we can fall back without aborting the session.
    if fg.enabled:
        fg, _backend, warn = try_init_framegen(fg, cfg.preset.width, cfg.preset.height)
        if _backend is not None:
            try:
                _backend.shutdown()
            except Exception:
                pass
        session.apply_frame_gen(fg, warning=warn)
    else:
        session.apply_frame_gen(fg)

    out_fps = fg.resolve_output_fps(cfg.preset.fps) if fg.enabled else float(cfg.preset.fps)
    click.echo(
        f"deepfake: {cfg.preset.width}x{cfg.preset.height} · camera {cfg.preset.fps} fps · "
        f"swap {cfg.swap_precision} · frame-gen {fg.describe()}"
        + (f" · output {out_fps:g} fps" if fg.enabled else ""),
        err=True,
    )
    sink = _make_sink(kwargs, cfg, fps=out_fps)
    cap = _parse_input_device(kwargs.get("input_device"), 0)
    session.mark_running()

    def on_stats(snap):
        session.publish_stats(snap)
        if session.stop_requested():
            raise KeyboardInterrupt

    try:
        final = run_realtime(
            cap,
            source,
            sink,
            cfg,
            fg,
            source_fps=float(cfg.preset.fps),
            print_stats=bool(kwargs.get("show_metrics", True)),
            on_stats=on_stats,
        )
    except KeyboardInterrupt:
        return
    click.echo("final: " + json.dumps(final), err=True)


def _probe_file_fps(raw: str | None) -> float | None:
    if not raw or raw.isdigit() or raw.startswith("/dev/") or not Path(raw).is_file():
        return None
    import cv2

    probe = cv2.VideoCapture(str(raw))
    try:
        fps = probe.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        probe.release()
    return float(fps) if 1.0 <= fps <= 240.0 else None


def _parse_input_device(raw: str | None, default: int = 0):
    if raw is None:
        return default
    if raw.isdigit():
        return int(raw)
    return raw


@click.group(invoke_without_command=False)
@click.version_option(__version__, prog_name="deepfake")
@click.pass_context
def main(ctx: click.Context) -> None:
    """deepfake — Real-time AI face swapping

    \b
    Usage:
      deepfake <command>

    \b
    Commands:
      webcam       Start real-time webcam face swapping
      virtualcam   Start a virtual face-swapped webcam
      video        Process an existing video
      devices      Show cameras and GPUs
      benchmark    Benchmark the pipeline
      models       Manage models
      doctor       Diagnose installation
      config       Manage preferences

    \b
    Examples:
      deepfake webcam
      deepfake webcam -f person.png
      deepfake virtualcam
      deepfake virtualcam -f person.png
      deepfake video -i input.mp4 -f person.png -o output.mp4

    Frame generation and the desktop widget start automatically.
    """
    ctx.ensure_object(dict)
    ensure_dirs()
    ensure_default_config()


@main.command("webcam")
@_common_io_options
def webcam_cmd(**kwargs):
    """Live webcam face swap (FrameGen + Quickshell on by default)."""
    kwargs = _apply_config_defaults(kwargs)
    require_consent(ack=kwargs.get("consent_ack", False), watermark=kwargs.get("watermark", True))
    source = _source_from_kwargs(kwargs)
    cfg = _cfg_from_kwargs(kwargs)
    try:
        _run_live(kwargs, source, cfg, mode="webcam")
    except click.ClickException:
        raise
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e


@main.command("video")
@click.argument("input_video", required=False, type=click.Path(path_type=Path, exists=True))
@click.option("-i", "--input", "input_opt", type=click.Path(path_type=Path, exists=True), default=None)
@_common_io_options
def video_cmd(input_video: Path | None, input_opt: Path | None, **kwargs):
    """Face-swap a video file.

    \b
      deepfake video -i input.mp4 -f person.png -o output.mp4
      deepfake video input.mp4 -f person.png
    """
    kwargs = _apply_config_defaults(kwargs)
    path = input_opt or input_video
    if path is None:
        raise click.UsageError("Provide an input video: deepfake video -i input.mp4 -f face.png -o out.mp4")
    require_consent(ack=kwargs.get("consent_ack", False), watermark=kwargs.get("watermark", True))
    source = _source_from_kwargs(kwargs)
    cfg = _cfg_from_kwargs(kwargs)
    if kwargs.get("output_video") is None and not kwargs.get("ffmpeg_out"):
        kwargs["output_video"] = path.with_name(path.stem + "-deepfake.mp4")
        if kwargs.get("preview") is None:
            kwargs["preview"] = False
    # never overwrite the original
    out = kwargs.get("output_video")
    if out is not None and Path(out).resolve() == path.resolve():
        raise click.ClickException("refusing to overwrite the input video; pick a different -o")
    if cfg.engine == "experimental" and cfg.model_backend != "liveportrait":
        return _run_experimental_video(path, source, cfg, kwargs)

    import cv2

    probe = cv2.VideoCapture(str(path))
    src_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or cfg.preset.width)
    h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or cfg.preset.height)
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe.release()

    kwargs["audio_from"] = path
    kwargs["encoder"] = kwargs.get("encoder") or ensure_default_config().encoder
    session = DeepfakeSession(
        "video",
        source_face=source,
        resolution=(w, h),
        widget=_widget_flag(kwargs),
    )
    session.bus.update(input_video=str(path), output_video=str(kwargs.get("output_video") or ""))
    session.start()
    fg = _fg_from_kwargs(kwargs, src_fps)
    if fg.enabled:
        fg, backend, warn = try_init_framegen(fg, w, h)
        if backend is not None:
            try:
                backend.shutdown()
            except Exception:
                pass
        session.apply_frame_gen(fg, warning=warn)
    else:
        session.apply_frame_gen(fg)

    try:
        from .realtime import run_offline

        session.mark_running()

        def progress_hook(snap_or_msg=None, **kw):
            # run_offline prints periodically; also publish rough progress via frames
            if isinstance(snap_or_msg, dict):
                session.publish_stats(snap_or_msg)

        summary = run_offline(
            str(path),
            source,
            lambda ww, hh, fps: _make_sink(kwargs, cfg, fps=fps, size=(ww, hh)),
            cfg,
            fg,
            print_stats=bool(kwargs.get("show_metrics", True)),
            on_stats=progress_hook,
            stop_requested=session.stop_requested,
        )
        if total > 0 and summary.get("source_frames"):
            session.publish_video_progress(
                pct=100.0 * summary["source_frames"] / max(1, total),
                eta_s=0.0,
                alphaface_fps=None,
                output_fps=summary.get("output_fps"),
                framegen_multiplier=fg.factor,
                generated_frames=summary.get("generated_frames"),
            )
        click.echo(json.dumps(summary, indent=2))
    except click.ClickException:
        raise
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    finally:
        session.close()


@main.command("virtualcam")
@_common_io_options
@click.option(
    "--v4l2",
    "v4l2_path",
    default=None,
    help="v4l2loopback device (default: auto-detect)",
)
def virtualcam_cmd(v4l2_path: str | None, **kwargs):
    """Stream swapped frames to a virtual webcam (OBS, browsers, calls)."""
    from .devices import LOOPBACK_HELP, find_loopback_device

    kwargs = _apply_config_defaults(kwargs)
    target = kwargs.get("output_device") or v4l2_path or find_loopback_device()
    if not target or not Path(target).exists():
        raise click.ClickException(LOOPBACK_HELP)
    if not os.access(target, os.W_OK):
        raise click.ClickException(f"{target} is not writable by you (need the 'video' group)")
    require_consent(ack=kwargs.get("consent_ack", False), watermark=kwargs.get("watermark", True))
    source = _source_from_kwargs(kwargs)
    kwargs["output_device"] = target
    click.echo(f"deepfake: virtual camera → {target}", err=True)
    if kwargs.get("preview") is None:
        kwargs["preview"] = False
    cfg = _cfg_from_kwargs(kwargs)
    try:
        _run_live(kwargs, source, cfg, mode="virtualcam")
    except click.ClickException:
        raise
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e


@main.command("help")
@click.argument("command", required=False)
@click.pass_context
def help_cmd(ctx: click.Context, command: str | None):
    """Show help (deepfake help webcam)."""
    parent = ctx.parent
    if command:
        cmd = main.get_command(parent, command)
        if cmd is None:
            raise click.ClickException(f"no such command: {command}")
        click.echo(cmd.get_help(click.Context(cmd, info_name=command, parent=parent)))
    else:
        click.echo(main.get_help(parent))


@main.command("devices")
@click.option("--json", "as_json", is_flag=True)
def devices_cmd(as_json: bool):
    """List V4L2 capture/output devices."""
    devices = list_v4l2_devices()
    if as_json:
        click.echo(
            json.dumps(
                [{"index": d.index, "path": d.path, "name": d.name, "kind": d.kind} for d in devices],
                indent=2,
            )
        )
        return
    click.echo(format_devices(devices))
    ids = list_fakeperson_identities()
    if ids:
        click.echo("\nfakeperson identities: " + ", ".join(ids))


@main.command("doctor")
def doctor_cmd():
    """Diagnose AlphaFace, CUDA, FrameGen, Quickshell, IPC, FFmpeg, v4l2loopback."""
    checks = run_doctor()
    click.echo(render_doctor(checks))
    critical = {"AlphaFace", "CUDA"}
    if any((not c.ok) and c.name in critical for c in checks):
        sys.exit(1)


@main.group("config")
def config_group():
    """Show or reset preference defaults (~/.config/deepfake/config.toml)."""


@config_group.command("show")
def config_show():
    cfg = load_config()
    click.echo(json.dumps(cfg.to_dict(), indent=2))


@config_group.command("reset")
def config_reset():
    path = save_config(DeepfakeConfig())
    click.echo(f"wrote {path}")


@main.command("benchmark")
@click.option("--resolution", "resolutions", multiple=True, type=click.Choice(["720p", "1080p"]))
@click.option("--camera-fps", default=30.0, show_default=True)
@click.option("--target-fps", default=60.0, show_default=True)
@click.option("--seconds", default=10.0, show_default=True)
@click.option("--frame-gen-model", type=click.Choice(sorted(RIFE_VARIANTS)), default="4.25", show_default=True)
@click.option("--swap-precision", type=click.Choice(["fp32", "bf16"]), default="bf16", show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.option("--consent-ack", is_flag=True, hidden=True)
def benchmark_cmd(resolutions, camera_fps, target_fps, seconds, frame_gen_model, swap_precision, as_json, consent_ack):
    """Benchmark AlphaFace only, FrameGen only, and the combined default pipeline."""
    require_consent(ack=consent_ack, watermark=True)
    import time

    from .benchmark import BenchConfig, run_benchmark
    from .session import save_framegen_cache

    bc = BenchConfig(
        resolutions=tuple(resolutions) or ("720p", "1080p"),
        camera_fps=camera_fps,
        target_fps=target_fps,
        seconds=seconds,
        variant=frame_gen_model,
        swap_precision=swap_precision,
    )
    try:
        report = run_benchmark(bc, progress=(lambda m: click.echo(m, err=True)))
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    # Cache recommendation for frame-gen auto
    try:
        rec = next(iter(report.get("recommendation", {}).values()))
        save_framegen_cache({"mode": rec["mode"], "reason": rec.get("reason"), "ts": time.time(), "from": "benchmark"})
    except Exception:
        pass
    if as_json:
        click.echo(json.dumps(report, indent=2))
        return
    from .report import render_benchmark

    render_benchmark(report)


@main.group("models")
def models_group():
    """List / install face-swap backends (never silent giant downloads)."""


@models_group.command("list")
@click.option("--json", "as_json", is_flag=True)
@click.option("--experimental", is_flag=True)
def models_list(as_json: bool, experimental: bool):
    rows = list_models()
    if experimental:
        rows = [r for r in rows if r.meta.get("experimental")]
    if as_json:
        click.echo(
            json.dumps(
                [{"name": r.name, "installed": r.installed, **r.meta} for r in rows],
                indent=2,
            )
        )
        return
    for r in rows:
        mark = "yes" if r.installed else "no"
        click.echo(f"{r.name}\tinstalled={mark}\tsize={r.meta.get('size_hint')}")
        click.echo(f"  source: {r.meta.get('source')}")
        click.echo(f"  code license: {r.meta.get('code_license')}")
        click.echo(f"  weights license: {r.meta.get('weights_license')}")
        if r.meta.get("notes"):
            click.echo(f"  notes: {r.meta['notes']}")


@models_group.command("install")
@click.argument("model")
@click.option("--yes", is_flag=True, help="Acknowledge license/size and proceed with download")
def models_install(model: str, yes: bool):
    try:
        msg = install_model(model, yes=yes)
    except (KeyError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(msg)


@models_group.command("info")
@click.argument("model")
def models_info(model):
    row = next((r for r in list_models() if r.name == model), None)
    if row is None:
        raise click.ClickException(f"Unknown model {model}")
    click.echo(json.dumps({"name": row.name, "installed": row.installed, **row.meta}, indent=2))


def _core_and_installed(all_models: bool, model: str | None) -> list[str]:
    from .model_store import CORE

    if all_models and model:
        raise click.UsageError("give a model name or --all, not both")
    if all_models:
        return list(CORE)
    if not model:
        raise click.UsageError("give a model name or --all")
    return [model]


def _cli_progress():
    state = {"file": None, "last": 0.0}

    def progress(filename, done, total):
        now = time.monotonic()
        if filename != state["file"] or now - state["last"] > 1.0 or (total and done >= total):
            state.update(file=filename, last=now)
            pct = f"{100 * done / total:5.1f}%" if total else ""
            click.echo(f"  {filename}: {done / 2**20:8.1f} MiB {pct}", err=True)

    return progress


@models_group.command("verify")
@click.argument("model", required=False)
@click.option("--all", "all_models", is_flag=True, help="Verify every core model (alphaface, inswapper, rife)")
@click.option("--load", is_flag=True, help="Also load each model on the GPU and run a sample inference")
@click.option("--json", "as_json", is_flag=True)
def models_verify(model, all_models, load, as_json):
    """Re-hash installed files against pinned / recorded SHA-256 (existence is not proof)."""
    from .models import verify_model

    results, failed = [], []
    for name in _core_and_installed(all_models, model):
        try:
            r = verify_model(name, load=load)
        except (KeyError, ValueError) as e:
            raise click.ClickException(str(e)) from e
        results.append(r)
        if not r["ok"]:
            failed.append(name)
        if not as_json:
            click.echo(f"{name}: {'OK' if r['ok'] else 'FAILED'}")
            for f in r.get("files", []):
                click.echo(f"  {'ok ' if f.get('ok') else 'BAD'} {f['path']}  [{f.get('trust')}]"
                           + (f"  {f['reason']}" if f.get("reason") else ""))
            for k in ("missing", "corrupt"):
                for f in r.get(k) or []:
                    click.echo(f"  BAD {f}  ({k})")
            if r.get("load_check"):
                click.echo("  load check: " + json.dumps(r["load_check"]))
    if as_json:
        click.echo(json.dumps(results if all_models else results[0], indent=2))
    if failed:
        raise click.ClickException(f"verification failed: {', '.join(failed)} "
                                   "(repair with: deepfake models reinstall <name> --yes)")


@models_group.command("reinstall")
@click.argument("model", required=False)
@click.option("--all", "all_models", is_flag=True, help="Redownload every core model")
@click.option("--yes", is_flag=True, help="Acknowledge licenses/size and redownload")
@click.option("--keep-rollback", is_flag=True, help="Keep the previous files under models/.rollback/")
@click.option("--no-load-check", is_flag=True, help="Skip the GPU load + sample inference gate")
def models_reinstall(model, all_models, yes, keep_rollback, no_load_check):
    """Force a fresh, staged download; old files stay in place until the new ones pass."""
    from .model_store import ModelStoreError, install

    names = _core_and_installed(all_models, model)
    if not yes:
        for n in names:
            m = CATALOG.get(n, {})
            click.echo(f"{n}: {m.get('size_hint')}\n  weights: {m.get('weights_license')}\n  source: {m.get('source')}")
        raise click.ClickException("re-run with --yes to redownload")
    failed = []
    for n in names:
        click.echo(f"== {n}")
        try:
            rec = install(n, force=True, progress=_cli_progress(), load_check=not no_load_check,
                          keep_rollback=keep_rollback, log=click.echo)
            check = rec.get("load_check")
            click.echo(f"{n}: OK, {len(rec['files'])} files" + (f"; load check: {json.dumps(check)}" if check else ""))
        except (ModelStoreError, KeyError, OSError) as e:
            failed.append(n)
            click.echo(f"{n}: FAILED: {e}", err=True)
    if failed:
        raise click.ClickException(f"reinstall failed for: {', '.join(failed)}")


@models_group.command("remove")
@click.argument("model")
@click.option("--yes", is_flag=True)
def models_remove(model, yes):
    from .models import remove_model

    try:
        click.echo(remove_model(model, yes=yes))
    except (KeyError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e


@main.group("experimental")
def experimental_group():
    """Inspect and configure optional, isolated character engines."""


@experimental_group.command("list-backends")
def experimental_list():
    from .engines import engine_status

    click.echo(json.dumps([r for r in engine_status() if r["engine"] == "experimental"], indent=2))


experimental_group.add_command(experimental_list, "capabilities")
experimental_group.add_command(experimental_list, "doctor")


@experimental_group.command("configure")
@click.option("--backend", required=True, type=click.Choice(["liveportrait", "personalive", "editalive"]))
@click.option("--checkout", required=True, type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--python", "python_path", default=None, type=click.Path(path_type=Path, exists=True))
def experimental_configure(backend, checkout, python_path):
    from .engines.experimental.adapters import configure_backend

    try:
        if python_path is None:
            if backend != "liveportrait":
                raise ValueError("PersonaLive and EditaLive require --python pointing to an isolated environment")
            python_path = Path(sys.executable)
        configure_backend(backend, checkout, python_path)
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Configured {backend}; run experimental doctor and models verify {backend} before rendering.")


@main.command("app")
def app_cmd():
    """Open the Quickshell control center without activating a camera."""
    import subprocess

    from .widget import find_qs, widget_shell_dir

    qs, shell = find_qs(), widget_shell_dir()
    if not qs or shell is None:
        raise click.ClickException("Install Quickshell and the packaged deepfake/quickshell assets")
    env = dict(os.environ)
    # Quickshell actions must resolve the same installed package as this CLI.
    env.setdefault("DEEPFAKE_CLI", os.environ.get("DEEPFAKE_EXECUTABLE", "deepfake"))
    shown = subprocess.run([qs, "ipc", "-p", str(shell), "call", "deepfake", "show"],
                           env=env, capture_output=True, check=False)
    if shown.returncode == 0:
        return
    subprocess.run([qs, "-n", "-d", "-p", str(shell)], env=env, check=True)


def _run_experimental_video(path, reference, cfg, kwargs):
    import tempfile
    import time

    import cv2

    from .engines.experimental.adapters import render
    from .pipeline import open_capture
    from .watermark import apply_watermark

    name = cfg.model_backend or {"head": "personalive", "fullbody": "editalive"}.get(cfg.mode)
    if name not in {"personalive", "editalive"} or (cfg.mode, name) not in {("head", "personalive"), ("fullbody", "editalive")}:
        raise click.ClickException("Choose head + personalive or fullbody + editalive. Hybrid is not implemented.")
    if kwargs.get("max_frames") or kwargs.get("output_fps") or kwargs.get("frame_gen") not in {None, "auto", "off"}:
        raise click.ClickException("Offline research adapters do not yet support frame limits or RIFE. Use --frame-gen off.")
    session = DeepfakeSession("video", source_face=reference, widget=_widget_flag(kwargs))
    session.start()
    session.bus.update(engine="experimental", backend=name, transformation_mode=cfg.mode)
    sink = cap = None
    try:
        with tempfile.TemporaryDirectory(prefix="deepfake-render-") as folder:
            generated, summary = render(name, reference, path.resolve(), Path(folder), session,
                                        prompt=kwargs.get("prompt", ""), fps=25 if name == "personalive" else 16)
            cap = open_capture(str(generated))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            kwargs["audio_from"] = path
            sink = _make_sink(kwargs, cfg, fps=fps, size=size)
            frames = 0
            start = time.monotonic()
            while True:
                if session.stop_requested():
                    raise InterruptedError("Render export cancelled")
                ok, frame = cap.read()
                if not ok:
                    break
                sink.write(apply_watermark(frame, enabled=cfg.watermark))
                frames += 1
            summary.update(output_frames=frames, output_timeline_fps=fps, export_wall_s=time.monotonic() - start,
                           streaming=False, framegen=False)
            summary.setdefault("native_inference_fps", None)
            session.bus.update(progress_pct=100, frames=frames)
            click.echo(json.dumps(summary, indent=2))
    except (RuntimeError, InterruptedError) as e:
        session.bus.update(error=str(e), state="error")
        raise click.ClickException(str(e)) from e
    finally:
        if cap is not None:
            cap.release()
        try:
            if sink is not None:
                sink.close()
        finally:
            session.close()


@main.group("desktop")
def desktop_group():
    """Quickshell control center bridge (protocol v1)."""


@desktop_group.command("request")
@click.argument("message")
def desktop_request(message):
    from .desktop import request

    try:
        response = request(json.loads(message))
        click.echo(json.dumps({"ok": True, **response}))
    except (ValueError, KeyError, RuntimeError, OSError) as e:
        click.echo(json.dumps({"ok": False, "error": str(e)}))


@desktop_group.command("install")
@click.option("--executable", required=True, type=click.Path(path_type=Path, exists=True))
def desktop_install(executable):
    """Update the existing launcher and Quickshell assets, with a backup."""
    from .desktop import install_desktop

    try:
        click.echo(json.dumps(install_desktop(executable), indent=2))
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e)) from e


@desktop_group.command("model-job", hidden=True)
@click.argument("action", type=click.Choice(["install_model", "verify_model", "remove_model", "reinstall_model"]))
@click.argument("model")
def desktop_model_job(action, model):
    from .experimental_models import install, manifests
    from .ipc import state_dir, write_json_atomic
    from .models import remove_model, verify_model

    path = state_dir() / "model-job.json"
    status = {"state": "running", "action": action, "model": model, "pid": os.getpid()}
    write_json_atomic(path, status)

    def progress(filename, done, total):
        status.update(file=filename, bytes=done, total_bytes=total)
        write_json_atomic(path, status)

    try:
        if action == "install_model":
            result = install(model, yes=True, progress=progress) if model in manifests() else install_model(model, yes=True, progress=progress)
        elif action == "reinstall_model":
            result = install_model(model, yes=True, force=True, progress=progress)
        elif action == "verify_model":
            status.update(file="re-hashing and loading on the GPU")
            write_json_atomic(path, status)
            result = verify_model(model, load=model in CATALOG)
            if not result["ok"]:
                bad = [f["path"] + ": " + str(f.get("reason")) for f in result.get("files", []) if not f.get("ok")]
                raise RuntimeError("; ".join(bad) or (result.get("load_check") or {}).get("error") or "verification failed")
        else:
            result = remove_model(model, yes=True)
        status.update(state="complete", result=result)
    except Exception as e:
        status.update(state="error", error=str(e))
    write_json_atomic(path, status)


if __name__ == "__main__":
    main(prog_name="deepfake")
