"""Click CLI for deepfake."""

from __future__ import annotations

import json
import os
import sys
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
from .models import install_model, list_models
from .outputs.sinks import create_sink
from .paths import ensure_dirs
from .pipeline import build_config, run_loop
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
    )


def _widget_flag(kwargs: dict) -> bool | None:
    return kwargs.get("widget")


def _run_live(kwargs: dict, source: Path, cfg, *, mode: str) -> None:
    """Webcam / virtualcam: session + FrameGen auto + Quickshell."""
    from .realtime import run_realtime

    session = DeepfakeSession(
        mode,
        source_face=source,
        input_device=str(kwargs.get("input_device") or ""),
        output_device=kwargs.get("output_device"),
        resolution=(cfg.preset.width, cfg.preset.height),
        widget=_widget_flag(kwargs),
    )
    session.start()
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
    finally:
        session.close()
    click.echo("final: " + json.dumps(final), err=True)


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
    source = _resolve_source(kwargs.get("source_path"), kwargs.get("identity"))
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
    source = _resolve_source(kwargs.get("source_path"), kwargs.get("identity"))
    cfg = _cfg_from_kwargs(kwargs)
    if kwargs.get("output_video") is None and not kwargs.get("ffmpeg_out"):
        kwargs["output_video"] = path.with_name(path.stem + "-deepfake.mp4")
        if kwargs.get("preview") is None:
            kwargs["preview"] = False
    # never overwrite the original
    out = kwargs.get("output_video")
    if out is not None and Path(out).resolve() == path.resolve():
        raise click.ClickException("refusing to overwrite the input video; pick a different -o")

    import cv2

    probe = cv2.VideoCapture(str(path))
    src_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or cfg.preset.width)
    h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or cfg.preset.height)
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe.release()

    kwargs["audio_from"] = path
    kwargs["encoder"] = ensure_default_config().encoder
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
        if not fg.enabled:
            sink = _make_sink(kwargs, cfg)
            session.mark_running()
            run_loop(str(path), source, sink, cfg)
            return
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
    source = _resolve_source(kwargs.get("source_path"), kwargs.get("identity"))
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
    from .benchmark import BenchConfig, run_benchmark
    from .session import save_framegen_cache
    import time

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
def models_list(as_json: bool):
    rows = list_models()
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


if __name__ == "__main__":
    main(prog_name="deepfake")
