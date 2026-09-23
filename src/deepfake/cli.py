"""Click CLI for deepfake."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from . import __version__
from .consent import require_consent
from .devices import format_devices, list_v4l2_devices, resolve_cuda
from .framegen.registry import BACKENDS as FRAMEGEN_BACKENDS
from .framegen.settings import PRESET_FRAME_GEN, FrameGenSettings, factor_for, parse_frame_gen
from .framegen.variants import VARIANTS as RIFE_VARIANTS
from .identity import list_fakeperson_identities, resolve_source_image
from .models import install_model, list_models
from .outputs.sinks import create_sink
from .paths import ensure_dirs
from .pipeline import build_config, run_loop
from .presets import PRESET_ALIASES, PRESETS, framegen_preset_name

PRESET_CHOICES = sorted([*PRESETS, *PRESET_ALIASES])


def _common_io_options(fn):
    opts = [
        click.option(
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
            default="balanced",
            show_default=True,
            help="latency (=low-latency) | balanced | quality (=high-quality)",
        ),
        click.option("--device", default="auto", show_default=True, help="cuda|cuda:0|cpu|auto"),
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
        click.option("--watermark/--no-watermark", default=True, show_default=True),
        click.option("--consent-ack", is_flag=True, hidden=True),
        click.option("--preview/--no-preview", default=True),
        click.option("-o", "--output", "output_video", type=click.Path(path_type=Path), default=None),
        click.option("--ffmpeg-out", multiple=True, help="Extra ffmpeg argv after raw input (repeatable)"),
        click.option("--gstreamer", default=None, help="GStreamer pipeline after videoconvert (optional)"),
        click.option("--max-frames", type=int, default=None),
        click.option("--show-metrics/--hide-metrics", default=True),
        click.option(
            "--frame-gen",
            "frame_gen",
            default=None,
            is_flag=False,
            flag_value="on",
            metavar="[2x|3x|4x|auto]",
            help="AI frame generation (RIFE interpolation between swapped frames). "
            "Bare flag = 2x; 'auto' benchmarks first and picks a mode. Off by default.",
        ),
        click.option("--no-frame-gen", "no_frame_gen", is_flag=True, help="Force frame generation off."),
        click.option(
            "--output-fps",
            type=float,
            default=None,
            help="Target presented fps (implies --frame-gen). E.g. 60 or 120.",
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
            help="RIFE variant (default from --preset: latency=4.25.lite, balanced=4.25, quality=4.26).",
        ),
        click.option(
            "--swap-precision",
            type=click.Choice(["auto", "fp32", "bf16"]),
            default="auto",
            show_default=True,
            help="AlphaFace compute precision (auto: bf16 for latency/balanced, fp32 for quality).",
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
    """--source / --identity, else the last face used, else ask (terminal only)."""
    settings = _load_settings()
    try:
        p = resolve_source_image(str(source_path) if source_path else None, identity)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    if p is None and settings.get("source") and Path(settings["source"]).is_file():
        p = Path(settings["source"])
        click.echo(f"deepfake: using last face {p} (change with --source)", err=True)
    if p is None:
        ids = list_fakeperson_identities()
        if not sys.stdin.isatty():
            raise click.ClickException(
                "No face image yet: pass --source face.jpg once (it is remembered)"
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


def _swap_precision(kwargs: dict) -> str:
    choice = kwargs.get("swap_precision") or "auto"
    if choice != "auto":
        return choice
    return str(PRESET_FRAME_GEN[framegen_preset_name(kwargs["preset"])]["swap_precision"])


def _fg_from_kwargs(kwargs: dict, source_fps: float) -> FrameGenSettings:
    """Resolve --frame-gen / --output-fps / --no-frame-gen into settings."""
    if kwargs.get("no_frame_gen"):
        return FrameGenSettings(enabled=False)
    try:
        enabled, factor = parse_frame_gen(kwargs.get("frame_gen"))
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--frame-gen") from e
    out_fps = kwargs.get("output_fps")
    if out_fps is not None:
        if out_fps <= source_fps:
            raise click.BadParameter(
                f"--output-fps {out_fps:g} must exceed the source rate ({source_fps:g} fps)",
                param_hint="--output-fps",
            )
        enabled = True
        factor = factor or factor_for(out_fps, source_fps)
    if not enabled:
        return FrameGenSettings(enabled=False)
    pre = PRESET_FRAME_GEN[framegen_preset_name(kwargs["preset"])]
    variant = kwargs.get("frame_gen_model") or str(pre["variant"])
    settings = FrameGenSettings(
        enabled=True,
        factor=factor,
        output_fps=out_fps,
        backend=kwargs.get("frame_gen_backend") or "rife",
        variant=variant,
        max_latency_ms=float(pre["max_latency_ms"]),
        flow_scale=pre.get("flow_scale"),  # type: ignore[arg-type]
    )
    if factor is None:  # --frame-gen auto
        settings = _auto_frame_gen(settings, kwargs, source_fps)
    return settings


def _auto_frame_gen(settings: FrameGenSettings, kwargs: dict, source_fps: float) -> FrameGenSettings:
    """Measure this machine briefly and pick a multiplier (or off)."""
    from dataclasses import replace

    from .benchmark import BenchConfig, run_benchmark

    cfg = _cfg_from_kwargs(kwargs)
    res = f"{cfg.preset.height}p"
    known = {"720p", "1080p"}
    bc = BenchConfig(
        resolutions=(res if res in known else "720p",),
        camera_fps=source_fps,
        target_fps=kwargs.get("output_fps") or 60.0,
        seconds=5.0,
        factors=(2, 3),
        variant=settings.variant,
        swap_precision=_swap_precision(kwargs),
    )
    click.echo("frame-gen auto: measuring this GPU (≈30 s) …", err=True)
    report = run_benchmark(bc, progress=lambda m: click.echo(f"  {m}", err=True))
    rec = next(iter(report["recommendation"].values()))
    click.echo(f"frame-gen auto → {rec['mode']} ({rec['reason']})", err=True)
    if rec["mode"] == "off":
        return FrameGenSettings(enabled=False)
    return replace(settings, factor=int(rec["mode"].rstrip("x")))


def _cfg_from_kwargs(kwargs: dict):
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
    # resolution overrides
    w, h, fps = kwargs.get("width"), kwargs.get("height"), kwargs.get("fps")
    if w or h or fps:
        from dataclasses import replace

        from .presets import Preset

        p = cfg.preset
        cfg = build_config(
            preset,
            device=cfg.device,
            backend=cfg.backend,
            face_index=cfg.face_index,
            multi_face=cfg.multi_face,
            blend_feather=cfg.blend_feather,
            color_match=cfg.color_match,
            temporal_smooth=cfg.temporal_smooth,
            watermark=cfg.watermark,
            show_metrics=cfg.show_metrics,
            max_frames=cfg.max_frames,
            allow_passthrough=cfg.allow_passthrough,
            swap_precision=cfg.swap_precision,
        )
        # rebuild preset with overrides via object replace on PipelineConfig.preset
        new_p = Preset(
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
        cfg.preset = new_p
        _ = replace  # silence lint if unused
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
    )


def _run_live(kwargs: dict, source: Path, cfg) -> None:
    """Webcam / virtualcam: threaded pipeline, optional frame generation."""
    from .framegen import FrameGenUnavailable
    from .realtime import run_realtime

    fg = _fg_from_kwargs(kwargs, float(cfg.preset.fps))
    out_fps = fg.resolve_output_fps(cfg.preset.fps) if fg.enabled else float(cfg.preset.fps)
    click.echo(
        f"deepfake: {cfg.preset.width}x{cfg.preset.height} camera {cfg.preset.fps} fps · "
        f"swap {cfg.swap_precision} · frame-gen {fg.describe()}"
        + (f" · output {out_fps:g} fps" if fg.enabled else ""),
        err=True,
    )
    sink = _make_sink(kwargs, cfg, fps=out_fps)
    cap = _parse_input_device(kwargs.get("input_device"), 0)
    try:
        final = run_realtime(
            cap, source, sink, cfg, fg, source_fps=float(cfg.preset.fps),
            print_stats=bool(kwargs.get("show_metrics", True)),
        )
    except FrameGenUnavailable as e:
        sink.close()
        raise click.ClickException(f"{e}  (or run without --frame-gen)") from e
    except KeyboardInterrupt:
        return
    click.echo("final: " + json.dumps(final), err=True)


def _parse_input_device(raw: str | None, default: int = 0):
    if raw is None:
        return default
    if raw.isdigit():
        return int(raw)
    return raw


@click.group()
@click.version_option(__version__, prog_name="deepfake")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Linux real-time face-swap CLI (AlphaFace research wrapper).

    \b
    Quick start:
      deepfake webcam --source face.jpg   first time (the face is remembered)
      deepfake webcam                     live preview
      deepfake virtualcam                 virtual camera for OBS / browsers / calls
      deepfake video clip.mp4             swap a file
      deepfake devices                    cameras and virtual cameras

    For research, VFX, avatars, filmmaking and consenting demos; output is
    watermarked "SYNTHETIC MEDIA" by default.
    """
    ctx.ensure_object(dict)
    ensure_dirs()


@main.command("webcam")
@_common_io_options
def webcam_cmd(**kwargs):
    """Live webcam face swap with optional --source person.jpg."""
    require_consent(ack=kwargs.get("consent_ack", False), watermark=kwargs.get("watermark", True))
    source = _resolve_source(kwargs.get("source_path"), kwargs.get("identity"))
    cfg = _cfg_from_kwargs(kwargs)
    try:
        _run_live(kwargs, source, cfg)
    except click.ClickException:
        raise
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e


@main.command("video")
@click.argument("input_video", type=click.Path(path_type=Path, exists=True))
@_common_io_options
def video_cmd(input_video: Path, **kwargs):
    """Face-swap an input video file."""
    require_consent(ack=kwargs.get("consent_ack", False), watermark=kwargs.get("watermark", True))
    source = _resolve_source(kwargs.get("source_path"), kwargs.get("identity"))
    cfg = _cfg_from_kwargs(kwargs)
    if kwargs.get("output_video") is None and not kwargs.get("ffmpeg_out"):
        # default save next to input
        kwargs["output_video"] = input_video.with_name(input_video.stem + "_swapped.mp4")
        kwargs["preview"] = kwargs.get("preview", False)
    import cv2

    probe = cv2.VideoCapture(str(input_video))
    src_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()
    fg = _fg_from_kwargs(kwargs, src_fps)
    try:
        if not fg.enabled:
            sink = _make_sink(kwargs, cfg)
            run_loop(str(input_video), source, sink, cfg)
            return
        from .realtime import run_offline

        summary = run_offline(
            str(input_video), source, lambda w, h, fps: _make_sink(kwargs, cfg, fps=fps, size=(w, h)),
            cfg, fg, print_stats=bool(kwargs.get("show_metrics", True)),
        )
        click.echo(json.dumps(summary, indent=2))
    except click.ClickException:
        raise
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e


@main.command("virtualcam")
@_common_io_options
@click.option(
    "--v4l2",
    "v4l2_path",
    default=None,
    help="v4l2loopback device (default: auto-detect, e.g. /dev/video10 'deepfake')",
)
def virtualcam_cmd(v4l2_path: str | None, **kwargs):
    """Stream swapped frames to a virtual webcam (OBS, browsers, video calls)."""
    from .devices import LOOPBACK_HELP, find_loopback_device

    target = kwargs.get("output_device") or v4l2_path or find_loopback_device()
    if not target or not Path(target).exists():
        raise click.ClickException(LOOPBACK_HELP)
    if not os.access(target, os.W_OK):
        raise click.ClickException(f"{target} is not writable by you (need the 'video' group)")
    require_consent(ack=kwargs.get("consent_ack", False), watermark=kwargs.get("watermark", True))
    source = _resolve_source(kwargs.get("source_path"), kwargs.get("identity"))
    kwargs["output_device"] = target
    click.echo(f"deepfake: virtual camera → {target} (select 'deepfake' in OBS / your browser)", err=True)
    kwargs["preview"] = kwargs.get("preview", False)
    cfg = _cfg_from_kwargs(kwargs)
    try:
        _run_live(kwargs, source, cfg)
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


@main.command("benchmark")
@click.option("--resolution", "resolutions", multiple=True, type=click.Choice(["720p", "1080p"]),
              help="Repeatable; default 720p and 1080p.")
@click.option("--camera-fps", default=30.0, show_default=True, help="Synthetic camera rate.")
@click.option("--target-fps", default=60.0, show_default=True, help="Output rate the recommendation aims for.")
@click.option("--seconds", default=10.0, show_default=True, help="Duration of each live-pipeline run.")
@click.option("--frame-gen-model", type=click.Choice(sorted(RIFE_VARIANTS)), default="4.25", show_default=True)
@click.option("--swap-precision", type=click.Choice(["fp32", "bf16"]), default="bf16", show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable report.")
@click.option("--consent-ack", is_flag=True, hidden=True)
def benchmark_cmd(resolutions, camera_fps, target_fps, seconds, frame_gen_model, swap_precision, as_json, consent_ack):
    """Measure swap-only vs swap+frame-generation on this GPU (real runs, no estimates).

    Uses a bundled fictional face on a synthetic real-time camera. Reports
    swap fps/latency, interpolation cost, presented fps, end-to-end latency
    and peak VRAM, then recommends a --frame-gen mode.
    """
    require_consent(ack=consent_ack, watermark=True)
    from .benchmark import BenchConfig, run_benchmark

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
