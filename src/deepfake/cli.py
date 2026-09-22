"""Click CLI for deepfake."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import __version__
from .consent import require_consent
from .devices import format_devices, list_v4l2_devices, resolve_cuda
from .identity import list_fakeperson_identities, resolve_source_image
from .models import install_model, list_models
from .outputs.sinks import create_sink
from .paths import ensure_dirs
from .pipeline import build_config, run_loop
from .presets import PRESETS


def _common_io_options(fn):
    opts = [
        click.option(
            "--source",
            "source_path",
            type=click.Path(path_type=Path, exists=False),
            default=None,
            help="Source identity face image (person.jpg)",
        ),
        click.option(
            "--identity",
            default=None,
            help="Optional fakeperson identity name (uses XDG identities dir)",
        ),
        click.option(
            "--preset",
            type=click.Choice(sorted(PRESETS.keys())),
            default="balanced",
            show_default=True,
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
        click.option("--consent-ack", is_flag=True, help="Acknowledge disclosed synthetic media / consent framing"),
        click.option("--preview/--no-preview", default=True),
        click.option("-o", "--output", "output_video", type=click.Path(path_type=Path), default=None),
        click.option("--ffmpeg-out", multiple=True, help="Extra ffmpeg argv after raw input (repeatable)"),
        click.option("--gstreamer", default=None, help="GStreamer pipeline after videoconvert (optional)"),
        click.option("--max-frames", type=int, default=None),
        click.option("--show-metrics/--hide-metrics", default=True),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _resolve_source(source_path: Path | None, identity: str | None) -> Path:
    try:
        p = resolve_source_image(str(source_path) if source_path else None, identity)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    if p is None:
        raise click.ClickException(
            "Provide --source person.jpg (or --identity <fakeperson-name>). "
            f"fakeperson identities found: {list_fakeperson_identities() or '(none)'}"
        )
    return p


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


def _make_sink(kwargs: dict, cfg):
    ffmpeg_args = list(kwargs.get("ffmpeg_out") or ())
    return create_sink(
        preview=bool(kwargs.get("preview", True)),
        output_video=kwargs.get("output_video"),
        ffmpeg_args=ffmpeg_args or None,
        gstreamer_pipeline=kwargs.get("gstreamer"),
        v4l2_device=kwargs.get("output_device"),
        width=cfg.preset.width,
        height=cfg.preset.height,
        fps=float(cfg.preset.fps),
    )


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

    Framing: research, VFX, avatars, filmmaking, consenting demos, disclosed
    synthetic media. No anonymity claims. Consent gates + optional watermark.
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
    sink = _make_sink(kwargs, cfg)
    cap = _parse_input_device(kwargs.get("input_device"), 0)
    try:
        run_loop(cap, source, sink, cfg)
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
    sink = _make_sink(kwargs, cfg)
    try:
        run_loop(str(input_video), source, sink, cfg)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e


@main.command("virtualcam")
@_common_io_options
@click.option(
    "--v4l2",
    "v4l2_path",
    default="/dev/video10",
    show_default=True,
    help="v4l2loopback device (OBS-friendly)",
)
def virtualcam_cmd(v4l2_path: str, **kwargs):
    """Stream swapped frames to a v4l2loopback virtual webcam."""
    require_consent(ack=kwargs.get("consent_ack", False), watermark=kwargs.get("watermark", True))
    source = _resolve_source(kwargs.get("source_path"), kwargs.get("identity"))
    kwargs["output_device"] = kwargs.get("output_device") or v4l2_path
    kwargs["preview"] = kwargs.get("preview", False)
    cfg = _cfg_from_kwargs(kwargs)
    sink = _make_sink(kwargs, cfg)
    cap = _parse_input_device(kwargs.get("input_device"), 0)
    try:
        run_loop(cap, source, sink, cfg)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e


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
@click.option("--frames", default=60, show_default=True)
@click.option("--preset", type=click.Choice(sorted(PRESETS.keys())), default="balanced")
@click.option("--device", default="auto")
@click.option("--backend", default="auto")
@click.option("--consent-ack", is_flag=True)
def benchmark_cmd(frames: int, preset: str, device: str, backend: str, consent_ack: bool):
    """Micro-benchmark detect+composite path (swap needs models)."""
    require_consent(ack=consent_ack, watermark=True)
    import numpy as np

    from .detect import create_detector
    from .metrics import MetricsTracker, format_metrics
    from .presets import get_preset

    p = get_preset(preset)
    det = create_detector("auto")
    mt = MetricsTracker()
    device_r = resolve_cuda(device)
    click.echo(f"device={device_r} preset={preset} backend={backend}")
    for _ in range(frames):
        t0 = __import__("time").perf_counter()
        frame = np.zeros((p.height, p.width, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        # synthetic face-ish blob so Haar may or may not hit — still measures overhead
        frame[p.height // 3 : p.height // 3 + 120, p.width // 3 : p.width // 3 + 120] = (200, 180, 160)
        _ = det.detect(frame)
        total = (__import__("time").perf_counter() - t0) * 1000
        m = mt.record(inference_ms=0.0, total_ms=total)
    click.echo(format_metrics(m))
    click.echo(
        "Note: full swap benchmark requires `models install`. "
        "This run measures detect/loop overhead only."
    )


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
