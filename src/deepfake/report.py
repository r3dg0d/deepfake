"""Human-readable benchmark report (Rich when available, plain text otherwise)."""

from __future__ import annotations

from typing import Any


def _v(d: dict[str, Any], key: str, unit: str = "") -> str:
    val = d.get(key)
    return "–" if val is None else f"{val}{unit}"


def _held(fg: dict[str, Any]) -> str:
    total = fg.get("keyframes", 0) + fg.get("generated_frames", 0) + fg.get("held_frames", 0)
    return "–" if not total else f"{fg.get('held_frames', 0) / total:.1%}"


def rows(report: dict[str, Any]) -> list[tuple[str, ...]]:
    out = []
    for res, r in report["results"].items():
        base = r["pipeline_no_framegen"]
        out.append(
            (
                res,
                "no frame-gen",
                _v(base, "swap_fps"),
                _v(base, "output_fps"),
                "–",
                "–",
                _v(base, "swap_latency_ms", " ms"),
                _v(base, "total_latency_ms", " ms"),
                _v(base, "peak_vram_mb_torch", " MB"),
            )
        )
        for key in sorted(k for k in r if k.startswith("pipeline_framegen_")):
            fg = r[key]
            factor = key.rsplit("_", 1)[1]
            only = r.get(f"framegen_only_{factor}", {})
            gen_ms = only.get("ms_per_generated_frame")
            out.append(
                (
                    res,
                    f"frame-gen {factor}",
                    _v(fg, "swap_fps"),
                    _v(fg, "output_fps"),
                    _held(fg),
                    "–" if gen_ms is None else f"{gen_ms:.1f} ms",
                    _v(fg, "swap_latency_ms", " ms"),
                    _v(fg, "total_latency_ms", " ms"),
                    _v(fg, "peak_vram_mb_torch", " MB"),
                )
            )
    return out


HEADERS = (
    "res",
    "mode",
    "swapped fps",
    "output fps",
    "held",
    "interp/frame",
    "swap latency",
    "total latency",
    "peak VRAM",
)


def render_benchmark(report: dict[str, Any]) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        _plain(report)
        return
    c = Console()
    c.print(
        Panel.fit(
            f"[bold cyan]deepfake benchmark[/]\n[dim]{report['gpu']} · torch {report['torch']} · "
            f"camera {report['camera_fps']:g} fps · swap {report['swap_precision']} · {report['framegen_backend']}[/]",
            border_style="cyan",
        )
    )
    t = Table(header_style="bold", border_style="dim")
    for h in HEADERS:
        t.add_column(h, justify="left" if h in ("res", "mode") else "right")
    for r in rows(report):
        t.add_row(*r)
    c.print(t)
    for res, rec in report["recommendation"].items():
        if rec["mode"] == "off":
            c.print(f"[yellow]![/] {res}: keep frame generation off — {rec['reason']}")
        else:
            c.print(
                f"[green]✔[/] {res}: recommended [bold]--frame-gen {rec['mode']}[/] → "
                f"{rec['expected_output_fps']} new frames/s, "
                f"+{rec['added_latency_ms']} ms latency ({rec['reason']})"
            )
    c.print(
        "[dim]output fps counts only new frames (real + interpolated); held = slots re-sent because a frame was late. "
        "Latency = capture → frame handed to the sink. Interp/frame from an isolated RIFE run.[/]"
    )


def _plain(report: dict[str, Any]) -> None:
    print(f"deepfake benchmark — {report['gpu']} · camera {report['camera_fps']:g} fps")
    widths = [max(len(str(x)) for x in col) for col in zip(HEADERS, *rows(report), strict=False)]
    for line in (HEADERS, *rows(report)):
        print("  ".join(str(x).ljust(w) for x, w in zip(line, widths, strict=False)))
    for res, rec in report["recommendation"].items():
        print(f"{res}: recommended --frame-gen {rec['mode']} ({rec['reason']})")
