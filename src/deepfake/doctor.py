"""Installation diagnostics for Deepfake (doctor command)."""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from typing import Callable

from .devices import list_v4l2_devices, resolve_cuda
from .models import list_models
from .paths import models_dir, vendor_dir
from .widget import find_qs, quickshell_available, widget_shell_dir


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor() -> list[Check]:
    checks: list[Check] = []

    # AlphaFace
    af = next((m for m in list_models() if m.name == "alphaface"), None)
    vendor_ok = vendor_dir().is_dir()
    weights = models_dir() / "alphaface"
    af_ok = bool(af and af.installed) or (vendor_ok and weights.is_dir())
    checks.append(Check("AlphaFace", af_ok, "available" if af_ok else "not installed (deepfake models install alphaface)"))

    # CUDA / GPU
    device = resolve_cuda("auto")
    cuda_ok = device.startswith("cuda")
    gpu_name = "n/a"
    try:
        out = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null").read().strip().splitlines()
        if out:
            gpu_name = out[0].strip()
    except OSError:
        pass
    checks.append(Check("CUDA", cuda_ok, device if cuda_ok else "unavailable"))
    checks.append(Check("GPU", bool(gpu_name and gpu_name != "n/a"), gpu_name))

    # FrameGen / RIFE
    rife = models_dir() / "rife"
    rife_ok = rife.is_dir() and any(rife.iterdir())
    torch_ok = importlib.util.find_spec("torch") is not None
    checks.append(Check("FrameGen", rife_ok and torch_ok, "RIFE + torch" if rife_ok and torch_ok else "missing RIFE weights or torch"))

    # Quickshell
    qs_ok, qs_detail = quickshell_available()
    checks.append(Check("Quickshell", qs_ok, qs_detail if qs_ok else qs_detail))
    shell = widget_shell_dir()
    checks.append(Check("Quickshell widget", shell is not None and (shell / "shell.qml").is_file(), str(shell) if shell else "shell.qml missing"))

    # IPC
    try:
        from .ipc import state_dir

        d = state_dir()
        checks.append(Check("IPC", d.is_dir(), str(d)))
    except OSError as e:
        checks.append(Check("IPC", False, str(e)))

    # FFmpeg / NVENC
    ffmpeg = shutil.which("ffmpeg")
    checks.append(Check("FFmpeg", bool(ffmpeg), ffmpeg or "not on PATH"))
    nvenc = False
    if ffmpeg:
        import subprocess

        r = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
        nvenc = "h264_nvenc" in (r.stdout or "")
    checks.append(Check("NVENC", nvenc, "h264_nvenc" if nvenc else "not available"))

    # v4l2loopback
    loop = [d for d in list_v4l2_devices() if d.kind == "output" or "loopback" in (d.name or "").lower()]
    # also check module
    loop_mod = Path_exists("/sys/module/v4l2loopback")
    checks.append(
        Check(
            "v4l2loopback",
            bool(loop) or loop_mod,
            (loop[0].path if loop else ("module loaded" if loop_mod else "no loopback device")),
        )
    )

    # qs binary noted
    _ = find_qs()
    return checks


def Path_exists(p: str) -> bool:
    from pathlib import Path

    return Path(p).exists()


def render_doctor(checks: list[Check], echo: Callable[[str], None] | None = None) -> str:
    lines = []
    for c in checks:
        mark = "✓" if c.ok else "✗"
        lines.append(f"{c.name}\n{mark} {c.detail}\n")
    text = "\n".join(lines).rstrip() + "\n"
    if echo:
        echo(text)
    return text
