"""Installation diagnostics for Deepfake (doctor command)."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

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
    checks.append(
        Check("AlphaFace", af_ok, "available" if af_ok else "not installed (deepfake models install alphaface)")
    )

    # CUDA / GPU
    device = resolve_cuda("auto")
    cuda_ok = device.startswith("cuda")
    gpu_name = "n/a"
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5
            )
            .stdout.strip()
            .splitlines()
        )
        if out:
            gpu_name = out[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    checks.append(Check("CUDA", cuda_ok, device if cuda_ok else "unavailable"))
    checks.append(Check("GPU", bool(gpu_name and gpu_name != "n/a"), gpu_name))

    # FrameGen / NVIDIA Optical Flow
    from .framegen.maxine import maxine_available
    from .framegen.nvof_api import optical_flow_available

    of_ok, of_detail = optical_flow_available()
    mx_ok, mx_detail = maxine_available()
    torch_ok = importlib.util.find_spec("torch") is not None
    fg_ok = (of_ok or mx_ok) and torch_ok
    fg_detail = of_detail if of_ok else (mx_detail if mx_ok else "NVIDIA Optical Flow / Maxine unavailable")
    if fg_ok and not torch_ok:
        fg_detail = "torch missing"
    checks.append(Check("FrameGen", fg_ok, fg_detail if fg_ok else fg_detail))
    checks.append(Check("Maxine VFG", mx_ok, mx_detail if mx_ok else mx_detail))

    # Quickshell
    qs_ok, qs_detail = quickshell_available()
    checks.append(Check("Quickshell", qs_ok, qs_detail if qs_ok else qs_detail))
    shell = widget_shell_dir()
    checks.append(
        Check(
            "Quickshell widget",
            shell is not None and (shell / "shell.qml").is_file(),
            str(shell) if shell else "shell.qml missing",
        )
    )

    from .neural_assets import verify_asset
    from .provenance import available, signing_paths

    for name, label in (("yunet", "Face detector (YuNet)"), ("bisenet", "Face parser"), ("xseg", "Occlusion engine")):
        ok = verify_asset(name)
        checks.append(Check(label, ok, "SHA-256 verified" if ok else f"missing; deepfake models install {name} --yes"))
    ort_ok = importlib.util.find_spec("onnxruntime") is not None
    checks.append(
        Check(
            "ONNX Runtime",
            ort_ok,
            "installed; CUDA provider must match torch/CUDA major" if ort_ok else "install [vision] or [vision-cuda12]",
        )
    )
    checks.append(Check("Mask optical flow", True, "OpenCV Farneback; per-track LK motion prediction"))
    checks.append(Check("Video segmentation", False, "SAM2/SAM-MT researched; not installed/enabled"))
    checks.append(Check("C2PA SDK", available(), "official c2pa-python" if available() else "install [provenance]"))
    cert, key = signing_paths()
    private = key.is_file() and not (key.stat().st_mode & 0o077)
    checks.append(
        Check(
            "Provenance signer",
            cert.is_file() and private,
            "configured (trust verified separately)"
            if cert.is_file() and private
            else "local development signer created on first export; not publicly trusted",
        )
    )
    from .invisible import ready

    checks.append(
        Check(
            "Invisible watermark",
            ready(),
            "TrustMark Q models verified" if ready() else "optional [watermark] + models install trustmark --yes",
        )
    )
    for tool in ("gst-launch-1.0", "pw-cli"):
        executable = shutil.which(tool)
        checks.append(Check(tool, bool(executable), executable or "optional tool not found"))

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
