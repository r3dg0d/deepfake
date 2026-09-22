"""Enumerate video capture / output devices (Linux)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoDevice:
    index: int
    path: str
    name: str
    kind: str  # capture | output | unknown


def _read_name(sys_path: Path) -> str:
    name_file = sys_path / "name"
    if name_file.is_file():
        try:
            return name_file.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            pass
    return sys_path.name


def list_v4l2_devices() -> list[VideoDevice]:
    """List /dev/video* with names from sysfs when available."""
    devices: list[VideoDevice] = []
    for path in sorted(Path("/dev").glob("video*")):
        m = re.match(r"video(\d+)$", path.name)
        if not m:
            continue
        idx = int(m.group(1))
        sys_path = Path("/sys/class/video4linux") / path.name
        name = _read_name(sys_path) if sys_path.exists() else path.name
        kind = "unknown"
        caps = sys_path / "device" / "interface"
        # Heuristic: v4l2loopback often has "Loopback" in name
        lower = name.lower()
        if "loopback" in lower or "virtual" in lower:
            kind = "output"
        else:
            kind = "capture"
        devices.append(VideoDevice(index=idx, path=str(path), name=name, kind=kind))
        _ = caps  # reserved for future capability probing
    return devices


def list_opencv_indices(max_probe: int = 8) -> list[int]:
    """Probe OpenCV capture indices (mockable; skips open when DEEPFAKE_MOCK_DEVICES=1)."""
    if os.environ.get("DEEPFAKE_MOCK_DEVICES") == "1":
        return [0]
    try:
        import cv2  # type: ignore
    except ImportError:
        return []
    found: list[int] = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i)
        try:
            if cap is not None and cap.isOpened():
                found.append(i)
        finally:
            if cap is not None:
                cap.release()
    return found


def format_devices(devices: list[VideoDevice] | None = None) -> str:
    devices = devices if devices is not None else list_v4l2_devices()
    if not devices:
        return "(no /dev/video* devices found — check permissions / plug in a camera)"
    lines = []
    for d in devices:
        lines.append(f"{d.index}\t{d.path}\t{d.kind}\t{d.name}")
    return "\n".join(lines)


def resolve_cuda(device: str | None) -> str:
    """Return 'cuda', 'cuda:N', or 'cpu'. Does not require torch for 'cpu'."""
    if device is None or device == "auto":
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device.startswith("cuda"):
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device
