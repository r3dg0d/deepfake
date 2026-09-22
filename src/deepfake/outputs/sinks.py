"""Output sinks: preview window, video file, FFmpeg pipe, GStreamer, v4l2loopback."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import numpy as np


class OutputSink(Protocol):
    def write(self, frame_bgr: np.ndarray) -> None: ...

    def close(self) -> None: ...


class NullSink:
    def write(self, frame_bgr: np.ndarray) -> None:
        _ = frame_bgr

    def close(self) -> None:
        return


class PreviewSink:
    def __init__(self, title: str = "deepfake") -> None:
        import cv2

        self._cv2 = cv2
        self.title = title

    def write(self, frame_bgr: np.ndarray) -> None:
        self._cv2.imshow(self.title, frame_bgr)
        self._cv2.waitKey(1)

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self.title)
        except Exception:
            pass


class VideoFileSink:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        import cv2

        self._cv2 = cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter for {path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        self._writer.write(frame_bgr)

    def close(self) -> None:
        self._writer.release()


class FFmpegPipeSink:
    """Raw BGR24 frames to ffmpeg stdin (OBS-friendly file or RTMP etc.)."""

    def __init__(self, ffmpeg_args: list[str], width: int, height: int, fps: float) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found on PATH")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            *ffmpeg_args,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._wh = (width, height)

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin closed")
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._wh:
            import cv2

            frame_bgr = cv2.resize(frame_bgr, self._wh)
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait(timeout=30)


class GStreamerPipeSink:
    def __init__(self, pipeline: str, width: int, height: int, fps: float) -> None:
        if not shutil.which("gst-launch-1.0"):
            raise RuntimeError(
                "GStreamer (gst-launch-1.0) not found. "
                "Install gstreamer1.0-tools / gst-plugins-*; documenting as optional. "
                "Example: gst-launch-1.0 fdsrc ! rawvideoparse ..."
            )
        cmd = [
            "gst-launch-1.0",
            "-q",
            "fdsrc",
            "fd=0",
            "!",
            f"rawvideoparse width={width} height={height} format=bgr framerate={int(fps)}/1",
            "!",
            "videoconvert",
            "!",
            pipeline,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._wh = (width, height)

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("gstreamer stdin closed")
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._wh:
            import cv2

            frame_bgr = cv2.resize(frame_bgr, self._wh)
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait(timeout=30)


class V4L2LoopbackSink:
    """Write frames to a v4l2loopback device (OBS-friendly virtual webcam)."""

    def __init__(self, device: str, width: int, height: int, fps: float) -> None:
        import cv2

        self._cv2 = cv2
        # Prefer pyv4l2 / ffmpeg; OpenCV VideoWriter with V4L2 may work on some systems.
        if not Path(device).exists():
            raise RuntimeError(
                f"{device} not found. Load v4l2loopback: "
                "sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=deepfake"
            )
        # Use ffmpeg v4l2 output for reliability
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg required for v4l2loopback sink")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-f",
            "v4l2",
            device,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._wh = (width, height)

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("v4l2 ffmpeg stdin closed")
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._wh:
            frame_bgr = self._cv2.resize(frame_bgr, self._wh)
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class MultiSink:
    def __init__(self, sinks: list[OutputSink]) -> None:
        self.sinks = sinks

    def write(self, frame_bgr: np.ndarray) -> None:
        for s in self.sinks:
            s.write(frame_bgr)

    def close(self) -> None:
        for s in self.sinks:
            s.close()


def create_sink(
    *,
    preview: bool = True,
    output_video: Path | None = None,
    ffmpeg_args: list[str] | None = None,
    gstreamer_pipeline: str | None = None,
    v4l2_device: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> OutputSink:
    sinks: list[OutputSink] = []
    if preview:
        try:
            sinks.append(PreviewSink())
        except Exception:
            # headless
            pass
    if output_video is not None:
        sinks.append(VideoFileSink(output_video, fps, (width, height)))
    if ffmpeg_args:
        sinks.append(FFmpegPipeSink(ffmpeg_args, width, height, fps))
    if gstreamer_pipeline:
        sinks.append(GStreamerPipeSink(gstreamer_pipeline, width, height, fps))
    if v4l2_device:
        sinks.append(V4L2LoopbackSink(v4l2_device, width, height, fps))
    if not sinks:
        return NullSink()
    if len(sinks) == 1:
        return sinks[0]
    return MultiSink(sinks)
