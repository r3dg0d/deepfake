"""NVIDIA Optical Flow (NVOFA) frame-generation backend.

Uses the hardware Optical Flow Accelerator via libnvidia-opticalflow, then
warps/blends on CUDA with PyTorch.

Sharpness notes (faces):
  * Prefer grid size 1 (balanced/quality) so flow is not blocky.
  * Occlusion / forward-backward inconsistency → pick nearest keyframe, do not average.
  * Nearer-frame bias so we do not soft-average two slightly misaligned warps on skin/eyes.
"""

from __future__ import annotations

import time

import numpy as np

from .base import (
    FrameGenBenchmark,
    FrameGenCapabilities,
    FrameGenerationBackend,
    FrameGenUnavailable,
    timesteps_for,
)
from .nvof_api import (
    NV_OF_PERF_LEVEL_FAST,
    NV_OF_PERF_LEVEL_MEDIUM,
    NV_OF_PERF_LEVEL_SLOW,
    NvOFSession,
    optical_flow_available,
)

_PERF = {
    "latency": NV_OF_PERF_LEVEL_FAST,
    "balanced": NV_OF_PERF_LEVEL_MEDIUM,
    "quality": NV_OF_PERF_LEVEL_SLOW,
    "fast": NV_OF_PERF_LEVEL_FAST,
    "medium": NV_OF_PERF_LEVEL_MEDIUM,
    "slow": NV_OF_PERF_LEVEL_SLOW,
}

# Finer grid = sharper faces (Ada supports 1). Latency uses 2 for speed.
_GRID = {
    "latency": 1,
    "balanced": 1,
    "quality": 1,
    "fast": 1,
    "medium": 1,
    "slow": 1,
}


class NvidiaOpticalFlowBackend(FrameGenerationBackend):
    name = "nvof"

    def __init__(
        self,
        variant: str = "balanced",
        *,
        device: str = "cuda",
        precision: str = "fp16",
        flow_scale: float | None = None,
        scene_change_threshold: float = 28.0,
        **_kw,
    ) -> None:
        self.variant = variant if variant in _PERF else "balanced"
        self.precision = precision
        self._device_str = device
        self._scene_thr = float(scene_change_threshold)
        self._session: NvOFSession | None = None
        self._prev_bgr: np.ndarray | None = None
        self._cur_bgr: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._cur_gray: np.ndarray | None = None
        self._prev_t = self._cur_t = None
        self._prev_gdev = self._cur_gdev = None
        self._torch = None
        self._grid_base = None
        self._w = self._h = 0
        self._shot = False
        self._context = None
        self._guard_enabled = False
        # Quality: forward+backward occlusion. Balanced/latency: forward-only + nearer bias.
        self._bidir = self.variant in ("quality", "slow")

    @property
    def capabilities(self) -> FrameGenCapabilities:
        return FrameGenCapabilities(
            name=self.name,
            variant=self.variant,
            max_factor=4,
            arbitrary_timestep=True,
            device="cuda",
            precision=self.precision,
            license="NVIDIA Optical Flow SDK headers (BSD-3); driver libnvidia-opticalflow",
            paper="NVIDIA Optical Flow Accelerator (NVOFA)",
            source="https://github.com/NVIDIA/NVIDIAOpticalFlowSDK",
        )

    @property
    def ready(self) -> bool:
        return self._prev_bgr is not None and self._cur_bgr is not None

    def initialize(self, width: int, height: int) -> None:
        ok, detail = optical_flow_available()
        if not ok:
            raise FrameGenUnavailable(f"NVIDIA Optical Flow unavailable: {detail}")
        try:
            import torch
        except ImportError as e:
            raise FrameGenUnavailable("PyTorch required for NvOF warp/blend") from e
        if not torch.cuda.is_available():
            raise FrameGenUnavailable("NvOF frame generation needs CUDA")
        self._torch = torch
        self._w, self._h = int(width), int(height)
        grid = _GRID.get(self.variant, 1)
        try:
            self._session = NvOFSession(
                self._w,
                self._h,
                perf_level=_PERF.get(self.variant, NV_OF_PERF_LEVEL_MEDIUM),
                grid=grid,
            )
        except Exception:  # noqa: BLE001
            # Fallback to grid 4 if finer unsupported on older GPUs.
            try:
                self._session = NvOFSession(
                    self._w,
                    self._h,
                    perf_level=_PERF.get(self.variant, NV_OF_PERF_LEVEL_MEDIUM),
                    grid=4,
                )
            except Exception as e2:  # noqa: BLE001
                raise FrameGenUnavailable(f"NvOF init failed: {e2}") from e2
        device = torch.device("cuda")
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, self._h, device=device),
            torch.linspace(-1, 1, self._w, device=device),
            indexing="ij",
        )
        self._grid_base = torch.stack([xx, yy], dim=-1).unsqueeze(0)

    def set_composite_context(self, context) -> None:
        """Carry target pixels and masks through the SAME flow as generated pixels."""
        self._context = context
        self._guard_enabled = True

    def push(self, frame_bgr: np.ndarray) -> None:
        import cv2

        torch = self._torch
        if frame_bgr.shape[1] != self._w or frame_bgr.shape[0] != self._h:
            frame_bgr = cv2.resize(frame_bgr, (self._w, self._h), interpolation=cv2.INTER_AREA)
        self._prev_bgr, self._cur_bgr = self._cur_bgr, np.ascontiguousarray(frame_bgr)
        if torch is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            self._prev_gray, self._cur_gray = self._cur_gray, gray
            if self._prev_gray is not None:
                mad = float(np.mean(np.abs(self._cur_gray.astype(np.int16) - self._prev_gray.astype(np.int16))))
                self._shot = mad >= self._scene_thr
            else:
                self._shot = False
            return
        t = (
            torch.from_numpy(self._cur_bgr)
            .to(device="cuda", dtype=torch.float32, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div_(255.0)
        )
        if self._guard_enabled:
            if self._context is None:
                target = t.clone()
                masks = torch.cat([torch.ones_like(t[:, :1]), torch.zeros_like(t[:, :1])], 1)
            else:
                original, visible, region = self._context
                if (
                    original.shape != frame_bgr.shape
                    or visible.shape != frame_bgr.shape[:2]
                    or region.shape != visible.shape
                ):
                    raise ValueError("FrameGen context must match keyframe dimensions")
                target = (
                    torch.from_numpy(np.ascontiguousarray(original))
                    .to(device="cuda", dtype=torch.float32)
                    .permute(2, 0, 1)[None]
                    / 255
                )
                masks = torch.from_numpy(np.stack([visible, region])).to(device="cuda", dtype=torch.float32)[None]
            t = torch.cat([t, target, masks], 1)
            self._context = None
        gray_f = 0.114 * t[0, 0] + 0.587 * t[0, 1] + 0.299 * t[0, 2]
        gray_u8 = (gray_f * 255.0).clamp(0, 255).to(torch.uint8).contiguous()
        self._prev_t, self._cur_t = self._cur_t, t
        self._prev_gdev, self._cur_gdev = self._cur_gdev, gray_u8
        if self._prev_t is not None:
            mad = float(torch.mean(torch.abs(self._cur_t[:, :3] - self._prev_t[:, :3])).item() * 255.0)
            self._shot = mad >= self._scene_thr
        else:
            self._shot = False

    def interpolate(self, timesteps: list[float]) -> list[np.ndarray]:
        if not self.ready or self._session is None:
            raise RuntimeError("NvOF backend not ready (need two keyframes)")
        if self._shot or not timesteps:
            # Scene cut: never invent a blend — emit the newest real frame.
            return [self._cur_bgr.copy() for _ in timesteps]

        if self._prev_gdev is not None and self._cur_gdev is not None:
            p0 = int(self._prev_gdev.data_ptr())
            p1 = int(self._cur_gdev.data_ptr())
            fwd = self._session.estimate_device(p0, p1, disable_temporal=False)
            bwd = self._session.estimate_device(p1, p0, disable_temporal=True) if self._bidir else None
        else:
            assert self._prev_gray is not None and self._cur_gray is not None
            fwd = self._session.estimate(self._prev_gray, self._cur_gray, disable_temporal=False)
            bwd = (
                self._session.estimate(self._cur_gray, self._prev_gray, disable_temporal=True) if self._bidir else None
            )
        return [self._warp_blend(t, fwd, bwd) for t in timesteps]

    def _warp_blend(self, t: float, fwd: np.ndarray, bwd: np.ndarray | None) -> np.ndarray:
        """Occlusion-aware warp with nearer-keyframe bias (keeps faces sharp)."""
        torch = self._torch
        assert torch is not None and self._grid_base is not None
        device = torch.device("cuda")
        h, w = self._h, self._w
        sx = 2.0 / max(w - 1, 1)
        sy = 2.0 / max(h - 1, 1)

        f = torch.from_numpy(np.ascontiguousarray(fwd)).to(device=device, dtype=torch.float32, non_blocking=True)

        g0 = self._grid_base.clone()
        g0[..., 0] = g0[..., 0] - f[..., 0] * t * sx
        g0[..., 1] = g0[..., 1] - f[..., 1] * t * sy

        img0 = self._prev_t
        img1 = self._cur_t
        if img0 is None or img1 is None:
            img0 = (
                torch.from_numpy(self._prev_bgr).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                / 255.0
            )
            img1 = (
                torch.from_numpy(self._cur_bgr).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                / 255.0
            )

        w0 = torch.nn.functional.grid_sample(img0, g0, mode="bilinear", padding_mode="border", align_corners=True)

        if bwd is not None:
            b = torch.from_numpy(np.ascontiguousarray(bwd)).to(device=device, dtype=torch.float32, non_blocking=True)
            g1 = self._grid_base.clone()
            g1[..., 0] = g1[..., 0] - b[..., 0] * (1.0 - t) * sx
            g1[..., 1] = g1[..., 1] - b[..., 1] * (1.0 - t) * sy
            w1 = torch.nn.functional.grid_sample(img1, g1, mode="bilinear", padding_mode="border", align_corners=True)
            correspondence = self._grid_base.clone()
            correspondence[..., 0] += f[..., 0] * sx
            correspondence[..., 1] += f[..., 1] * sy
            aligned_b = torch.nn.functional.grid_sample(
                b.permute(2, 0, 1)[None], correspondence, padding_mode="border", align_corners=True
            )[0].permute(1, 2, 0)
            cons = torch.linalg.vector_norm(f + aligned_b, dim=-1)
            reliable = (cons < 1.5).to(dtype=w0.dtype).unsqueeze(0).unsqueeze(0)
        else:
            # Forward-only: blend warped prev with current keyframe (not a soft 50/50 of two warps).
            g1 = self._grid_base.clone()
            g1[..., 0] += f[..., 0] * (1.0 - t) * sx
            g1[..., 1] += f[..., 1] * (1.0 - t) * sy
            w1 = torch.nn.functional.grid_sample(img1, g1, padding_mode="border", align_corners=True)
            # Large forward motion → prefer nearer keyframe to avoid smear.
            mag = torch.linalg.vector_norm(f, dim=-1)
            reliable = (mag < 12.0).to(dtype=w0.dtype).unsqueeze(0).unsqueeze(0)

        # Nearer-frame bias keeps faces crisp (avoid soft-averaging misaligned eyes/teeth).
        if t <= 0.5:
            blend = (t / 0.5) * 0.30
            alpha = blend * reliable
            out = (1.0 - alpha) * w0 + alpha * w1
            out = torch.where(reliable.bool(), out, w0)
        else:
            blend = ((1.0 - t) / 0.5) * 0.30
            alpha = blend * reliable
            out = (1.0 - alpha) * w1 + alpha * w0
            out = torch.where(reliable.bool(), out, w1)

        protected_target = None
        protection = None
        if out.shape[1] == 8:
            protected_target = out[:, 3:6]
            visible = torch.minimum(w0[:, 6:7], w1[:, 6:7])
            region = torch.maximum(w0[:, 7:8], w1[:, 7:8])
            protection = (region * (1 - visible)).clamp(0, 1)
            out = out[:, :3]

        # Mild detail restore: unsharp mask gated by local Laplacian (skip flat skin noise,
        # boost mid/high detail — eyes/hair/mouth edges).
        # Light detail restore (faces): stronger for balanced/quality, mild for latency.
        amt = 0.25 if self.variant in ("latency", "fast") else 0.45
        gray = 0.114 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.299 * out[:, 2:3]
        blur = torch.nn.functional.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        detail = gray - blur
        gate = (detail.abs() > 0.008).to(out.dtype)
        out = (out + amt * detail.repeat(1, 3, 1, 1) * gate).clamp(0, 1)

        if protected_target is not None:
            # Apply AFTER sharpening so target foreground is never hallucinated
            # or sharpened by the swap. The masks use exactly g0/g1 above.
            out = out * (1 - protection) + protected_target * protection
        arr = (out.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0).round().to(torch.uint8).contiguous()
        return arr.cpu().numpy()

    def benchmark(self, width: int, height: int, factor: int, iterations: int = 60) -> FrameGenBenchmark:
        self.initialize(width, height)
        a = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        self.push(a)
        self.push(np.roll(a, 3, axis=1))
        old_thr = self._scene_thr
        self._scene_thr = 255.0
        ts = timesteps_for(factor)
        for _ in range(3):
            self.interpolate(ts)
        t0 = time.perf_counter()
        for i in range(iterations):
            self.push(np.roll(a, (i % 7) + 1, axis=1))
            self.interpolate(ts)
        elapsed = time.perf_counter() - t0
        self._scene_thr = old_thr
        n_gen = max(1, len(ts)) * iterations
        ms = (elapsed / n_gen) * 1000
        vram = None
        if self._torch is not None:
            vram = self._torch.cuda.max_memory_allocated() / (1024 * 1024)
        self.shutdown()
        return FrameGenBenchmark(
            backend=self.name,
            variant=self.variant,
            width=width,
            height=height,
            factor=factor,
            ms_per_generated_frame=ms,
            ms_per_source_interval=ms * max(1, len(ts)),
            generated_fps_capacity=1000.0 / ms if ms > 0 else 0.0,
            peak_vram_mb=vram,
            samples=iterations,
        )

    def shutdown(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._prev_bgr = self._cur_bgr = None
        self._prev_gray = self._cur_gray = None
        self._prev_t = self._cur_t = None
        self._prev_gdev = self._cur_gdev = None
        self._grid_base = None
