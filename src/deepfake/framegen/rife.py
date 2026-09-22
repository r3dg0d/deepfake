"""RIFE (Practical-RIFE v4.25/4.26) CUDA frame-generation backend."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..paths import models_dir
from .base import (
    FrameGenBenchmark,
    FrameGenCapabilities,
    FrameGenerationBackend,
    FrameGenUnavailable,
    timesteps_for,
)
from .variants import VARIANTS

DEFAULT_VARIANT = "4.25"


def rife_dir() -> Path:
    return models_dir() / "rife"


def weights_path(variant: str) -> Path | None:
    """Installed weights for ``variant`` (safetensors preferred), or None."""
    base = rife_dir()
    for cand in (base / f"flownet_v{variant}.safetensors", base / f"flownet_v{variant}.pkl"):
        if cand.is_file() and cand.stat().st_size > 1_000_000:
            return cand
    return None


def _load_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    import torch

    # weights_only refuses arbitrary pickled objects (plain tensors only).
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k.replace("module.", ""): v for k, v in sd.items() if k.startswith("module.")}


class RifeBackend(FrameGenerationBackend):
    name = "rife"

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        *,
        device: str = "cuda",
        precision: str = "fp16",
        flow_scale: float | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown RIFE variant {variant!r}; choose from {sorted(VARIANTS)}")
        self.variant = VARIANTS[variant]
        self.precision = precision
        self._device_str = device
        self._flow_scale = flow_scale
        self._net = None
        self._head = None
        self._stream = None
        self._prev = None  # (tensor, features)
        self._cur = None
        self._w = self._h = 0

    @property
    def capabilities(self) -> FrameGenCapabilities:
        return FrameGenCapabilities(
            name=self.name,
            variant=self.variant.name,
            max_factor=4,
            arbitrary_timestep=True,
            device="cuda",
            precision=self.precision,
            license="MIT (Practical-RIFE code+weights, hzwer; vs-rife refactor, HolyWu)",
            paper="Huang et al., ECCV 2022 — arXiv:2011.06294",
            source="https://github.com/hzwer/Practical-RIFE",
        )

    @property
    def ready(self) -> bool:
        return self._prev is not None and self._cur is not None

    def initialize(self, width: int, height: int) -> None:
        try:
            import torch
        except ImportError as e:  # pragma: no cover - depends on host
            raise FrameGenUnavailable("PyTorch is required for RIFE frame generation") from e
        if not torch.cuda.is_available() or not self._device_str.startswith("cuda"):
            raise FrameGenUnavailable("RIFE frame generation needs a CUDA GPU (CPU would add, not hide, latency)")
        path = weights_path(self.variant.name)
        if path is None:
            raise FrameGenUnavailable(
                f"RIFE {self.variant.name} weights not installed. Run: deepfake models install rife --yes"
            )
        from .rife_arch import Head, IFNet

        self._torch = torch
        self._dev = torch.device(self._device_str)
        self._dtype = torch.float16 if self.precision == "fp16" else torch.float32
        # From 1080p up, estimate flow at half resolution (upstream's advice for large
        # frames): measured ~13 → ~8 ms per generated 1080p frame on an RTX 4090.
        scale = self._flow_scale if self._flow_scale is not None else (0.5 if width * height >= 1920 * 1080 else 1.0)
        self._scale = scale
        sd = _load_state_dict(path)
        net = IFNet(self.variant, scale=scale)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        head_sd = {k[len("encode.") :]: v for k, v in sd.items() if k.startswith("encode.")}
        head = Head()
        head.load_state_dict(head_sd)
        bad = [k for k in missing if not k.startswith("encode.")]
        if bad:
            raise FrameGenUnavailable(f"RIFE weights do not match variant {self.variant.name}: missing {bad[:4]}")
        self._net = net.eval().to(self._dev, self._dtype)
        self._head = head.eval().to(self._dev, self._dtype)
        self._stream = torch.cuda.Stream(self._dev)
        self._configure(width, height)
        self.warmup()

    def warmup(self, factor: int = 3) -> None:
        """Run the kernels once so cuDNN/allocator setup does not land on live frames."""
        blank = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        self.push(blank)
        self.push(blank)
        self.interpolate(timesteps_for(factor))
        self._prev = self._cur = None

    def _configure(self, width: int, height: int) -> None:
        torch = self._torch
        self._w, self._h = width, height
        mod = max(self.variant.modulo, int(self.variant.modulo / self._scale))
        self._pw = math.ceil(width / mod) * mod
        self._ph = math.ceil(height / mod) * mod
        pw, ph = self._pw, self._ph
        with torch.cuda.stream(self._stream):
            self._flow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], device=self._dev)
            gh = torch.linspace(-1.0, 1.0, pw, device=self._dev).view(1, 1, 1, pw).expand(-1, -1, ph, -1)
            gv = torch.linspace(-1.0, 1.0, ph, device=self._dev).view(1, 1, ph, 1).expand(-1, -1, -1, pw)
            self._grid = torch.cat([gh, gv], 1)
        # Pinned staging buffers make host<->device copies asynchronous.
        self._in_pinned = torch.empty((height, width, 3), dtype=torch.uint8).pin_memory()
        self._out_pinned: list = []
        self._prev = self._cur = None

    def _upload(self, frame_bgr: np.ndarray):
        torch = self._torch
        F = torch.nn.functional
        self._in_pinned.numpy()[...] = frame_bgr
        x = self._in_pinned.to(self._dev, non_blocking=True)
        # BGR→RGB is irrelevant to RIFE (channel-symmetric), keep BGR end to end.
        x = x.permute(2, 0, 1).unsqueeze(0).to(self._dtype).div_(255.0)
        return F.pad(x, (0, self._pw - self._w, 0, self._ph - self._h))

    def push(self, frame_bgr: np.ndarray) -> None:
        if self._net is None:
            raise RuntimeError("initialize() first")
        torch = self._torch
        h, w = frame_bgr.shape[:2]
        if (w, h) != (self._w, self._h):
            self._configure(w, h)
        with torch.inference_mode(), torch.cuda.stream(self._stream):
            img = self._upload(np.ascontiguousarray(frame_bgr))
            feat = self._head(img)
        self._stream.synchronize()  # the pinned input buffer is reused on the next push
        self._prev, self._cur = self._cur, (img, feat)

    def interpolate(self, timesteps: list[float]) -> list[np.ndarray]:
        if not self.ready:
            return []
        torch = self._torch
        (img0, f0), (img1, f1) = self._prev, self._cur
        while len(self._out_pinned) < len(timesteps):
            self._out_pinned.append(torch.empty((self._h, self._w, 3), dtype=torch.uint8).pin_memory())
        with torch.inference_mode(), torch.cuda.stream(self._stream):
            for i, t in enumerate(timesteps):
                ts = torch.full((1, 1, self._ph, self._pw), float(t), dtype=self._dtype, device=self._dev)
                out = self._net(img0, img1, ts, self._flow_div, self._grid, f0, f1)[:, :, : self._h, : self._w]
                u8 = out.clamp_(0, 1).mul_(255).round_().to(torch.uint8)[0].permute(1, 2, 0)
                self._out_pinned[i].copy_(u8, non_blocking=True)
        self._stream.synchronize()
        return [self._out_pinned[i].numpy().copy() for i in range(len(timesteps))]

    def benchmark(self, width: int, height: int, factor: int, iterations: int = 60) -> FrameGenBenchmark:
        torch = self._torch
        if self._net is None:
            self.initialize(width, height)
        rng = np.random.default_rng(0)
        base = (rng.random((height + 16, width + 16, 3)) * 255).astype(np.uint8)
        frames = [np.ascontiguousarray(base[i : i + height, i : i + width]) for i in range(8)]
        steps = timesteps_for(factor)
        for i in range(6):  # warm-up (cuDNN algorithm selection, allocator)
            self.push(frames[i % 8])
            self.interpolate(steps)
        torch.cuda.reset_peak_memory_stats(self._dev)
        lat = []
        t_all = time.perf_counter()
        for i in range(iterations):
            t0 = time.perf_counter()
            self.push(frames[i % 8])
            self.interpolate(steps)
            lat.append((time.perf_counter() - t0) * 1000.0)
        total = time.perf_counter() - t_all
        lat.sort()
        p50 = lat[len(lat) // 2]
        n_gen = max(1, len(steps))
        return FrameGenBenchmark(
            backend=self.name,
            variant=self.variant.name,
            width=width,
            height=height,
            factor=factor,
            ms_per_generated_frame=p50 / n_gen,
            ms_per_source_interval=p50,
            generated_fps_capacity=iterations * n_gen / total,
            peak_vram_mb=torch.cuda.max_memory_allocated(self._dev) / 2**20,
            samples=iterations,
            extra={"ms_p95": lat[int(len(lat) * 0.95)], "flow_scale": self._scale},
        )

    def shutdown(self) -> None:
        self._prev = self._cur = None
        self._net = self._head = None
        self._out_pinned = []
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass
