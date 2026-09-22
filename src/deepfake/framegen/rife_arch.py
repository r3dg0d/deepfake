"""RIFE IFNet (HDv3, v4.25 / v4.25-lite / v4.26) network definition.

Vendored from Practical-RIFE (https://github.com/hzwer/Practical-RIFE,
MIT, Copyright (c) 2021 hzwer) via vs-rife
(https://github.com/HolyWu/vs-rife, MIT, Copyright (c) 2021 HolyWu), which
refactored the forward pass to take the backwarp grid and per-frame encoder
features as explicit inputs. The three variants share one architecture and
differ only in the last block width, the pyramid scales and the weights, so
they are expressed here as one parameterised class. See NOTICE.

Paper: Huang et al., "Real-Time Intermediate Flow Estimation for Video Frame
Interpolation", ECCV 2022, arXiv:2011.06294.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .variants import VARIANTS, RifeVariant  # noqa: F401  (re-exported)


def warp(img: torch.Tensor, flow: torch.Tensor, flow_div: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    dtype = img.dtype
    flow = flow.float()
    flow = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    out = F.grid_sample(img.float(), g, mode="bilinear", padding_mode="border", align_corners=True)
    return out.to(dtype)


def _conv(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=True), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    """Per-frame feature encoder; run once per source frame and cached."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, cin: int, c: int) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(_conv(cin, c // 2, 3, 2, 1), _conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor, flow: torch.Tensor | None, scale: float):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    def __init__(self, variant: RifeVariant, scale: float = 1.0) -> None:
        super().__init__()
        c_in = 8 + 4 + 8 + 8
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(c_in, c=128)
        self.block2 = IFBlock(c_in, c=96)
        self.block3 = IFBlock(c_in, c=64)
        self.block4 = IFBlock(c_in, c=variant.last_block_width)
        self.encode = Head()
        self.scale_list = [p / scale for p in variant.pyramid]

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: torch.Tensor,
        flow_div: torch.Tensor,
        grid: torch.Tensor,
        f0: torch.Tensor,
        f1: torch.Tensor,
    ) -> torch.Tensor:
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        warped0, warped1 = img0, img1
        flow = mask = feat = None
        blocks = (self.block0, self.block1, self.block2, self.block3, self.block4)
        for i, block in enumerate(blocks):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, self.scale_list[i])
            else:
                wf0 = warp(f0, flow[:, :2], flow_div, grid)
                wf1 = warp(f1, flow[:, 2:4], flow_div, grid)
                fd, mask, feat = block(
                    torch.cat((warped0, warped1, wf0, wf1, timestep, mask, feat), 1), flow, self.scale_list[i]
                )
                flow = flow + fd
            warped0 = warp(img0, flow[:, :2], flow_div, grid)
            warped1 = warp(img1, flow[:, 2:4], flow_div, grid)
        m = torch.sigmoid(mask)
        return warped0 * m + warped1 * (1 - m)
